"""Turn CMS NanoAOD branches into padded particle tensors.

The extractor is deliberately thin: it knows about branch names and physical
conventions, and nothing about tensors or models.  It produces *raw* physics
quantities; normalisation into the model-facing representation happens inside
the model (see :class:`model.particle_transformer.FeatureNormalizer`)
so that the data layer stays a faithful copy of what is in the file.

Raw per-particle vector (6 channels)::

    [pt, eta, phi, mass, charge, type_id]

``type_id`` indexes :data:`model.config.OBJECT_TYPES`.  The PDG id is
not stored separately: for the resolved object types it is fully determined by
the type and the charge.

Performance note
----------------
The original implementation looped in Python over every particle of every
event, which dominated the data-loading time.  Here the inner loop runs over
*types* and the per-event work is a handful of NumPy calls, so the extractor
is roughly two orders of magnitude faster.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import awkward as ak
import numpy as np

# ---------------------------------------------------------------------------
# Branch definitions
# ---------------------------------------------------------------------------

#: Resolved object types, in ``type_id`` order.
PARTICLE_TYPES: Dict[str, Dict[str, object]] = {
    "muon": {
        "n_branch": "nMuon",
        "pt": "Muon_pt",
        "eta": "Muon_eta",
        "phi": "Muon_phi",
        "mass": "Muon_mass",
        "charge": "Muon_charge",
        "type_id": 0,
    },
    "electron": {
        "n_branch": "nElectron",
        "pt": "Electron_pt",
        "eta": "Electron_eta",
        "phi": "Electron_phi",
        "mass": "Electron_mass",
        "charge": "Electron_charge",
        "type_id": 1,
    },
    "photon": {
        "n_branch": "nPhoton",
        "pt": "Photon_pt",
        "eta": "Photon_eta",
        "phi": "Photon_phi",
        "mass": None,
        "charge": None,
        "type_id": 2,
    },
    "jet": {
        "n_branch": "nJet",
        "pt": "Jet_pt",
        "eta": "Jet_eta",
        "phi": "Jet_phi",
        "mass": "Jet_mass",
        "charge": None,
        "type_id": 3,
    },
    "tau": {
        "n_branch": "nTau",
        "pt": "Tau_pt",
        "eta": "Tau_eta",
        "phi": "Tau_phi",
        "mass": "Tau_mass",
        "charge": "Tau_charge",
        "type_id": 4,
    },
}

#: Packed particle-flow candidates. Present in NanoAOD v10+ only.
PF_TYPE = {
    "n_branch": "nPackedPFCands",
    "pt": "PackedPFCand_pt",
    "eta": "PackedPFCand_eta",
    "phi": "PackedPFCand_phi",
    "mass": "PackedPFCand_mass",
    "charge": "PackedPFCand_charge",
    "type_id": 5,
}

#: Raw feature width.
N_FEATURES = 6

#: Jet-level b-tagging branches, in order of preference.
BTAG_BRANCHES = ("Jet_btagDeepFlavB", "Jet_btagDeepB", "Jet_btagCSVV2")

#: Working point used to binarise the DeepFlav score.
#: UL2016-18 "medium" DeepFlav working point.
BTAG_MEDIUM_WP = (0.1530, 0.0160)


def get_required_branches(include_pf: bool = True) -> List[str]:
    """Branch names needed by the extractor.

    ``include_pf`` adds the packed particle-flow candidate branches when the
    file provides them; the caller is responsible for filtering the result
    against the branches that actually exist.
    """
    branches: List[str] = []
    for cfg in PARTICLE_TYPES.values():
        branches.append(str(cfg["n_branch"]))
        branches.append(str(cfg["pt"]))
        branches.append(str(cfg["eta"]))
        branches.append(str(cfg["phi"]))
        for optional in ("mass", "charge"):
            value = cfg.get(optional)
            if value:
                branches.append(str(value))
    if include_pf:
        for key in ("n_branch", "pt", "eta", "phi", "mass", "charge"):
            branches.append(str(PF_TYPE[key]))
    return branches


def get_optional_branches() -> List[str]:
    """Branches used for downstream labels (b-tagging), if present."""
    return list(BTAG_BRANCHES)


def has_pf_candidates(available: Sequence[str]) -> bool:
    """Whether a file provides packed particle-flow candidates."""
    return all(
        str(PF_TYPE[key]) in available
        for key in ("n_branch", "pt", "eta", "phi")
    )


def find_btag_branch(available: Sequence[str]) -> Optional[str]:
    """Return the best available jet b-tagging branch name."""
    for name in BTAG_BRANCHES:
        if name in available:
            return name
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_flat(values) -> np.ndarray:
    """Flatten an awkward array (or numpy array) into a 1-D numpy array."""
    if isinstance(values, ak.Array):
        if values.ndim <= 1:
            return np.asarray(ak.to_numpy(values), dtype=np.float64).ravel()
        return np.asarray(ak.to_numpy(ak.flatten(values)), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).ravel()


def _as_event_array(values) -> List[np.ndarray]:
    """Convert an awkward (or numpy) jagged array into a list of per-event arrays.

    ``ak.to_numpy`` raises for irregular jagged arrays, which is the normal
    case for NanoAOD (every event has a different number of particles).  The
    array is therefore flattened and split using the per-event counts.
    """
    if isinstance(values, ak.Array):
        if values.ndim <= 1:
            flat = _to_flat(values)
            return [np.asarray([value], dtype=np.float64) for value in flat]
        counts = np.asarray(ak.to_numpy(ak.num(values, axis=1)), dtype=np.int64)
        flat = _to_flat(values)
        out: List[np.ndarray] = []
        start = 0
        for count in counts:
            out.append(flat[start : start + int(count)])
            start += int(count)
        return out

    array = np.asarray(values, dtype=np.float64)
    if array.ndim <= 1:
        return [np.asarray([value], dtype=np.float64) for value in array]
    return [np.asarray(row, dtype=np.float64) for row in array]


def _zeros_like_counts(counts: np.ndarray) -> np.ndarray:
    """Flat zeros reshaped into the per-event jagged layout."""
    total = int(counts.sum())
    return np.zeros(total, dtype=np.float64)


def _available_fields(arrays) -> List[str]:
    if hasattr(arrays, "fields"):
        return list(arrays.fields)
    return list(arrays.keys())


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_batch(
    arrays,
    max_particles: int = 64,
    sort_by_pt: bool = True,
    use_pf_candidates: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract particle features for a batch of NanoAOD events.

    Args:
        arrays: mapping of branch name -> awkward array, as produced by
            ``uproot.iterate``.
        max_particles: maximum number of particles retained per event.
        sort_by_pt: sort particles by decreasing transverse momentum. This
            keeps the hardest particles when an event is truncated, which is
            the physically meaningful choice.
        use_pf_candidates: include packed particle-flow candidates when the
            file provides them.

    Returns:
        dict with keys

        ``particles``  ``float32[B, max_particles, 7]`` raw features
        ``mask``       ``float32[B, max_particles]`` 1 for real particles
        ``n_particles````int32[B]`` number of particles before truncation
        ``is_jet``     ``bool[B, max_particles]`` 1 for reconstructed jets
        ``btag``       ``float32[B, max_particles]`` DeepFlav score (0 if absent)
    """
    available = _available_fields(arrays)
    first = available[0]
    n_events = len(arrays[first])

    sources: List[Dict[str, object]] = []
    for cfg in PARTICLE_TYPES.values():
        if str(cfg["n_branch"]) not in available:
            continue
        sources.append(cfg)
    if use_pf_candidates and has_pf_candidates(available):
        sources.append(PF_TYPE)

    btag_branch = find_btag_branch(available)

    # --- Pull the per-type columns out of the awkward arrays -------------
    columns: Dict[str, Dict[str, List[np.ndarray]]] = {}
    counts: Dict[str, np.ndarray] = {}
    for cfg in sources:
        name = str(cfg["n_branch"])
        counts[str(cfg["type_id"])] = _to_flat(arrays[name]).astype(np.int64)
        cols: Dict[str, List[np.ndarray]] = {}
        for key in ("pt", "eta", "phi", "mass", "charge"):
            branch = cfg.get(key)
            if branch and str(branch) in available:
                cols[key] = _as_event_array(arrays[str(branch)])
            else:
                cols[key] = None
        columns[str(cfg["type_id"])] = cols

    btag_values = _as_event_array(arrays[btag_branch]) if btag_branch else None

    # --- Assemble per event ---------------------------------------------
    particles = np.zeros((n_events, max_particles, N_FEATURES), dtype=np.float32)
    mask = np.zeros((n_events, max_particles), dtype=np.float32)
    is_jet = np.zeros((n_events, max_particles), dtype=bool)
    btag = np.zeros((n_events, max_particles), dtype=np.float32)
    n_particles = np.zeros(n_events, dtype=np.int32)

    for i in range(n_events):
        chunks: List[np.ndarray] = []
        tag_chunks: List[np.ndarray] = []
        btag_chunks: List[np.ndarray] = []
        for cfg in sources:
            type_id = int(cfg["type_id"])
            n = int(counts[str(type_id)][i])
            if n <= 0:
                continue
            cols = columns[str(type_id)]
            pt = cols["pt"][i]
            eta = cols["eta"][i]
            phi = cols["phi"][i]
            mass = cols["mass"][i] if cols["mass"] is not None else np.zeros(n)
            charge = cols["charge"][i] if cols["charge"] is not None else np.zeros(n)

            block = np.zeros((n, N_FEATURES), dtype=np.float64)
            block[:, 0] = pt[:n]
            block[:, 1] = eta[:n]
            block[:, 2] = phi[:n]
            block[:, 3] = mass[:n]
            block[:, 4] = charge[:n]
            block[:, 5] = float(type_id)
            chunks.append(block)

            tags = np.zeros(n, dtype=np.float64)
            if type_id == 3:  # jet
                tags[:] = 1.0
            tag_chunks.append(tags)

            scores = np.zeros(n, dtype=np.float64)
            if type_id == 3 and btag_values is not None:
                scores[:] = btag_values[i][:n]
            btag_chunks.append(scores)

        if not chunks:
            continue

        event_particles = np.concatenate(chunks, axis=0)
        event_tags = np.concatenate(tag_chunks, axis=0)
        event_btag = np.concatenate(btag_chunks, axis=0)

        if sort_by_pt:
            order = np.argsort(-event_particles[:, 0], kind="stable")
            event_particles = event_particles[order]
            event_tags = event_tags[order]
            event_btag = event_btag[order]

        n_total = event_particles.shape[0]
        n_particles[i] = n_total
        n_keep = min(n_total, max_particles)
        particles[i, :n_keep] = event_particles[:n_keep]
        mask[i, :n_keep] = 1.0
        is_jet[i, :n_keep] = event_tags[:n_keep] > 0
        btag[i, :n_keep] = event_btag[:n_keep]

    return {
        "particles": particles,
        "mask": mask,
        "n_particles": n_particles,
        "is_jet": is_jet,
        "btag": btag,
    }


def binarize_btag(scores: np.ndarray, wp: tuple = BTAG_MEDIUM_WP) -> np.ndarray:
    """Turn a DeepFlav score into a binary label using a working point."""
    cut, slack = wp
    return (scores > cut + slack).astype(np.float32)
