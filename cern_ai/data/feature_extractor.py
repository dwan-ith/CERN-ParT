"""
CMS NanoAOD Feature Extractor
Converts uproot awkward arrays into padded particle feature tensors.
Target: RTX 4050 6GB — all operations are lightweight numpy.
"""
import numpy as np

# Branch name mapping for CMS NanoAOD Run 2 (2016-2018)
PARTICLE_TYPES = {
    "Muon": {
        "n_branch": "nMuon",
        "pt": "Muon_pt",
        "eta": "Muon_eta",
        "phi": "Muon_phi",
        "mass": "Muon_mass",
        "charge": "Muon_charge",
        "pdg_id": 13,
        "type_id": 0,
    },
    "Electron": {
        "n_branch": "nElectron",
        "pt": "Electron_pt",
        "eta": "Electron_eta",
        "phi": "Electron_phi",
        "mass": "Electron_mass",
        "charge": "Electron_charge",
        "pdg_id": 11,
        "type_id": 1,
    },
    "Photon": {
        "n_branch": "nPhoton",
        "pt": "Photon_pt",
        "eta": "Photon_eta",
        "phi": "Photon_phi",
        "mass": None,
        "charge": None,
        "pdg_id": 22,
        "type_id": 2,
    },
    "Jet": {
        "n_branch": "nJet",
        "pt": "Jet_pt",
        "eta": "Jet_eta",
        "phi": "Jet_phi",
        "mass": "Jet_mass",
        "charge": None,
        "pdg_id": 0,
        "type_id": 3,
    },
    "Tau": {
        "n_branch": "nTau",
        "pt": "Tau_pt",
        "eta": "Tau_eta",
        "phi": "Tau_phi",
        "mass": "Tau_mass",
        "charge": None,
        "pdg_id": 15,
        "type_id": 4,
    },
}

N_FEATURES = 7  # [pt, eta, phi, mass, charge, pdg_id, type_id]


def get_required_branches():
    """Return list of branch names needed for feature extraction."""
    branches = []
    for cfg in PARTICLE_TYPES.values():
        branches.append(cfg["n_branch"])
        branches.append(cfg["pt"])
        branches.append(cfg["eta"])
        branches.append(cfg["phi"])
        if cfg["mass"]:
            branches.append(cfg["mass"])
        if cfg["charge"]:
            branches.append(cfg["charge"])
    return branches


def extract_batch(arrays, max_particles=64):
    """
    Extract particle features from uproot awkward arrays.

    Args:
        arrays: dict of awkward arrays from uproot.iterate()
        max_particles: maximum number of particles per event

    Returns:
        dict with keys:
            particles: np.ndarray [B, max_particles, 7]
            mask: np.ndarray [B, max_particles]
            n_particles: np.ndarray [B]
    """
    # Handle both dict and awkward.Array
    if hasattr(arrays, 'fields'):
        first_field = arrays.fields[0]
    else:
        first_field = list(arrays.keys())[0]
    n_events = len(arrays[first_field])
    batch_particles = []
    batch_mask = []
    batch_counts = []

    for i in range(n_events):
        particles = _extract_single_event(arrays, i)
        n = len(particles)

        result = np.zeros((max_particles, N_FEATURES), dtype=np.float32)
        mask = np.zeros(max_particles, dtype=np.float32)

        if n > 0:
            n_keep = min(n, max_particles)
            result[:n_keep] = np.array(particles[:n_keep], dtype=np.float32)
            mask[:n_keep] = 1.0

        batch_particles.append(result)
        batch_mask.append(mask)
        batch_counts.append(n)

    return {
        "particles": np.stack(batch_particles),
        "mask": np.stack(batch_mask),
        "n_particles": np.array(batch_counts, dtype=np.int32),
    }


def _extract_single_event(arrays, event_idx):
    """Extract all particles from a single event as a list of [7]-feature vectors."""
    particles = []

    # Get available fields
    if hasattr(arrays, 'fields'):
        available = set(arrays.fields)
    else:
        available = set(arrays.keys())

    for ptype, cfg in PARTICLE_TYPES.items():
        n_branch = cfg["n_branch"]
        if n_branch not in available:
            continue

        n = int(arrays[n_branch][event_idx])
        if n == 0:
            continue

        pts = np.asarray(arrays[cfg["pt"]][event_idx], dtype=np.float32)
        etas = np.asarray(arrays[cfg["eta"]][event_idx], dtype=np.float32)
        phis = np.asarray(arrays[cfg["phi"]][event_idx], dtype=np.float32)

        masses = (
            np.asarray(arrays[cfg["mass"]][event_idx], dtype=np.float32)
            if cfg["mass"]
            else np.zeros(n, dtype=np.float32)
        )
        charges = (
            np.asarray(arrays[cfg["charge"]][event_idx], dtype=np.float32)
            if cfg["charge"]
            else np.zeros(n, dtype=np.float32)
        )

        pdg_base = float(cfg["pdg_id"])
        type_id = float(cfg["type_id"])

        for j in range(n):
            pt = float(pts[j])
            eta = float(etas[j])
            phi = float(phis[j])
            mass = float(masses[j])
            charge = float(charges[j])

            if cfg["charge"] and charge != 0:
                pdg = pdg_base * np.sign(charge)
            else:
                pdg = pdg_base

            particles.append([pt, eta, phi, mass, charge, pdg, type_id])

    if not particles:
        return np.zeros((0, N_FEATURES), dtype=np.float32).tolist()

    arr = np.array(particles, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr.tolist()
