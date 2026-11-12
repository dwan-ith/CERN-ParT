"""Unit tests for the feature extractor.

These use synthetic awkward arrays, so they run offline and fast.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import awkward as ak
except ImportError:  # pragma: no cover
    ak = None

from data.feature_extractor import (
    N_FEATURES,
    PARTICLE_TYPES,
    binarize_btag,
    extract_batch,
    find_btag_branch,
    get_required_branches,
    has_pf_candidates,
)

pytestmark = pytest.mark.skipif(ak is None, reason="awkward is required")


def _make_arrays(n_events=3, seed=0):
    """Build a small synthetic NanoAOD-like mapping of branches."""
    rng = np.random.default_rng(seed)

    def jagged(counts, lo=1.0, hi=100.0, size=None):
        out = []
        for n in counts:
            out.append(rng.uniform(lo, hi, size=n) if n else np.zeros(0))
        return ak.Array(out)

    counts = rng.integers(0, 6, size=n_events)
    arrays = {}
    for name, cfg in PARTICLE_TYPES.items():
        arrays[cfg["n_branch"]] = ak.Array(list(counts))
        arrays[cfg["pt"]] = jagged(list(counts))
        arrays[cfg["eta"]] = jagged(list(counts), lo=-3, hi=3)
        arrays[cfg["phi"]] = jagged(list(counts), lo=-np.pi, hi=np.pi)
        if cfg.get("mass"):
            arrays[cfg["mass"]] = jagged(list(counts), lo=0.0, hi=5.0)
        if cfg.get("charge"):
            arrays[cfg["charge"]] = ak.Array(
                [rng.choice([-1.0, 1.0], size=n) for n in counts]
            )
    arrays["Jet_btagDeepFlavB"] = jagged(list(counts), lo=0.0, hi=1.0)
    return arrays


def test_all_channels_are_initialised():
    """Every raw channel must carry a real value, not uninitialised memory.

    Regression test: an earlier version allocated the per-particle block with
    ``np.empty`` for a 7-wide vector while only filling 6 channels, which
    leaked arbitrary garbage into the tensor (and overflowed the float32
    cast).
    """
    arrays = _make_arrays(n_events=6)
    batch = extract_batch(arrays, max_particles=32)
    particles = batch["particles"]
    assert np.all(np.isfinite(particles)), "features must be finite"
    for i in range(6):
        n = int(batch["mask"][i].sum())
        assert np.all(np.isfinite(particles[i, :n]))
    # The type channel must always be a valid type id.
    valid = set(range(6))
    types = set(np.unique(particles[batch["mask"] > 0, 5]).tolist())
    assert types <= valid


def test_extract_batch_shapes():
    arrays = _make_arrays(n_events=4)
    batch = extract_batch(arrays, max_particles=16)
    assert batch["particles"].shape == (4, 16, N_FEATURES)
    assert batch["mask"].shape == (4, 16)
    assert batch["is_jet"].shape == (4, 16)
    assert batch["btag"].shape == (4, 16)
    assert batch["particles"].dtype == np.float32


def test_mask_matches_particle_count():
    arrays = _make_arrays(n_events=5)
    batch = extract_batch(arrays, max_particles=32)
    for i in range(5):
        n = batch["n_particles"][i]
        assert batch["mask"][i].sum() == min(n, 32)
        # Padding must be exactly zero.
        assert np.all(batch["particles"][i, int(batch["mask"][i].sum()):] == 0.0)


def test_particles_sorted_by_pt():
    arrays = _make_arrays(n_events=3)
    batch = extract_batch(arrays, max_particles=64, sort_by_pt=True)
    for i in range(3):
        n = int(batch["mask"][i].sum())
        pts = batch["particles"][i, :n, 0]
        assert np.all(np.diff(pts) <= 1e-6), "particles must be sorted by decreasing pT"


def test_truncation_keeps_highest_pt():
    arrays = _make_arrays(n_events=3)
    batch = extract_batch(arrays, max_particles=2)
    for i in range(3):
        n = int(batch["mask"][i].sum())
        if n > 2:
            kept = batch["particles"][i, :2, 0]
            everything = batch["particles"][i, :n, 0]
            assert kept.max() <= everything.max() + 1e-6
            assert np.allclose(np.sort(kept)[::-1], np.sort(everything)[::-1][:2])


def test_jet_flags_and_btag_align():
    arrays = _make_arrays(n_events=3)
    batch = extract_batch(arrays, max_particles=64)
    for i in range(3):
        n = int(batch["mask"][i].sum())
        types = batch["particles"][i, :n, 5]
        is_jet = batch["is_jet"][i, :n]
        assert np.all((types == 3.0) == is_jet)


def test_binarize_btag_uses_working_point():
    scores = np.array([0.0, 0.10, 0.16, 0.17, 0.90])
    labels = binarize_btag(scores)
    # medium WP: cut 0.1530 + slack 0.0160 -> threshold 0.169
    assert labels.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_branch_helpers():
    branches = get_required_branches()
    assert "Muon_pt" in branches
    assert "nJet" in branches
    assert find_btag_branch(["Jet_btagDeepFlavB"]) == "Jet_btagDeepFlavB"
    assert find_btag_branch(["nothing"]) is None
    assert has_pf_candidates(["nPackedPFCands", "PackedPFCand_pt"]) is False
    assert (
        has_pf_candidates(
            [
                "nPackedPFCands",
                "PackedPFCand_pt",
                "PackedPFCand_eta",
                "PackedPFCand_phi",
            ]
        )
        is True
    )


def test_empty_event_is_all_padding():
    arrays = _make_arrays(n_events=2)
    arrays["nMuon"] = ak.Array([0, 0])
    arrays["nElectron"] = ak.Array([0, 0])
    arrays["nPhoton"] = ak.Array([0, 0])
    arrays["nJet"] = ak.Array([0, 0])
    arrays["nTau"] = ak.Array([0, 0])
    arrays["Muon_pt"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Muon_eta"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Muon_phi"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Electron_pt"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Electron_eta"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Electron_phi"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Photon_pt"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Photon_eta"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Photon_phi"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Jet_pt"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Jet_eta"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Jet_phi"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Tau_pt"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Tau_eta"] = ak.Array([np.zeros(0), np.zeros(0)])
    arrays["Tau_phi"] = ak.Array([np.zeros(0), np.zeros(0)])
    batch = extract_batch(arrays, max_particles=8)
    assert batch["mask"].sum() == 0
    assert batch["n_particles"].tolist() == [0, 0]


def test_pf_candidates_are_included_when_present():
    arrays = _make_arrays(n_events=2)
    arrays["nPackedPFCands"] = ak.Array([3, 4])
    arrays["PackedPFCand_pt"] = ak.Array([np.ones(3) * 5, np.ones(4) * 7])
    arrays["PackedPFCand_eta"] = ak.Array([np.zeros(3), np.zeros(4)])
    arrays["PackedPFCand_phi"] = ak.Array([np.zeros(3), np.zeros(4)])
    arrays["PackedPFCand_mass"] = ak.Array([np.zeros(3), np.zeros(4)])
    arrays["PackedPFCand_charge"] = ak.Array([np.ones(3), -np.ones(4)])
    batch = extract_batch(arrays, max_particles=64, use_pf_candidates=True)
    types = batch["particles"][batch["mask"] > 0, 5]
    assert (types == 5).sum() == 7
