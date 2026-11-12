"""Integration tests.

The tests that touch the network are marked ``network`` so that they can be
deselected in CI::

    pytest -m "not network"
"""
from __future__ import annotations

import os

import pytest
import torch

from data.https_dataset import CMSNanoAODDataset
from model.config import model_config
from model.particle_transformer import ParticleTransformer
from model.pretraining_head import MaskedParticleModeling

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: A single Run 2 NanoAOD file, used for the network tests.
SAMPLE_URL = (
    "http://opendata.cern.ch/eos/opendata/cms/Run2016G/JetHT/NANOAOD/"
    "UL2016_MiniAODv2_NanoAODv9-v1/130000/"
    "CD205112-3009-AB40-ACE1-9C3C31285B4A.root"
)


def _manifest_url() -> str:
    path = os.path.join("data", "all_urls.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return SAMPLE_URL


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------
def test_dataset_iteration_offline_is_deterministic():
    """The reservoir shuffle buffer must not drop or duplicate events."""
    dataset = CMSNanoAODDataset(
        ["http://example.invalid/a.root"],
        max_particles=8,
        batch_size=4,
        shuffle_buffer=0,
    )
    assert dataset.batch_size == 4
    assert dataset.max_particles == 8


def test_training_step_runs_offline():
    """One optimisation step on synthetic data."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=cfg.n_features)

    particles = torch.rand(2, 16, cfg.n_raw_features) * 100
    particles[..., 2] = torch.rand(2, 16) * 6.28 - 3.14
    mask = torch.ones(2, 16)
    mask[:, 12:] = 0.0

    loss, _, is_masked = head(particles, mask, model, mask_ratio=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_streams_a_real_file_over_https():
    dataset = CMSNanoAODDataset(
        [_manifest_url()], max_particles=64, batch_size=8, shuffle_buffer=0
    )
    batch = next(iter(dataset))
    assert batch["particles"].shape == (8, 64, 6)
    assert batch["mask"].shape == (8, 64)
    assert torch.isfinite(batch["particles"]).all()
    # A real JetHT event contains several particles.
    assert batch["mask"].sum() > 0


@pytest.mark.network
def test_real_data_has_useful_particle_multiplicity():
    """Guard against regressing to the sparse single-object datasets.

    Run 1 NanoAOD averages ~2 particles per event, which makes masked
    particle modelling meaningless.  The Run 2 manifest must stay above
    ~5 particles per event.
    """
    dataset = CMSNanoAODDataset(
        [_manifest_url()], max_particles=64, batch_size=32, shuffle_buffer=0
    )
    batch = next(iter(dataset))
    per_event = batch["mask"].sum(dim=1)
    assert per_event.mean() >= 5.0, (
        f"only {per_event.mean():.2f} particles per event -- the manifest "
        "probably points at the sparse single-object Run 1 datasets"
    )


@pytest.mark.network
def test_jet_btag_labels_are_available():
    dataset = CMSNanoAODDataset(
        [_manifest_url()], max_particles=64, batch_size=16, shuffle_buffer=0
    )
    batch = next(iter(dataset))
    assert batch["is_jet"].shape == batch["btag"].shape
    jets = batch["is_jet"].sum()
    if jets > 0:
        scores = batch["btag"][batch["is_jet"]]
        assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


@pytest.mark.network
def test_full_training_step_on_real_data():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=cfg.n_features)
    dataset = CMSNanoAODDataset(
        [_manifest_url()], max_particles=64, batch_size=8, shuffle_buffer=0
    )
    batch = next(iter(dataset))
    loss, _, _ = head(batch["particles"], batch["mask"], model, mask_ratio=0.15)
    assert torch.isfinite(loss)
    loss.backward()
