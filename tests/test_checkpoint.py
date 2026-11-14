"""Tests for checkpointing and resumption."""
from __future__ import annotations

import os

import torch

from checkpoint import latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
from model.config import model_config
from model.particle_transformer import ParticleTransformer
from model.pretraining_head import MaskedParticleModeling


def _build():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=cfg.n_features)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-3
    )
    return cfg, model, head, optimizer


def test_roundtrip_restores_weights(tmp_path):
    cfg, model, head, optimizer = _build()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        path,
        step=42,
        model=model,
        head=head,
        optimizer=optimizer,
        loss=0.5,
        model_config=cfg.to_dict(),
        training_config={"lr": 1e-3},
    )
    assert os.path.exists(path)

    _, model2, head2, optimizer2 = _build()
    payload = load_checkpoint(path, model=model2, head=head2, optimizer=optimizer2)
    assert payload["step"] == 42
    assert payload["loss"] == 0.5
    assert payload["model_config"]["d_model"] == cfg.d_model

    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(head.parameters(), head2.parameters()):
        assert torch.allclose(a, b)


def test_optimizer_state_is_restored(tmp_path):
    cfg, model, head, optimizer = _build()
    path = str(tmp_path / "ckpt.pt")
    # Take a step so the optimizer actually has state.
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    optimizer.step()
    save_checkpoint(path, step=1, model=model, head=head, optimizer=optimizer)

    _, model2, head2, optimizer2 = _build()
    load_checkpoint(path, model=model2, head=head2, optimizer=optimizer2)
    assert len(optimizer2.state_dict()["state"]) > 0


def test_optimizer_can_be_omitted(tmp_path):
    cfg, model, head, optimizer = _build()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, step=1, model=model, head=head, optimizer=optimizer,
                    save_optimizer=False)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert "model" in payload


def test_checkpoint_is_written_atomically(tmp_path):
    """No partially written checkpoint must be left behind."""
    cfg, model, head, optimizer = _build()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, step=1, model=model, head=head)
    assert not os.path.exists(path + ".tmp")
    assert os.path.exists(path)


def test_latest_checkpoint_prefers_the_highest_step(tmp_path):
    cfg, model, head, optimizer = _build()
    for step in (100, 900, 500):
        save_checkpoint(
            str(tmp_path / f"step_{step:06d}.pt"), step=step, model=model, head=head
        )
    found = latest_checkpoint(str(tmp_path))
    assert found.endswith("step_000900.pt")


def test_latest_checkpoint_returns_none_for_empty_dir(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None


def test_prune_checkpoints_keeps_the_newest(tmp_path):
    """Checkpoints are gigabytes for the larger presets; old ones must go."""
    cfg, model, head, optimizer = _build()
    for step in (100, 200, 300, 400, 500):
        save_checkpoint(
            str(tmp_path / f"step_{step:06d}.pt"), step=step, model=model, head=head
        )
    prune_checkpoints(str(tmp_path), keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_000400.pt", "step_000500.pt"]


def test_prune_checkpoints_is_a_noop_when_fewer_than_keep(tmp_path):
    cfg, model, head, optimizer = _build()
    save_checkpoint(str(tmp_path / "step_000100.pt"), step=100, model=model, head=head)
    prune_checkpoints(str(tmp_path), keep=5)
    assert len(list(tmp_path.glob("step_*.pt"))) == 1


def test_prune_checkpoints_disabled_with_zero(tmp_path):
    cfg, model, head, optimizer = _build()
    for step in (100, 200):
        save_checkpoint(
            str(tmp_path / f"step_{step:06d}.pt"), step=step, model=model, head=head
        )
    prune_checkpoints(str(tmp_path), keep=0)
    assert len(list(tmp_path.glob("step_*.pt"))) == 2
