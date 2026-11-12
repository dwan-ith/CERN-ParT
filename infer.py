"""Inference demo: event embeddings and masked-particle reconstruction.

Two demonstrations:

1. a **synthetic** batch, so the script runs without network access;
2. a **real** event streamed from CERN Open Data, if a URL is given.

Usage::

    python infer.py                                   # synthetic demo
    python infer.py --checkpoint checkpoints/final.pt
    python infer.py --url-file data/btag_val.txt      # stream real events
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch

from data.https_dataset import CMSNanoAODDataset
from model.config import N_RAW_FEATURES, ModelConfig, model_config
from model.particle_transformer import ParticleTransformer
from model.pretraining_head import MaskedParticleModeling


def _synthetic_particles(batch: int = 2, n: int = 16, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physically plausible random particles.

    The raw layout is ``[pt, eta, phi, mass, charge, type_id]`` with
    ``type_id`` indexing ``model.config.OBJECT_TYPES``.
    """
    generator = torch.Generator().manual_seed(seed)
    pt = torch.rand(batch, n, generator=generator) * 150 + 0.5
    eta = torch.rand(batch, n, generator=generator) * 6 - 3
    phi = torch.rand(batch, n, generator=generator) * 2 * torch.pi - torch.pi
    mass = torch.rand(batch, n, generator=generator) * 5
    charge = torch.randint(-1, 2, (batch, n), generator=generator).float()
    type_id = torch.randint(0, 5, (batch, n), generator=generator).float()
    particles = torch.stack([pt, eta, phi, mass, charge, type_id], dim=-1)
    mask = torch.ones(batch, n)
    mask[:, -3:] = 0.0  # three padding slots
    return particles, mask


def load_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    preset: str = "base",
):
    """Build the model, restoring the architecture stored in the checkpoint.

    Checkpoints written by this repository embed their ``model_config``; when
    one is present it is used, so that a checkpoint always comes back with the
    architecture it was trained with.
    """
    cfg: ModelConfig = model_config(preset)
    mask_strategy = "kinematic"
    payload: Dict = {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored = payload.get("model_config") or {}
        if stored:
            cfg = ModelConfig.from_dict(stored)
        training = payload.get("training_config") or {}
        mask_strategy = training.get("mask_strategy", mask_strategy)
        print(
            f"[infer] checkpoint {checkpoint_path}: step {payload.get('step', '?')}, "
            f"loss {payload.get('loss', float('nan')):.4f}, "
            f"d_model {cfg.d_model}, layers {cfg.n_layers}"
        )
    else:
        print(
            f"[infer] no checkpoint given -- running a randomly initialised "
            f"'{preset}' model (architecture demo mode)"
        )

    model = ParticleTransformer(cfg=cfg, mask_strategy=mask_strategy).to(device)
    head = MaskedParticleModeling(
        d_model=cfg.d_model,
        n_features=cfg.n_features,
        hidden_channels=model.normalizer.masked_channel_mask.nonzero().flatten().tolist(),
    ).to(device)

    if payload:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            print(
                f"[infer] warning: {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys -- the checkpoint does not "
                "match this architecture exactly"
            )
        if "head" in payload:
            head.load_state_dict(payload["head"], strict=False)

    model.eval()
    head.eval()
    return model, head, cfg


def run_demo(
    checkpoint_path: Optional[str] = None,
    url_file: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device: {device}")

    model, head, cfg = load_model(checkpoint_path, device=device)

    print("\n" + "=" * 62)
    print("DEMO 1: forward pass and event embeddings")
    print("=" * 62)

    particles, mask = _synthetic_particles()
    particles = particles.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        cls_out, particle_out = model(particles, mask)

    print(f"input particles:       {tuple(particles.shape)}  (B, N, {N_RAW_FEATURES})")
    print(f"event embedding [CLS]: {tuple(cls_out.shape)}  -- one vector per event")
    print(f"particle embeddings:   {tuple(particle_out.shape)}  -- one vector per particle")

    print("\n" + "=" * 62)
    print("DEMO 2: masked particle reconstruction")
    print("=" * 62)
    with torch.no_grad():
        loss, preds, is_masked = head(particles, mask, model, mask_ratio=0.25)
    print(f"masked particles:      {int(is_masked.sum())} / {particles.shape[0] * particles.shape[1]}")
    print(f"reconstruction loss:   {float(loss):.6f} (MSE on the hidden channels)")
    print(f"predictions:           {tuple(preds.shape)}")

    if url_file:
        print("\n" + "=" * 62)
        print("DEMO 3: a real event streamed from CERN Open Data")
        print("=" * 62)
        urls = _read_urls(url_file)
        if not urls:
            print(f"[infer] no URLs in {url_file}")
            return
        dataset = CMSNanoAODDataset(urls[:1], max_particles=64, batch_size=4,
                                     shuffle_buffer=0)
        try:
            batch = next(iter(dataset))
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[infer] could not stream the file: {exc}")
            return
        real_particles = batch["particles"].to(device)
        real_mask = batch["mask"].to(device)
        with torch.no_grad():
            cls_real, particle_real = model(real_particles, real_mask)
        n_jets = int(batch["is_jet"].sum())
        print(f"events streamed:       {real_particles.shape[0]}")
        print(f"particles per event:   {float(real_mask.sum(1).mean()):.2f}")
        print(f"jets in the batch:     {n_jets}")
        print(f"event embedding:       {tuple(cls_real.shape)}")
        print(f"particle embeddings:   {tuple(particle_real.shape)}")

    print("\n" + "=" * 62)
    print("done")
    print("=" * 62)


def _read_urls(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        ]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="", help="path to a checkpoint")
    parser.add_argument("--url-file", default="", help="file with NanoAOD URLs to stream")
    parser.add_argument("--preset", default="base", help="preset used without a checkpoint")
    args = parser.parse_args(argv)
    run_demo(args.checkpoint or None, args.url_file or None)


if __name__ == "__main__":
    main()
