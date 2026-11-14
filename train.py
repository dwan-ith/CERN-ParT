"""Pretraining loop for the Particle Transformer.

Run it with::

    python train.py --preset base --n-steps 5000
    python train.py --preset base --resume checkpoints/step_005000.pt

The loop is written so that it can be interrupted at any point: checkpoints
are written atomically and contain the optimizer state, so ``--resume``
continues a run rather than restarting it.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    write_run_config,
)
from data.https_dataset import CMSNanoAODDataset
from model.config import (
    DataConfig,
    TrainingConfig,
    cosine_lr,
    estimate_memory,
    model_config,
)
from model.particle_transformer import ParticleTransformer
from model.pretraining_head import MaskedParticleModeling


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] {name} — {total:.1f} GB")
        return torch.device("cuda")
    print("[device] CPU (no CUDA available)")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    parameters, cfg: TrainingConfig, name: str = "adamw"
) -> torch.optim.Optimizer:
    """Create the optimizer.

    ``adamw8bit`` keeps the optimizer state at one byte per moment, which is
    what makes the larger presets fit on a 6 GB card.  If ``bitsandbytes`` is
    not installed we fall back to regular AdamW and say so.
    """
    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(
                parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[warn] adamw8bit unavailable ({exc}); falling back to adamw")
            return torch.optim.AdamW(
                parameters, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
            )
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise KeyError(f"unknown optimizer {name!r}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def resolve_urls(args: argparse.Namespace, data_cfg: DataConfig) -> List[str]:
    if args.urls:
        return [u.strip() for u in args.urls.split(",") if u.strip()]
    if args.url_file:
        with open(args.url_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if not data_cfg.default_urls:
        raise SystemExit(
            "no URLs given: pass --urls or --url-file, or generate "
            "data/all_urls.txt with `python generate_urls.py`"
        )
    return list(data_cfg.default_urls)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_pretraining_loss(
    model: ParticleTransformer,
    head: MaskedParticleModeling,
    dataset,
    device: torch.device,
    n_batches: int = 20,
    mask_ratio: float = 0.15,
    seed: int = 1234,
) -> float:
    """Masked-particle-modelling loss on a held-out stream.

    The masking RNG is re-seeded for every batch so that the number is
    comparable between evaluations.
    """
    was_training = model.training
    model.eval()
    head.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    total = 0.0
    count = 0
    for i, batch in enumerate(dataset):
        if i >= n_batches:
            break
        particles = batch["particles"].to(device)
        mask = batch["mask"].to(device)
        loss, _, _ = head(particles, mask, model, mask_ratio, generator=generator)
        total += float(loss.item())
        count += 1

    if was_training:
        model.train()
        head.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def save_and_report(path: str, **kwargs) -> None:
    """Save a checkpoint and report how much disk it used."""
    save_checkpoint(path, **kwargs)
    size = os.path.getsize(path) / 1e9
    print(f"  [saved] {path} ({size:.2f} GB)")


def train(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    args = args or parse_args()

    model_cfg = model_config(args.preset)
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    train_cfg.n_steps = args.n_steps
    train_cfg.batch_size = args.batch_size
    train_cfg.lr = args.lr
    train_cfg.min_lr = args.min_lr
    train_cfg.mask_ratio = args.mask_ratio
    train_cfg.mask_strategy = args.mask_strategy
    train_cfg.grad_accum = args.grad_accum
    train_cfg.max_particles = args.max_particles
    train_cfg.warmup_steps = args.warmup_steps
    train_cfg.save_every = args.save_every
    train_cfg.log_every = args.log_every
    train_cfg.checkpoint_dir = args.out_dir
    train_cfg.amp = not args.no_amp
    train_cfg.seed = args.seed

    if args.no_interaction:
        model_cfg.use_interaction_bias = False

    set_seed(train_cfg.seed)
    device = get_device()

    # --- Model ---------------------------------------------------------
    model = ParticleTransformer(cfg=model_cfg, mask_strategy=train_cfg.mask_strategy).to(device)
    head = MaskedParticleModeling(
        d_model=model_cfg.d_model,
        n_features=model_cfg.n_features,
        hidden_channels=train_cfg.masked_channels,
    ).to(device)

    n_params = model.count_parameters() + sum(
        p.numel() for p in head.parameters() if p.requires_grad
    )
    print(f"[model] preset={args.preset} params={n_params / 1e6:.2f}M")

    # --- Memory budget -------------------------------------------------
    memory = estimate_memory(model_cfg, train_cfg, optimizer=args.optimizer, n_params=n_params)
    print(
        f"[memory] estimated peak {memory['total']:.2f} GB "
        f"(params {memory['params']:.2f} + grads {memory['grads']:.2f} + "
        f"optim {memory['optimizer']:.2f} + act {memory['activations']:.2f})"
    )
    if device.type == "cuda":
        capacity = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory["total"] > capacity * 0.95:
            print(
                f"[warn] estimated peak exceeds {capacity:.1f} GB of VRAM; "
                "consider a smaller --preset, --no-amp or --optimizer adamw8bit"
            )

    # --- Optimizer -----------------------------------------------------
    optimizer = build_optimizer(
        list(model.parameters()) + list(head.parameters()), train_cfg, args.optimizer
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if (train_cfg.amp and device.type == "cuda")
        else None
    )

    # --- Data ----------------------------------------------------------
    urls = resolve_urls(args, data_cfg)
    if args.max_files:
        urls = urls[: args.max_files]
    dataset = CMSNanoAODDataset(
        urls,
        max_particles=train_cfg.max_particles,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        shuffle_buffer=args.shuffle_buffer,
        tree=data_cfg.tree,
        seed=train_cfg.seed,
        cache_dir=args.cache_dir or None,
        download_workers=args.download_workers,
    )

    # --- Optional held-out stream --------------------------------------
    val_dataset = None
    val_urls: List[str] = []
    if args.val_urls:
        val_urls = [u.strip() for u in args.val_urls.split(",") if u.strip()]
    elif args.val_url_file:
        with open(args.val_url_file, "r", encoding="utf-8") as handle:
            val_urls = [
                line.strip()
                for line in handle
                if line.strip() and not line.startswith("#")
            ]
    if val_urls and args.val_every:
        val_dataset = CMSNanoAODDataset(
            val_urls,
            max_particles=train_cfg.max_particles,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            shuffle_buffer=0,
            tree=data_cfg.tree,
            seed=train_cfg.seed + 1,
            cache_dir=args.cache_dir or None,
        )
        print(f"[val] {len(val_urls)} held-out file(s)")

    # --- Resume --------------------------------------------------------
    start_step = 0
    if args.resume:
        path = args.resume if args.resume != "auto" else latest_checkpoint(train_cfg.checkpoint_dir)
        if path and os.path.exists(path):
            payload = load_checkpoint(
                path,
                model=model,
                head=head,
                optimizer=optimizer,
                scaler=scaler,
                map_location=str(device),
            )
            start_step = int(payload.get("step", 0))
            print(f"[resume] {path} -> continuing at step {start_step}")
        else:
            print(f"[resume] no checkpoint found at {args.resume!r}; starting fresh")

    run_config = {
        "args": {k: v for k, v in vars(args).items()},
        "model": model_cfg.to_dict(),
        "training": train_cfg.to_dict(),
        "memory_estimate_gb": memory,
        "n_params": n_params,
        "n_urls": len(urls),
    }
    write_run_config(os.path.join(train_cfg.checkpoint_dir, "run_config.json"), run_config)

    # --- Loop ----------------------------------------------------------
    print(
        f"[train] {len(urls)} files, {train_cfg.n_steps} optimizer steps, "
        f"batch={train_cfg.batch_size} x accum={train_cfg.grad_accum}, "
        f"mask={train_cfg.mask_ratio:.0%} ({train_cfg.mask_strategy})"
    )

    model.train()
    head.train()
    optimizer_step = start_step
    micro_step = 0
    running = 0.0
    running_count = 0
    t0 = time.time()
    last_log_loss = float("nan")

    while optimizer_step < train_cfg.n_steps:
        for batch in dataset:
            if optimizer_step >= train_cfg.n_steps:
                break

            particles = batch["particles"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            lr = cosine_lr(
                optimizer_step,
                train_cfg.warmup_steps,
                train_cfg.n_steps,
                train_cfg.lr,
                train_cfg.min_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, _, is_masked = head(
                        particles, mask, model, train_cfg.mask_ratio
                    )
                scaler.scale(loss / train_cfg.grad_accum).backward()
            else:
                loss, _, is_masked = head(particles, mask, model, train_cfg.mask_ratio)
                (loss / train_cfg.grad_accum).backward()

            running += float(loss.item())
            running_count += 1
            micro_step += 1

            if micro_step % train_cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % train_cfg.log_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    avg_loss = running / max(running_count, 1)
                    last_log_loss = avg_loss
                    events = train_cfg.batch_size * train_cfg.grad_accum * running_count
                    # Fraction of *real* particles that were masked.  Dividing
                    # by the padded tensor width would understate it badly.
                    real = mask.sum().clamp(min=1.0)
                    mask_frac = float(is_masked.sum().item() / real.item())
                    print(
                        f"  step {optimizer_step:>6d}/{train_cfg.n_steps} | "
                        f"loss {avg_loss:.4f} | mask {mask_frac:.3f} | "
                        f"{events / elapsed:.1f} evt/s | lr {lr:.2e}"
                    )
                    running = 0.0
                    running_count = 0
                    t0 = time.time()

                if optimizer_step % train_cfg.save_every == 0:
                    path = os.path.join(
                        train_cfg.checkpoint_dir, f"step_{optimizer_step:06d}.pt"
                    )
                    save_and_report(
                        path,
                        step=optimizer_step,
                        model=model,
                        head=head,
                        optimizer=optimizer,
                        scaler=scaler,
                        loss=last_log_loss,
                        model_config=model_cfg.to_dict(),
                        training_config=train_cfg.to_dict(),
                        save_optimizer=not args.no_save_optimizer,
                    )
                    prune_checkpoints(train_cfg.checkpoint_dir, args.keep_last)

                if (
                    val_dataset is not None
                    and args.val_every
                    and optimizer_step % args.val_every == 0
                ):
                    val_loss = evaluate_pretraining_loss(
                        model,
                        head,
                        val_dataset,
                        device,
                        n_batches=args.val_batches,
                        mask_ratio=train_cfg.mask_ratio,
                    )
                    print(
                        f"  [val] step {optimizer_step} | "
                        f"masked-particle loss {val_loss:.4f}"
                    )

    # --- Final checkpoint ----------------------------------------------
    final_path = os.path.join(train_cfg.checkpoint_dir, "final.pt")
    save_checkpoint(
        final_path,
        step=optimizer_step,
        model=model,
        head=head,
        optimizer=optimizer,
        scaler=scaler,
        loss=last_log_loss,
        model_config=model_cfg.to_dict(),
        training_config=train_cfg.to_dict(),
        save_optimizer=not args.no_save_optimizer,
    )
    stats = dict(dataset.stats)
    stats["particles_per_event"] = (
        stats.get("particles", 0.0) / stats["events"] if stats.get("events") else 0.0
    )
    print(f"[done] {optimizer_step} steps | final checkpoint: {final_path}")
    print(f"[stats] events={int(stats.get('events', 0))} "
          f"particles/event={stats['particles_per_event']:.2f}")
    return {"model": model, "head": head, "step": optimizer_step, "stats": stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the Particle Transformer with masked particle modelling"
    )
    parser.add_argument("--preset", default="base", help="model preset (see config.py)")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument(
        "--mask-strategy", default="kinematic", choices=["kinematic", "full"]
    )
    parser.add_argument("--max-particles", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--urls", default="", help="comma-separated list of ROOT file URLs")
    parser.add_argument("--url-file", default="", help="file with one URL per line")
    parser.add_argument("--val-urls", default="", help="comma-separated held-out URLs")
    parser.add_argument("--val-url-file", default="", help="file of held-out URLs")
    parser.add_argument("--val-every", type=int, default=0,
                        help="evaluate on the held-out stream every N steps")
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0, help="only use the first N files")
    parser.add_argument(
        "--cache-dir",
        default="",
        help="download the files into this directory first (much faster; "
        "ROOT files must be streamed whole over HTTP otherwise)",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "sgd"])
    parser.add_argument("--no-save-optimizer", action="store_true",
                        help="do not store the optimizer state (much smaller "
                        "checkpoints, but --resume can no longer continue the "
                        "optimiser)")
    parser.add_argument("--keep-last", type=int, default=3,
                        help="how many step checkpoints to keep on disk")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-interaction", action="store_true", help="ablate the ParT interaction bias")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from this checkpoint (default: the latest in --out-dir)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Console-script entry point (``cern-train``)."""
    return train(parse_args(argv))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    train()
