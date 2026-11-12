"""
Training loop for CERN Foundation Model (Particle Transformer).
Streams CMS NanoAOD via HTTPS, trains with Masked Particle Modeling.

RTX 4050 6GB settings:
    d_model=128, n_layers=4, batch_size=16, AMP=True

Usage:
    python -m cern_ai.train
    python -m cern_ai.train --n_steps 5000 --lr 5e-4
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from .model.config import ModelConfig, TrainingConfig, DataConfig
from .model.particle_transformer import ParticleTransformer
from .model.pretraining_head import MaskedParticleModeling
from .data.https_dataset import CMSNanoAODDataset


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] {torch.cuda.get_device_name(0)} — "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("[device] CPU (no CUDA)")
    return dev


def get_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


def save_checkpoint(model, head, optimizer, scaler, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save model + head only (optimizer ~2GB for 259M params, skip to save disk)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "head": head.state_dict(),
        "loss": loss,
    }, path)


def train(args=None):
    if args is None:
        args = parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    if args:
        train_cfg.n_steps = args.n_steps
        train_cfg.batch_size = args.batch_size
        train_cfg.lr = args.lr
        train_cfg.mask_ratio = args.mask_ratio

    device = get_device()

    # --- Model ---
    model = ParticleTransformer(cfg=model_cfg).to(device)
    head = MaskedParticleModeling(d_model=model_cfg.d_model,
                                  n_features=model_cfg.n_features).to(device)

    n_params = model.count_parameters() + sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[model] {n_params / 1e6:.2f}M parameters")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda") if (train_cfg.amp and device.type == "cuda") else None

    # --- Data ---
    urls = data_cfg.default_urls
    if args and args.urls:
        urls = args.urls.split(",")
    dataset = CMSNanoAODDataset(
        urls,
        max_particles=train_cfg.max_particles,
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )

    # --- Training ---
    print(f"[train] Streaming from {len(urls)} file(s), "
          f"{train_cfg.n_steps} steps, batch={train_cfg.batch_size}")

    model.train()
    head.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    while step < train_cfg.n_steps:
        for batch in dataset:
            if step >= train_cfg.n_steps:
                break

            particles = batch["particles"].to(device)
            mask = batch["mask"].to(device)

            # LR schedule
            lr = get_lr(step, train_cfg.warmup_steps, train_cfg.n_steps, train_cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, preds, is_masked = head(particles, mask, model, train_cfg.mask_ratio)
                if is_masked.sum() > 0:
                    scaler.scale(loss / train_cfg.grad_accum).backward()
            else:
                loss, preds, is_masked = head(particles, mask, model, train_cfg.mask_ratio)
                if is_masked.sum() > 0:
                    (loss / train_cfg.grad_accum).backward()

            running_loss += loss.item()

            if (step + 1) % train_cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            # Logging
            if (step + 1) % train_cfg.log_every == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / train_cfg.log_every
                events_per_sec = (train_cfg.log_every * train_cfg.batch_size) / max(elapsed, 1e-6)
                mask_frac = is_masked.float().mean().item() if is_masked is not None else 0
                print(f"  step {step+1:>6d}/{train_cfg.n_steps} | "
                      f"loss {avg_loss:.4f} | "
                      f"mask {mask_frac:.2%} | "
                      f"{events_per_sec:.0f} evt/s | "
                      f"lr {lr:.2e}")
                running_loss = 0.0
                t0 = time.time()

            # Checkpoint
            if (step + 1) % train_cfg.save_every == 0:
                ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"step_{step+1:06d}.pt")
                save_checkpoint(model, head, optimizer, scaler, step + 1, avg_loss, ckpt_path)
                print(f"  [saved] {ckpt_path}")

            step += 1

    # Final save
    final_path = os.path.join(train_cfg.checkpoint_dir, "final.pt")
    save_checkpoint(model, head, optimizer, scaler, step, running_loss, final_path)
    print(f"[done] {step} steps, final checkpoint: {final_path}")

    return model, head


def parse_args():
    parser = argparse.ArgumentParser(description="Train CERN Particle Transformer")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--urls", type=str, default="",
                        help="Comma-separated list of ROOT file URLs")
    return parser.parse_args()


if __name__ == "__main__":
    train()
