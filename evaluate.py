"""Downstream benchmark: jet b-tagging with the pretrained encoder.

This is the experiment that decides whether the pretraining is worth
anything.  Three modes are supported and use exactly the same data, the same
schedule and the same head, so that the only difference is the encoder:

``scratch``
    Encoder and head are trained from random initialisation.  This is the
    baseline.
``finetune``
    The encoder is initialised from a pretrained checkpoint and trained
    jointly with the head.
``linear``
    The encoder is frozen and only the head is trained (a linear probe).
    This isolates how much of the representation is usable without letting
    the encoder move.

Additionally ``--no-interaction`` ablates the ParT pairwise interaction bias,
which is the architectural contribution of the model.

Usage::

    python evaluate.py --checkpoint checkpoints/final.pt --mode finetune
    python evaluate.py --mode scratch
    python evaluate.py --checkpoint checkpoints/final.pt --mode linear
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from data.feature_extractor import BTAG_MEDIUM_WP, binarize_btag
from data.https_dataset import CMSNanoAODDataset
from model.config import DataConfig, cosine_lr, model_config
from model.particle_transformer import ParticleTransformer
from model.task_head import JetTaggingHead, roc_auc


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_urls(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        ]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] {name} — {total:.1f} GB")
        return torch.device("cuda")
    print("[device] CPU")
    return torch.device("cpu")


def build_datasets(args, data_cfg: DataConfig):
    train_urls = read_urls(args.train_urls)
    val_urls = read_urls(args.val_urls)
    train = CMSNanoAODDataset(
        train_urls,
        max_particles=args.max_particles,
        batch_size=args.batch_size,
        shuffle=True,
        shuffle_buffer=args.shuffle_buffer,
        tree=data_cfg.tree,
        seed=args.seed,
        cache_dir=args.cache_dir or None,
        download_workers=args.download_workers,
    )
    val = CMSNanoAODDataset(
        val_urls,
        max_particles=args.max_particles,
        batch_size=args.batch_size,
        shuffle=False,
        shuffle_buffer=0,
        tree=data_cfg.tree,
        seed=args.seed,
        cache_dir=args.cache_dir or None,
        download_workers=args.download_workers,
    )
    return train, val


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def load_encoder(
    model: ParticleTransformer,
    checkpoint: str,
    device: torch.device,
) -> None:
    """Load a pretrained encoder, failing loudly on an architecture mismatch.

    ``strict=False`` is used as a fallback so that a checkpoint from a
    slightly different architecture still loads the parts that match, but a
    complete mismatch is an error rather than a silently random model.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(payload["model"])
    except (RuntimeError, KeyError) as exc:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if len(missing) >= len(model.state_dict()) // 2:
            raise RuntimeError(
                f"checkpoint {checkpoint} does not match this architecture "
                f"({len(missing)} missing / {len(unexpected)} unexpected keys): {exc}"
            ) from exc
        print(
            f"[init] warning: checkpoint partially loaded "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )
    return payload


def train_one_epoch(
    model: ParticleTransformer,
    head: JetTaggingHead,
    loader,
    optimizer,
    scaler,
    device,
    args,
    epoch: int,
    lr: float,
) -> Dict[str, float]:
    model.train()
    head.train()
    if getattr(args, "freeze_encoder", False):
        # A frozen encoder must stay in eval mode: dropout would otherwise
        # make the features used for training differ from the ones used for
        # evaluation.
        model.eval()
    total_loss = 0.0
    n_batches = 0
    n_jets = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        if i >= args.max_batches_per_epoch:
            break
        particles = batch["particles"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        is_jet = batch["is_jet"].to(device, non_blocking=True)
        btag = batch["btag"].to(device, non_blocking=True)
        labels = torch.from_numpy(binarize_btag(btag.cpu().numpy(), BTAG_MEDIUM_WP)).to(device)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                _, particle_embeds = model(particles, mask)
                loss, _ = head.loss(
                    particle_embeds,
                    is_jet,
                    labels,
                    min_pt=args.min_jet_pt,
                    particle_pt=particles[..., 0],
                )
            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            _, particle_embeds = model(particles, mask)
            loss, _ = head.loss(
                particle_embeds,
                is_jet,
                labels,
                min_pt=args.min_jet_pt,
                particle_pt=particles[..., 0],
            )
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        n_batches += 1
        n_jets += int((is_jet & (particles[..., 0] >= args.min_jet_pt)).sum().item())

    return {
        "loss": total_loss / max(n_batches, 1),
        "batches": n_batches,
        "jets": n_jets,
        "epoch_time": time.time() - t0,
    }


@torch.no_grad()
def evaluate(model: ParticleTransformer, head: JetTaggingHead, loader, device, args):
    model.eval()
    head.eval()
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for i, batch in enumerate(loader):
        if i >= args.max_eval_batches:
            break
        particles = batch["particles"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        is_jet = batch["is_jet"].to(device, non_blocking=True)
        btag = batch["btag"].to(device, non_blocking=True)
        target = torch.from_numpy(binarize_btag(btag.cpu().numpy(), BTAG_MEDIUM_WP)).to(device)

        _, particle_embeds = model(particles, mask)
        logits = head(particle_embeds)

        select = is_jet.bool()
        if args.min_jet_pt > 0:
            select = select & (particles[..., 0] >= args.min_jet_pt)
        if select.sum() == 0:
            continue
        scores.append(logits[select].float().cpu().numpy())
        labels.append(target[select].float().cpu().numpy())

    if not scores:
        return float("nan"), 0
    score_array = np.concatenate(scores)
    label_array = np.concatenate(labels)
    return roc_auc(score_array, label_array), int(label_array.size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> Dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="", help="pretrained encoder checkpoint")
    parser.add_argument(
        "--mode", default="finetune", choices=["scratch", "finetune", "linear"]
    )
    parser.add_argument("--preset", default="base")
    parser.add_argument("--train-urls", default="data/btag_train.txt")
    parser.add_argument("--val-urls", default="data/btag_val.txt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-particles", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=0.2)
    parser.add_argument("--min-jet-pt", type=float, default=20.0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=400)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument(
        "--cache-dir",
        default="",
        help="download the files into this directory first (much faster)",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="freeze the encoder (implies --mode linear behaviour)")
    parser.add_argument("--out", default="", help="write a JSON result file")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    data_cfg = DataConfig()

    model_cfg = model_config(args.preset)
    if args.no_interaction:
        model_cfg.use_interaction_bias = False

    model = ParticleTransformer(cfg=model_cfg).to(device)
    head = JetTaggingHead(d_model=model_cfg.d_model).to(device)

    pretrained = False
    if args.mode in ("finetune", "linear"):
        if not args.checkpoint:
            raise SystemExit(f"--mode {args.mode} requires --checkpoint")
        payload = load_encoder(model, args.checkpoint, device)
        pretrained = True
        if args.mode == "linear":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            args.freeze_encoder = True
            print(f"[init] loaded frozen encoder from {args.checkpoint}")
        else:
            print(
                f"[init] loaded encoder from {args.checkpoint} "
                f"(step {payload.get('step')})"
            )
    else:
        print("[init] training from random initialisation (scratch baseline)")

    train_loader, val_loader = build_datasets(args, data_cfg)
    print(
        f"[data] {len(train_loader.urls)} train files / "
        f"{len(val_loader.urls)} val files"
    )

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters + list(head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    total_steps = max(args.epochs * args.max_batches_per_epoch, 1)
    warmup = int(args.warmup_epochs * args.max_batches_per_epoch)

    best_auc = float("nan")
    history: List[Dict[str, float]] = []
    step = 0
    for epoch in range(args.epochs):
        lr = cosine_lr(step, warmup, total_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats = train_one_epoch(
            model, head, train_loader, optimizer, scaler, device, args, epoch, lr
        )
        step += args.max_batches_per_epoch
        auc, n_jets = evaluate(model, head, val_loader, device, args)
        best_auc = auc if math.isnan(best_auc) else max(best_auc, auc)
        record = {
            "epoch": epoch,
            "loss": stats["loss"],
            "val_auc": auc,
            "val_jets": n_jets,
            "lr": lr,
        }
        history.append(record)
        print(
            f"  epoch {epoch}: loss {stats['loss']:.4f} | "
            f"val AUC {auc:.4f} ({n_jets} jets) | lr {lr:.2e}"
        )

    result = {
        "mode": args.mode,
        "preset": args.preset,
        "pretrained": pretrained,
        "interaction_bias": not args.no_interaction,
        "best_val_auc": best_auc,
        "final_val_auc": history[-1]["val_auc"] if history else float("nan"),
        "epochs": args.epochs,
        "seed": args.seed,
        "history": history,
    }
    print(
        f"[result] mode={args.mode} pretrained={pretrained} "
        f"interaction={not args.no_interaction} "
        f"final val AUC = {result['final_val_auc']:.4f}"
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"[result] written to {args.out}")
    return result


if __name__ == "__main__":
    main()
