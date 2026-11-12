"""
Inference and Feature Extraction Demo for CERN Particle Transformer.

Runs inference on streamed CMS NanoAOD collision events or synthetic particles,
extracts high-dimensional event representations ([CLS] embeddings), and performs
Masked Particle Modeling (MPM) reconstruction.

Usage:
    python infer.py
    python infer.py --checkpoint checkpoints/final.pt
"""

import argparse
import os
import torch

from model.config import ModelConfig, N_MODEL_FEATURES, N_RAW_FEATURES
from model.particle_transformer import ParticleTransformer
from model.pretraining_head import MaskedParticleModeling
from data.https_dataset import CMSNanoAODDataset


def load_model(checkpoint_path=None, device="cpu"):
    cfg = ModelConfig()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[infer] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_config" in ckpt and ckpt["model_config"]:
            try:
                cfg = ModelConfig(**ckpt["model_config"])
            except Exception:
                pass
        model = ParticleTransformer(cfg=cfg).to(device)
        head = MaskedParticleModeling(d_model=cfg.d_model, n_features=N_MODEL_FEATURES).to(device)
        try:
            model.load_state_dict(ckpt["model"])
            head.load_state_dict(ckpt["head"])
            step = ckpt.get("step", "unknown")
            loss = ckpt.get("loss", "unknown")
            print(f"[infer] Checkpoint loaded (Step: {step}, Loss: {loss})")
        except Exception as e:
            print(f"[infer] Note: Checkpoint architecture differs ({e}). Initialized fresh weights.")
    else:
        model = ParticleTransformer(cfg=cfg).to(device)
        head = MaskedParticleModeling(d_model=cfg.d_model, n_features=N_MODEL_FEATURES).to(device)
        print("[infer] Initializing model with uncheckpointed weights (architecture demo mode)")

    model.eval()
    head.eval()
    return model, head, cfg


def run_demo(checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Running inference on device: {device}")

    model, head, cfg = load_model(checkpoint_path, device=device)

    # Demo 1: Synthetic batch with particle candidates
    print("\n" + "=" * 60)
    print("DEMO 1: Forward Pass & Event Embeddings")
    print("=" * 60)

    B, N, F = 2, 8, N_RAW_FEATURES
    dummy_particles = torch.randn(B, N, F, device=device)
    dummy_mask = torch.ones(B, N, device=device)
    dummy_mask[:, -2:] = 0.0  # 2 padding particles

    with torch.no_grad():
        cls_out, particle_out = model(dummy_particles, dummy_mask)

    print(f"Input batch shape:       {dummy_particles.shape} (B={B}, Particles={N}, Features={F})")
    print(f"Event embedding [CLS]:   {cls_out.shape} -> Suitable for downstream Jet/Event classification")
    print(f"Per-particle embeddings: {particle_out.shape} -> Suitable for constituent tagging")

    # Demo 2: Masked Particle Reconstruction
    print("\n" + "=" * 60)
    print("DEMO 2: Masked Particle Modeling (MPM) Reconstruction")
    print("=" * 60)

    with torch.no_grad():
        loss, preds, is_masked = head(dummy_particles, dummy_mask, model, mask_ratio=0.25)

    print(f"Masked particles count:  {is_masked.sum().item()} / {B * N}")
    print(f"Reconstruction MSE loss: {loss.item():.4f}")
    print(f"Predictions shape:       {preds.shape}")

    print("\n" + "=" * 60)
    print("Inference demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Transformer Inference Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt",
                        help="Path to PyTorch checkpoint file")
    args = parser.parse_args()
    run_demo(args.checkpoint)

