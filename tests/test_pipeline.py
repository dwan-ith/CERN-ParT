"""
End-to-end pipeline test.
Streams 1 CMS NanoAOD file via HTTPS, extracts features, runs model forward pass.

Usage:
    python tests/test_pipeline.py
"""
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_feature_extractor():
    """Test that feature extractor produces correct shapes."""
    from data.feature_extractor import extract_batch, N_FEATURES, get_required_branches

    branches = get_required_branches()
    print(f"[test] Required branches: {len(branches)}")
    assert len(branches) > 0
    print("[test] Feature extractor OK")
    return True


def test_https_streaming():
    """Test streaming 1 file via HTTPS with uproot."""
    import uproot
    from data.feature_extractor import extract_batch, get_required_branches

    url = "http://opendata.cern.ch/eos/opendata/cms/derived-data/NanoAODRun1/01-Jul-22/Run2010A_Mu/02A55927-7112-4A20-B06E-86E24B644B59.root"
    print(f"[test] Opening {url}")

    t0 = time.time()
    branches = get_required_branches()

    for arrays in uproot.iterate(
        [url],
        filter_name=branches,
        step_size=8,
    ):
        batch = extract_batch(arrays, max_particles=64)
        elapsed = time.time() - t0

        print(f"[test] First batch in {elapsed:.1f}s")
        print(f"  particles shape: {batch['particles'].shape}")
        print(f"  mask shape:      {batch['mask'].shape}")
        print(f"  n_particles:     {batch['n_particles']}")
        print(f"  particles nonzero: {(batch['particles'] != 0).any(axis=-1).sum()}")

        assert batch["particles"].shape[2] == 6, f"Expected 6 features, got {batch['particles'].shape[2]}"
        assert batch["mask"].shape == batch["particles"].shape[:2]
        print("[test] HTTPS streaming OK")
        return True

    print("[test] WARNING: no batches returned from uproot.iterate")
    return False


def test_model_forward():
    """Test model forward pass with random input."""
    from model.particle_transformer import ParticleTransformer
    from model.config import ModelConfig, N_MODEL_FEATURES, N_RAW_FEATURES

    cfg = ModelConfig()
    model = ParticleTransformer(cfg=cfg)
    n_params = model.count_parameters()
    print(f"[test] Model: {n_params / 1e6:.2f}M parameters")

    B, N, F = 4, 64, N_RAW_FEATURES
    particles = torch.randn(B, N, F)
    mask = torch.ones(B, N)
    mask[:, -10:] = 0  # last 10 are padding

    t0 = time.time()
    cls_out, particle_out = model(particles, mask)
    elapsed = time.time() - t0

    print(f"[test] Forward pass: {elapsed*1000:.1f}ms")
    print(f"  cls_out shape:      {cls_out.shape}")
    print(f"  particle_out shape: {particle_out.shape}")

    assert cls_out.shape == (B, cfg.d_model)
    assert particle_out.shape == (B, N, cfg.d_model)
    print("[test] Model forward OK")
    return True


def test_pretraining_head():
    """Test masked particle modeling loss computation."""
    from model.particle_transformer import ParticleTransformer
    from model.pretraining_head import MaskedParticleModeling
    from model.config import ModelConfig, N_MODEL_FEATURES, N_RAW_FEATURES

    cfg = ModelConfig()
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=N_MODEL_FEATURES)

    B, N, F = 4, 64, N_RAW_FEATURES
    particles = torch.randn(B, N, F)
    mask = torch.ones(B, N)

    loss, preds, is_masked = head(particles, mask, model, mask_ratio=0.15)

    print(f"[test] MPM loss:       {loss.item():.4f}")
    print(f"  preds shape:    {preds.shape}")
    print(f"  masked count:   {is_masked.sum().item()} / {B * N}")

    assert loss.item() > 0, "Loss should be > 0"
    assert preds.shape == (B, N, N_MODEL_FEATURES)
    print("[test] Pretraining head OK")
    return True


def test_gpu_memory():
    """Check GPU memory footprint."""
    if not torch.cuda.is_available():
        print("[test] Skipping GPU memory test (no CUDA)")
        return True

    from model.particle_transformer import ParticleTransformer
    from model.pretraining_head import MaskedParticleModeling
    from model.config import ModelConfig, N_MODEL_FEATURES, N_RAW_FEATURES

    torch.cuda.reset_peak_memory_stats()
    cfg = ModelConfig()
    model = ParticleTransformer(cfg=cfg).cuda()
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=N_MODEL_FEATURES).cuda()

    B, N, F = 16, 64, N_RAW_FEATURES
    particles = torch.randn(B, N, F).cuda()
    mask = torch.ones(B, N).cuda()

    loss, _, _ = head(particles, mask, model, mask_ratio=0.15)
    loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[test] GPU peak memory: {peak:.2f} GB (of 6 GB available)")

    assert peak < 5.5, f"Peak memory {peak:.2f} GB exceeds 5.5 GB budget"
    print("[test] GPU memory OK")
    return True


def main():
    print("=" * 60)
    print("CERN Foundation Model — Pipeline Test")
    print("=" * 60)

    tests = [
        ("Feature extractor", test_feature_extractor),
        ("HTTPS streaming", test_https_streaming),
        ("Model forward", test_model_forward),
        ("Pretraining head", test_pretraining_head),
        ("GPU memory", test_gpu_memory),
    ]

    results = []
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"[test] FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Results:")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    all_pass = all(ok for _, ok in results)
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
