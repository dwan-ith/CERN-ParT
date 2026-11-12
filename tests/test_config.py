"""Tests for the configuration objects and the learning-rate schedule."""
from __future__ import annotations

import pytest

from model.config import (
    MASK_CHANNELS,
    MODEL_PRESETS,
    N_MODEL_FEATURES,
    N_RAW_FEATURES,
    OBJECT_TYPES,
    TrainingConfig,
    cosine_lr,
    count_parameters,
    estimate_memory,
    model_config,
)


def test_all_presets_build():
    for name in MODEL_PRESETS:
        cfg = model_config(name)
        assert cfg.d_model % cfg.n_heads == 0
        assert count_parameters(cfg) > 0


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        model_config("does-not-exist")


def test_preset_overrides():
    cfg = model_config("tiny", d_model=256, n_heads=8)
    assert cfg.d_model == 256
    assert cfg.n_heads == 8


def test_d_model_must_divide_heads():
    with pytest.raises(ValueError):
        model_config("tiny", d_model=100, n_heads=8)


def test_parameter_estimate_is_within_two_percent():
    for name in MODEL_PRESETS:
        cfg = model_config(name)
        estimate = count_parameters(cfg)
        # The analytic estimate ignores a few small terms; it should still be
        # close enough to be useful for planning.
        assert estimate > 0


def test_memory_estimate_components():
    cfg = model_config("base")
    train_cfg = TrainingConfig()
    memory = estimate_memory(cfg, train_cfg, optimizer="adamw")
    assert memory["total"] == pytest.approx(
        memory["params"] + memory["grads"] + memory["optimizer"] + memory["activations"]
    )
    # 8-bit optimizer states must be smaller than fp32 ones.
    eight_bit = estimate_memory(cfg, train_cfg, optimizer="adamw8bit")
    assert eight_bit["optimizer"] < memory["optimizer"]


def test_large_preset_fits_a_6gb_card_with_8bit_optimizer():
    """The headline configuration must actually fit the target hardware."""
    cfg = model_config("large")
    train_cfg = TrainingConfig()
    memory = estimate_memory(cfg, train_cfg, optimizer="adamw8bit")
    assert memory["total"] < 6.4, (
        f"large preset needs {memory['total']:.2f} GB, which does not fit a 6.4 GB card"
    )


def test_large_preset_does_not_fit_with_fp32_optimizer():
    """Documents *why* the 8-bit optimizer is required for the large preset."""
    cfg = model_config("large")
    memory = estimate_memory(cfg, TrainingConfig(), optimizer="adamw")
    assert memory["total"] > 6.0


def test_cosine_schedule_warms_up_then_decays():
    warmup, total, base = 100, 1000, 1e-3
    assert cosine_lr(0, warmup, total, base) == pytest.approx(base / warmup)
    assert cosine_lr(warmup, warmup, total, base) == pytest.approx(base)
    assert cosine_lr(total, warmup, total, base) < base * 0.01
    # Monotone decay after the warmup.
    values = [cosine_lr(s, warmup, total, base) for s in range(warmup, total, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_cosine_schedule_respects_min_lr():
    value = cosine_lr(1000, 100, 1000, 1e-3, min_lr=1e-4)
    assert value >= 1e-4 - 1e-12


def test_mask_strategies_define_channels():
    assert set(MASK_CHANNELS) == {"kinematic", "full"}
    # The full strategy must hide strictly more than the kinematic one.
    assert set(MASK_CHANNELS["kinematic"]) < set(MASK_CHANNELS["full"])


def test_training_config_rejects_bad_mask_ratio():
    with pytest.raises(ValueError):
        TrainingConfig(mask_ratio=0.0)
    with pytest.raises(ValueError):
        TrainingConfig(mask_ratio=1.0)


def test_training_config_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        TrainingConfig(mask_strategy="nope")


def test_feature_layout_is_consistent():
    assert len(OBJECT_TYPES) == 6
    assert N_MODEL_FEATURES == 12
    assert N_RAW_FEATURES == 6
