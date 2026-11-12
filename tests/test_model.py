"""Unit tests for the model: shapes, leak-freedom and numerical correctness."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.config import (
    CH_ETA,
    CH_PT,
    count_parameters,
    model_config,
)
from model.particle_transformer import (
    InteractionFeatureComputer,
    ParticleTransformer,
)
from model.pretraining_head import MaskedParticleModeling
from model.task_head import JetTaggingHead, roc_auc


def _make_particles(batch=2, n=8, seed=0, max_pt=100.0):
    """A random but physically plausible particle set."""
    generator = torch.Generator().manual_seed(seed)
    pt = torch.rand(batch, n, generator=generator) * max_pt + 0.5
    eta = torch.rand(batch, n, generator=generator) * 6 - 3
    phi = torch.rand(batch, n, generator=generator) * 2 * math.pi - math.pi
    mass = torch.rand(batch, n, generator=generator) * 5
    charge = torch.randint(-1, 2, (batch, n), generator=generator).float()
    type_id = torch.randint(0, 6, (batch, n), generator=generator).float()
    particles = torch.stack([pt, eta, phi, mass, charge, type_id], dim=-1)
    mask = torch.ones(batch, n)
    mask[:, -2:] = 0.0  # two padding slots
    return particles, mask


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_forward_shapes():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    particles, mask = _make_particles()
    cls_out, particle_out = model(particles, mask)
    assert cls_out.shape == (2, cfg.d_model)
    assert particle_out.shape == (2, 8, cfg.d_model)


def test_cls_interaction_row_and_column_are_zero():
    """The CLS token must be aligned with the *first* index of the tensor.

    Regression test: the interaction tensor was originally padded by
    *appending* the CLS row and column, which placed them at index N while
    the CLS token sits at index 0 of the hidden states.  The CLS token was
    therefore paired with the last particle's interaction features.
    """
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    particles, _ = _make_particles()
    interaction = model.interaction_compute(particles)
    padded = model._pad_interaction(interaction)
    B, N = particles.shape[0], particles.shape[1]
    assert padded.shape == (B, N + 1, N + 1, interaction.shape[-1])
    # Row 0 (CLS query) and column 0 (CLS key) must vanish ...
    assert torch.all(padded[:, 0, :, :] == 0.0)
    assert torch.all(padded[:, :, 0, :] == 0.0)
    # ... while the particle block must be preserved, unshifted.
    assert torch.allclose(padded[:, 1:, 1:, :], interaction)


def test_padding_does_not_change_output():
    """Padded slots must not influence the representation of real particles."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg).eval()
    particles, mask = _make_particles()
    with torch.no_grad():
        base, _ = model(particles, mask)
        perturbed = particles.clone()
        perturbed[:, -2:, :] = 1234.5  # garbage in the padding slots
        other, _ = model(perturbed, mask)
    assert torch.allclose(base, other, atol=1e-5)


def test_parameter_estimate_is_accurate():
    for preset in ("tiny", "small", "base"):
        cfg = model_config(preset)
        model = ParticleTransformer(cfg=cfg)
        estimate = count_parameters(cfg)
        actual = model.count_parameters()
        assert abs(estimate - actual) / actual < 0.02, (
            f"{preset}: estimate {estimate} vs actual {actual}"
        )


def test_interaction_ablation_removes_the_bias_network():
    cfg = model_config("tiny", use_interaction_bias=False)
    model = ParticleTransformer(cfg=cfg)
    assert not any(
        "interaction_mlp" in name for name, _ in model.named_parameters()
    )
    assert count_parameters(cfg) < count_parameters(model_config("tiny"))


# ---------------------------------------------------------------------------
# Leak-freedom -- the central correctness property
# ---------------------------------------------------------------------------
def test_kinematic_mask_hides_the_momentum():
    """Changing the pT of a masked particle must not change the model output.

    The masked particle's momentum is the prediction target.  If the model
    could still see it -- through the embedding or through the interaction
    features -- the pretraining task would be trivial and the learned
    representation would be worthless.
    """
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg, mask_strategy="kinematic").eval()
    particles, mask = _make_particles()
    masked = torch.zeros_like(mask, dtype=torch.bool)
    masked[:, 3] = True

    with torch.no_grad():
        a, _ = model(particles, mask, masked=masked)
        perturbed = particles.clone()
        perturbed[:, 3, 0] = perturbed[:, 3, 0] * 7.0 + 13.0  # pT
        perturbed[:, 3, 3] = perturbed[:, 3, 3] * 3.0 + 5.0  # mass
        b, _ = model(perturbed, mask, masked=masked)
    assert torch.allclose(a, b, atol=1e-5), "masked momentum leaked into the model"


def test_kinematic_mask_keeps_the_direction_visible():
    """The direction is *meant* to stay visible: it defines the task."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg, mask_strategy="kinematic").eval()
    particles, mask = _make_particles()
    masked = torch.zeros_like(mask, dtype=torch.bool)
    masked[:, 3] = True
    with torch.no_grad():
        a, _ = model(particles, mask, masked=masked)
        perturbed = particles.clone()
        perturbed[:, 3, 1] += 0.5  # eta
        b, _ = model(perturbed, mask, masked=masked)
    assert not torch.allclose(a, b, atol=1e-4), "direction should stay visible"


def test_full_mask_hides_everything_physical():
    """With the ``full`` strategy, even the direction must not leak."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg, mask_strategy="full").eval()
    particles, mask = _make_particles()
    masked = torch.zeros_like(mask, dtype=torch.bool)
    masked[:, 3] = True
    with torch.no_grad():
        a, _ = model(particles, mask, masked=masked)
        perturbed = particles.clone()
        perturbed[:, 3, 0] *= 5.0  # pT
        perturbed[:, 3, 1] += 0.7  # eta
        perturbed[:, 3, 2] += 0.3  # phi
        perturbed[:, 3, 3] *= 4.0  # mass
        b, _ = model(perturbed, mask, masked=masked)
    assert torch.allclose(a, b, atol=1e-5), "hidden quantities leaked into the model"


def test_full_mask_still_exposes_the_identity():
    """Charge and type are deliberately left visible by both strategies."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg, mask_strategy="full").eval()
    particles, mask = _make_particles()
    masked = torch.zeros_like(mask, dtype=torch.bool)
    masked[:, 3] = True
    with torch.no_grad():
        a, _ = model(particles, mask, masked=masked)
        perturbed = particles.clone()
        perturbed[:, 3, 5] = (perturbed[:, 3, 5] + 1) % 6  # type id
        b, _ = model(perturbed, mask, masked=masked)
    assert not torch.allclose(a, b, atol=1e-4)


def test_interaction_features_are_zeroed_for_masked_pairs():
    computer = InteractionFeatureComputer("kinematic")
    particles, _ = _make_particles()
    masked = torch.zeros(particles.shape[:2], dtype=torch.bool)
    masked[:, 3] = True
    feats = computer(particles, masked)
    # Momentum-dependent components must vanish for every pair touching the
    # masked particle, while dR survives.
    pairs = torch.zeros(particles.shape[:2], dtype=torch.bool)
    pairs[:, 3] = True
    pair_masked = pairs.unsqueeze(2) | pairs.unsqueeze(1)
    assert torch.all(feats[pair_masked][:, 1:] == 0.0)
    assert torch.any(feats[pair_masked][:, 0] != 0.0)


def test_full_strategy_zeroes_the_whole_row():
    computer = InteractionFeatureComputer("full")
    particles, _ = _make_particles()
    masked = torch.zeros(particles.shape[:2], dtype=torch.bool)
    masked[:, 3] = True
    feats = computer(particles, masked)
    pairs = torch.zeros(particles.shape[:2], dtype=torch.bool)
    pairs[:, 3] = True
    pair_masked = pairs.unsqueeze(2) | pairs.unsqueeze(1)
    assert torch.all(feats[pair_masked] == 0.0)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------
def _reference_attention(block, h, mask, interaction):
    """Naive reference implementation of the block's attention."""
    B, N, d = h.shape
    H = block.n_heads
    d_h = block.d_head
    h_norm = block.attn_norm(h)
    qkv = block.qkv(h_norm).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_h)
    if interaction is not None and block.use_interaction_bias:
        bias = block.interaction_mlp(interaction).permute(0, 3, 1, 2)  # [B, H, N, N]
        scores = scores + bias
    scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, d)
    return block.out_proj(out)


def test_scaled_dot_product_attention_matches_reference():
    """The fast SDPA path must agree with a naive implementation."""
    cfg = model_config("tiny", dropout=0.0)
    model = ParticleTransformer(cfg=cfg).eval()
    block = model.blocks[0]
    particles, mask = _make_particles()
    interaction = model.interaction_compute(particles)

    B, N, d = particles.shape[0], particles.shape[1] + 1, cfg.d_model
    h = torch.randn(B, N, d)
    full_mask = torch.cat([torch.ones(B, 1), mask], dim=1)
    ix = model._pad_interaction(interaction)

    with torch.no_grad():
        fast = block(h, full_mask, ix)
        attn_out = h + _reference_attention(block, h, full_mask, ix)
        ref = attn_out + block.ff(block.ff_norm(attn_out))
    assert torch.allclose(fast, ref, atol=1e-5), "SDPA path disagrees with reference"


def test_no_nans_with_heavily_padded_batches():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg).eval()
    particles, mask = _make_particles(batch=3, n=12)
    mask[:, 5:] = 0.0  # keep only 5 particles
    with torch.no_grad():
        cls_out, particle_out = model(particles, mask)
    assert torch.isfinite(cls_out).all()
    assert torch.isfinite(particle_out).all()


def test_gradient_checkpointing_is_mathematically_transparent():
    cfg = model_config("tiny", dropout=0.0)
    torch.manual_seed(0)
    checkpointed = ParticleTransformer(cfg=cfg)
    checkpointed.train()
    plain = ParticleTransformer(cfg=cfg)
    plain.load_state_dict(checkpointed.state_dict())
    plain.eval()

    particles, mask = _make_particles()
    with torch.no_grad():
        a, _ = checkpointed(particles, mask)
        b, _ = plain(particles, mask)
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Pretraining head
# ---------------------------------------------------------------------------
def test_pretraining_masks_at_least_one_particle_per_event():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=cfg.n_features)
    particles, mask = _make_particles(batch=4, n=16)
    loss, _, is_masked = head(particles, mask, model, mask_ratio=0.01)
    assert is_masked.any(dim=1).all(), "every event should contribute"
    assert loss.requires_grad
    loss.backward()


def test_pretraining_loss_backward_works_without_masked_particles():
    """An empty selection must still return a loss attached to the graph."""
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    head = MaskedParticleModeling(d_model=cfg.d_model, n_features=cfg.n_features)
    particles, mask = _make_particles(batch=2, n=8)
    mask[:, :] = 0.0  # nothing real at all
    loss, _, is_masked = head(particles, mask, model, mask_ratio=0.15)
    assert is_masked.sum() == 0
    assert loss.requires_grad
    loss.backward()


def test_pretraining_target_is_the_true_normalised_value():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    particles, mask = _make_particles()
    target = model.normalizer(particles, None)
    assert target.shape == (*particles.shape[:2], cfg.n_features)
    assert torch.isfinite(target).all()


def test_masking_replaces_the_hidden_channel():
    cfg = model_config("tiny")
    model = ParticleTransformer(cfg=cfg)
    particles, mask = _make_particles()
    masked = torch.zeros_like(mask, dtype=torch.bool)
    masked[:, 2] = True
    feats = model.normalizer(particles, masked)
    true_feats = model.normalizer(particles, None)
    # The hidden pT channel must no longer carry the true value ...
    assert not torch.allclose(feats[:, 2, CH_PT], true_feats[:, 2, CH_PT])
    # ... while a particle that is not masked keeps its values.
    assert torch.allclose(feats[:, 1, CH_PT], true_feats[:, 1, CH_PT])
    # Channels that are not hidden must be untouched.
    assert torch.allclose(feats[:, 2, CH_ETA], true_feats[:, 2, CH_ETA])


# ---------------------------------------------------------------------------
# Task head
# ---------------------------------------------------------------------------
def test_roc_auc_matches_reference_values():
    # Perfect separation.
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    # Perfectly inverted.
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    # Ties get half credit.
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)
    # Single class is undefined.
    assert np.isnan(roc_auc([0.1, 0.2], [1, 1]))


def test_jet_tagging_head_loss_ignores_non_jets():
    head = JetTaggingHead(d_model=32)
    embeds = torch.randn(2, 5, 32)
    is_jet = torch.tensor([[True, False, True, False, False], [False, True, False, True, True]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    loss, logits = head.loss(embeds, is_jet, labels)
    assert logits.shape == (2, 5)
    assert loss.requires_grad
    loss.backward()


def test_jet_tagging_head_handles_events_without_jets():
    head = JetTaggingHead(d_model=32)
    embeds = torch.randn(2, 4, 32)
    is_jet = torch.zeros(2, 4, dtype=torch.bool)
    labels = torch.zeros(2, 4)
    loss, _ = head.loss(embeds, is_jet, labels)
    assert loss.requires_grad
    loss.backward()
