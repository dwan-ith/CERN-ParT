"""Masked Particle Modelling (MPM) pretraining objective.

The model is asked to reconstruct the *hidden channels* of randomly selected
particles from the rest of the event.  Only the hidden channels contribute to
the loss, so the model is never rewarded for simply copying a visible input.

Two details that matter in practice:

* **Every event contributes.**  With a 15% mask ratio and the particle
  multiplicities of real NanoAOD events, a large fraction of events end up
  with *no* masked particle at all and are silently dropped from the
  gradient.  Here each event with at least one real particle is guaranteed to
  have at least one masked particle.
* **The loss is always attached to the graph.**  An empty selection returns a
  zero loss that is still connected to the computation graph, so
  ``loss.backward()`` never raises.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MASK_CHANNELS, N_MODEL_FEATURES


class MaskedParticleModeling(nn.Module):
    """Self-supervised pretraining head.

    Args:
        d_model: model dimension.
        n_features: width of the normalised feature vector to predict.
        hidden_channels: indices of the channels that are hidden and
            therefore reconstructed.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_features: int = N_MODEL_FEATURES,
        hidden_channels: Sequence[int] = MASK_CHANNELS["kinematic"],
    ) -> None:
        super().__init__()
        self.hidden_channels = tuple(hidden_channels)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_features),
        )

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        model,
        mask_ratio: float = 0.15,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            particles: ``[B, N, 7]`` raw particle features.
            mask: ``[B, N]`` 1 for real particles, 0 for padding.
            model: the :class:`ParticleTransformer` to pretrain.
            mask_ratio: fraction of real particles to hide.
            generator: optional RNG for reproducible masking.

        Returns:
            ``(loss, predictions, is_masked)``.
        """
        B, N, _ = particles.shape
        device = particles.device

        rand = torch.rand(B, N, device=device, generator=generator)
        real = mask.bool()
        is_masked = (rand < mask_ratio) & real

        # Guarantee at least one masked particle per event that has any
        # particle at all, otherwise the event contributes no gradient.
        empty = real.any(dim=1) & ~is_masked.any(dim=1)
        if empty.any():
            rescue = torch.rand(B, N, device=device, generator=generator)
            rescue = rescue.masked_fill(~real, float("inf"))
            pick = rescue.argmin(dim=1)
            is_masked[empty, pick[empty]] = True

        _, particle_embeds = model(particles, mask, masked=is_masked)
        predictions = self.pred_head(particle_embeds)

        # The target is the *true* (unmasked) normalised feature vector.
        target = model.normalizer(particles, None)

        if is_masked.sum() == 0:
            zero = predictions.sum() * 0.0
            return zero, predictions, is_masked

        hidden = list(self.hidden_channels)
        pred_sel = predictions[is_masked][:, hidden]
        target_sel = target[is_masked][:, hidden]
        loss = F.mse_loss(pred_sel, target_sel)
        return loss, predictions, is_masked
