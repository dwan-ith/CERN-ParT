"""The Particle Transformer (ParT) architecture.

Reference
---------
Qu, Li, Moutinho, Wang, Zhang, "Particle Transformer for Jet Tagging",
arXiv:2206.11833 (2022).

The defining feature of ParT is that the pairwise interaction between two
particles is not learned *implicitly* from the individual particle
representations, but is injected explicitly into the attention logits as a
bias computed from physical interaction variables.  For particles ``i`` and
``j`` those are

===============  =========================================
``dR``           ``sqrt(deta^2 + dphi^2)``
``kT``           ``min(pt_i, pt_j) * dR``
``z``            ``pt_i / (pt_i + pt_j)``
``dm``           ``|mass_i - mass_j|``
===============  =========================================

Masking design
--------------
Masked-particle-modelling has to be *leak free*: the model must not be able to
recover the hidden quantity from the quantities that remain visible.  Two
strategies are implemented and both are covered by tests.

``kinematic`` (default)
    The particle's *direction* (eta, phi) and its identity (charge, type) stay
    visible; its momentum (``pt``) and ``mass`` are hidden.  The interaction
    features that depend on momentum (``kT``, ``z``, ``dm``) are zeroed for
    every pair that involves the masked particle, while the purely geometric
    ``dR`` is kept.  The task is therefore *"reconstruct the momentum of a
    particle whose direction you can see"* -- well posed, and physically the
    closest analogue of masked language modelling for collider events.

``full``
    Everything except charge and type is hidden, and *all* interaction
    features involving the masked particle are zeroed, including ``dR``.  The
    model has to infer both where the particle was and how fast it was.

In both cases the hidden channels are replaced by learned values, so the
model always knows that a channel is hidden rather than genuinely zero.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import (
    CH_COSPHI,
    CH_ETA,
    CH_MASS,
    CH_PT,
    CH_SINPHI,
    CH_TYPE0,
    IX_DM,
    IX_DR,
    IX_KT,
    IX_Z,
    MASK_CHANNELS,
    N_MODEL_FEATURES,
    N_RAW_FEATURES,
    OBJECT_TYPES,
    ModelConfig,
)


class FeatureNormalizer(nn.Module):
    """Map raw particle features onto a well-conditioned model input.

    The momenta are log-scaled, the azimuthal angle is encoded as
    ``sin``/``cos`` (so that it is continuous across the +/-pi boundary) and
    the object type is one-hot encoded.  Every channel ends up O(1), which
    matters a lot: feeding raw ``pT`` values in the hundreds of GeV into the
    embedding makes the reconstruction loss explode and the optimiser
    unstable.

    The module also implements *channel masking*: for particles selected by
    the pretraining objective, the hidden channels are overwritten with
    learned values so that the model can tell "hidden" from "genuinely zero".
    """

    def __init__(
        self,
        n_raw_features: int = N_RAW_FEATURES,
        n_features: int = N_MODEL_FEATURES,
        masked_channels: Tuple[int, ...] = MASK_CHANNELS["kinematic"],
        n_types: int = len(OBJECT_TYPES),
    ) -> None:
        super().__init__()
        if n_raw_features != N_RAW_FEATURES:
            raise ValueError(
                f"expected {N_RAW_FEATURES} raw feature channels, got {n_raw_features}"
            )
        self.n_features = n_features
        self.n_types = n_types

        # Learned replacement values for the hidden channels.
        self.mask_value = nn.Parameter(torch.zeros(n_features))

        channel_mask = torch.zeros(n_features, dtype=torch.bool)
        for channel in masked_channels:
            channel_mask[channel] = True
        self.register_buffer("masked_channel_mask", channel_mask, persistent=False)

    def forward(
        self, particles: torch.Tensor, masked: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:
            particles: ``[B, N, 7]`` raw features.
            masked: optional ``[B, N]`` boolean mask of hidden particles.
        """
        pt = particles[..., 0].clamp(min=0.0)
        eta = particles[..., 1]
        phi = particles[..., 2]
        mass = particles[..., 3].clamp(min=0.0)
        charge = particles[..., 4]
        type_id = particles[..., 5]

        feats = particles.new_zeros(*particles.shape[:-1], self.n_features)
        feats[..., CH_PT] = torch.log1p(pt) / 5.0
        feats[..., CH_ETA] = eta / 5.0
        feats[..., CH_SINPHI] = torch.sin(phi)
        feats[..., CH_COSPHI] = torch.cos(phi)
        feats[..., CH_MASS] = torch.log1p(mass) / 5.0
        for t in range(self.n_types):
            feats[..., CH_TYPE0 + t] = (type_id == float(t)).to(feats.dtype)

        if masked is not None:
            hidden = masked.unsqueeze(-1) & self.masked_channel_mask
            feats = torch.where(hidden, self.mask_value.to(feats.dtype), feats)
        return feats


class InteractionFeatureComputer(nn.Module):
    """Compute the pairwise ParT interaction features.

    The features are computed from the *raw* physical quantities (not from the
    normalised model input) and are rescaled to be O(1) before they enter the
    interaction network.
    """

    def __init__(self, strategy: str = "kinematic") -> None:
        super().__init__()
        if strategy not in MASK_CHANNELS:
            raise KeyError(f"unknown masking strategy {strategy!r}")
        self.strategy = strategy

        # Which interaction components survive for a pair that involves a
        # masked particle.  ``kinematic`` keeps the geometric ``dR`` only.
        keep = torch.zeros(4)
        if strategy == "kinematic":
            keep[IX_DR] = 1.0
        self.register_buffer("keep_components", keep, persistent=False)

    def forward(
        self, particles: torch.Tensor, masked: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:
            particles: ``[B, N, 7]`` raw features.
            masked: optional ``[B, N]`` boolean mask of hidden particles.

        Returns:
            ``[B, N, N, 4]`` interaction features.
        """
        pt = particles[..., 0].clamp(min=0.0)
        eta = particles[..., 1]
        phi = particles[..., 2]
        mass = particles[..., 3].clamp(min=0.0)

        pt_i = pt.unsqueeze(2)
        pt_j = pt.unsqueeze(1)
        eta_i = eta.unsqueeze(2)
        eta_j = eta.unsqueeze(1)
        phi_i = phi.unsqueeze(2)
        phi_j = phi.unsqueeze(1)
        mass_i = mass.unsqueeze(2)
        mass_j = mass.unsqueeze(1)

        deta = eta_i - eta_j
        dphi = phi_i - phi_j
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
        dR = torch.sqrt(deta**2 + dphi**2 + 1e-8)

        kT = torch.minimum(pt_i, pt_j) * dR
        z = pt_i / (pt_i + pt_j + 1e-8)
        dm = torch.abs(mass_i - mass_j)

        feats = torch.stack(
            [
                dR / 5.0,
                torch.log1p(kT),
                z,
                torch.log1p(dm),
            ],
            dim=-1,
        )

        if masked is not None:
            pair_masked = masked.unsqueeze(2) | masked.unsqueeze(1)  # [B, N, N]
            keep = self.keep_components.to(feats.dtype)
            feats = feats * keep
            # Zero the momentum-dependent components for pairs that involve a
            # masked particle; ``dR`` is kept in the kinematic strategy.
            drop = pair_masked.unsqueeze(-1) & (keep < 0.5)
            feats = feats.masked_fill(drop, 0.0)
        return feats


class ParticleEmbedding(nn.Module):
    """Two-layer MLP that maps normalised features onto the model dimension."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionBlock(nn.Module):
    """Pre-norm transformer block with pairwise interaction attention bias.

    The bias is produced by a small MLP from the four interaction features.
    Following ParT, the network outputs one bias per attention head rather
    than a single shared bias, which lets different heads specialise on
    different interaction variables.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        d_interaction: int = 128,
        dropout: float = 0.1,
        use_interaction_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_interaction_bias = use_interaction_bias

        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        if use_interaction_bias:
            self.interaction_mlp = nn.Sequential(
                nn.Linear(4, n_heads * d_interaction),
                nn.GELU(),
                nn.Linear(n_heads * d_interaction, n_heads),
            )

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        interaction: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Args:
            h: ``[B, N, d_model]``
            mask: ``[B, N]`` 1 for real particles (and for the CLS token).
            interaction: ``[B, N, N, 4]`` or ``None``.
        """
        B, N, d = h.shape
        H = self.n_heads
        d_h = self.d_head

        h_norm = self.attn_norm(h)
        qkv = self.qkv(h_norm).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if self.use_interaction_bias and interaction is not None:
            bias = self.interaction_mlp(interaction)  # [B, N, N, H]
            bias = bias.permute(0, 3, 1, 2)  # [B, H, N, N]
            attn_mask = bias

        # Padding is enforced by masking the *keys*.  The CLS token is always
        # a valid key, so no row is ever fully masked and the softmax can
        # never produce NaNs.
        key_pad = mask.unsqueeze(1).unsqueeze(2) == 0  # [B, 1, 1, N]
        if attn_mask is None:
            attn_mask = torch.zeros(
                B, H, N, N, device=h.device, dtype=q.dtype
            ).masked_fill(key_pad, float("-inf"))
        else:
            attn_mask = attn_mask.masked_fill(key_pad, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = self.attn_drop(out)
        out = out.transpose(1, 2).reshape(B, N, d)
        out = self.out_proj(out)
        h = h + self.dropout(out)

        h = h + self.ff(self.ff_norm(h))
        return h


class ParticleTransformer(nn.Module):
    """Particle Transformer encoder.

    Input is a set of particles plus a validity mask; output is a per-event
    embedding (the CLS token) and per-particle embeddings.

    Args:
        cfg: architecture configuration.
        mask_strategy: which channels the pretraining objective hides.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, mask_strategy: str = "kinematic", **overrides):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig(**overrides)
        elif overrides:
            cfg = ModelConfig(**{**cfg.to_dict(), **overrides})
        self.cfg = cfg
        self.mask_strategy = mask_strategy

        self.normalizer = FeatureNormalizer(
            n_raw_features=cfg.n_raw_features,
            n_features=cfg.n_features,
            masked_channels=MASK_CHANNELS[mask_strategy],
            n_types=len(OBJECT_TYPES),
        )
        self.embedding = ParticleEmbedding(cfg.n_features, cfg.d_model)
        self.interaction_compute = InteractionFeatureComputer(mask_strategy)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        #: Added to the embedding of every particle whose features are
        #: hidden, so the model can tell "hidden" from "genuinely zero".
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    d_interaction=cfg.d_interaction,
                    dropout=cfg.dropout,
                    use_interaction_bias=cfg.use_interaction_bias,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Scale the residual projections down (GPT-2 style) so that deep
        # stacks do not blow up at initialisation.
        for block in self.blocks:
            nn.init.normal_(block.out_proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            # ``ff`` is Linear -> GELU -> Dropout -> Linear -> Dropout
            nn.init.normal_(block.ff[3].weight, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def _pad_interaction(self, interaction: torch.Tensor) -> torch.Tensor:
        """Insert the CLS token into the interaction tensor.

        The CLS token is prepended to the hidden states, so it has to be
        *prepended* to the interaction tensor as well: a zero row for the
        CLS query and a zero column for the CLS key.  Appending them at the
        end instead would silently pair the CLS token with the last
        particle's interaction features.
        """
        B, N, _, F = interaction.shape
        zeros_row = interaction.new_zeros(B, 1, N, F)
        interaction = torch.cat([zeros_row, interaction], dim=1)
        zeros_col = interaction.new_zeros(B, N + 1, 1, F)
        return torch.cat([zeros_col, interaction], dim=2)

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        masked: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            particles: ``[B, N, 7]`` raw particle features.
            mask: ``[B, N]`` 1 for real particles, 0 for padding.
            masked: optional ``[B, N]`` boolean mask of particles whose
                features are hidden.

        Returns:
            ``(cls_out, particle_out)`` of shapes ``[B, d_model]`` and
            ``[B, N, d_model]``.
        """
        B, N, _ = particles.shape

        features = self.normalizer(particles, masked)
        h = self.embedding(features)
        h = h * mask.unsqueeze(-1).to(h.dtype)
        if masked is not None:
            h = h + masked.unsqueeze(-1).to(h.dtype) * self.mask_embedding

        if self.cfg.use_interaction_bias:
            interaction = self.interaction_compute(particles, masked)
        else:
            interaction = None

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        if interaction is not None:
            # The CLS token has no physical counterpart: its row and column
            # of interaction features are zero.
            interaction = self._pad_interaction(interaction)

        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                h = checkpoint(
                    block, h, full_mask, interaction, use_reentrant=False
                )
            else:
                h = block(h, full_mask, interaction)

        h = self.final_norm(h)
        return h[:, 0], h[:, 1:]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
