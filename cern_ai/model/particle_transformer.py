"""
Particle Transformer (ParT) — scaled for RTX 4050 6GB.

Architecture from "Particle Transformer for Jet Tagging" (Qu et al. 2022).
Key innovation: pairwise interaction features (dR, kT, z, dm) injected as
attention bias before softmax.

Uses gradient checkpointing to fit ~200M+ params in 6 GB VRAM.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ParticleEmbedding(nn.Module):
    """Map raw particle features [pt, eta, phi, mass, charge, pdg, type] → d_model."""

    def __init__(self, n_features=7, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x, mask):
        h = self.net(x)
        h = h * mask.unsqueeze(-1)
        return h


class InteractionFeatureComputer(nn.Module):
    """
    Compute pairwise interaction features between all particle pairs.
    Features: [dR, kT, z, |dm|] — the 4 key features from ParT paper.
    """

    def forward(self, particles):
        pt = particles[:, :, 0]
        eta = particles[:, :, 1]
        phi = particles[:, :, 2]
        mass = particles[:, :, 3]

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
        dR = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-8)

        kT = torch.min(pt_i, pt_j) * dR
        z = pt_i / (pt_i + pt_j + 1e-8)
        dm = torch.abs(mass_i - mass_j)

        return torch.stack([dR, kT, z, dm], dim=-1)


class InteractionBlock(nn.Module):
    """Transformer block with pairwise interaction attention bias."""

    def __init__(self, d_model=128, n_heads=4, d_ff=512, d_interaction=64, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(4, d_interaction),
            nn.GELU(),
            nn.Linear(d_interaction, 1),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h, mask, interaction):
        B, N, d = h.shape
        H = self.n_heads
        d_h = self.d_head

        # --- Self-attention ---
        h_norm = self.attn_norm(h)
        qkv = self.qkv(h_norm).reshape(B, N, 3, H, d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_h)

        # Add interaction bias (shared across heads)
        bias = self.interaction_mlp(interaction).squeeze(-1)  # [B, N, N]
        attn = attn + bias.unsqueeze(1)

        # Mask: zero out padding positions
        attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, d)
        out = self.out_proj(out)
        h = h + out

        # --- Feed-forward ---
        h = h + self.ff(self.ff_norm(h))

        return h


class ParticleTransformer(nn.Module):
    """
    Full Particle Transformer model with gradient checkpointing.

    Input:  particles [B, N, 7], mask [B, N]
    Output: event_embedding [B, d_model], particle_embeddings [B, N, d_model]
    """

    def __init__(self, cfg=None, n_features=7, d_model=128, n_heads=4,
                 n_layers=4, d_ff=512, d_interaction=64, max_particles=64,
                 dropout=0.1, gradient_checkpointing=True):
        super().__init__()
        if cfg is not None:
            n_features = cfg.n_features
            d_model = cfg.d_model
            n_heads = cfg.n_heads
            n_layers = cfg.n_layers
            d_ff = cfg.d_ff
            d_interaction = cfg.d_interaction
            max_particles = cfg.max_particles
            dropout = cfg.dropout
            gradient_checkpointing = getattr(cfg, 'gradient_checkpointing', True)

        self.gradient_checkpointing = gradient_checkpointing
        self.embedding = ParticleEmbedding(n_features, d_model)
        self.interaction_compute = InteractionFeatureComputer()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            InteractionBlock(d_model, n_heads, d_ff, d_interaction, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, particles, mask):
        B, N, _ = particles.shape

        interaction = self.interaction_compute(particles)

        h = self.embedding(particles, mask)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones(B, 1, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        # Expand interaction for cls token (zeros for cls-row/col)
        extra = torch.zeros(B, N + 1, 1, 4, device=particles.device)
        interaction = torch.cat([interaction, extra[:, :N, :, :]], dim=2)
        extra2 = torch.zeros(B, 1, N + 1, 4, device=particles.device)
        interaction = torch.cat([interaction, extra2], dim=1)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(block, h, full_mask, interaction, use_reentrant=False)
            else:
                h = block(h, full_mask, interaction)

        h = self.final_norm(h)

        cls_out = h[:, 0]
        particle_out = h[:, 1:]

        return cls_out, particle_out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
