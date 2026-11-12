"""
Masked Particle Modeling (MPM) pretraining head.
Randomly masks 15% of particles; model predicts the original 4-vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedParticleModeling(nn.Module):
    """
    Self-supervised pretraining objective for particle physics.
    - Mask 15% of input particles with a learnable [MASK] token
    - Predict the original features (pt, eta, phi, mass, charge, pdg, type)
    - Loss = MSE on masked particles only
    """

    def __init__(self, d_model=128, n_features=7):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, n_features) * 0.02)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_features),
        )

    def forward(self, particles, mask, model, mask_ratio=0.15):
        """
        Args:
            particles: [B, N, 7] raw particle features
            mask: [B, N] binary mask (1 = real particle, 0 = padding)
            model: ParticleTransformer
            mask_ratio: fraction of real particles to mask

        Returns:
            loss: scalar MSE loss
            predictions: [B, N, 7] predicted features
            is_masked: [B, N] bool, which particles were masked
        """
        B, N, _ = particles.shape

        # Select particles to mask (only among real particles)
        rand = torch.rand(B, N, device=particles.device)
        is_masked = (rand < mask_ratio) & mask.bool()

        # Replace masked particles with mask token
        masked_particles = particles.clone()
        mask_expanded = is_masked.unsqueeze(-1).expand_as(masked_particles)
        masked_particles = torch.where(mask_expanded,
                                       self.mask_token.expand(B, N, -1),
                                       masked_particles)

        # Forward through model
        _, particle_embeds = model(masked_particles, mask)

        # Predict original features
        predictions = self.pred_head(particle_embeds)

        # Loss only on masked particles
        if is_masked.sum() == 0:
            return torch.tensor(0.0, device=particles.device), predictions, is_masked

        loss = F.mse_loss(predictions[is_masked], particles[is_masked])

        return loss, predictions, is_masked
