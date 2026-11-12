"""Downstream task head: jet b-tagging from the event context.

The pretraining objective learns a representation of *events*.  To show that
the representation is useful we attach a small classifier to the per-jet
embeddings and ask it to predict whether a jet is b-tagged.

The classification is per jet rather than per event so that a single NanoAOD
event contributes many training targets, which is what makes the task
tractable on the amount of data that can be streamed in a few minutes.

Note on what this task is (and is not)
--------------------------------------
The published CMS open data used here stores *resolved objects* (muons,
electrons, photons, jets, taus) rather than the individual particle-flow
candidates that a jet is made of.  The classifier therefore sees the jet as a
single particle among the other event objects.  It is not a
state-of-the-art jet tagger and it is not meant to be -- the point of the
benchmark is the *relative* comparison between a pretrained encoder and a
randomly initialised one, trained identically.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class JetTaggingHead(nn.Module):
    """Binary classifier attached to the per-particle embeddings.

    Args:
        d_model: model dimension.
        dropout: dropout probability inside the head.
    """

    def __init__(self, d_model: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(d_model // 4, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, particle_embeds: torch.Tensor) -> torch.Tensor:
        """Args:
            particle_embeds: ``[B, N, d_model]``.

        Returns:
            ``[B, N]`` logits.
        """
        return self.net(particle_embeds).squeeze(-1)

    def loss(
        self,
        particle_embeds: torch.Tensor,
        is_jet: torch.Tensor,
        labels: torch.Tensor,
        min_pt: float = 0.0,
        particle_pt: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-entropy over the jets of the batch.

        Args:
            particle_embeds: ``[B, N, d_model]``.
            is_jet: ``[B, N]`` boolean, true for jets.
            labels: ``[B, N]`` binary labels for jets.
            min_pt: optional minimum jet pT in GeV.
            particle_pt: optional ``[B, N]`` transverse momenta, required if
                ``min_pt > 0``.

        Returns:
            ``(loss, logits)``.
        """
        logits = self(particle_embeds)
        select = is_jet.bool()
        if min_pt > 0.0:
            if particle_pt is None:
                raise ValueError("particle_pt is required when min_pt > 0")
            select = select & (particle_pt >= min_pt)
        if select.sum() == 0:
            zero = logits.sum() * 0.0
            return zero, logits
        loss = F.binary_cross_entropy_with_logits(
            logits[select].float(), labels[select].float()
        )
        return loss, logits


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve, computed without external dependencies.

    Ties are handled by the standard (Mann-Whitney) formulation.  Returns
    ``nan`` when there is only one class present.
    """
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = average_rank
        i = j + 1
    positive_rank_sum = ranks[sorted_labels > 0.5].sum()
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
