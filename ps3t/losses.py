"""
Multi-Objective Physics-Guided Loss
=====================================

Implements Section 2.4 / Eqs. (12)-(18).

    L_total = lambda1 * L_proj + lambda2 * L_recon + lambda3 * L_poisson

L_proj    (Eq. 13): L1 loss between predicted and reference sinograms.
L_recon   (Eq. 15): L1 loss between FBP(S_pred) and the NDCT reference image.
L_poisson (Eq. 18): Poisson negative log-likelihood between predicted and
                     reference photon counts, Eqs. (16)-(17).

Default weights follow the paper's reported training configuration:
lambda1 = 0.4, lambda2 = 0.4, lambda3 = 0.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def projection_loss(s_pred: torch.Tensor, s_full: torch.Tensor) -> torch.Tensor:
    """Eq. (13): L_proj = || S_pred - S_full ||_1"""
    return torch.nn.functional.l1_loss(s_pred, s_full)


def reconstruction_consistency_loss(ct_pred: torch.Tensor, ct_ndct: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.l1_loss(ct_pred, ct_ndct)


def poisson_physics_loss(
    s_pred: torch.Tensor, s_full: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Eqs. (16)-(18): Poisson negative log-likelihood between predicted and
    reference photon counts.

        P_pred = exp(-S_pred)          Eq. (16)
        P_ref  = exp(-S_full)          Eq. (17)
        L_poisson = sum( P_pred - P_ref * log(P_pred) )   Eq. (18)
    """
    p_pred = torch.exp(-s_pred)
    p_ref = torch.exp(-s_full)
    log_p_pred = torch.log(p_pred.clamp_min(eps))
    loss = (p_pred - p_ref * log_p_pred).mean()
    return loss


@dataclass
class LossWeights:
    lambda1: float = 0.4  # projection loss
    lambda2: float = 0.4  # reconstruction consistency loss
    lambda3: float = 0.2  # Poisson physics loss


class PS3TLoss(nn.Module):
    """Combined multi-objective loss, Eq. (12)."""

    def __init__(self, weights: LossWeights | None = None, poisson_eps: float = 1e-6):
        super().__init__()
        self.weights = weights or LossWeights()
        self.poisson_eps = poisson_eps

    def forward(
        self,
        s_pred: torch.Tensor,
        s_full: torch.Tensor,
        ct_pred: torch.Tensor,
        ct_ndct: torch.Tensor,
    ):
        l_proj = projection_loss(s_pred, s_full)
        l_recon = reconstruction_consistency_loss(ct_pred, ct_ndct)
        l_poisson = poisson_physics_loss(s_pred, s_full, eps=self.poisson_eps)

        total = (
            self.weights.lambda1 * l_proj
            + self.weights.lambda2 * l_recon
            + self.weights.lambda3 * l_poisson
        )

        return total, {
            "loss_total": total.detach(),
            "loss_proj": l_proj.detach(),
            "loss_recon": l_recon.detach(),
            "loss_poisson": l_poisson.detach(),
        }
