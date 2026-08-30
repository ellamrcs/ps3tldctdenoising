"""
Photon-Aware Physics Attention
===============================

Implements Section 2.2 / Figure 2 / Eqs. (2)-(6).

Photon estimation (Beer-Lambert law), Eq. (2):
    P_t = exp(-S_t)

Uncertainty map, Eq. (3):
    U_t = 1 / sqrt(P_t + eps)

Q, K, V projections, Eq. (4):
    Q_t = W_Q x_t,  K_t = W_K x_t,  V_t = W_V x_t

Photon-aware attention, Eq. (5):
    A_t = softmax( (Q_t K_t^T) / sqrt(d) - alpha * U_t )

Output, Eq. (6):
    z_t = A_t V_t

The uncertainty term penalizes attention over photon-starved (high
uncertainty) projections, emphasizing measurements with higher photon
reliability -- consistent with the multiplicative e^{-U_t} gating shown
in Figure 2 (softmax(QK^T (.) e^{-U_t}) V), implemented here in additive
log-space form for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotonEstimator(nn.Module):
    """Computes the photon-count estimate P_t and uncertainty map U_t
    directly from the raw (noisy) sinogram, Eqs. (2)-(3)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sinogram: (B, 1, A, D) raw sinogram patch (line-integral / -log
                      attenuation values, S_t).

        Returns:
            U: (B, A) per-angle uncertainty map, averaged over the detector axis.
        """
        P = torch.exp(-sinogram)  # Eq. (2)
        P = P.mean(dim=-1).squeeze(1)  # (B, A) average photon estimate per angle
        U = 1.0 / torch.sqrt(P + self.eps)  # Eq. (3)
        return U


class PhotonAwareAttention(nn.Module):
    """Multi-head self-attention over projection angles, dynamically
    conditioned on the photon-estimation uncertainty map (Eqs. 4-6)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0, eps: float = 1e-3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.photon_estimator = PhotonEstimator(eps=eps)

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable scaling parameter alpha (Eq. 5)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sinogram: torch.Tensor):
        """
        Args:
            x:        (B, A, dim) projection-embedded token sequence.
            sinogram: (B, 1, A, D) raw sinogram, used to compute U_t.

        Returns:
            z:  (B, A, dim) photon-aware attended features (Eq. 6).
            U:  (B, A) uncertainty map, returned for reuse by the
                downstream state-space module (Eqs. 7-9).
        """
        B, A, _ = x.shape
        U = self.photon_estimator(sinogram)  # (B, A)

        q = self.w_q(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, A, A)

        # Penalize attention to keys (angles) with high photon uncertainty.
        uncertainty_bias = self.alpha * U.unsqueeze(1).unsqueeze(2)  # (B,1,1,A)
        logits = logits - uncertainty_bias

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        z = torch.matmul(attn, v)  # (B, H, A, head_dim)
        z = z.transpose(1, 2).contiguous().view(B, A, self.dim)
        z = self.out_proj(z)
        return z, U
