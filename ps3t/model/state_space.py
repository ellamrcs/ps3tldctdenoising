"""
Sequential State Space Evolution
==================================

Implements Section 2.3 / Eqs. (7)-(11).

Hidden attenuation state recurrence, Eq. (7):
    h_t = A(U_t) h_{t-1} + B(U_t) z_t

Uncertainty-conditioned transition matrices, Eqs. (8)-(9):
    A(U_t) = A0 + alpha_A * U_t
    B(U_t) = B0 + alpha_B * U_t

Denoised projection features, Eq. (10):
    y_t = C h_t + D z_t

The state evolution runs sequentially (linear complexity) across the
angular/projection dimension, so that photon-starved views borrow more
information from neighboring angles (via a larger effective A(U_t)) while
reliable views retain stronger self-dependence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequentialStateSpaceBlock(nn.Module):
    """Uncertainty-conditioned linear state-space recurrence over the
    projection-angle axis, with a residual connection + LayerNorm
    (see Figure 1 caption: "selective state update + residual connection
    + LayerNorm")."""

    def __init__(self, dim: int, state_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # Base transition matrices A0 (state_dim x state_dim) and
        # B0 (state_dim x dim).
        self.A0 = nn.Parameter(torch.eye(state_dim) * 0.9 + 0.01 * torch.randn(state_dim, state_dim))
        self.B0 = nn.Parameter(0.01 * torch.randn(state_dim, dim))

        # Learnable scalar modulation coefficients alpha_A, alpha_B (Eqs. 8-9).
        self.alpha_A = nn.Parameter(torch.tensor(0.1))
        self.alpha_B = nn.Parameter(torch.tensor(0.1))

        # Output matrices C (dim x state_dim) and D (dim x dim), Eq. (10).
        self.C = nn.Linear(state_dim, dim, bias=False)
        self.D = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, A, dim) photon-aware attended features (Eq. 6 output).
            U: (B, A) uncertainty map (shared with the attention module).

        Returns:
            y: (B, A, dim) denoised projection features after residual +
               LayerNorm.
        """
        B_, A_, _ = z.shape
        device, dtype = z.device, z.dtype

        h = torch.zeros(B_, self.state_dim, device=device, dtype=dtype)
        ys = []

        # U broadcast scalar per angle modulates the base transition matrices.
        for t in range(A_):
            z_t = z[:, t, :]  # (B, dim)
            u_t = U[:, t].view(B_, 1, 1)  # (B, 1, 1)

            A_t = self.A0.unsqueeze(0) + self.alpha_A * u_t  # (B, state_dim, state_dim)
            B_t = self.B0.unsqueeze(0) + self.alpha_B * u_t  # (B, state_dim, dim)

            h = torch.bmm(A_t, h.unsqueeze(-1)).squeeze(-1) + torch.bmm(
                B_t, z_t.unsqueeze(-1)
            ).squeeze(-1)  # Eq. (7)

            y_t = self.C(h) + self.D(z_t)  # Eq. (10)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, A, dim)
        y = self.dropout(y)
        y = self.norm(y + z)  # residual connection + LayerNorm
        return y
