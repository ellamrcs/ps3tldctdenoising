"""
Projection Embedding + Feature Embedding
=========================================

Implements the front-end of PS3T described in Section 2.1 / Figure 1:

    Projection Embedding : Conv2D + GELU + LayerNorm
    Feature Embedding    : linear projection + positional encoding + LayerNorm

and Eq. (1):

    x_t = W_E * S_t + b_E

The sinogram is treated as a sequence of projection "views" (angles) each
carrying a 1D detector-response vector. We patchify the sinogram into
(angle, detector) tokens with a small Conv2D stem (to pick up local
detector-axis structure), then flatten to a sequence over the angular
dimension and add a learned positional encoding over projection angles.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the angular (sequence) axis."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class ProjectionEmbedding(nn.Module):
    """Conv2D + GELU + LayerNorm projection embedding (Eq. 1), followed by a
    linear "Feature Embedding" head with positional encoding + LayerNorm.

    Input:  sinogram patch  (B, 1, A, D)   A = angles, D = detector bins
    Output: token sequence  (B, A, dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 128,
        conv_channels: int = 32,
        kernel_size: int = 3,
        max_angles: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()

        pad = kernel_size // 2
        # Conv2D + GELU + LayerNorm (Eq. 1: x_t = W_E S_t + b_E realized as a
        # small conv stem operating jointly across angle/detector axes).
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(conv_channels)

        # Feature Embedding: linear projection + positional encoding + LayerNorm
        self.linear_proj = nn.Linear(conv_channels, dim)
        self.pos_encoding = PositionalEncoding(dim, max_len=max_angles)
        self.feature_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sinogram: (B, 1, A, D) raw / low-dose sinogram patch.

        Returns:
            tokens: (B, A, dim) sequence of per-angle projection embeddings,
                    pooled over the detector axis.
        """
        feat = self.conv_embed(sinogram)  # (B, C, A, D)
        # Pool over detector axis to obtain one token per projection angle.
        feat = feat.mean(dim=-1)  # (B, C, A)
        feat = feat.transpose(1, 2)  # (B, A, C)
        feat = self.conv_norm(feat)

        tokens = self.linear_proj(feat)  # (B, A, dim)
        tokens = self.pos_encoding(tokens)
        tokens = self.feature_norm(tokens)
        tokens = self.dropout(tokens)
        return tokens
