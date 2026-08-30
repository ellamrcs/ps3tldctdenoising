"""
Projection Reconstruction
===========================

Implements the "Projection Reconstruction" block of Figure 1
(linear decoder + feature reshaping + sinogram reconstruction) and Eq. (11):

    S_pred = W_r y_t + b_r
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionReconstruction(nn.Module):
    """Maps the denoised per-angle feature sequence back to a full-resolution
    sinogram patch via a linear decoder followed by a 1x1 convolutional
    reconstruction layer (Eq. 11)."""

    def __init__(self, dim: int, detector_bins: int, out_channels: int = 1):
        super().__init__()
        self.detector_bins = detector_bins
        self.decoder = nn.Linear(dim, detector_bins)
        self.reconstruct = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, A, dim) denoised projection features.

        Returns:
            S_pred: (B, 1, A, D) reconstructed / denoised sinogram patch.
        """
        feat = self.decoder(y)  # (B, A, D)
        feat = feat.unsqueeze(1)  # (B, 1, A, D)
        S_pred = self.reconstruct(feat)  # Eq. (11)
        return S_pred
