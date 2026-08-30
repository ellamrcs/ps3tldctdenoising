"""
Differentiable Filtered Backprojection (FBP)
==============================================

Implements the differentiable reconstruction operator R(.) used in
Eq. (14):

    CT_pred = R(S_pred)

for a standard 2D parallel-beam acquisition geometry. The operator applies
a ramp filter along the detector axis in the frequency domain, followed by
a differentiable backprojection realized with bilinear grid sampling
(`torch.nn.functional.grid_sample`), so gradients flow from the
reconstruction-consistency loss (Eq. 15) back into the projection-domain
network.

Notes
-----
* Angular sampling (number of views per rotation, total rotation extent)
  **varies per patient/scanner/protocol** and, for a cropped sinogram
  patch, the patch's starting angle also varies per sample -- see
  `scripts/preprocess_mayo.py` and `ps3t.dataset.MayoLDCTDataset`. This
  module defaults to `angle_range` evenly spaced over the full patch
  (constructor default `angle_range=pi`) but accepts per-sample
  `angular_step_rad` / `angle_start_rad` overrides in `forward()` so each
  sample can use its own true acquisition angles instead of one dataset-wide
  assumption.
* This is a lightweight, dependency-free FBP suitable for research use.
  For clinical-grade fan/cone-beam geometries, prefer `FanBeamFBP`
  (`ps3t/model/fan_fbp.py`) or swap in a dedicated projector (e.g., ASTRA,
  TIGRE, torch-radon) while keeping the same `forward(sinogram) -> image`
  interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ramp_filter(detector_bins: int, device, dtype) -> torch.Tensor:
    """Ram-Lak ramp filter in the frequency domain (length = next pow2 >= 2*D)."""
    n = 1
    while n < 2 * detector_bins:
        n *= 2
    freqs = torch.fft.fftfreq(n, d=1.0).to(device=device, dtype=dtype)
    ramp = 2.0 * torch.abs(freqs)
    return ramp, n


def _as_batch_tensor(value, default, batch_size: int, device, dtype) -> torch.Tensor:
    """Normalizes `value` (None / python scalar / tensor of shape () or (B,))
    into a (B,) tensor, falling back to `default` when `value` is None."""
    if value is None:
        value = default
    if torch.is_tensor(value):
        v = value.to(device=device, dtype=dtype).reshape(-1)
        if v.numel() == 1:
            v = v.expand(batch_size)
        assert v.numel() == batch_size, "geometry tensor must be scalar or length B"
        return v
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


class DifferentiableFBP(nn.Module):
    """Differentiable ramp-filter + backprojection operator for parallel-beam
    sinograms, R(.) in Eq. (14)."""

    def __init__(self, image_size: int | None = None, angle_range: float = math.pi):
        super().__init__()
        self.image_size = image_size
        self.angle_range = angle_range

    def forward(
        self,
        sinogram: torch.Tensor,
        angular_step_rad: torch.Tensor | float | None = None,
        angle_start_rad: torch.Tensor | float | None = None,
        **_ignored,  # tolerate fan-beam-only kwargs (dso_mm, dsd_mm, ...) so
                     # the same `geometry` dict from a batch can be passed to
                     # either FBP variant without branching in caller code.
    ) -> torch.Tensor:
        """
        Args:
            sinogram: (B, 1, A, D) sinogram (angles x detector bins).
            angular_step_rad: optional per-sample angular spacing between
                consecutive views (radians/view), e.g.
                `batch["angular_step_rad"]`. Falls back to
                `self.angle_range / A` when omitted.
            angle_start_rad: optional per-sample starting angle (radians) of
                this sinogram/patch's first view, e.g.
                `batch["angle_start_rad"]`. Defaults to 0.

        Returns:
            image: (B, 1, H, W) reconstructed image, H = W = image_size
                   (defaults to D, the detector bin count).
        """
        B, C, A, D = sinogram.shape
        assert C == 1, "DifferentiableFBP expects a single-channel sinogram"
        device, dtype = sinogram.device, sinogram.dtype
        img_size = self.image_size or D

        ang_step = _as_batch_tensor(angular_step_rad, self.angle_range / A, B, device, dtype)
        ang_start = _as_batch_tensor(angle_start_rad, 0.0, B, device, dtype)

        # ---- 1. Ramp filtering along the detector axis ----
        ramp, n_fft = _ramp_filter(D, device, dtype)
        sino = sinogram.squeeze(1)  # (B, A, D)
        sino_padded = F.pad(sino, (0, n_fft - D))
        sino_fft = torch.fft.fft(sino_padded, dim=-1)
        sino_filtered_fft = sino_fft * ramp.view(1, 1, -1)
        sino_filtered = torch.fft.ifft(sino_filtered_fft, dim=-1).real
        sino_filtered = sino_filtered[..., :D]  # (B, A, D)

        # ---- 2. Backprojection via grid_sample (per-sample angles) ----
        # Build a coordinate grid over the output image in [-1, 1].
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, img_size, device=device, dtype=dtype),
            torch.linspace(-1, 1, img_size, device=device, dtype=dtype),
            indexing="ij",
        )  # (H, W) each
        xs = xs.unsqueeze(0)  # (1, H, W)
        ys = ys.unsqueeze(0)  # (1, H, W)

        view_idx = torch.arange(A, device=device, dtype=dtype).view(1, -1)  # (1, A)
        angles = ang_start.view(-1, 1) + view_idx * ang_step.view(-1, 1)  # (B, A)

        recon = torch.zeros(B, img_size, img_size, device=device, dtype=dtype)

        for a_idx in range(A):
            theta = angles[:, a_idx].view(-1, 1, 1)  # (B, 1, 1), per-sample angle
            t = xs * torch.cos(theta) + ys * torch.sin(theta)  # (B, H, W) in [-1, 1]

            row = sino_filtered[:, a_idx, :].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
            grid_y = torch.zeros_like(t)
            grid = torch.stack([t, grid_y], dim=-1)  # (B, H, W, 2)

            sampled = F.grid_sample(
                row, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )  # (B, 1, H, W)
            recon += sampled.squeeze(1)

        recon = recon * ang_step.view(-1, 1, 1)
        recon = recon.unsqueeze(1)  # (B, 1, H, W)
        return recon
