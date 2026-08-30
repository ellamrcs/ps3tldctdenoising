"""
Fan-Beam Differentiable FBP (equispaced detector)
====================================================

`DifferentiableFBP` in `fbp.py` implements a parallel-beam approximation.
This module implements the standard equispaced-detector fan-beam filtered
backprojection algorithm (Kak & Slaney, *Principles of Computerized
Tomographic Imaging*, Ch. 3, Eqs. 3.60-3.61).

Acquisition geometry -- DSO (source-to-isocenter), DSD (source-to-detector),
and detector pixel spacing -- **varies per patient/series** in the Mayo
Clinic LDCT and Projection Dataset (different scanners/protocols across the
299 patients), so this module accepts geometry either as fixed defaults
(constructor args, used as a fallback) or as **per-sample tensors passed to
`forward()`**, sourced from each patient's own DICOM headers via
`scripts/preprocess_mayo.py` / `ps3t/dataset.py`. Do not assume a single
global (DSO, DSD, spacing) triplet is correct for the whole dataset.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fbp import _ramp_filter


def _as_batch_tensor(value, default: float, batch_size: int, device, dtype) -> torch.Tensor:
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


class FanBeamFBP(nn.Module):
    """Equispaced-detector fan-beam differentiable FBP, R(.) in Eq. (14),
    supporting per-sample (per-patient) scanner geometry.
    """

    def __init__(
        self,
        dso_mm: float,
        dsd_mm: float,
        detector_spacing_mm: float,
        num_detectors: int,
        image_size: int,
        pixel_spacing_mm: float | None = None,
        angle_range: float = 2 * 3.141592653589793,
    ):
        """
        Args:
            dso_mm, dsd_mm, detector_spacing_mm: DEFAULT/fallback geometry,
                used whenever `forward()` is not given per-sample overrides
                (e.g. for the synthetic dataset, or if a patient's DICOM
                metadata is unavailable). Prefer passing real per-sample
                values to `forward()` whenever possible.
            pixel_spacing_mm: reconstruction pixel spacing; defaults to
                detector_spacing_mm if not given.
        """
        super().__init__()
        self.dso_default = dso_mm
        self.dsd_default = dsd_mm
        self.detector_spacing_default = detector_spacing_mm
        self.num_detectors = num_detectors
        self.image_size = image_size
        self.pixel_spacing_default = pixel_spacing_mm or detector_spacing_mm
        self.angle_range = angle_range

    def forward(
        self,
        sinogram: torch.Tensor,
        dso_mm: torch.Tensor | float | None = None,
        dsd_mm: torch.Tensor | float | None = None,
        detector_spacing_mm: torch.Tensor | float | None = None,
        pixel_spacing_mm: torch.Tensor | float | None = None,
        angular_step_rad: torch.Tensor | float | None = None,
        angle_start_rad: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            sinogram: (B, 1, A, D) fan-beam sinogram, D == num_detectors.
            dso_mm, dsd_mm, detector_spacing_mm, pixel_spacing_mm: optional
                per-sample geometry overrides, each either a python float
                (applied to the whole batch) or a (B,) tensor (one value per
                patient/sample in the batch -- e.g. from
                `batch["dso_mm"]` produced by `MayoLDCTDataset`). Falls back
                to this module's constructor defaults when omitted.
            angular_step_rad: optional per-sample angular spacing between
                consecutive views (radians/view), e.g. `batch["angular_step_rad"]`.
                Number of projection views per rotation and total rotation
                extent both vary by patient/scanner/protocol -- see
                `scripts/preprocess_mayo.py`. Falls back to
                `self.angle_range / A` (this module's constructor default)
                when omitted.
            angle_start_rad: optional per-sample starting angle (radians) of
                this sinogram/patch's first view within the full acquisition,
                e.g. `batch["angle_start_rad"]` (needed because sinograms are
                often trained on as a cropped angular patch, not the full
                rotation -- see `MayoLDCTDataset`). Defaults to 0.

        Returns:
            image: (B, 1, H, W), H = W = image_size.
        """
        B, C, A, D = sinogram.shape
        assert C == 1, "FanBeamFBP expects a single-channel sinogram"
        device, dtype = sinogram.device, sinogram.dtype

        dso = _as_batch_tensor(dso_mm, self.dso_default, B, device, dtype)  # (B,)
        dsd = _as_batch_tensor(dsd_mm, self.dsd_default, B, device, dtype)  # (B,)
        ds = _as_batch_tensor(
            detector_spacing_mm, self.detector_spacing_default, B, device, dtype
        )  # (B,)
        px = _as_batch_tensor(
            pixel_spacing_mm, self.pixel_spacing_default, B, device, dtype
        )  # (B,)
        ang_step = _as_batch_tensor(
            angular_step_rad, self.angle_range / A, B, device, dtype
        )  # (B,)
        ang_start = _as_batch_tensor(angle_start_rad, 0.0, B, device, dtype)  # (B,)

        # Cosine weighting (per-sample detector spacing / DSD) ----
        det_idx = torch.arange(D, device=device, dtype=dtype) - (D - 1) / 2.0  # (D,)
        t = det_idx.view(1, -1) * ds.view(-1, 1)  # (B, D), physical mm coordinate
        cos_weight = dsd.view(-1, 1) / torch.sqrt(dsd.view(-1, 1) ** 2 + t ** 2)  # (B, D)
        sino = sinogram.squeeze(1) * cos_weight.unsqueeze(1)  # (B, A, D)


        ramp, n_fft = _ramp_filter(D, device, dtype)
        sino_padded = F.pad(sino, (0, n_fft - D))
        sino_fft = torch.fft.fft(sino_padded, dim=-1)
        sino_filtered = torch.fft.ifft(sino_fft * ramp.view(1, 1, -1), dim=-1).real
        sino_filtered = sino_filtered[..., :D]  # (B, A, D)

        unit_lin = torch.linspace(-1, 1, self.image_size, device=device, dtype=dtype)
        uy, ux = torch.meshgrid(unit_lin, unit_lin, indexing="ij")  # (H, W), unit square

        half_extent = (self.image_size * px) / 2.0  # (B,), physical mm half-width
        xs = ux.unsqueeze(0) * half_extent.view(-1, 1, 1)  # (B, H, W)
        ys = uy.unsqueeze(0) * half_extent.view(-1, 1, 1)  # (B, H, W)

        view_idx = torch.arange(A, device=device, dtype=dtype).view(1, -1)  # (1, A)
        angles = ang_start.view(-1, 1) + view_idx * ang_step.view(-1, 1)  # (B, A)

        recon = torch.zeros(B, self.image_size, self.image_size, device=device, dtype=dtype)

        dso_b = dso.view(-1, 1, 1)
        dsd_b = dsd.view(-1, 1, 1)
        ds_b = ds.view(-1, 1, 1)

        for a_idx in range(A):
            beta = angles[:, a_idx].view(-1, 1, 1)  # (B, 1, 1), per-sample angle
            cos_b, sin_b = torch.cos(beta), torch.sin(beta)

            x_r = xs * cos_b + ys * sin_b
            y_r = -xs * sin_b + ys * cos_b

            U = (dso_b + y_r) / dso_b
            t_prime = dsd_b * x_r / (dso_b + y_r)

            t_norm = t_prime / ((D - 1) / 2.0 * ds_b)  # (B, H, W)
            grid_y = torch.zeros_like(t_norm)
            grid = torch.stack([t_norm, grid_y], dim=-1)  # (B, H, W, 2)

            row = sino_filtered[:, a_idx, :].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
            sampled = F.grid_sample(
                row, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).squeeze(1)  # (B, H, W)

            recon += sampled / (U ** 2)

        recon = recon * (ang_step.view(-1, 1, 1) / 2.0)
        return recon.unsqueeze(1)  # (B, 1, H, W)
