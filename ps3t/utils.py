"""
Utilities: reproducibility, metrics (PSNR / SSIM / RMSE), checkpointing.

Metrics follow Section 2.5: PSNR, SSIM, and RMSE are computed slice-wise
and averaged across the test set.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix the random seed (paper uses seed = 42, Section 2.6) across
    Python, NumPy, and PyTorch for reproducible data splits and
    initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr.item()


@torch.no_grad()
def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


@torch.no_grad()
def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Single-scale SSIM (Wang et al., 2004) for a batch of (B, 1, H, W)
    images, implemented with a Gaussian window convolution."""
    device, dtype = pred.device, pred.dtype
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    pad = window_size // 2
    mu_p = torch.nn.functional.conv2d(pred, window, padding=pad)
    mu_t = torch.nn.functional.conv2d(target, window, padding=pad)

    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad) - mu_p_sq
    sigma_t_sq = torch.nn.functional.conv2d(target * target, window, padding=pad) - mu_t_sq
    sigma_pt = torch.nn.functional.conv2d(pred * target, window, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    )
    return ssim_map.mean().item()


def evaluate_metrics(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> dict:
    return {
        "psnr": compute_psnr(pred, target, data_range),
        "ssim": compute_ssim(pred, target, data_range),
        "rmse": compute_rmse(pred, target),
    }


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def batch_geometry(batch: dict, device) -> dict:
    """Extracts the per-sample scanner + angular geometry tensors from a
    `MayoLDCTDataset` / `SyntheticSinogramDataset` batch and moves them to
    `device`, in the kwarg format expected by
    `PS3T.forward(..., geometry=...)` / `FanBeamFBP.forward(...)` /
    `DifferentiableFBP.forward(...)` (the latter ignores the fan-beam-only
    keys via **_ignored, so this same dict works for either beam type).
    """
    return {
        "dso_mm": batch["dso_mm"].to(device),
        "dsd_mm": batch["dsd_mm"].to(device),
        "detector_spacing_mm": batch["detector_spacing_mm"].to(device),
        "angular_step_rad": batch["angular_step_rad"].to(device),
        "angle_start_rad": batch["angle_start_rad"].to(device),
    }
