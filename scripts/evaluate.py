#!/usr/bin/env python
"""
Evaluate a trained PS3T checkpoint on the held-out test split, reporting
per-slice PSNR / SSIM / RMSE, mean values, and 95% confidence intervals
(Section 2.5, Tables 2/4/6).

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/ps3t_last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps3t.dataset import MayoLDCTDataset, SyntheticSinogramDataset
from ps3t.model import PS3T
from ps3t.utils import compute_psnr, compute_rmse, compute_ssim, load_checkpoint, set_seed, batch_geometry


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PS3T")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--use-synthetic", action="store_true")
    return p.parse_args()


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    ci_half = 1.96 * sem
    return mean, mean - ci_half, mean + ci_half


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.use_synthetic:
        cfg["data"]["use_synthetic"] = True

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["data"].get("use_synthetic", False):
        test_ds = SyntheticSinogramDataset(
            length=64, angles=cfg["model"]["angles"], detector_bins=cfg["model"]["detector_bins"],
            image_size=cfg["model"]["image_size"],
        )
    else:
        test_ds = MayoLDCTDataset(
            data_root=cfg["data"]["data_root"], anatomy=cfg["data"]["anatomy"], split="test",
            patch_size=tuple(cfg["data"]["patch_size"]), seed=cfg["seed"],
            train_frac=cfg["data"]["train_frac"], val_frac=cfg["data"]["val_frac"],
        )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = PS3T(**cfg["model"], geometry=cfg.get("geometry")).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    psnrs, ssims, rmses = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            low = batch["low_dose_sinogram"].to(device)
            ndct = batch["ndct_image"].to(device)
            geometry = batch_geometry(batch, device)
            pred = model(low, geometry=geometry)

            psnrs.append(compute_psnr(pred, ndct))
            ssims.append(compute_ssim(pred, ndct))
            rmses.append(compute_rmse(pred, ndct))

    for name, values in [("PSNR", psnrs), ("SSIM", ssims), ("RMSE", rmses)]:
        mean, lo, hi = mean_ci95(values)
        print(f"{name}: {mean:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]  (n={len(values)})")


if __name__ == "__main__":
    main()
