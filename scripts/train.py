#!/usr/bin/env python
"""
Train PS3T on the Mayo Clinic LDCT and Projection Dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --use-synthetic   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ps3t.dataset import MayoLDCTDataset, SyntheticSinogramDataset
from ps3t.losses import LossWeights, PS3TLoss
from ps3t.model import PS3T
from ps3t.utils import evaluate_metrics, save_checkpoint, set_seed, batch_geometry


def parse_args():
    p = argparse.ArgumentParser(description="Train PS3T")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--use-synthetic", action="store_true", help="Override config: use synthetic data")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    return p.parse_args()


def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    use_synth = data_cfg.get("use_synthetic", False)

    if use_synth:
        train_ds = SyntheticSinogramDataset(
            length=256, angles=model_cfg["angles"], detector_bins=model_cfg["detector_bins"],
            image_size=model_cfg["image_size"],
        )
        val_ds = SyntheticSinogramDataset(
            length=32, angles=model_cfg["angles"], detector_bins=model_cfg["detector_bins"],
            image_size=model_cfg["image_size"],
        )
    else:
        train_ds = MayoLDCTDataset(
            data_root=data_cfg["data_root"], anatomy=data_cfg["anatomy"], split="train",
            patch_size=tuple(data_cfg["patch_size"]), seed=cfg["seed"],
            train_frac=data_cfg["train_frac"], val_frac=data_cfg["val_frac"],
        )
        val_ds = MayoLDCTDataset(
            data_root=data_cfg["data_root"], anatomy=data_cfg["anatomy"], split="val",
            patch_size=tuple(data_cfg["patch_size"]), seed=cfg["seed"],
            train_frac=data_cfg["train_frac"], val_frac=data_cfg["val_frac"],
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=data_cfg.get("num_workers", 4), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.use_synthetic:
        cfg["data"]["use_synthetic"] = True

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)

    model = PS3T(**cfg["model"], geometry=cfg.get("geometry")).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    criterion = PS3TLoss(weights=LossWeights(**cfg["loss"]))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        running = {"loss_total": 0.0, "loss_proj": 0.0, "loss_recon": 0.0, "loss_poisson": 0.0}

        for step, batch in enumerate(train_loader):
            low = batch["low_dose_sinogram"].to(device, non_blocking=True)
            full = batch["full_dose_sinogram"].to(device, non_blocking=True)
            ndct = batch["ndct_image"].to(device, non_blocking=True)
            geometry = batch_geometry(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                _, s_pred, ct_pred = model(low, return_intermediate=True, geometry=geometry)
                loss, logs = criterion(s_pred, full, ct_pred, ndct)

            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in logs.items():
                running[k] += v.item()

            if step % cfg["train"]["log_every"] == 0:
                print(
                    f"epoch {epoch:03d} step {step:04d} "
                    + " ".join(f"{k}={v.item():.4f}" for k, v in logs.items())
                )

        scheduler.step()
        n_steps = max(1, len(train_loader))
        print(
            f"[epoch {epoch:03d}] "
            + " ".join(f"{k}={v / n_steps:.4f}" for k, v in running.items())
            + f" time={time.time() - t0:.1f}s"
        )

        if (epoch + 1) % cfg["train"].get("val_every", 1) == 0:
            validate(model, val_loader, device)

        save_checkpoint(checkpoint_dir / f"ps3t_epoch{epoch:03d}.pt", model, optimizer, epoch)
        save_checkpoint(checkpoint_dir / "ps3t_last.pt", model, optimizer, epoch)

    print("Training complete.")


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    metrics_sum = {"psnr": 0.0, "ssim": 0.0, "rmse": 0.0}
    n = 0
    for batch in val_loader:
        low = batch["low_dose_sinogram"].to(device)
        ndct = batch["ndct_image"].to(device)
        geometry = batch_geometry(batch, device)
        pred = model(low, geometry=geometry)
        m = evaluate_metrics(pred, ndct)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
        n += 1
    n = max(1, n)
    print("[val] " + " ".join(f"{k}={v / n:.4f}" for k, v in metrics_sum.items()))
    model.train()


if __name__ == "__main__":
    main()
