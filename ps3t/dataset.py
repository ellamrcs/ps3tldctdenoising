"""
Dataset utilities
====================

Loader for the Mayo Clinic LDCT and Projection Dataset (via TCIA,
https://doi.org/10.7937/9NPB-2637), following the patient-level
70% / 15% / 15% split described in Section 2.6 / Table 1.

Expected on-disk layout (after you export the TCIA DICOM/projection data
to per-patient numpy arrays -- see `scripts/preprocess_mayo.py` for a
starting point):

    data_root/
      abdomen/
        L001/
          low_dose_sinograms.npy   # (num_slices, A, D)  float32
          full_dose_sinograms.npy  # (num_slices, A, D)  float32
          ndct_images.npy          # (num_slices, H, W)  float32
        L002/
          ...
      chest/
        C001/
          ...
      head/
        N001/
          ...

Because the raw Mayo Clinic dataset requires a TCIA data-use agreement
and per-site DICOM-to-array preprocessing that is outside the scope of
this repository, `MayoLDCTDataset` operates on the preprocessed layout
above. A lightweight `SyntheticSinogramDataset` is also provided so the
full pipeline (training loop, losses, metrics) can be exercised without
downloading any data.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

Anatomy = Literal["abdomen", "chest", "head"]
Split = Literal["train", "val", "test"]


def patient_level_split(
    patient_ids: Sequence[str], seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.15
) -> dict:
    """Reproduces the patient-level 70/15/15 split of Table 1 with a fixed
    random seed (seed = 42 in the paper) to prevent any slice-level leakage
    between subsets.
    """
    ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


class MayoLDCTDataset(Dataset):
    """Patient-level-partitioned dataset over preprocessed Mayo Clinic LDCT
    sinogram/image triplets for a single anatomical region.
    """

    def __init__(
        self,
        data_root: str | Path,
        anatomy: Anatomy,
        split: Split,
        patch_size: tuple[int, int] = (64, 64),
        seed: int = 42,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ):
        self.data_root = Path(data_root) / anatomy
        self.patch_size = patch_size

        patient_dirs = sorted(p.name for p in self.data_root.iterdir() if p.is_dir())
        splits = patient_level_split(
            patient_dirs, seed=seed, train_frac=train_frac, val_frac=val_frac
        )
        self.patient_ids: List[str] = splits[split]

        # Index of (patient_id, slice_idx) pairs, built strictly after
        # patient-level assignment (Section 2.6).
        self.index: List[tuple[str, int]] = []
        for pid in self.patient_ids:
            low = np.load(self.data_root / pid / "low_dose_sinograms.npy", mmap_mode="r")
            for s in range(low.shape[0]):
                self.index.append((pid, s))

    def __len__(self) -> int:
        return len(self.index)

    def _load_patient(self, pid: str):
        low = np.load(self.data_root / pid / "low_dose_sinograms.npy", mmap_mode="r")
        full = np.load(self.data_root / pid / "full_dose_sinograms.npy", mmap_mode="r")
        ndct = np.load(self.data_root / pid / "ndct_images.npy", mmap_mode="r")
        return low, full, ndct

    def _load_geometry(self, pid: str, slice_idx: int) -> dict:
        """Loads this slice's per-patient scanner geometry (DSO, DSD,
        detector pixel spacing, angular step), extracted from DICOM /
        projection-data headers by `scripts/preprocess_mayo.py`. Falls back
        to the confirmed Mayo Clinic dataset defaults if `geometry.npz` is
        missing for this patient (e.g. not yet (re-)preprocessed with
        geometry extraction).
        """
        geom_path = self.data_root / pid / "geometry.npz"
        if not geom_path.exists():
            return {
                "dso_mm": 595.0,
                "dsd_mm": 1085.6,
                "detector_spacing_mm": 0.7421875,
                # UNVERIFIED literature-derived placeholder (~1160 views per
                # full rotation) -- see scripts/preprocess_mayo.py
                # DEFAULT_ANGULAR_FALLBACK. Confirm against your own raw
                # projection-data headers.
                "angular_step_rad": (2 * np.pi) / 1160.0,
            }
        geom = np.load(geom_path)
        return {
            "dso_mm": float(np.asarray(geom["dso_mm"]).reshape(-1)[slice_idx]),
            "dsd_mm": float(np.asarray(geom["dsd_mm"]).reshape(-1)[slice_idx]),
            "detector_spacing_mm": float(
                np.asarray(geom["detector_spacing_mm"]).reshape(-1)[slice_idx]
            ),
            "angular_step_rad": float(
                np.asarray(geom["angular_step_rad"]).reshape(-1)[slice_idx]
            ),
        }

    def __getitem__(self, idx: int):
        pid, s = self.index[idx]
        low, full, ndct = self._load_patient(pid)
        geometry = self._load_geometry(pid, s)

        low_sino = np.asarray(low[s]).astype(np.float32)
        full_sino = np.asarray(full[s]).astype(np.float32)
        ndct_img = np.asarray(ndct[s]).astype(np.float32)

        ph, pw = self.patch_size
        low_patch, full_patch, angle_start_idx = _random_matching_patch(
            low_sino, full_sino, ph, pw
        )
        # Physical starting angle (radians) of this patch's first view within
        # the full sinogram -- needed so FanBeamFBP/DifferentiableFBP place
        # the patch's views at their true acquisition angles rather than
        # always starting from angle 0.
        angle_start_rad = angle_start_idx * geometry["angular_step_rad"]

        return {
            "low_dose_sinogram": torch.from_numpy(low_patch).unsqueeze(0),
            "full_dose_sinogram": torch.from_numpy(full_patch).unsqueeze(0),
            "ndct_image": torch.from_numpy(ndct_img).unsqueeze(0),
            "patient_id": pid,
            # Per-sample scanner + angular geometry (varies by
            # patient/scanner/protocol) -- feed these into
            # PS3T.forward(..., geometry={...}) so each sample reconstructs
            # with its own true acquisition geometry instead of one
            # dataset-wide default.
            "dso_mm": torch.tensor(geometry["dso_mm"], dtype=torch.float32),
            "dsd_mm": torch.tensor(geometry["dsd_mm"], dtype=torch.float32),
            "detector_spacing_mm": torch.tensor(
                geometry["detector_spacing_mm"], dtype=torch.float32
            ),
            "angular_step_rad": torch.tensor(
                geometry["angular_step_rad"], dtype=torch.float32
            ),
            "angle_start_rad": torch.tensor(angle_start_rad, dtype=torch.float32),
        }


def _random_matching_patch(low: np.ndarray, full: np.ndarray, ph: int, pw: int):
    """Crops a random matching (angle, detector) patch from a pair of
    sinograms and returns the angle-axis start index alongside it, so the
    caller can compute this patch's true starting acquisition angle
    (see `MayoLDCTDataset.__getitem__`)."""
    A, D = low.shape
    ph = min(ph, A)
    pw = min(pw, D)
    top = random.randint(0, A - ph)
    left = random.randint(0, D - pw)
    return (
        low[top : top + ph, left : left + pw].copy(),
        full[top : top + ph, left : left + pw].copy(),
        top,
    )


class SyntheticSinogramDataset(Dataset):
    """Generates synthetic low/full-dose sinogram pairs and a matching NDCT
    image for smoke-testing the training pipeline without any real data.
    """

    def __init__(
        self,
        length: int = 256,
        angles: int = 64,
        detector_bins: int = 64,
        image_size: int | None = None,
        seed: int = 42,
    ):
        self.length = length
        self.angles = angles
        self.detector_bins = detector_bins
        self.image_size = image_size or detector_bins
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(idx)
        full_sino = np.abs(
            rng.randn(self.angles, self.detector_bins).astype(np.float32)
        ) + 0.5
        poisson_scale = 40.0  # simulated low-dose photon-count scale
        photon_counts = rng.poisson(
            np.exp(-full_sino) * poisson_scale
        ).astype(np.float32)
        photon_counts = np.clip(photon_counts, 1.0, None)
        low_sino = -np.log(photon_counts / poisson_scale)

        ndct_img = rng.rand(self.image_size, self.image_size).astype(np.float32)

        return {
            "low_dose_sinogram": torch.from_numpy(low_sino).unsqueeze(0),
            "full_dose_sinogram": torch.from_numpy(full_sino).unsqueeze(0),
            "ndct_image": torch.from_numpy(ndct_img).unsqueeze(0),
            "patient_id": f"SYN{idx:04d}",
            # Fixed placeholder geometry (Mayo Clinic dataset defaults) --
            # synthetic data has no real scanner behind it.
            "dso_mm": torch.tensor(595.0, dtype=torch.float32),
            "dsd_mm": torch.tensor(1085.6, dtype=torch.float32),
            "detector_spacing_mm": torch.tensor(0.7421875, dtype=torch.float32),
            "angular_step_rad": torch.tensor((2 * np.pi) / 1160.0, dtype=torch.float32),
            "angle_start_rad": torch.tensor(0.0, dtype=torch.float32),
        }

