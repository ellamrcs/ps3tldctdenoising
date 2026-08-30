#!/usr/bin/env python
"""
Preprocess raw Mayo Clinic LDCT and Projection Dataset files (as downloaded
from TCIA: https://doi.org/10.7937/9NPB-2637) into the per-patient .npy
layout expected by `ps3t.dataset.MayoLDCTDataset`:

    <output_root>/<anatomy>/<patient_id>/low_dose_sinograms.npy
    <output_root>/<anatomy>/<patient_id>/full_dose_sinograms.npy
    <output_root>/<anatomy>/<patient_id>/ndct_images.npy
    <output_root>/<anatomy>/<patient_id>/geometry.npz   # per-slice scanner + angular geometry

Two SEPARATE input directories
---------------------------------
The TCIA release ships reconstructed DICOM *images* and raw projection
(sinogram) data as two distinct deliverables, typically organized in their
own directory trees per patient (naming/layout varies by TCIA collection
version -- adjust the per-patient globbing in `load_raw_patient()` below
to match what you actually downloaded). This script therefore takes two
separate root paths:

    --dicom-root      root of the reconstructed-image DICOM series
                       (low-dose + full-dose CT images, per patient)
    --projection-root root of the raw projection/sinogram data
                       (per patient, vendor-specific format)

Both are assumed to use the same per-patient subdirectory naming (e.g.
`L001/`, `C001/`, `N001/`) so a given patient's image and projection data
can be paired up by directory name; adjust `load_raw_patient()` if your
local copy of the dataset uses different per-patient identifiers between
the two deliverables.

Because exact file naming and projection-data formats vary by scanner
(Siemens vs. GE) and by TCIA collection version, this script is
intentionally a *template*: fill in `load_raw_patient()` with the logic
appropriate to your local copy of the dataset, then run:

    python scripts/preprocess_mayo.py \
        --dicom-root /path/to/tcia/images \
        --projection-root /path/to/tcia/projections \
        --output-root ./data/mayo_ldct \
        --anatomy abdomen

This keeps data acquisition/licensing (TCIA data-use agreement) separate
from the modeling code in this repository.

Per-patient scanner + angular geometry
-----------------------------------------
Two families of acquisition parameters are extracted per patient/slice and
saved in `geometry.npz`, because BOTH can vary across patients, scanners,
and protocols -- neither is a safe dataset-wide constant:

1. Spatial geometry (`extract_geometry_from_dicom`): Distance Source to
   Detector, Distance Source to Patient, detector pixel spacing. These sit
   in standard DICOM tags, read from `--dicom-root`.

2. Angular sampling (`extract_angular_sampling`): number of projection
   views per rotation and the total rotation extent (e.g. 2*pi for a full
   gantry rotation, less for a short scan). **Unlike (1), this information
   is typically NOT in standard image-DICOM tags** -- it lives in the raw
   projection-data file's own header/metadata under `--projection-root`,
   which is vendor-proprietary (e.g. Siemens IMA projection headers, GE
   proprietary formats) and varies by TCIA collection version.
   `extract_angular_sampling()` below is therefore a stub you MUST fill in
   once you've inspected your actual projection-data files; it documents
   the fields to look for and a literature-derived fallback, but that
   fallback should be treated as a rough placeholder, not a verified
   value, until you've checked it against your own files.

Both are consumed by `MayoLDCTDataset` and fed per-sample into
`FanBeamFBP` / `DifferentiableFBP`, instead of assuming fixed values for
the whole dataset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


# ---------------------------------------------------------------------------
# 1. Spatial geometry (standard DICOM tags, from --dicom-root)
# ---------------------------------------------------------------------------

# DICOM tags used for geometry extraction (see `pydicom.dataset.Dataset`):
#   (0018,1110) Distance Source to Detector
#   (0018,1111) Distance Source to Patient
#   (0018,1164) or (0028,0030) Pixel Spacing (detector-side vs. image-side;
#     which tag holds the *detector* pixel spacing depends on whether you're
#     reading the projection-data header or the reconstructed-image header --
#     verify against your specific TCIA file for each modality/series)
DEFAULT_GEOMETRY_FALLBACK = {
    "dso_mm": 595.0,
    "dsd_mm": 1085.6,
    "detector_spacing_mm": 0.7421875,
}


def extract_geometry_from_dicom(ds) -> dict:
    """Reads (DSO, DSD, detector pixel spacing) from a single pydicom
    Dataset, falling back to `DEFAULT_GEOMETRY_FALLBACK` per-field when a
    tag is missing (some series may omit projection geometry tags on the
    reconstructed-image DICOMs, only on the raw projection-data headers).

    Args:
        ds: a `pydicom.dataset.Dataset` (result of `pydicom.dcmread(path)`),
            typically read from a file under `--dicom-root`.

    Returns:
        dict with keys "dso_mm", "dsd_mm", "detector_spacing_mm".
    """
    geometry = dict(DEFAULT_GEOMETRY_FALLBACK)

    dsd = getattr(ds, "DistanceSourceToDetector", None)  # (0018,1110)
    dso = getattr(ds, "DistanceSourceToPatient", None)  # (0018,1111)
    pixel_spacing = getattr(ds, "PixelSpacing", None)  # (0028,0030), [row, col] mm

    if dsd is not None:
        geometry["dsd_mm"] = float(dsd)
    if dso is not None:
        geometry["dso_mm"] = float(dso)
    if pixel_spacing is not None and len(pixel_spacing) >= 1:
        # Detector pixel spacing is typically isotropic; use the first
        # (row) component. Verify this assumption against your vendor's
        # projection-data DICOM if reading raw sinogram headers instead.
        geometry["detector_spacing_mm"] = float(pixel_spacing[0])

    return geometry


# ---------------------------------------------------------------------------
# 2. Angular sampling (NOT standard DICOM -- from --projection-root)
# ---------------------------------------------------------------------------

# Literature-derived placeholder for this collection's typical protocol
# (commonly-cited approximate values for the Siemens scanners used in this
# dataset -- e.g. ~1160 views per full rotation). 
DEFAULT_ANGULAR_FALLBACK = {
    "num_views": 1160,
    "rotation_range_rad": 2 * math.pi,  # full gantry rotation
}


def extract_angular_sampling(raw_projection_patient_dir: Path) -> dict:
    """STUB -- fill in for your local copy of the raw projection data.

    Angular sampling (number of projection views per rotation, and the
    total rotation extent) generally lives in the raw projection-data
    file's own vendor-specific header under `--projection-root`, not in the
    reconstructed-image DICOM tags used by `extract_geometry_from_dicom`.
    Typical fields to look for once you've inspected your actual files:
      - a view/frame count (often mirrors the sinogram's angular axis
        length directly, in which case you may not even need this
        function -- just read `low_dose_sinograms.shape[-2]` per slice)
      - a rotation-angle or gantry-angle-increment field
      - start-angle / stop-angle fields for non-full-rotation acquisitions

    Args:
        raw_projection_patient_dir: this patient's directory under
            `--projection-root`, containing the raw projection-data file(s).

    Returns:
        dict with keys "num_views" (int) and "rotation_range_rad" (float).
    """
    raise NotImplementedError(
        "Fill in extract_angular_sampling() once you've inspected your local "
        "TCIA projection-data file format under --projection-root. Until "
        "then, DEFAULT_ANGULAR_FALLBACK is used -- verify it against your "
        "data before trusting FBP reconstructions."
    )


def angular_step_from_sampling(sampling: dict) -> float:
    """Converts (num_views, rotation_range_rad) into the per-view angular
    step (radians) consumed by `FanBeamFBP` / `DifferentiableFBP`."""
    return sampling["rotation_range_rad"] / sampling["num_views"]


def load_raw_patient(dicom_patient_dir: Path, projection_patient_dir: Path):
    """Template hook: implement reading of one patient's raw low-dose /
    full-dose projection data (from `projection_patient_dir`) and NDCT
    DICOM series (from `dicom_patient_dir`).

    Args:
        dicom_patient_dir: this patient's directory under `--dicom-root`.
        projection_patient_dir: this patient's directory under
            `--projection-root`.

    Returns:
        low_dose_sinograms:  (num_slices, A, D) float32 array
        full_dose_sinograms: (num_slices, A, D) float32 array
        ndct_images:         (num_slices, H, W) float32 array, normalized to [0, 1]
        geometry:            dict of (num_slices,) arrays with keys
                              "dso_mm", "dsd_mm", "detector_spacing_mm"
                              (see `extract_geometry_from_dicom`, sourced
                              from `dicom_patient_dir`) plus
                              "angular_step_rad" (see
                              `extract_angular_sampling` /
                              `angular_step_from_sampling`, sourced from
                              `projection_patient_dir`), one value per slice.
    """
    raise NotImplementedError(
        "Fill in load_raw_patient() with a reader for your local TCIA download: "
        "pydicom + extract_geometry_from_dicom() on dicom_patient_dir for "
        "spatial geometry, extract_angular_sampling() on "
        "projection_patient_dir for angular sampling (after you've "
        "implemented it for your projection-data format), and the vendor "
        "projection-data reader for the sinograms themselves."
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess Mayo Clinic LDCT dataset")
    parser.add_argument(
        "--dicom-root", type=str, required=True,
        help="Root directory of reconstructed-image DICOM series (per patient).",
    )
    parser.add_argument(
        "--projection-root", type=str, required=True,
        help="Root directory of raw projection/sinogram data (per patient).",
    )
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--anatomy", type=str, required=True, choices=["abdomen", "chest", "head"])
    args = parser.parse_args()

    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")

    dicom_root = Path(args.dicom_root)
    projection_root = Path(args.projection_root)
    out_root = Path(args.output_root) / args.anatomy
    out_root.mkdir(parents=True, exist_ok=True)

    # Pair up patients present in BOTH directory trees (by directory name).
    dicom_patients = {p.name for p in dicom_root.iterdir() if p.is_dir()}
    projection_patients = {p.name for p in projection_root.iterdir() if p.is_dir()}
    missing_dicom = projection_patients - dicom_patients
    missing_projection = dicom_patients - projection_patients
    if missing_dicom:
        print(f"WARNING: {len(missing_dicom)} patient(s) have projection data but no "
              f"DICOM images (skipped): {sorted(missing_dicom)[:5]}{'...' if len(missing_dicom) > 5 else ''}")
    if missing_projection:
        print(f"WARNING: {len(missing_projection)} patient(s) have DICOM images but no "
              f"projection data (skipped): {sorted(missing_projection)[:5]}{'...' if len(missing_projection) > 5 else ''}")

    patient_ids = sorted(dicom_patients & projection_patients)
    for pid in patient_ids:
        print(f"Processing patient {pid} ...")
        low, full, ndct, geometry = load_raw_patient(
            dicom_root / pid, projection_root / pid
        )

        patient_out = out_root / pid
        patient_out.mkdir(parents=True, exist_ok=True)
        np.save(patient_out / "low_dose_sinograms.npy", low.astype(np.float32))
        np.save(patient_out / "full_dose_sinograms.npy", full.astype(np.float32))
        np.save(patient_out / "ndct_images.npy", ndct.astype(np.float32))
        np.savez(
            patient_out / "geometry.npz",
            dso_mm=np.asarray(geometry["dso_mm"], dtype=np.float32),
            dsd_mm=np.asarray(geometry["dsd_mm"], dtype=np.float32),
            detector_spacing_mm=np.asarray(geometry["detector_spacing_mm"], dtype=np.float32),
            angular_step_rad=np.asarray(geometry["angular_step_rad"], dtype=np.float32),
        )

    print(f"Done. Preprocessed data written to {out_root}")


if __name__ == "__main__":
    main()
