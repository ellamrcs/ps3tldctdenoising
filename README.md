# PS³T — Physics-Guided Sequential State Space Transformer

> Marcos, L.; Babyn, P.; Alirezaie, J. **Physics-Guided Sequential State Space Transformer (PS³T) for Projection Domain LDCT Denoising.** *Signals [accepted]*, 2026.

PS³T is a projection-domain (sinogram) denoising framework for Low-Dose CT (LDCT) that combines:

1. **Photon-aware attention** — attention scores are conditioned on a per-projection photon-uncertainty map derived from the Beer–Lambert law, so noisy (photon-starved) views contribute less.
2. **Sequential state-space evolution** — a linear-complexity recurrence over projection angles (instead of quadratic self-attention) that adaptively propagates information across angles based on local uncertainty.
3. **A differentiable Filtered Backprojection (FBP) layer** — enforces reconstruction-domain consistency end-to-end during training.
4. **An image-domain ViT refinement stage** — a final encoder–decoder transformer that sharpens anatomical structures on the FBP output.

---

## Architecture

```
Input Sinogram (S)
       │
       ▼
┌─────────────────────┐     ┌──────────────────────┐
│ Projection Embedding │────▶│  Photon-Aware Attn    │◀── Uncertainty map U_t = 1/√(P_t+ε), P_t = e^(-S_t)
│ Conv2D+GELU+LayerNorm│     │  softmax(QKᵀ/√d − αU) │
└─────────────────────┘     └──────────┬────────────┘
                                        ▼
                          ┌─────────────────────────────┐
                          │ Sequential State-Space Block │
                          │ h_t = A(U_t) h_{t-1}+B(U_t)z_t│
                          │ y_t = C h_t + D z_t           │
                          └──────────────┬────────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │ Projection Reconstruction     │──▶ S_pred
                          └──────────────┬────────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │ Differentiable FBP  R(·)      │──▶ CT_pred
                          └──────────────┬────────────────┘
                                         ▼
                          ┌─────────────────────────────┐
                          │ Image-Domain ViT Refinement    │──▶ NDCT-quality output
                          │ (encoder–decoder, Fig. 3)       │
                          └─────────────────────────────┘
```

Every module maps directly to a section of the paper:

| Module | File | Paper reference |
|---|---|---|
| Projection / Feature Embedding | `ps3t/model/projection_embedding.py` | §2.2, Eq. (1) |
| Photon-Aware Attention | `ps3t/model/photon_attention.py` | §2.2, Eqs. (2)–(6), Fig. 2 |
| Sequential State Space Evolution | `ps3t/model/state_space.py` | §2.3, Eqs. (7)–(10) |
| Projection Reconstruction | `ps3t/model/projection_reconstruction.py` | §2.3, Eq. (11) |
| Differentiable FBP | `ps3t/model/fbp.py` | §2.4, Eq. (14) |
| Image-Domain ViT Refinement | `ps3t/model/vit_refine.py` | §2.1, Fig. 3 |
| Full model assembly | `ps3t/model/ps3t.py` | §2.1, Fig. 1 |
| Multi-objective loss | `ps3t/losses.py` | §2.4, Eqs. (12)–(18) |
| Metrics (PSNR/SSIM/RMSE) | `ps3t/utils.py` | §2.5 |
| Dataset / patient-level split | `ps3t/dataset.py` | §2.6, Table 1 |

## Repository layout

```
ps3tldctdenoising/
├── ps3t/                     # installable package
│   ├── model/
│   │   ├── projection_embedding.py
│   │   ├── photon_attention.py
│   │   ├── state_space.py
│   │   ├── projection_reconstruction.py
│   │   ├── fbp.py
│   │   ├── vit_refine.py
│   │   └── ps3t.py
│   ├── dataset.py
│   ├── losses.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess_mayo.py    # template for TCIA raw-data -> .npy conversion
├── configs/
│   └── default.yaml
├── tests/
│   └── test_model.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/ellamrcs/ps3tldctdenoising.git
cd ps3t
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
# or, for an editable install of the package itself:
pip install -e .
```

GPU training requires a CUDA-enabled PyTorch build — install the appropriate wheel from
[pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version before
`pip install -r requirements.txt` if the default PyPI wheel doesn't match your system.

### Verify the install

```bash
pytest -q
```

This runs lightweight shape/gradient smoke tests (`tests/test_model.py`) on a tiny model
configuration and does not require any dataset.

## Quickstart (synthetic data)

To exercise the full training/evaluation pipeline without downloading any data:

```bash
python scripts/train.py --config configs/default.yaml --use-synthetic
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint checkpoints/ps3t_last.pt --use-synthetic
```

This trains on procedurally generated sinogram/image pairs (`ps3t/dataset.py:SyntheticSinogramDataset`)
purely to confirm the model, losses, and training loop run end-to-end on your machine.

## Training on the Mayo Clinic LDCT and Projection Dataset

1. **Obtain the data.** The dataset is hosted on The Cancer Imaging Archive (TCIA)
   under a data-use agreement: <https://doi.org/10.7937/9NPB-2637> (299 patients:
   100 abdomen, 100 chest, 99 head; McCollough et al., 2021).
2. **Preprocess.** The TCIA release ships reconstructed-image DICOMs and raw
   projection (sinogram) data as two separate deliverables/directory trees.
   Convert both into the per-patient `.npy` + `geometry.npz` layout expected
   by `MayoLDCTDataset` using `scripts/preprocess_mayo.py` — you'll need to
   fill in `load_raw_patient()` (and `extract_angular_sampling()`) for
   local copy of the data, since projection-data formats vary by scanner
   vendor and TCIA collection version:

   ```bash
   python scripts/preprocess_mayo.py \
       --dicom-root /path/to/tcia/images \
       --projection-root /path/to/tcia/projections \
       --output-root ./data/mayo_ldct \
       --anatomy abdomen
   ```

   Resulting layout (paths default from `configs/default.yaml`'s `raw_data:`
   / `data.data_root` keys):

   ```
   data/mayo_ldct/
     abdomen/<patient_id>/{low_dose_sinograms,full_dose_sinograms,ndct_images}.npy
     abdomen/<patient_id>/geometry.npz   # per-slice DSO/DSD/spacing/angular step
     chest/<patient_id>/...
     head/<patient_id>/...
   ```

3. **Edit `configs/default.yaml`** — set `data.data_root` and `data.anatomy`
   (`abdomen` | `chest` | `head`; the paper trains one model per anatomical region).
4. **Train:**

   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

5. **Evaluate** (reports mean PSNR/SSIM/RMSE with 95% CIs on the patient-level test split):

   ```bash
   python scripts/evaluate.py --config configs/default.yaml \
       --checkpoint checkpoints/ps3t_last.pt
   ```

Dataset partitioning is performed **at the patient level** (not slice level) with a
fixed seed (`seed: 42`), matching the paper's 70% / 15% / 15% train/val/test split
(Table 1) and preventing any slice leakage across subsets.

## Configuration

All hyperparameters live in `configs/default.yaml`. Batch size, optimizer,
epochs, and loss weights mirror the paper's reported training setup (§2.6);
architectural dimensions not stated in the paper have been set as follows:

| Setting | Paper value | Config key |
|---|---|---|
| Sinogram patch size | 64 × 64 | `data.patch_size` |
| Batch size | 8 | `train.batch_size` |
| Optimizer | Adam, lr = 1e-4, cosine decay | `train.lr`, `train.lr_schedule` |
| Epochs | 50 | `train.epochs` |
| Mixed precision | FP16 (AMP) | `train.amp` |
| Loss weights (λ1, λ2, λ3) | 0.4, 0.4, 0.2 | `loss.lambda1/2/3` |
| Random seed | 42 | `seed` |
| Embedding dim | 256 | `model.embed_dim` |
| SSM state dim | 128 | `model.state_dim` |
| Attention heads | 8(see note below) | `model.num_attn_heads` |
| ViT encoder/decoder depth | 3 / 3 | `model.num_vit_encoder_layers/decoder_layers` |
| ViT patch size | 8 × 8 | `model.vit_patch_size` |
| Photon-estimation ε (Eq. 3) | 1e-3 | `model.photon_epsilon` |

> **Note on heads=8:** 8 heads with `embed_dim=256` gives an even head
> dimension of 32. 6 heads (an earlier candidate value) does not divide 256
> evenly (256/6 ≈ 42.67) and both `PhotonAwareAttention` and PyTorch's
> `nn.MultiheadAttention` require `embed_dim % num_heads == 0`; if you want
> heads=6 exactly, set `embed_dim` to a multiple of 6 (e.g. 252) instead.
>

## Loss function

```
L_total = λ1 · L_proj + λ2 · L_recon + λ3 · L_poisson
```

- **L_proj** — L1 loss between predicted and full-dose reference sinograms (Eq. 13).
- **L_recon** — L1 loss between `R(S_pred)` (differentiable FBP reconstruction) and
  the NDCT reference image (Eq. 15).
- **L_poisson** — Poisson negative log-likelihood between predicted and reference
  photon counts, `P = exp(-S)` (Eqs. 16–18), enforcing photon-statistical consistency.

## Reproducing paper results

This implementation follows the manuscript's equations and module descriptions as
closely as possible, but a few architectural details are under-specified in the
paper (e.g. exact number of transformer layers/heads, SSM state dimensionality, the
precise geometry — fan- vs. parallel-beam — used for FBP). Defaults in
`configs/default.yaml` are reasonable choices but may need tuning to match the
paper's reported metrics. In particular:

- `ps3t/model/fbp.py` the Mayo Clinic dataset was acquired with fan-beam
  clinical scanners, and acquisition geometry varies per patient/scanner/protocol
  (not a single dataset-wide constant). `ps3t/model/fan_fbp.py` provides a
  fan-beam-aware alternative (`FanBeamFBP`) that accepts per-sample geometry:

  | Parameter | DICOM tag | Fallback default (if unavailable) |
  |---|---|---|
  | Distance Source to Detector (DSD) | (0018,1110) | 1085.6 mm |
  | Distance Source to Patient (DSO) | (0018,1111) | 595.0 mm |
  | Detector pixel spacing | Pixel Spacing | 0.7421875 mm |

  The fallback values above were confirmed against one real study in this
  collection but **should not be assumed to hold for all 299 patients** —
  different scanners (Siemens vs. GE) and protocols can use different geometry.

  **Per-patient extraction pipeline:**
  1. `scripts/preprocess_mayo.py` reads each patient's own DICOM headers via
     `extract_geometry_from_dicom()` and writes a per-slice `geometry.npz`
     alongside the sinogram/image `.npy` files.
  2. `MayoLDCTDataset` loads that patient's `geometry.npz` and returns
     `dso_mm` / `dsd_mm` / `detector_spacing_mm` as part of every batch
     (falling back to the defaults above only if `geometry.npz` is missing).
  3. `scripts/train.py` / `scripts/evaluate.py` extract these via
     `ps3t.utils.batch_geometry()` and pass them to
     `PS3T.forward(..., geometry=...)`, so **each sample in a batch
     reconstructs with its own patient-specific geometry**, not one global
     default — this works even with mixed-geometry batches.

  Set `geometry.beam_type: "fan"` in `configs/default.yaml` to activate
  `FanBeamFBP`; it stays off (parallel-beam) by default. 

- **Angular sampling** (number of projection views per rotation, total
  rotation extent) is a scan-protocol parameter and **also varies per
  patient/scanner/protocol**, exactly like the spatial geometry above — so
  it gets the same per-patient extraction treatment rather than a fixed
  constant. Unlike DSD/DSO/pixel-spacing, though, this information
  typically does not live in standard image-DICOM tags — it's in the
  raw projection-data file's own vendor-specific header (Siemens/GE
  proprietary formats), so `scripts/preprocess_mayo.py:extract_angular_sampling()`
  is left as a stub for you to fill in once you've inspected your actual
  projection files.

  Sinograms are also trained on as randomly-cropped angular *patches*, not
  the full rotation, so each sample additionally carries its own patch
  starting angle (`angle_start_rad`, computed in `MayoLDCTDataset` from the
  random crop offset × the patient's angular step) — both `FanBeamFBP` and
  `DifferentiableFBP` accept `angular_step_rad` / `angle_start_rad` as
  per-sample tensors in `forward()`, wired through the same `geometry` dict
  as the spatial parameters via `ps3t.utils.batch_geometry()`.


## Citation

If you use this code, please cite the original paper:

```bibtex
@article{marcos2026ps3t,
  title   = {Physics-Guided Sequential State Space Transformer (PS3T) for Projection Domain LDCT Denoising},
  author  = {Marcos, Luella and Babyn, Paul and Alirezaie, Javad},
  journal = {Signals},
  year    = {2026},
  note    = {Submitted}
}
```

and, if helpful, this implementation:

```bibtex
@software{ps3t_implementation,
  title  = {PS3T: Implementation},
  year   = {2026},
  url    = {https://github.com/ellamrcs/ps3tldctdenoising}
}
```

## Data & license

- **Code** in this repository is released under the [MIT License](LICENSE).
- The **Mayo Clinic LDCT and Projection Dataset** is distributed separately via TCIA
  under its own data-use terms: <https://doi.org/10.7937/9NPB-2637>. This repository
  does not redistribute any patient data.


## Disclaimer

This software is provided for research purposes only. It has **not** been validated
for clinical use, has not undergone diagnostic-performance evaluation (the original
paper explicitly notes this limitation and calls for future reader studies), and
must not be used to inform patient care.

