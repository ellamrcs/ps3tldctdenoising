"""
Smoke tests: verify the PS3T model runs end-to-end with correct output
shapes and that gradients flow back to the projection-embedding input,
using tiny dimensions for speed.

Run with:  pytest -q
"""

import torch

from ps3t.dataset import SyntheticSinogramDataset
from ps3t.losses import LossWeights, PS3TLoss
from ps3t.model import PS3T
from ps3t.model.fan_fbp import FanBeamFBP


def _tiny_model():
    return PS3T(
        angles=16,
        detector_bins=16,
        embed_dim=32,
        state_dim=16,
        num_attn_heads=4,
        num_vit_encoder_layers=1,
        num_vit_decoder_layers=1,
        vit_patch_size=4,
        image_size=16,
    )


def test_forward_shapes():
    model = _tiny_model()
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)


def test_forward_intermediate_shapes():
    model = _tiny_model()
    x = torch.randn(2, 1, 16, 16)
    ndct_pred, s_pred, ct_pred = model(x, return_intermediate=True)
    assert ndct_pred.shape == (2, 1, 16, 16)
    assert s_pred.shape == (2, 1, 16, 16)
    assert ct_pred.shape == (2, 1, 16, 16)


def test_backward_pass():
    model = _tiny_model()
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    full = torch.randn(2, 1, 16, 16)
    ndct = torch.randn(2, 1, 16, 16)

    ndct_pred, s_pred, ct_pred = model(x, return_intermediate=True)
    criterion = PS3TLoss(weights=LossWeights())
    loss, logs = criterion(s_pred, full, ct_pred, ndct)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(loss)
    for k, v in logs.items():
        assert torch.isfinite(v)


def test_fan_beam_model_forward():
    # Uses the confirmed Mayo Clinic scanner geometry.
    model = PS3T(
        angles=16,
        detector_bins=16,
        embed_dim=32,
        state_dim=16,
        num_attn_heads=4,
        num_vit_encoder_layers=1,
        num_vit_decoder_layers=1,
        vit_patch_size=4,
        image_size=16,
        geometry={
            "beam_type": "fan",
            "distance_source_to_detector_mm": 1085.6,
            "distance_source_to_patient_mm": 595.0,
            "detector_pixel_spacing_mm": 0.7421875,
        },
    )
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_fan_beam_fbp_standalone():
    fbp = FanBeamFBP(
        dso_mm=595.0,
        dsd_mm=1085.6,
        detector_spacing_mm=0.7421875,
        num_detectors=16,
        image_size=16,
    )
    sino = torch.rand(2, 1, 16, 16)
    img = fbp(sino)
    assert img.shape == (2, 1, 16, 16)
    assert torch.isfinite(img).all()


def test_fan_beam_fbp_per_sample_geometry():
    # Two samples with genuinely different scanner geometry AND different
    # angular sampling / patch start angle -- exercises the per-sample
    # (per-patient) override path end-to-end.
    fbp = FanBeamFBP(
        dso_mm=595.0, dsd_mm=1085.6, detector_spacing_mm=0.7421875,
        num_detectors=16, image_size=16,
    )
    sino = torch.rand(2, 1, 16, 16)
    dso = torch.tensor([595.0, 610.0])
    dsd = torch.tensor([1085.6, 1100.0])
    spacing = torch.tensor([0.7421875, 0.8])
    angular_step = torch.tensor([(2 * 3.14159265 / 1160.0), (3.14159265 / 800.0)])
    angle_start = torch.tensor([0.0, 0.3])

    img = fbp(
        sino, dso_mm=dso, dsd_mm=dsd, detector_spacing_mm=spacing,
        angular_step_rad=angular_step, angle_start_rad=angle_start,
    )
    assert img.shape == (2, 1, 16, 16)
    assert torch.isfinite(img).all()
    # Different geometry per sample should (generically) produce different
    # reconstructions -- a basic sanity check that overrides are actually used.
    assert not torch.allclose(img[0], img[1])


def test_ps3t_with_batch_geometry():
    from ps3t.utils import batch_geometry

    model = PS3T(
        angles=16, detector_bins=16, embed_dim=32, state_dim=16,
        num_attn_heads=4, num_vit_encoder_layers=1, num_vit_decoder_layers=1,
        vit_patch_size=4, image_size=16,
        geometry={
            "beam_type": "fan",
            "distance_source_to_detector_mm": 1085.6,
            "distance_source_to_patient_mm": 595.0,
            "detector_pixel_spacing_mm": 0.7421875,
        },
    )
    ds = SyntheticSinogramDataset(length=2, angles=16, detector_bins=16, image_size=16)
    batch = {
        "low_dose_sinogram": torch.stack([ds[0]["low_dose_sinogram"], ds[1]["low_dose_sinogram"]]),
        "dso_mm": torch.stack([ds[0]["dso_mm"], ds[1]["dso_mm"]]),
        "dsd_mm": torch.stack([ds[0]["dsd_mm"], ds[1]["dsd_mm"]]),
        "detector_spacing_mm": torch.stack(
            [ds[0]["detector_spacing_mm"], ds[1]["detector_spacing_mm"]]
        ),
        "angular_step_rad": torch.stack(
            [ds[0]["angular_step_rad"], ds[1]["angular_step_rad"]]
        ),
        "angle_start_rad": torch.stack(
            [ds[0]["angle_start_rad"], ds[1]["angle_start_rad"]]
        ),
    }
    geometry = batch_geometry(batch, device=torch.device("cpu"))
    out = model(batch["low_dose_sinogram"], geometry=geometry)
    assert out.shape == (2, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_synthetic_dataset_batch():
    ds = SyntheticSinogramDataset(length=4, angles=16, detector_bins=16, image_size=16)
    batch = ds[0]
    assert batch["low_dose_sinogram"].shape == (1, 16, 16)
    assert batch["full_dose_sinogram"].shape == (1, 16, 16)
    assert batch["ndct_image"].shape == (1, 16, 16)
    assert "angular_step_rad" in batch and "angle_start_rad" in batch
