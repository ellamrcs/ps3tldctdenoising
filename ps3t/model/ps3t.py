"""
PS3T: Physics-Guided Sequential State Space Transformer
==========================================================

End-to-end assembly of the pipeline shown in Figure 1:

    Input Sinogram
        -> Projection Embedding
        -> Photon-Aware Attention  (uses Feature Embedding output + raw sinogram)
        -> Sequential State Space Evolution
        -> Projection Reconstruction  (-> S_pred)
        -> Differentiable FBP          (-> CT_pred)
        -> Image-Domain ViT Refinement (-> NDCT output)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fbp import DifferentiableFBP
from .fan_fbp import FanBeamFBP
from .photon_attention import PhotonAwareAttention
from .projection_embedding import ProjectionEmbedding
from .projection_reconstruction import ProjectionReconstruction
from .state_space import SequentialStateSpaceBlock
from .vit_refine import ImageViTRefinement


class PS3T(nn.Module):
    def __init__(
        self,
        angles: int = 64,
        detector_bins: int = 64,
        embed_dim: int = 128,
        state_dim: int = 64,
        num_attn_heads: int = 4,
        num_vit_encoder_layers: int = 2,
        num_vit_decoder_layers: int = 2,
        vit_patch_size: int = 8,
        image_size: int | None = None,
        dropout: float = 0.0,
        photon_epsilon: float = 1e-3,
        geometry: dict | None = None,
    ):
        """
        Args:
            photon_epsilon: numerical-stability constant eps in the photon
                uncertainty map U_t = 1/sqrt(P_t + eps) (Eq. 3).
            geometry: optional dict to select/parameterize the reconstruction
                operator R(.). If omitted, defaults to the parallel-beam
                `DifferentiableFBP`. To use the confirmed Mayo Clinic scanner
                geometry, pass e.g.:
                    geometry = {
                        "beam_type": "fan",
                        "distance_source_to_detector_mm": 1085.6,
                        "distance_source_to_patient_mm": 595.0,
                        "detector_pixel_spacing_mm": 0.7421875,
                    }
        """
        super().__init__()
        self.angles = angles
        self.detector_bins = detector_bins
        self.image_size = image_size or detector_bins

        self.projection_embedding = ProjectionEmbedding(
            in_channels=1, dim=embed_dim, max_angles=angles, dropout=dropout
        )
        self.photon_attention = PhotonAwareAttention(
            dim=embed_dim, num_heads=num_attn_heads, dropout=dropout, eps=photon_epsilon
        )
        self.state_space = SequentialStateSpaceBlock(
            dim=embed_dim, state_dim=state_dim, dropout=dropout
        )
        self.projection_reconstruction = ProjectionReconstruction(
            dim=embed_dim, detector_bins=detector_bins
        )
        geometry = geometry or {}
        self.beam_type = geometry.get("beam_type", "parallel")
        if self.beam_type == "fan":
            self.fbp = FanBeamFBP(
                dso_mm=geometry.get("distance_source_to_patient_mm", 595.0),
                dsd_mm=geometry.get("distance_source_to_detector_mm", 1085.6),
                detector_spacing_mm=geometry.get("detector_pixel_spacing_mm", 0.7421875),
                num_detectors=detector_bins,
                image_size=self.image_size,
            )
        else:
            self.fbp = DifferentiableFBP(image_size=self.image_size)
        self.vit_refine = ImageViTRefinement(
            image_size=self.image_size,
            patch_size=vit_patch_size,
            dim=embed_dim,
            num_heads=num_attn_heads,
            num_encoder_layers=num_vit_encoder_layers,
            num_decoder_layers=num_vit_decoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        sinogram: torch.Tensor,
        return_intermediate: bool = False,
        geometry: dict | None = None,
    ):
        """
        Args:
            sinogram: (B, 1, A, D) low-dose sinogram patch.
            return_intermediate: if True, also return S_pred and CT_pred
                (needed to compute Lproj and Lrecon during training).
            geometry: optional per-sample geometry override dict, merged
                with **geometry into self.fbp(s_pred, **geometry). Expected
                keys match `FanBeamFBP.forward` / `DifferentiableFBP.forward`:
                "dso_mm", "dsd_mm", "detector_spacing_mm", "pixel_spacing_mm",
                "angular_step_rad", "angle_start_rad" -- each either a python
                float or a (B,) tensor, e.g. from `ps3t.utils.batch_geometry(batch, device)`
                built from a `MayoLDCTDataset` batch, so every sample in the
                batch reconstructs with its own patient-specific scanner and
                angular geometry rather than a single dataset-wide default.
                `DifferentiableFBP` (parallel-beam) uses only the angular
                keys and ignores the fan-beam-only ones via **_ignored, so
                the same dict works for either beam type.

        Returns:
            ndct_pred: (B, 1, H, W) final refined NDCT-quality image.
            (optionally) S_pred: (B, 1, A, D) denoised sinogram.
            (optionally) ct_pred: (B, 1, H, W) pre-refinement FBP reconstruction.
        """
        tokens = self.projection_embedding(sinogram)  # (B, A, dim)
        z, U = self.photon_attention(tokens, sinogram)  # (B, A, dim), (B, A)
        y = self.state_space(z, U)  # (B, A, dim)
        s_pred = self.projection_reconstruction(y)  # (B, 1, A, D)

        if geometry:
            ct_pred = self.fbp(s_pred, **geometry)
        else:
            ct_pred = self.fbp(s_pred)  # (B, 1, H, W)

        ndct_pred = self.vit_refine(ct_pred)  # (B, 1, H, W)

        if return_intermediate:
            return ndct_pred, s_pred, ct_pred
        return ndct_pred
