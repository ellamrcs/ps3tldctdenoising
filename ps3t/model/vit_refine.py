"""
Image-Domain ViT Refinement Block
====================================

Implements Figure 3: an encoder-decoder Vision Transformer that refines
anatomical structures and suppresses residual artifacts in the
FBP-reconstructed image, producing the final NDCT-quality reconstruction.

Encoder: LayerNorm -> Multi-Head Self-Attention -> Concat(skip) -> LayerNorm
         -> residual add -> Feed-Forward Network -> residual add
Decoder: Multi-Head Self-Attention -> Concat(skip) -> LayerNorm -> residual
         add -> LayerNorm -> Feed-Forward Network -> residual add -> reshape
         back to image space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MHSABlock(nn.Module):
    """LayerNorm -> MHSA -> Concat(skip) -> LayerNorm -> residual, matching
    the repeated sub-block used in both the encoder and decoder of Fig. 3."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.concat_proj = nn.Linear(dim * 2, dim)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        concat = torch.cat([x, attn_out], dim=-1)  # "Concat" block in Fig. 3
        merged = self.concat_proj(concat)
        merged = self.post_norm(merged)
        return x + merged  # residual connection


class ViTEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mhsa_block = MHSABlock(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        y = self.mhsa_block(x_hat)
        y_hat = y + self.ffn(y)
        return y_hat


class ViTDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mhsa_block = MHSABlock(dim, num_heads, dropout)
        self.pre_ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        z = self.mhsa_block(y_hat)
        z_normed = self.pre_ffn_norm(z)
        out = z_normed + self.ffn(z_normed)
        return out


class ImageViTRefinement(nn.Module):
    """Patch-based encoder-decoder ViT that refines an FBP-reconstructed
    CT image, per Figure 3."""

    def __init__(
        self,
        image_size: int,
        patch_size: int = 8,
        dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.0,
        in_channels: int = 1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_side = image_size // patch_size
        self.num_patches = self.num_patches_side ** 2
        self.dim = dim

        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        self.encoder = nn.ModuleList(
            [ViTEncoderBlock(dim, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [ViTDecoderBlock(dim, num_heads, dropout) for _ in range(num_decoder_layers)]
        )

        self.patch_unembed = nn.Linear(dim, patch_dim)

    def _to_patches(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        p = self.patch_size
        x = image.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)
        return x  # (B, num_patches, patch_dim)

    def _from_patches(self, patches: torch.Tensor, image_shape) -> torch.Tensor:
        B, C, H, W = image_shape
        p = self.patch_size
        n_side = self.num_patches_side
        x = patches.view(B, n_side, n_side, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 1, H, W) FBP-reconstructed CT image.

        Returns:
            refined: (B, 1, H, W) refined NDCT-quality output, with a
                     global residual connection to the input image.
        """
        shape = image.shape
        x_hat = self._to_patches(image)  # (B, N, patch_dim)
        x_hat = self.patch_embed(x_hat) + self.pos_embed

        y_hat = x_hat
        for enc_layer in self.encoder:
            y_hat = enc_layer(y_hat)

        z = y_hat
        for dec_layer in self.decoder:
            z = dec_layer(z)

        patches_out = self.patch_unembed(z)  # (B, N, patch_dim)
        residual_img = self._from_patches(patches_out, shape)

        return image + residual_img
