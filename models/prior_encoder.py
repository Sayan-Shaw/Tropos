"""
models/prior_encoder.py
========================

Lightweight CNN that encodes the coarse geometric prior mask into a
feature map compatible with SAM2's image encoder output.

Input:  coarse_mask (B, 1, H, W) — the Stage 1 projected mask
Output: prior_features (B, embed_dim, H/16, W/16)

These features are ADDED element-wise to the image encoder output
before it enters the mask decoder.

Total trainable parameters: ~50K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):
    """
    5-layer CNN that progressively downsamples the coarse mask
    from (H, W) to (H/16, W/16) and projects to embed_dim channels.

    Architecture:
        Conv(1→16, 3, s=1)  → GELU → (H, W)
        Conv(16→32, 3, s=2) → GELU → (H/2, W/2)
        Conv(32→64, 3, s=2) → GELU → (H/4, W/4)
        Conv(64→64, 3, s=2) → GELU → (H/8, W/8)
        Conv(64→C, 3, s=2)  → GELU → (H/16, W/16)

    where C = embed_dim (256 for Hiera-Tiny).
    """

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coarse_mask):
        """
        Parameters
        ----------
        coarse_mask : (B, 1, H, W) float tensor, values in {0, 1}

        Returns
        -------
        features : (B, embed_dim, H/16, W/16)
        """
        return self.encoder(coarse_mask)

    def forward_and_fuse(self, coarse_mask, image_features):
        """
        Encode the mask and add to image features.
        Handles spatial size mismatch via interpolation.

        Parameters
        ----------
        coarse_mask : (B, 1, H, W)
        image_features : (B, embed_dim, Hf, Wf) from frozen image encoder

        Returns
        -------
        fused : (B, embed_dim, Hf, Wf)
        """
        prior_feat = self.encoder(coarse_mask)
        if prior_feat.shape[-2:] != image_features.shape[-2:]:
            prior_feat = F.interpolate(
                prior_feat, size=image_features.shape[-2:],
                mode='bilinear', align_corners=False)
        return image_features + prior_feat
