"""
Gaussian Network (G) – DrivingForward AAAI-2025  (Section 3.3)

Independently predicts 3D Gaussian primitive parameters for each
input image.  For each pixel the network outputs:
  • s  – scaling factor   ∈ R^3
  • r  – rotation         ∈ R^4  (quaternion)
  • c  – colour           ∈ R^{3*(d+1)^2}  (SH coefficients, degree d)
  • o  – opacity          ∈ (0, 1)

Position μ is obtained via scale-aware localization (unprojection),
not predicted by this network (see driving_forward.py).

Architecture:
  Encoder E  – shares the ResNet backbone with DepthNetwork
  Aggregation A – combines depth feature F_depth and image feature F_image
  Decoder heads – separate conv heads for each attribute
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: int = 1, act: bool = True):
        super().__init__()
        layers: list = [nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
                        nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.ELU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnAct(ch, ch),
            ConvBnAct(ch, ch, act=False),
        )
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── Feature Aggregation ───────────────────────────────────────────────────────

class FeatureAggregation(nn.Module):
    """
    Combines depth feature F_depth and image feature F_image.

    From paper (Section 3.3):
      "we use the previous scale-aware depth and image feature from the
       depth network as input for the Gaussian network."

    A simple channel-concatenation followed by convolution is used.
    The shared image feature ensures multi-view consistency.
    """

    def __init__(self, depth_feat_dim: int, img_feat_dim: int, out_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBnAct(depth_feat_dim + img_feat_dim, out_dim),
            ResBlock(out_dim),
            ResBlock(out_dim),
        )

    def forward(self, f_depth: torch.Tensor,
                f_image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_depth, f_image], dim=1)
        return self.fuse(x)


# ── Parameter Decoder Heads ──────────────────────────────────────────────────

class ScaleHead(nn.Module):
    """Predict log-scale s ∈ R^3 (exponentiated to positive scale)."""
    def __init__(self, in_ch: int, scale_min: float = 0.001, scale_max: float = 10.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.conv = nn.Conv2d(in_ch, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sigmoid then rescale to [scale_min, scale_max]
        return self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(self.conv(x))


class RotationHead(nn.Module):
    """Predict unit quaternion r ∈ R^4."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.conv(x)
        # Normalize in FP32: F.normalize in FP16 overflows when ||q|| is large
        return F.normalize(q.float(), dim=1).to(q.dtype)


class ColourHead(nn.Module):
    """
    Predict spherical harmonics coefficients for colour.
    SH degree d → (d+1)^2 coefficients per channel → 3*(d+1)^2 total.
    """
    def __init__(self, in_ch: int, sh_degree: int = 1):
        super().__init__()
        num_coeff = (sh_degree + 1) ** 2
        self.conv = nn.Conv2d(in_ch, 3 * num_coeff, 1)
        self.num_coeff = num_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh bounds SH coefficients to (-3, 3) — safe in FP16 (max ~65504)
        # and covers the useful range; raw conv output grows unbounded over epochs.
        return torch.tanh(self.conv(x)) * 3.0


class OpacityHead(nn.Module):
    """Predict opacity o ∈ (0, 1)."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


# ── Full Gaussian Network ─────────────────────────────────────────────────────

class GaussianNetwork(nn.Module):
    """
    Gaussian Network G.

    For each input image independently predicts per-pixel Gaussian attributes.
    Positions are NOT predicted here — they come from depth unprojection in
    the main DrivingForward model.

    Args:
        depth_feat_dim : channel count of F_depth from DepthNetwork.decoder output
        img_feat_dim   : channel count of F_image (same as depth_feat_dim by default)
        hidden_dim     : internal channel dimension for aggregation + decoders
        sh_degree      : spherical harmonics degree for colour (1 = 4 coeffs, 3 = 16 coeffs)
        scale_min/max  : clamp for predicted scale
    """

    def __init__(
        self,
        depth_feat_dim: int = 64,
        img_feat_dim:   int = 64,
        hidden_dim:     int = 128,
        sh_degree:      int = 1,
        scale_min:      float = 0.001,
        scale_max:      float = 10.0,
    ):
        super().__init__()

        self.aggregation = FeatureAggregation(depth_feat_dim, img_feat_dim, hidden_dim)
        self.scale_head   = ScaleHead(hidden_dim, scale_min, scale_max)
        self.rot_head     = RotationHead(hidden_dim)
        self.colour_head  = ColourHead(hidden_dim, sh_degree)
        self.opacity_head = OpacityHead(hidden_dim)
        self.sh_degree    = sh_degree

    def forward(
        self,
        f_depth: torch.Tensor,
        f_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            f_depth : (B, depth_feat_dim, H, W)  depth feature
            f_image : (B, img_feat_dim,   H, W)  image feature

        Returns dict with keys:
            'scale'   : (B, 3, H, W)           – Gaussian scale (σ)
            'rotation': (B, 4, H, W)           – rotation quaternion (unit)
            'colour'  : (B, 3*(d+1)^2, H, W)  – SH coefficients
            'opacity' : (B, 1, H, W)           – opacity ∈ (0,1)
        """
        h = self.aggregation(f_depth, f_image)
        return {
            "scale":    self.scale_head(h),
            "rotation": self.rot_head(h),
            "colour":   self.colour_head(h),
            "opacity":  self.opacity_head(h),
        }

    @property
    def num_sh_coeffs(self) -> int:
        return (self.sh_degree + 1) ** 2
