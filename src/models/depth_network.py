"""
Depth Network (D) – DrivingForward AAAI-2025  (Section 3.2 & 3.3)

Architecture: ResNet encoder → U-Net decoder
Outputs:
  • Depth map  D  ∈ R^{H×W}         – scale-aware metric depth [min_depth, max_depth]
  • Image features F ∈ R^{C×H×W}   – shared with Gaussian network encoder

Self-supervised training signal comes from the photometric warp loss
L_repro (see losses.py), using spatial and temporal context frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Tuple, Optional
import timm


# ── Building blocks ──────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Bilinear-upsample + skip-connection + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Encoder ──────────────────────────────────────────────────────────────────

class DepthEncoder(nn.Module):
    """
    ResNet feature pyramid (stages 0-4).
    Supports gradient checkpointing to save VRAM on RTX 3050.
    """

    # Channel counts per ResNet variant
    _CHANNELS = {
        "resnet18":  [64,  64,  128,  256,  512],
        "resnet50":  [64, 256,  512, 1024, 2048],
        "resnet101": [64, 256,  512, 1024, 2048],
    }

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        net = timm.create_model(backbone, pretrained=pretrained,
                                features_only=True, out_indices=(0, 1, 2, 3, 4))

        # Split into 5 stages for per-stage checkpoint
        self.stages = nn.ModuleList(net.children() if hasattr(net, '__iter__')
                                    else [net])
        # Use timm feature info
        self._net = net
        self.out_channels: List[int] = self._CHANNELS.get(backbone,
                                       self._CHANNELS["resnet18"])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.grad_checkpoint and self.training:
            return checkpoint(self._net, x, use_reentrant=False)
        return self._net(x)  # list of 5 feature maps


# ── Decoder ──────────────────────────────────────────────────────────────────

class DepthDecoder(nn.Module):
    """
    U-Net decoder.
    Returns:
      depths : dict {scale: (B,1,H/2^scale,W/2^scale)} for multi-scale supervision
      feat   : (B, feat_dim, H, W) – feature map passed to Gaussian network
    """

    def __init__(self, enc_channels: List[int], feat_dim: int = 64,
                 min_depth: float = 0.1, max_depth: float = 80.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        dec_ch = [feat_dim, feat_dim * 2, feat_dim * 4, feat_dim * 8]

        # 4 up-blocks (enc stages 4→3→2→1→0)
        self.up4 = UpBlock(enc_channels[4], enc_channels[3], dec_ch[3])
        self.up3 = UpBlock(dec_ch[3],       enc_channels[2], dec_ch[2])
        self.up2 = UpBlock(dec_ch[2],       enc_channels[1], dec_ch[1])
        self.up1 = UpBlock(dec_ch[1],       enc_channels[0], dec_ch[0])

        # Multi-scale depth heads (sigmoid → rescale)
        self.depth_head4 = nn.Conv2d(dec_ch[3], 1, 1)
        self.depth_head3 = nn.Conv2d(dec_ch[2], 1, 1)
        self.depth_head2 = nn.Conv2d(dec_ch[1], 1, 1)
        self.depth_head1 = nn.Conv2d(dec_ch[0], 1, 1)

        # Feature head (full resolution, shared with Gaussian network)
        self.feat_head = ConvBnRelu(dec_ch[0], feat_dim)

    def _sigmoid_depth(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(x)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[Dict[int, torch.Tensor],
                                                           torch.Tensor]:
        f0, f1, f2, f3, f4 = feats

        x4 = self.up4(f4, f3)
        x3 = self.up3(x4, f2)
        x2 = self.up2(x3, f1)
        x1 = self.up1(x2, f0)

        depths = {
            4: self._sigmoid_depth(self.depth_head4(x4)),
            3: self._sigmoid_depth(self.depth_head3(x3)),
            2: self._sigmoid_depth(self.depth_head2(x2)),
            1: self._sigmoid_depth(self.depth_head1(x1)),
        }
        feat = self.feat_head(x1)
        return depths, feat


# ── Full Depth Network ────────────────────────────────────────────────────────

class DepthNetwork(nn.Module):
    """
    Depth Network D.

    At training time: call with multi-scale output for L_loc computation.
    At inference time: only the finest depth (scale 1) and feat are used.

    Args:
        backbone       : timm backbone name
        pretrained     : load ImageNet weights
        min_depth      : minimum depth in metres
        max_depth      : maximum depth in metres
        feat_dim       : feature dimension shared with Gaussian network
        grad_checkpoint: use gradient checkpointing (saves ~30% VRAM)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        feat_dim: int = 64,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.encoder = DepthEncoder(backbone, pretrained, grad_checkpoint)
        self.decoder = DepthDecoder(
            self.encoder.out_channels, feat_dim, min_depth, max_depth
        )
        self.feat_dim = feat_dim

    def forward(
        self,
        image: torch.Tensor,
        return_all_scales: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
        """
        Args:
            image            : (B, 3, H, W)
            return_all_scales: if True also return multi-scale depth dict

        Returns:
            depth  : (B, 1, H, W)  finest-scale depth
            feat   : (B, C, H, W)  image features
            scales : dict or None
        """
        feats = self.encoder(image)
        depths, feat = self.decoder(feats)
        depth = depths[1]

        if return_all_scales:
            return depth, feat, depths
        return depth, feat, None
