"""
Pose Network (P) – DrivingForward AAAI-2025  (Section 3.2)

Predicts the relative camera motion T^{t'→t} between consecutive
multi-camera frames.  The camera-to-vehicle transform E is fixed for
each camera (given by the dataset), so only the vehicle ego-motion
needs to be predicted.

Architecture follows the PoseCNN design from Monodepth2 (Godard et al. 2019):
  • Encoder : lightweight ResNet shared weight across all input frames
  • Pose head : global avg-pool → FC → 6-DOF axis-angle + translation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import timm


class PoseEncoder(nn.Module):
    """Shared backbone for pose estimation."""

    _CHANNELS = {
        "resnet18": 512,
        "resnet50": 2048,
    }

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 num_input_frames: int = 2):
        super().__init__()
        self.num_input_frames = num_input_frames
        # First conv accepts (3 * num_frames) channels
        net = timm.create_model(backbone, pretrained=pretrained, in_chans=3)
        # Replace stem conv
        old = net.conv1
        net.conv1 = nn.Conv2d(
            3 * num_input_frames,
            old.out_channels, old.kernel_size, old.stride,
            old.padding, bias=False
        )
        if pretrained:
            # Average pre-trained weights across duplicated input channels
            with torch.no_grad():
                net.conv1.weight[:] = old.weight.repeat(1, num_input_frames, 1, 1) / num_input_frames

        self.net = net
        self.out_dim = self._CHANNELS.get(backbone, 512)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, 3*T, H, W)"""
        return self.net.forward_features(frames)  # (B, C, h, w)


class PoseHead(nn.Module):
    """Global-pool → FC → axis-angle + translation."""

    def __init__(self, in_dim: int, num_pairs: int = 1):
        super().__init__()
        self.num_pairs = num_pairs
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6 * num_pairs),  # axis-angle (3) + translation (3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_pairs, 6)."""
        x = self.pool(x)
        return self.fc(x).view(x.size(0), self.num_pairs, 6)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation  (B, 3) → (B, 3, 3)."""
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    axis  = axis_angle / angle
    c, s  = torch.cos(angle), torch.sin(angle)
    t     = 1.0 - c

    ax, ay, az = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    R = torch.stack([
        t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay,
        t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax,
        t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c,
    ], dim=-1).view(*axis_angle.shape[:-1], 3, 3)
    return R


def pose_vec_to_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    6-DOF vector → 4×4 transform matrix.
    vec: (B, 6)  — [axis_angle(3) | translation(3)]
    Returns: (B, 4, 4)
    """
    B = vec.size(0)
    R = axis_angle_to_matrix(vec[:, :3])        # (B, 3, 3)
    t = vec[:, 3:].unsqueeze(-1)                # (B, 3, 1)

    T = torch.eye(4, device=vec.device, dtype=vec.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = t.squeeze(-1)
    return T


class PoseNetwork(nn.Module):
    """
    Pose Network P.

    Input : two consecutive image stacks from all N cameras,
            concatenated along channel dimension.
            Shape: (B, 3*2, H, W)   (each frame is 3-channel RGB)

    Output: relative pose T^{t'→t}  as 4×4 matrix (B, 4, 4)
            and its inverse T^{t→t'} for warping in both directions.

    Note: Because the paper uses fixed camera-to-vehicle extrinsics E,
          the pose network only predicts vehicle ego-motion, which is then
          composed with E to get world-space camera poses.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_input_frames: int = 2,
        rotation_scale: float = 0.01,
        translation_scale: float = 0.1,
    ):
        super().__init__()
        self.rotation_scale    = rotation_scale
        self.translation_scale = translation_scale

        self.encoder = PoseEncoder(backbone, pretrained, num_input_frames)
        # Predict one pair per call (t-1→t or t+1→t)
        self.head = PoseHead(self.encoder.out_dim, num_pairs=1)

    def forward(
        self,
        frame_src: torch.Tensor,
        frame_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_src: (B, 3, H, W)  source frame I_{t'}
            frame_tgt: (B, 3, H, W)  target frame I_t

        Returns:
            T_src2tgt: (B, 4, 4)   T^{t'→t}
            T_tgt2src: (B, 4, 4)   T^{t→t'} (inverse)
        """
        # Concatenate [src | tgt] along channel axis (order matters)
        x = torch.cat([frame_src, frame_tgt], dim=1)
        feat = self.encoder(x)
        vec  = self.head(feat).squeeze(1)  # (B, 6)

        # Scale outputs as in Monodepth2 for training stability
        vec[:, :3] = vec[:, :3] * self.rotation_scale
        vec[:, 3:] = vec[:, 3:] * self.translation_scale

        T_src2tgt = pose_vec_to_matrix(vec)
        T_tgt2src = torch.linalg.inv(T_src2tgt.float()).to(T_src2tgt.dtype)
        return T_src2tgt, T_tgt2src
