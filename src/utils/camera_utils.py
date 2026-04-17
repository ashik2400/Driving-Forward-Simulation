"""
Camera geometry utilities for DrivingForward.

Covers:
  • Depth unprojection  μ = Π^{-1}(pixel, depth)   (Eq. 8)
  • Image warping for photometric loss               (Eq. 3)
  • 3-D covariance construction from scale + rotation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple


# ── Pixel grid ────────────────────────────────────────────────────────────────

def pixel_grid(H: int, W: int, device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return (H*W, 3) homogeneous pixel coordinates [u, v, 1]."""
    v, u = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype),
                          indexing="ij")
    ones = torch.ones_like(u)
    return torch.stack([u, v, ones], dim=-1).reshape(-1, 3)   # (H*W, 3)


# ── Unprojection  (Eq. 8) ─────────────────────────────────────────────────────

def unproject_depth(
    depth:  torch.Tensor,
    K:      torch.Tensor,
    E_c2v:  torch.Tensor,
) -> torch.Tensor:
    """
    μ_i = Π^{-1}(pixel_i, D_i)  – scale-aware Gaussian centres.

    Projects each pixel through the inverse intrinsic matrix to get a
    3-D ray, scales by depth to get a camera-frame 3-D point, then
    transforms to vehicle frame via E_c2v.

    Args:
        depth  : (B, 1, H, W)  – metric depth
        K      : (B, 3, 3)     – camera intrinsics
        E_c2v  : (B, 4, 4)     – camera → vehicle transform

    Returns:
        mu     : (B, H*W, 3)   – 3-D point positions in vehicle frame
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Pixel grid (shared across batch; then unsqueeze for bmm)
    pix = pixel_grid(H, W, device, depth.dtype)          # (HW, 3)
    pix = pix.unsqueeze(0).expand(B, -1, -1)             # (B, HW, 3)

    # Back-project to unit camera rays
    # Cast to float32 before inversion: torch.inverse in FP16 is numerically
    # unstable and produces NaN for well-conditioned matrices after many epochs.
    K_inv  = torch.linalg.inv(K.float()).to(K.dtype)      # (B, 3, 3)
    rays   = (K_inv @ pix.permute(0, 2, 1)).permute(0, 2, 1)  # (B, HW, 3)

    # Scale by depth
    d_flat = depth.view(B, 1, H * W).permute(0, 2, 1)    # (B, HW, 1)
    pts_c  = rays * d_flat                                 # (B, HW, 3)  camera frame

    # Homogeneous → vehicle frame
    ones   = torch.ones(B, H * W, 1, device=device, dtype=depth.dtype)
    pts_h  = torch.cat([pts_c, ones], dim=-1)              # (B, HW, 4)
    pts_v  = (E_c2v @ pts_h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, HW, 4)

    return pts_v[..., :3]                                  # (B, HW, 3)


# ── Image warping  (Eq. 3) ────────────────────────────────────────────────────

def warp_image(
    src_image: torch.Tensor,
    depth_tgt: torch.Tensor,
    K_src:     torch.Tensor,
    K_tgt:     torch.Tensor,
    T_tgt2src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp src_image to target viewpoint using target depth and relative pose.

    Implements I_trg = K_src[T^{trg→src} Π^{-1}(I_trg, D_trg)]  (Eq. 3).

    Args:
        src_image : (B, 3, H, W)   source image I_src
        depth_tgt : (B, 1, H, W)   depth of target image
        K_src     : (B, 3, 3)      source camera intrinsics
        K_tgt     : (B, 3, 3)      target camera intrinsics
        T_tgt2src : (B, 4, 4)      transform target→source (camera frames)

    Returns:
        warped    : (B, 3, H, W)   warped source image
        valid_mask: (B, 1, H, W)   mask of valid (in-bounds) pixels
    """
    B, _, H, W = src_image.shape
    device = src_image.device

    # Upsample depth to image resolution if the depth network outputs at a
    # lower stride (e.g. stride-2 decoder head gives H/2 × W/2).
    if depth_tgt.shape[-2:] != (H, W):
        depth_tgt = F.interpolate(depth_tgt, size=(H, W),
                                  mode="bilinear", align_corners=False)

    # Back-project target depth to target camera frame
    pix   = pixel_grid(H, W, device, depth_tgt.dtype).unsqueeze(0).expand(B, -1, -1)
    K_tgt_inv = torch.linalg.inv(K_tgt.float()).to(K_tgt.dtype)
    rays  = (K_tgt_inv @ pix.permute(0, 2, 1)).permute(0, 2, 1)
    d_flat = depth_tgt.view(B, 1, H * W).permute(0, 2, 1)
    pts_tgt = rays * d_flat                                    # (B, HW, 3)

    # Transform to source camera frame
    ones   = torch.ones(B, H * W, 1, device=device, dtype=depth_tgt.dtype)
    pts_h  = torch.cat([pts_tgt, ones], dim=-1)                # (B, HW, 4)
    pts_src = (T_tgt2src @ pts_h.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]  # (B, HW, 3)

    # Project to source image plane
    pts_proj = (K_src @ pts_src.permute(0, 2, 1)).permute(0, 2, 1)  # (B, HW, 3)
    z        = pts_proj[..., 2:3].clamp(min=1e-4)
    uv       = pts_proj[..., :2] / z                               # (B, HW, 2)

    # Normalise to [-1, 1] for grid_sample
    uv_norm = uv.view(B, H, W, 2)
    uv_norm[..., 0] = (uv_norm[..., 0] / (W - 1)) * 2 - 1
    uv_norm[..., 1] = (uv_norm[..., 1] / (H - 1)) * 2 - 1

    warped     = F.grid_sample(src_image, uv_norm, mode="bilinear",
                               padding_mode="border", align_corners=True)
    valid_mask = ((uv_norm[..., 0].abs() <= 1) &
                  (uv_norm[..., 1].abs() <= 1)).unsqueeze(1).float()

    return warped, valid_mask


# ── 3-D covariance from scale + rotation ─────────────────────────────────────

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (N, 4) [w, x, y, z] → rotation matrix (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).view(-1, 3, 3)
    return R


def build_covariance_3d(
    scales:    torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """
    Build 3-D covariance matrix Σ_k = R S S^T R^T.

    Args:
        scales    : (N, 3)   – per-axis scale
        rotations : (N, 4)   – unit quaternion

    Returns:
        cov3d     : (N, 6)   – upper-triangle of symmetric 3×3 matrix
    """
    R = quaternion_to_matrix(rotations)                # (N, 3, 3)
    S = torch.diag_embed(scales)                       # (N, 3, 3)
    M = R @ S                                          # (N, 3, 3)
    cov = M @ M.transpose(-1, -2)                      # (N, 3, 3)
    # Return upper triangle as 6-vector [00, 01, 02, 11, 12, 22]
    return torch.stack([
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
    ], dim=-1)
