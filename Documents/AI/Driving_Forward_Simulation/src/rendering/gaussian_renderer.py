"""
Gaussian Renderer – wraps diff-gaussian-rasterization for DrivingForward.

At inference, all N-camera Gaussian sets are merged into a single scene
and rendered from any target viewpoint.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GaussianRenderer(nn.Module):
    """
    Differentiable 3-D Gaussian splat renderer.

    Wraps diff-gaussian-rasterization when available;
    otherwise uses the lightweight fallback in DrivingForward.
    """

    def __init__(self, sh_degree: int = 1, bg_color: Tuple = (0, 0, 0)):
        super().__init__()
        self.sh_degree = sh_degree
        self.bg_color  = bg_color

        try:
            from diff_gaussian_rasterization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
            self._GRS  = GaussianRasterizationSettings
            self._GR   = GaussianRasterizer
            self._avail = True
        except ImportError:
            self._avail = False
            print("[GaussianRenderer] diff-gaussian-rasterization unavailable – "
                  "will use point-composite fallback.")

    # ── Merge Gaussians from multiple cameras ─────────────────────────────────

    @staticmethod
    def merge(gaussians: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Concatenate Gaussians from N cameras along the primitive axis."""
        keys = gaussians[0].keys()
        return {k: torch.cat([g[k] for g in gaussians], dim=1) for k in keys}

    # ── Render ────────────────────────────────────────────────────────────────

    def forward(
        self,
        gaussians:   Dict[str, torch.Tensor],
        K:           torch.Tensor,
        E_v2c:       torch.Tensor,
        H:           int,
        W:           int,
        item:        int = 0,
    ) -> torch.Tensor:
        """
        Render a single view from a merged Gaussian set.

        Args:
            gaussians : dict with means3D (B, N, 3), scales (B, N, 3),
                        rotations (B, N, 4), colours (B, N, C), opacities (B, N)
            K         : (3, 3) target camera intrinsics
            E_v2c     : (4, 4) vehicle → camera transform
            H, W      : output image resolution
            item      : batch index to render

        Returns:
            rendered  : (3, H, W) float [0, 1]
        """
        if self._avail:
            return self._render_3dgs(gaussians, K, E_v2c, H, W, item)
        return self._render_fallback(gaussians, K, E_v2c, H, W, item)

    def _render_3dgs(self, gaussians, K, E_v2c, H, W, item):
        # diff-gaussian-rasterization CUDA kernel only accepts Float32.
        # Disable autocast for the entire rasterizer call so that matmul and
        # other eligible ops don't silently downcast to BFloat16/FP16.
        with torch.autocast(device_type="cuda", enabled=False):
            means3D   = gaussians["means3D"][item].float()
            scales    = gaussians["scales"][item].float()
            rotations = gaussians["rotations"][item].float()
            colours   = gaussians["colours"][item].float()
            opacities = gaussians["opacities"][item].float().unsqueeze(-1)

            device = means3D.device
            bg     = torch.tensor(self.bg_color, dtype=torch.float32, device=device)

            K     = K.float()
            E_v2c = E_v2c.float()

            tanfovx = float(0.5 * W / K[0, 0])
            tanfovy = float(0.5 * H / K[1, 1])

            # Build full projection matrix (4×4)
            proj = _build_proj_matrix(K, H, W, near=0.1, far=100.0).to(device)

            settings = self._GRS(
                image_height  = H,
                image_width   = W,
                tanfovx       = tanfovx,
                tanfovy       = tanfovy,
                bg            = bg,
                scale_modifier= 1.0,
                viewmatrix    = E_v2c.T.contiguous(),
                projmatrix    = (proj @ E_v2c).T.contiguous(),
                sh_degree     = self.sh_degree,
                campos        = (-E_v2c[:3, :3].T @ E_v2c[:3, 3]),
                prefiltered   = False,
                debug         = False,
                antialiasing  = False,
            )
            rast = self._GR(settings)
            num_coeff = (self.sh_degree + 1) ** 2
            shs = colours.view(-1, num_coeff, 3)

            rendered, *_ = rast(
                means3D         = means3D,
                means2D         = torch.zeros_like(means3D, requires_grad=True),
                shs             = shs,
                colors_precomp  = None,
                opacities       = opacities,
                scales          = scales,
                rotations       = rotations,
                cov3D_precomp   = None,
            )
        return rendered.clamp(0, 1)

    def _render_fallback(self, gaussians, K, E_v2c, H, W, item):
        """Simple depth-sorted point composite (no gradients through render)."""
        means   = gaussians["means3D"][item]          # (N, 3)
        colours = gaussians["colours"][item, :, :3]   # (N, 3) DC term
        colours = torch.sigmoid(colours)

        pts_c = (E_v2c[:3, :3] @ means.T + E_v2c[:3, 3:]).T  # (N, 3)
        z     = pts_c[:, 2]
        valid = z > 0.1

        u = (pts_c[valid, 0] / z[valid] * K[0, 0] + K[0, 2]).long()
        v = (pts_c[valid, 1] / z[valid] * K[1, 1] + K[1, 2]).long()
        c = colours[valid]
        z_v = z[valid]

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, c, z_v = u[in_bounds], v[in_bounds], c[in_bounds], z_v[in_bounds]

        canvas = torch.zeros(3, H, W, device=means.device)
        z_buf  = torch.full((H, W), float("inf"), device=means.device)

        order  = z_v.argsort()
        for i in order:
            vi, ui = v[i].item(), u[i].item()
            if z_v[i] < z_buf[vi, ui]:
                canvas[:, vi, ui] = c[i]
                z_buf[vi, ui]     = z_v[i]

        return canvas


# ── Projection matrix helper ──────────────────────────────────────────────────

def _build_proj_matrix(K: torch.Tensor, H: int, W: int,
                        near: float, far: float) -> torch.Tensor:
    """Build OpenGL-style 4×4 projection matrix from intrinsics."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    P = torch.zeros(4, 4, dtype=K.dtype)
    P[0, 0] =  2 * fx / W
    P[1, 1] =  2 * fy / H
    P[0, 2] = -(2 * cx / W - 1)
    P[1, 2] = -(2 * cy / H - 1)
    P[2, 2] =  (far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] =  1.0
    return P
