"""
DrivingForward – Top-level model  (AAAI-2025)

Integrates:
  1. DepthNetwork   D  → depth maps + image features
  2. PoseNetwork    P  → relative ego-motion (training only)
  3. GaussianNetwork G → per-pixel 3DGS attributes
  4. Scale-aware localisation: μ = Π^{-1}(pixel, depth)
  5. 3DGS rendering via diff-gaussian-rasterization

Two inference modes (Section 4.1 of paper):
  • SF (single-frame): only depth + Gaussian networks, one image at a time.
                       Flexible input count at test time.
  • MF (multi-frame):  adds pose network; uses t-1, t, t+1 frames.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_network    import DepthNetwork
from .pose_network     import PoseNetwork
from .gaussian_network import GaussianNetwork
from ..utils.camera_utils import unproject_depth, build_covariance_3d


class DrivingForward(nn.Module):
    """
    DrivingForward model.

    Args:
        cfg: namespace / dict loaded from YAML config
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.mode           = m.get("mode", "SF")
        self.min_depth      = m["min_depth"]
        self.max_depth      = m["max_depth"]
        self.sh_degree      = m["num_sh_degree"]
        self.scale_min      = m["gaussian_scale_min"]
        self.scale_max      = m["gaussian_scale_max"]

        backbone        = m["backbone"]
        pretrained      = m.get("pretrained", True)
        feat_dim        = 64
        grad_ckpt       = cfg["training"].get("grad_checkpoint", False)

        # Shared modules
        self.depth_net = DepthNetwork(
            backbone=backbone,
            pretrained=pretrained,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            feat_dim=feat_dim,
            grad_checkpoint=grad_ckpt,
        )
        self.gaussian_net = GaussianNetwork(
            depth_feat_dim=feat_dim,
            img_feat_dim=feat_dim,
            hidden_dim=128,
            sh_degree=self.sh_degree,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )

        if self.mode == "MF":
            self.pose_net = PoseNetwork(backbone=backbone, pretrained=pretrained)
        else:
            self.pose_net = None

        # Try to import differentiable Gaussian rasterizer
        try:
            from diff_gaussian_rasterization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
            self._rasterizer_cls      = GaussianRasterizer
            self._rast_settings_cls   = GaussianRasterizationSettings
            self._has_rasterizer      = True
        except ImportError:
            self._has_rasterizer = False
            print("[DrivingForward] WARNING: diff-gaussian-rasterization not found. "
                  "Rendering will use fallback (alpha blending approximation).")

    # ── Per-image forward pass ────────────────────────────────────────────────

    def encode_image(
        self,
        image: torch.Tensor,
        return_all_scales: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Run depth + Gaussian network for a single camera image.

        Args:
            image: (B, 3, H, W)

        Returns:
            depth   : (B, 1, H, W)
            params  : dict with keys scale, rotation, colour, opacity
            scales  : multi-scale depth dict (training only)
        """
        depth, feat, scales = self.depth_net(image, return_all_scales)
        params = self.gaussian_net(feat, feat)   # F_depth == F_image (shared)
        return depth, params, scales

    # ── Scale-aware localisation: μ = Π^{-1}(pixel, depth) ──────────────────

    def localise_gaussians(
        self,
        depth:    torch.Tensor,
        K:        torch.Tensor,
        E_c2v:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. 8 in paper: μ_i = Π^{-1}(I_i, D_i)

        Unprojects depth to 3-D positions in vehicle frame.

        Args:
            depth  : (B, 1, H, W)    metric depth
            K      : (B, 3, 3)       camera intrinsics
            E_c2v  : (B, 4, 4)       camera-to-vehicle extrinsics

        Returns:
            mu     : (B, H*W, 3)     3-D Gaussian centres
        """
        return unproject_depth(depth, K, E_c2v)

    # ── Assemble Gaussian primitives ─────────────────────────────────────────

    @staticmethod
    def assemble_gaussians(
        mu:     torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Flatten spatial dimensions and build full Gaussian set.

        Args:
            mu     : (B, H*W, 3)
            params : output of GaussianNetwork.forward — spatial maps

        Returns dict ready for rendering with keys:
            means3D, scales, rotations, colours, opacities
        """
        B = mu.shape[0]

        def flat(t: torch.Tensor) -> torch.Tensor:
            # (B, C, H, W) → (B, H*W, C)
            return t.flatten(2).permute(0, 2, 1)

        return {
            "means3D":   mu,
            "scales":    flat(params["scale"]),
            "rotations": flat(params["rotation"]),
            "colours":   flat(params["colour"]),
            "opacities": flat(params["opacity"]).squeeze(-1),
        }

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        gaussians:   Dict[str, torch.Tensor],
        K_target:    torch.Tensor,
        E_c2v_target: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Render a target view from a set of Gaussian primitives.

        When diff-gaussian-rasterization is available uses the full
        differentiable splat rasteriser; otherwise falls back to a
        point-cloud depth-composite approximation for testing.

        Args:
            gaussians    : dict from assemble_gaussians
            K_target     : (3, 3)  target camera intrinsics
            E_c2v_target : (4, 4)  target cam-to-vehicle extrinsic
            H, W         : output image size

        Returns:
            rendered : (3, H, W)
        """
        if self._has_rasterizer:
            return self._render_3dgs(gaussians, K_target, E_c2v_target, H, W)
        return self._render_fallback(gaussians, K_target, E_c2v_target, H, W)

    def _render_3dgs(
        self,
        gaussians:    Dict[str, torch.Tensor],
        K:            torch.Tensor,
        E_c2v:        torch.Tensor,
        H: int, W: int,
    ) -> torch.Tensor:
        """Full differentiable 3DGS rendering (single item from batch)."""
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        from ..rendering.gaussian_renderer import _build_proj_matrix

        # diff-gaussian-rasterization CUDA kernel only accepts Float32.
        # Disable autocast so matmul / eligible ops don't silently downcast.
        with torch.autocast(device_type="cuda", enabled=False):
            means3D   = gaussians["means3D"][0].float()
            scales    = gaussians["scales"][0].float()
            rotations = gaussians["rotations"][0].float()
            colours   = gaussians["colours"][0].float()
            opacities = gaussians["opacities"][0].float().unsqueeze(-1)

            K     = K.float()
            E_c2v = E_c2v.float()

            # Vehicle → camera transform
            E_v2c = torch.linalg.inv(E_c2v)
            device = means3D.device

            bg = torch.zeros(3, device=device)
            tanfovx = float(0.5 * W / K[0, 0])
            tanfovy = float(0.5 * H / K[1, 1])

            # Full 4×4 OpenGL-style projection matrix required by the CUDA kernel.
            proj = _build_proj_matrix(K, H, W, near=0.1, far=100.0).to(device)

            settings = GaussianRasterizationSettings(
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
            rast = self._rasterizer_cls(settings)
            rendered, _ = rast(
                means3D        = means3D,
                means2D        = torch.zeros_like(means3D, requires_grad=True),
                shs            = colours.view(-1, (self.sh_degree + 1) ** 2, 3),
                colors_precomp = None,
                opacities      = opacities,
                scales         = scales,
                rotations      = rotations,
                cov3D_precomp  = None,
            )
        return rendered.clamp(0, 1)

    def _render_fallback(
        self,
        gaussians:    Dict[str, torch.Tensor],
        K:            torch.Tensor,
        E_c2v:        torch.Tensor,
        H: int, W: int,
    ) -> torch.Tensor:
        """
        Differentiable-free fallback: project Gaussians as points and
        alpha-blend by depth.  Used for testing without rasterizer.
        """
        device = gaussians["means3D"].device
        means  = gaussians["means3D"][0]        # (N, 3) in vehicle frame

        # Transform to camera frame
        E_v2c  = torch.linalg.inv(E_c2v.float()).to(E_c2v.dtype)
        pts_c  = (E_v2c[:3, :3] @ means.T + E_v2c[:3, 3:]).T  # (N, 3)

        # Perspective projection
        z     = pts_c[:, 2].clamp(min=self.min_depth)
        u     = (pts_c[:, 0] / z * K[0, 0] + K[0, 2]).long()
        v     = (pts_c[:, 1] / z * K[1, 1] + K[1, 2]).long()

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        rendered = torch.zeros(3, H, W, device=device)

        # Simple z-composite (closest wins) — not differentiable
        z_buf = torch.full((H, W), float("inf"), device=device)
        raw_col = gaussians["colours"][0, :, :3]  # (N, 3) DC term
        raw_col = torch.sigmoid(raw_col)

        idx_sorted = z[valid].argsort(descending=True)
        u_v, v_v, c_v = u[valid][idx_sorted], v[valid][idx_sorted], raw_col[valid][idx_sorted]
        for i in range(u_v.shape[0]):
            rendered[:, v_v[i], u_v[i]] = c_v[i]

        return rendered

    # ── Main forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        batch: Dict,
    ) -> Dict:
        """
        Main forward pass.

        batch keys:
          'images'   : (B, N, 3, H, W)        – N camera images at time t
          'K'        : (B, N, 3, 3)            – intrinsics per camera
          'E_c2v'    : (B, N, 4, 4)            – cam-to-vehicle extrinsics
          'images_prev': (B, N, 3, H, W)       – t-1 (MF only)
          'images_next': (B, N, 3, H, W)       – t+1 (MF only)

        Returns dict with:
          'depths'       : list of (B, 1, H, W) per camera
          'depth_scales' : list of multi-scale dicts (training)
          'gaussians'    : list of Gaussian dicts per camera
          'poses'        : list of (T_prev, T_next) tuples (MF only)
        """
        images  = batch["images"]      # (B, N, 3, H, W)
        K       = batch["K"]           # (B, N, 3, 3)
        E_c2v   = batch["E_c2v"]       # (B, N, 4, 4)

        B, N, _, H, W = images.shape
        is_training = self.training

        depths      = []
        all_params  = []
        depth_scales = []
        all_gaussians = []

        for i in range(N):
            img_i   = images[:, i]        # (B, 3, H, W)
            depth_i, params_i, scales_i = self.encode_image(img_i, return_all_scales=is_training)

            mu_i = self.localise_gaussians(depth_i, K[:, i], E_c2v[:, i])
            g_i  = self.assemble_gaussians(mu_i, params_i)

            depths.append(depth_i)
            all_params.append(params_i)
            depth_scales.append(scales_i)
            all_gaussians.append(g_i)

        out = {
            "depths":       depths,
            "depth_scales": depth_scales,
            "params":       all_params,
            "gaussians":    all_gaussians,
        }

        # Pose prediction (MF mode, training)
        if self.mode == "MF" and self.pose_net is not None:
            poses = []
            imgs_prev = batch.get("images_prev")
            imgs_next = batch.get("images_next")
            for i in range(N):
                T_prev, T_prev_inv, T_next, T_next_inv = None, None, None, None
                if imgs_prev is not None:
                    T_prev, T_prev_inv = self.pose_net(imgs_prev[:, i], images[:, i])
                if imgs_next is not None:
                    T_next, T_next_inv = self.pose_net(imgs_next[:, i], images[:, i])
                poses.append((T_prev, T_prev_inv, T_next, T_next_inv))
            out["poses"] = poses

        return out
