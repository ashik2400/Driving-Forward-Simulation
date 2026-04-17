"""
Loss functions for DrivingForward  (AAAI-2025, Section 3.2 & 3.3)

L_repro  – photometric reconstruction loss with SSIM  (Eq. 2)
L_loc    – scale-aware localisation: L_repro + λ_sp·L_sp + λ_sp-tm·L_sp-tm
           + λ_smooth·L_smooth   (Eq. 7)
L_render – rendering loss: L2 + LPIPS  (Eq. 13 region)
Total    – L_loc + λ_render·L_render
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import lpips as lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False
    print("[losses] lpips not installed – LPIPS loss will be skipped.")


# ── SSIM ─────────────────────────────────────────────────────────────────────

def ssim(x: torch.Tensor, y: torch.Tensor,
         C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """
    Structural Similarity Index (Wang et al. 2004).
    Returns per-pixel SSIM map, averaged over channels.
    """
    mu_x  = F.avg_pool2d(x, 3, 1, 1)
    mu_y  = F.avg_pool2d(y, 3, 1, 1)
    mu_xx = F.avg_pool2d(x * x, 3, 1, 1)
    mu_yy = F.avg_pool2d(y * y, 3, 1, 1)
    mu_xy = F.avg_pool2d(x * y, 3, 1, 1)

    sig_x  = mu_xx - mu_x * mu_x
    sig_y  = mu_yy - mu_y * mu_y
    sig_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return (num / den.clamp(min=1e-8)).mean(dim=1, keepdim=True)


# ── Photometric loss  (Eq. 2) ─────────────────────────────────────────────────

class PhotometricLoss(nn.Module):
    """
    L_repro = η·(1 - SSIM(I_trg, Î_trg)) / 2 + (1-η)·||I_trg - Î_trg||₁

    η = ssim_weight (default 0.85, as in paper)
    """

    def __init__(self, ssim_weight: float = 0.85):
        super().__init__()
        self.eta = ssim_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred   : (B, 3, H, W)  warped / synthesised image
            target : (B, 3, H, W)  ground-truth target
            mask   : (B, 1, H, W)  valid region (1 = valid)

        Returns scalar loss.
        """
        l1   = (pred - target).abs()                          # (B, 3, H, W)
        ssim_map = (1.0 - ssim(pred, target)) * 0.5           # (B, 1, H, W)

        per_pixel = self.eta * ssim_map + (1 - self.eta) * l1.mean(dim=1, keepdim=True)

        if mask is not None:
            per_pixel = per_pixel * mask
            return per_pixel.sum() / mask.sum().clamp(min=1)
        return per_pixel.mean()


# ── Depth smoothness ─────────────────────────────────────────────────────────

class SmoothnessLoss(nn.Module):
    """
    Edge-aware depth smoothness loss (Godard et al. 2019).
    L_smooth = |∂D/∂u|·e^{-|∂I/∂u|} + |∂D/∂v|·e^{-|∂I/∂v|}
    """

    def forward(self, depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # Align image to depth resolution if the depth network outputs at a
        # lower stride (e.g. stride-2 head gives H/2 × W/2).
        if image.shape[-2:] != depth.shape[-2:]:
            image = F.interpolate(image, size=depth.shape[-2:],
                                  mode="bilinear", align_corners=False)

        # Mean-normalise depth
        mean_d = depth.mean(dim=[2, 3], keepdim=True).clamp(min=1e-4)
        depth  = depth / mean_d

        d_dx = (depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs()
        d_dy = (depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs()

        i_dx = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(dim=1, keepdim=True)
        i_dy = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(dim=1, keepdim=True)

        loss = (d_dx * torch.exp(-i_dx)).mean() + (d_dy * torch.exp(-i_dy)).mean()
        return loss


# ── Scale-aware localisation loss  (Eq. 7) ────────────────────────────────────

class LocalisationLoss(nn.Module):
    """
    L_loc = L_repro + λ_sp·L_sp + λ_sp-tm·L_sp-tm + λ_smooth·L_smooth

    All three L_repro variants use the same PhotometricLoss but with
    different context pairs:
      temporal  – same camera, adjacent timesteps (t-1 or t+1)
      spatial   – adjacent cameras at same timestep
      spatial-temporal – adjacent cameras at adjacent timesteps
    """

    def __init__(
        self,
        ssim_weight:   float = 0.85,
        lambda_sp:     float = 0.5,
        lambda_sp_tm:  float = 0.5,
        lambda_smooth: float = 0.001,
    ):
        super().__init__()
        self.photo    = PhotometricLoss(ssim_weight)
        self.smooth   = SmoothnessLoss()
        self.lam_sp   = lambda_sp
        self.lam_sptm = lambda_sp_tm
        self.lam_sm   = lambda_smooth

    def forward(
        self,
        warped_temporal:      Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        warped_spatial:       Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        warped_spatial_temp:  Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        depths:               List[torch.Tensor],
        images:               torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Each warped_* is a list of (pred, target, mask) tuples per camera.
        """
        device = depths[0].device
        losses: Dict[str, torch.Tensor] = {}

        def _avg_photo(pairs):
            if not pairs:
                return torch.tensor(0.0, device=device)
            vals = [self.photo(p, t, m) for p, t, m in pairs]
            return torch.stack(vals).mean()

        losses["l_temporal"]  = _avg_photo(warped_temporal  or [])
        losses["l_sp"]        = _avg_photo(warped_spatial   or [])
        losses["l_sp_tm"]     = _avg_photo(warped_spatial_temp or [])

        # Smoothness per camera
        sm_vals = [self.smooth(d, images[:, i]) for i, d in enumerate(depths)]
        losses["l_smooth"] = torch.stack(sm_vals).mean()

        losses["l_loc"] = (
            losses["l_temporal"]
            + self.lam_sp   * losses["l_sp"]
            + self.lam_sptm * losses["l_sp_tm"]
            + self.lam_sm   * losses["l_smooth"]
        )
        return losses


# ── Rendering loss  (L2 + LPIPS) ─────────────────────────────────────────────

class RenderLoss(nn.Module):
    """
    L_render = L2(rendered, target) + λ_lpips·LPIPS(rendered, target)

    LPIPS uses VGG features (Alex net also works; VGG is more accurate).
    """

    def __init__(self, lambda_lpips: float = 0.05):
        super().__init__()
        self.lam = lambda_lpips
        if _HAS_LPIPS:
            self.lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False)
        else:
            self.lpips_fn = None

    def forward(
        self,
        rendered: torch.Tensor,
        target:   torch.Tensor,
        mask:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rendered : (B, 3, H, W)  model output
            target   : (B, 3, H, W)  ground truth
            mask     : (B, 1, H, W)  optional valid mask
        """
        diff = (rendered - target) ** 2
        if mask is not None:
            diff = diff * mask
            l2 = diff.sum() / mask.sum().clamp(min=1)
        else:
            l2 = diff.mean()

        if self.lpips_fn is not None:
            # LPIPS expects [-1, 1] images.
            # Force FP32: VGG16 activations overflow FP16 (~epoch 19+).
            r_n = (rendered * 2 - 1).float()
            t_n = (target   * 2 - 1).float()
            if mask is not None:
                r_n = r_n * mask.float()
                t_n = t_n * mask.float()
            with torch.amp.autocast("cuda", enabled=False):
                lp = self.lpips_fn(r_n, t_n).mean()
        else:
            lp = torch.tensor(0.0, device=rendered.device)

        return l2 + self.lam * lp


# ── Colour consistency loss ───────────────────────────────────────────────────

# SH DC-to-colour constant (same as in the CUDA kernel)
_SH_C0 = 0.28209479177387814


class ColourConsistencyLoss(nn.Module):
    """
    Directly supervises the SH DC coefficient to reproduce the input pixel colour.

    The 3DGS CUDA kernel evaluates colour as:
        colour = clamp(SH_C0 * sh_dc + higher_order + 0.5, 0, 1)

    We approximate (ignoring higher-order terms, which are zero-init):
        colour_pred = sigmoid(SH_C0 * sh_dc + 0.5)

    Loss = MSE(colour_pred, image_pixel)

    This gives the colour head a dense per-pixel gradient that is
    independent of whether the render pipeline succeeds, which is critical
    when the render loss is sparse or weak.
    """

    def forward(
        self,
        sh_dc: torch.Tensor,   # (B, 3, H, W)  – first 3 channels of colour output
        image: torch.Tensor,   # (B, 3, H, W)  – input RGB image [0, 1]
    ) -> torch.Tensor:
        if sh_dc.shape[-2:] != image.shape[-2:]:
            image = F.interpolate(image, size=sh_dc.shape[-2:],
                                  mode="bilinear", align_corners=False)
        colour_pred = torch.sigmoid(sh_dc.float() * _SH_C0 + 0.5)
        return F.mse_loss(colour_pred, image.float())


# ── Total loss ────────────────────────────────────────────────────────────────

class TotalLoss(nn.Module):
    """
    Combines L_loc and L_render with configurable weights.

    Usage:
        criterion = TotalLoss(cfg["loss"])
        total, breakdown = criterion(loc_losses, rendered, target)
    """

    def __init__(self, loss_cfg: dict):
        super().__init__()
        self.loc_loss    = LocalisationLoss(
            ssim_weight   = loss_cfg.get("ssim_weight",   0.85),
            lambda_sp     = loss_cfg.get("lambda_sp",     0.5),
            lambda_sp_tm  = loss_cfg.get("lambda_sp_tm",  0.5),
            lambda_smooth = loss_cfg.get("lambda_smooth", 0.001),
        )
        self.render_loss  = RenderLoss(loss_cfg.get("lambda_lpips", 0.05))
        self.colour_loss  = ColourConsistencyLoss()
        self.lam_render   = loss_cfg.get("lambda_render",  1.0)
        self.lam_colour   = loss_cfg.get("lambda_colour",  0.0)

    def forward(
        self,
        warped_temporal:     Optional[List],
        warped_spatial:      Optional[List],
        warped_spatial_temp: Optional[List],
        depths:              List[torch.Tensor],
        images:              torch.Tensor,
        rendered:            Optional[torch.Tensor] = None,
        target:              Optional[torch.Tensor] = None,
        render_mask:         Optional[torch.Tensor] = None,
        all_params:          Optional[List[Dict]]   = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        loc_losses = self.loc_loss(
            warped_temporal, warped_spatial, warped_spatial_temp, depths, images
        )
        breakdown  = {**loc_losses}

        render = torch.tensor(0.0, device=depths[0].device)
        if rendered is not None and target is not None:
            render = self.render_loss(rendered, target, render_mask)
        breakdown["l_render"] = render

        # Colour consistency: supervise SH DC coefficients directly from input pixels.
        colour = torch.tensor(0.0, device=depths[0].device)
        if self.lam_colour > 0.0 and all_params is not None:
            N = images.shape[1]
            vals = []
            for i in range(N):
                # colour output: (B, 3*num_coeff, H, W); first 3 channels = DC term
                sh_dc = all_params[i]["colour"][:, :3, :, :]
                vals.append(self.colour_loss(sh_dc, images[:, i]))
            colour = torch.stack(vals).mean()
        breakdown["l_colour"] = colour

        total = loc_losses["l_loc"] + self.lam_render * render + self.lam_colour * colour
        breakdown["total"] = total

        return total, breakdown
