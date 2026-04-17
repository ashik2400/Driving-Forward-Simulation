"""
Evaluation metrics: PSNR, SSIM, LPIPS
(Paper results: PSNR 26.06 / SSIM 0.781 / LPIPS 0.215 in MF mode)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

try:
    import lpips as lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio. Both tensors ∈ [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM. Tensors (B, 3, H, W) or (3, H, W)."""
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x   = F.avg_pool2d(pred,   3, 1, 1)
    mu_y   = F.avg_pool2d(target, 3, 1, 1)
    mu_xx  = F.avg_pool2d(pred   * pred,   3, 1, 1)
    mu_yy  = F.avg_pool2d(target * target, 3, 1, 1)
    mu_xy  = F.avg_pool2d(pred   * target, 3, 1, 1)

    sig_x  = mu_xx - mu_x * mu_x
    sig_y  = mu_yy - mu_y * mu_y
    sig_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return ((num / den.clamp(min=1e-8)).mean()).item()


class Evaluator:
    """Accumulates per-sample metrics and computes averages."""

    def __init__(self):
        self.reset()
        if _HAS_LPIPS:
            self._lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False)
        else:
            self._lpips_fn = None

    def reset(self):
        self._psnr  = []
        self._ssim  = []
        self._lpips = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: (3, H, W) or (B, 3, H, W) in [0, 1]."""
        if pred.dim() == 3:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred   = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        self._psnr.append(psnr(pred, target))
        self._ssim.append(ssim_score(pred, target))

        if self._lpips_fn is not None:
            lp = self._lpips_fn(pred * 2 - 1, target * 2 - 1).mean().item()
            self._lpips.append(lp)

    def compute(self) -> Dict[str, float]:
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        out = {
            "PSNR":  mean(self._psnr),
            "SSIM":  mean(self._ssim),
        }
        if self._lpips:
            out["LPIPS"] = mean(self._lpips)
        return out
