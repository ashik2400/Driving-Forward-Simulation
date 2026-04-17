"""
Visualisation utilities for DrivingForward.

• save_surround_view  – 2×3 grid of surround camera images
• save_depth_map      – colour-mapped depth with colour bar
• save_novel_view     – GT vs rendered side-by-side
• make_comparison_grid – full evaluation grid for a scene
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Colour map helpers ────────────────────────────────────────────────────────

def depth_to_colormap(
    depth: torch.Tensor,
    vmin: float = 0.1,
    vmax: float = 80.0,
    cmap: str = "magma",
) -> np.ndarray:
    """(1, H, W) or (H, W) torch tensor → (H, W, 3) uint8 RGB."""
    d = depth.squeeze().detach().cpu().numpy()
    d = (d - vmin) / (vmax - vmin + 1e-8)
    d = np.clip(d, 0.0, 1.0)
    rgba = (cm.get_cmap(cmap)(d) * 255).astype(np.uint8)
    return rgba[..., :3]


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8 RGB."""
    return (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


# ── Surround-view grid ────────────────────────────────────────────────────────

CAMERA_LABELS = [
    "Front", "Front-Right", "Back-Right",
    "Back",  "Back-Left",   "Front-Left",
]

def save_surround_view(
    images: torch.Tensor,
    save_path: str,
    title: str = "Surround View",
) -> None:
    """
    Args:
        images: (N, 3, H, W)  N=6 cameras
        save_path: output file path
    """
    N = images.shape[0]
    assert N == 6, "Expected 6 cameras for nuScenes surround view"

    rows, cols = 2, 3
    H, W = images.shape[-2], images.shape[-1]
    grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for i in range(N):
        r, c = divmod(i, cols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = tensor_to_uint8(images[i])

    fig, ax = plt.subplots(figsize=(cols * 4, rows * 3))
    ax.imshow(grid)
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    # Camera labels
    for i, label in enumerate(CAMERA_LABELS):
        r, c = divmod(i, cols)
        ax.text(c * W + W // 2, r * H + 15, label,
                color="white", fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Depth visualisation ───────────────────────────────────────────────────────

def save_depth_map(
    depth: torch.Tensor,
    save_path: str,
    image: Optional[torch.Tensor] = None,
    title: str = "Predicted Depth",
) -> None:
    """
    depth : (1, H, W) metric depth
    image : (3, H, W) optional overlay image
    """
    depth_rgb = depth_to_colormap(depth)
    nplots = 2 if image is not None else 1
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 4))
    if nplots == 1:
        axes = [axes]

    if image is not None:
        axes[0].imshow(tensor_to_uint8(image))
        axes[0].set_title("Input Image")
        axes[0].axis("off")

    im = axes[-1].imshow(depth_rgb)
    axes[-1].set_title(title)
    axes[-1].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(0.1, 80.0),
            cmap=plt.cm.magma,
        ),
        ax=axes[-1], fraction=0.046, pad=0.04, label="Depth (m)",
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Novel view comparison ─────────────────────────────────────────────────────

def save_novel_view_comparison(
    gt: torch.Tensor,
    rendered: torch.Tensor,
    save_path: str,
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
) -> None:
    """GT vs Rendered side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(tensor_to_uint8(gt))
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    title = "Rendered"
    if psnr is not None:
        title += f"  PSNR={psnr:.2f}"
    if ssim is not None:
        title += f"  SSIM={ssim:.3f}"

    axes[1].imshow(tensor_to_uint8(rendered))
    axes[1].set_title(title)
    axes[1].axis("off")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Full scene comparison grid ────────────────────────────────────────────────

def make_scene_grid(
    gt_images:       torch.Tensor,
    rendered_images: torch.Tensor,
    depth_maps:      torch.Tensor,
    save_path:       str,
) -> None:
    """
    3-row grid: GT | Rendered | Depth  for each of N cameras.

    gt_images, rendered_images, depth_maps : (N, 3/1, H, W)
    """
    N = gt_images.shape[0]
    rows = 3

    fig, axes = plt.subplots(rows, N, figsize=(N * 3, rows * 2))
    row_labels = ["Ground Truth", "Rendered", "Depth"]

    for i in range(N):
        axes[0, i].imshow(tensor_to_uint8(gt_images[i]))
        axes[1, i].imshow(tensor_to_uint8(rendered_images[i]))
        axes[2, i].imshow(depth_to_colormap(depth_maps[i]))
        for r in range(rows):
            axes[r, i].axis("off")
            if i == 0:
                axes[r, i].set_ylabel(row_labels[r], fontsize=10)

    for i, label in enumerate(CAMERA_LABELS[:N]):
        axes[0, i].set_title(label, fontsize=9)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
