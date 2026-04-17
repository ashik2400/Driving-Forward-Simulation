"""
Memory optimisation utilities – tuned for RTX 3050 (4 GB VRAM).

Key strategies employed:
  1. AMP (BF16) via torch.amp — BF16 has the same exponent range as FP32
     so it never overflows, unlike FP16 which overflows at 65504.
     RTX 3050 (Ampere GA107) supports BF16 natively.
     GradScaler is disabled: BF16 does not need loss scaling.
  2. Gradient checkpointing  (already wired into DepthNetwork)
  3. Per-camera sequential processing  (reduces peak VRAM)
  4. Explicit cache clearing after each batch
  5. Model weight sharding helper for eval
"""

from __future__ import annotations

import gc
import contextlib
from typing import Generator

import torch
import torch.nn as nn


# ── VRAM reporting ────────────────────────────────────────────────────────────

def vram_summary(device: torch.device = None) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return "CUDA not available"
    alloc  = torch.cuda.memory_allocated(device) / 1e9
    reserv = torch.cuda.memory_reserved(device)  / 1e9
    total  = torch.cuda.get_device_properties(device).total_memory / 1e9
    return (f"VRAM  allocated={alloc:.2f} GB  "
            f"reserved={reserv:.2f} GB  "
            f"total={total:.2f} GB")


def clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── AMP context ───────────────────────────────────────────────────────────────

@contextlib.contextmanager
def maybe_autocast(enabled: bool = True) -> Generator:
    """Wrap forward pass in BF16 autocast when enabled.

    BF16 vs FP16:
      FP16 — 5 exponent bits, overflows at 65504 → NaN in deep nets after warmup
      BF16 — 8 exponent bits (same as FP32), never overflows, same memory as FP16
    """
    if enabled and torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


# ── Gradient scaler factory ───────────────────────────────────────────────────

def build_scaler(enabled: bool) -> torch.amp.GradScaler:
    # BF16 has the same dynamic range as FP32 — loss scaling is not needed.
    # GradScaler is kept in the call sites but always disabled here.
    return torch.amp.GradScaler("cuda", enabled=False)


# ── FP16 model conversion ─────────────────────────────────────────────────────

def to_half(model: nn.Module) -> nn.Module:
    """Convert all floating-point params to FP16 (inference only)."""
    return model.half()


# ── Per-camera sequential inference ──────────────────────────────────────────

def process_cameras_sequential(model, images, K, E_c2v, device):
    """
    Process each camera one at a time to reduce peak VRAM.

    Args:
        model   : DrivingForward instance
        images  : (B, N, 3, H, W)
        K       : (B, N, 3, 3)
        E_c2v   : (B, N, 4, 4)

    Returns list of (depth, params) per camera.
    """
    N = images.shape[1]
    results = []
    for i in range(N):
        img_i = images[:, i].to(device)
        with torch.no_grad():
            depth_i, params_i, _ = model.encode_image(img_i)
        results.append((depth_i.cpu(), {k: v.cpu() for k, v in params_i.items()}))
        clear_cache()
    return results


# ── Model size report ─────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def print_model_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size  = model_size_mb(model)
    print(f"  Parameters : {total:,}  (trainable: {train:,})")
    print(f"  Model size : {size:.1f} MB")
