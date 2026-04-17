"""
Scan all checkpoints and report which ones have NaN/Inf weights.
Run: python scripts/find_clean_checkpoint.py
"""
import torch
from pathlib import Path

ckpt_dir = Path("checkpoints")
checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))

print(f"Scanning {len(checkpoints)} checkpoints...\n")

last_clean = None
for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    nan_keys, inf_keys = [], []
    for k, v in state.items():
        if v.is_floating_point():
            if torch.isnan(v).any():
                nan_keys.append(k)
            elif torch.isinf(v).any():
                inf_keys.append(k)

    if nan_keys or inf_keys:
        status = "CORRUPT"
        detail = ""
        if nan_keys:
            detail += f"  NaN in: {nan_keys[0]}"
            if len(nan_keys) > 1:
                detail += f" (+{len(nan_keys)-1} more)"
        if inf_keys:
            detail += f"  Inf in: {inf_keys[0]}"
    else:
        status = "CLEAN"
        last_clean = ckpt_path
        detail = ""

    print(f"  {ckpt_path.name}  [{status}]{detail}")

print()
if last_clean:
    print(f"Last clean checkpoint: {last_clean}")
    print(f"\nResume command:")
    print(f"  python train.py --config configs/rtx3050.yaml --resume {last_clean}")
else:
    print("No clean checkpoints found — must train from scratch.")
    print("  python train.py --config configs/rtx3050.yaml")
