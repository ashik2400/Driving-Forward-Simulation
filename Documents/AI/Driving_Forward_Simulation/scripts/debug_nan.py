"""
Trace exactly where NaN first appears in the forward pass.
Run AFTER find_clean_checkpoint.py with the last clean checkpoint.

Usage:
    python scripts/debug_nan.py --checkpoint checkpoints/epoch_XXX.pt
"""
import argparse, sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import DrivingForward
from src.data   import get_dataloader
from src.utils.camera_utils import warp_image

def check(name, tensor):
    if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
        print(f"  *** NaN/Inf first appears in: {name}  shape={tuple(tensor.shape)}")
        return True
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/rtx3050.yaml")
    args = p.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable AMP for this diagnostic — run in pure FP32
    cfg["training"]["amp"] = False
    print("AMP disabled for diagnostic (pure FP32)\n")

    model = DrivingForward(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = get_dataloader(cfg, split="train")
    batch  = next(iter(loader))

    images = batch["images"].to(device)
    K      = batch["K"].to(device)
    E_c2v  = batch["E_c2v"].to(device)
    B, N, _, H, W = images.shape

    print(f"Input images  finite: {torch.isfinite(images).all().item()}")
    print(f"Input K       finite: {torch.isfinite(K).all().item()}")
    print(f"Input E_c2v   finite: {torch.isfinite(E_c2v).all().item()}\n")

    found = False
    with torch.no_grad():
        out = model({"images": images, "K": K, "E_c2v": E_c2v})

        for i, d in enumerate(out["depths"]):
            if check(f"depth[{i}]", d): found = True

        for i, g in enumerate(out["gaussians"]):
            for k, v in g.items():
                if check(f"gaussians[{i}].{k}", v): found = True

        # Spatial warp
        for i in range(N):
            j = (i + 1) % N
            E_i2j = (E_c2v[:, j].float() @
                     torch.linalg.inv(E_c2v[:, i].float())).to(E_c2v.dtype)
            if check(f"E_i2j[{i}]", E_i2j): found = True

            w, m = warp_image(images[:, j], out["depths"][i],
                              K[:, i], K[:, j], E_i2j)
            if check(f"warped_spatial[{i}]", w): found = True

    if not found:
        print("No NaN found in FP32 forward pass.")
        print("=> NaN is AMP-specific. Cause: FP16 overflow in a specific layer.")
        print("   Fix: reduce lr or add per-layer FP32 casting.")
    else:
        print("\nFix the layer above and re-run this script to verify.")

if __name__ == "__main__":
    main()
