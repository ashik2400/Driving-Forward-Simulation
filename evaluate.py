"""
evaluate.py – Run evaluation on nuScenes val split.

Usage:
    python evaluate.py --config configs/rtx3050.yaml --checkpoint checkpoints/epoch_010.pt

Reports PSNR / SSIM / LPIPS for:
  • SF mode (single-frame)
  • Saves comparison images to outputs/eval/
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models          import DrivingForward
from src.data            import get_dataloader
from src.rendering       import GaussianRenderer
from src.utils.metrics   import Evaluator
from src.utils.visualization import save_novel_view_comparison, make_scene_grid
from src.utils.memory_utils  import clear_cache, vram_summary


def parse_args():
    p = argparse.ArgumentParser("DrivingForward Evaluation")
    p.add_argument("--config",     default="configs/rtx3050.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir",    default="outputs/eval")
    p.add_argument("--max_scenes", type=int, default=50)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  DrivingForward – Evaluation")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(device)}")
        print(f"  {vram_summary(device)}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model = DrivingForward(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    renderer  = GaussianRenderer(sh_degree=cfg["model"]["num_sh_degree"])
    evaluator = Evaluator()

    val_loader = get_dataloader(cfg, split="val")
    dc = cfg["dataset"]
    H, W = dc["image_height"], dc["image_width"]

    with torch.no_grad():
        for scene_idx, batch in enumerate(val_loader):
            if scene_idx >= args.max_scenes:
                break

            images = batch["images"].to(device)   # (1, N, 3, H, W)
            K      = batch["K"].to(device)
            E_c2v  = batch["E_c2v"].to(device)
            token  = batch["token"][0] if isinstance(batch["token"], (list, tuple)) \
                     else batch["token"]

            out    = model({"images": images, "K": K, "E_c2v": E_c2v})
            depths = out["depths"]
            gs     = out["gaussians"]
            N_cam  = images.shape[1]

            # Per-camera render (no merge) — keeps VRAM within 4 GB budget.
            rendered_list = []
            for i in range(N_cam):
                E_v2c = torch.linalg.inv(E_c2v[0, i].float())
                clear_cache()
                try:
                    ren = renderer(gs[i], K[0, i], E_v2c, H, W)
                except torch.cuda.OutOfMemoryError:
                    clear_cache()
                    ren = renderer._render_fallback(gs[i], K[0, i], E_v2c, H, W, item=0)
                rendered_list.append(ren)
                evaluator.update(ren.cpu(), images[0, i].cpu())

            # Save scene grid
            if scene_idx < 10:
                rendered_t = torch.stack(rendered_list)           # (N, 3, H, W)
                depth_t    = torch.stack([d[0] for d in depths])  # (N, 1, H, W)
                grid_path  = os.path.join(args.out_dir,
                                          f"scene_{scene_idx:04d}_{token[:8]}.png")
                make_scene_grid(images[0].cpu(), rendered_t.cpu(),
                                depth_t.cpu(), grid_path)
                print(f"  Saved grid: {grid_path}")

            clear_cache()

    metrics = evaluator.compute()
    print(f"\n{'='*40}")
    print("  Evaluation Results")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.4f}")
    print(f"{'='*40}\n")

    # Save metrics
    import json
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
