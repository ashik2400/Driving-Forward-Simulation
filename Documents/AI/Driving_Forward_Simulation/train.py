"""
train.py – DrivingForward training entry point.

Usage (RTX 3050):
    python train.py --config configs/rtx3050.yaml

Usage (full paper settings, multi-GPU):
    torchrun --nproc_per_node=8 train.py --config configs/base.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import yaml

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.models    import DrivingForward
from src.data      import get_dataloader
from src.losses    import TotalLoss
from src.rendering import GaussianRenderer
from src.utils.memory_utils  import (
    build_scaler, clear_cache, maybe_autocast, print_model_summary, vram_summary
)
from src.utils.camera_utils  import warp_image


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("DrivingForward Training")
    p.add_argument("--config",    default="configs/rtx3050.yaml")
    p.add_argument("--resume",    default=None, help="checkpoint path to resume")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


# ── Config loader ─────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Optimiser & scheduler ─────────────────────────────────────────────────────

def build_optim(model: nn.Module, cfg: dict):
    tc = cfg["training"]
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if "encoder" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": float(tc["lr_backbone"])},
        {"params": other_params,    "lr": float(tc["lr"])},
    ], weight_decay=float(tc["weight_decay"]))
    return opt


def build_scheduler(opt, cfg: dict, total_steps: int, last_epoch: int = -1):
    from torch.optim.lr_scheduler import OneCycleLR
    tc = cfg["training"]
    return OneCycleLR(
        opt,
        max_lr          = [tc["lr_backbone"], tc["lr"]],
        total_steps     = total_steps,
        pct_start       = tc["warmup_steps"] / total_steps,
        anneal_strategy = "cos",
        last_epoch      = last_epoch,
    )


# ── One training step ─────────────────────────────────────────────────────────

def train_step(model, batch, criterion, opt, scaler, scheduler,
               cfg, device, step, renderer):
    tc = cfg["training"]

    images    = batch["images"].to(device)
    K         = batch["K"].to(device)
    E_c2v     = batch["E_c2v"].to(device)

    B, N, _, H, W = images.shape

    amp_enabled = tc["amp"] and device.type == "cuda"

    # Precompute relative extrinsics in float32 OUTSIDE autocast.
    # torch.inverse inside FP16 autocast is numerically unstable and is the
    # primary source of NaN loss after prolonged training.
    E_i2j_list = []
    for i in range(N):
        j = (i + 1) % N
        E_i2j = (E_c2v[:, j].float() @
                 torch.linalg.inv(E_c2v[:, i].float())).to(E_c2v.dtype)
        E_i2j_list.append(E_i2j)

    with maybe_autocast(amp_enabled):
        out = model(batch | {"images": images, "K": K, "E_c2v": E_c2v})
        depths      = out["depths"]
        gs_list     = out["gaussians"]
        all_params  = out["params"]

        # ── Build photometric pairs for L_loc ─────────────────────────────
        warped_temporal, warped_spatial = [], []

        imgs_prev = batch.get("images_prev")
        imgs_next = batch.get("images_next")
        poses     = out.get("poses", [])

        for i in range(N):
            # Temporal pairs (MF mode)
            for j, (imgs_ctx, pose_key) in enumerate([(imgs_prev, 0), (imgs_next, 2)]):
                if imgs_ctx is not None and poses:
                    T_ctx2tgt = poses[i][pose_key]
                    if T_ctx2tgt is not None:
                        src = imgs_ctx[:, i].to(device)
                        warped, mask = warp_image(
                            src, depths[i], K[:, i], K[:, i], T_ctx2tgt
                        )
                        warped_temporal.append((warped, images[:, i], mask))

            # Spatial pairs (adjacent camera same timestep)
            w, m = warp_image(images[:, (i + 1) % N], depths[i],
                              K[:, i], K[:, (i + 1) % N], E_i2j_list[i])
            warped_spatial.append((w, images[:, i], m))

        # ── L_render: render one camera per step (stochastic over cameras) ──
        # Rendering all 6 cameras per step OOMs on 4 GB; one camera per step
        # gives full coverage every 6 steps while keeping peak VRAM safe.
        rendered_img, target_img = None, None
        render_cam = step % N
        try:
            E_v2c_r = torch.linalg.inv(E_c2v[0, render_cam].float())
            ren = renderer(gs_list[render_cam], K[0, render_cam], E_v2c_r, H, W)
            rendered_img = ren.unsqueeze(0)          # (1, 3, H, W)
            target_img   = images[0, render_cam].unsqueeze(0).float()  # (1, 3, H, W)
        except torch.cuda.OutOfMemoryError:
            clear_cache()   # skip render loss this step if VRAM is tight

        total_loss, breakdown = criterion(
            warped_temporal     = warped_temporal,
            warped_spatial      = warped_spatial,
            warped_spatial_temp = None,
            depths              = depths,
            images              = images,
            rendered            = rendered_img,
            target              = target_img,
            all_params          = all_params,
        )

    # ── NaN guard: skip this batch rather than corrupt weights ────────────────
    if not torch.isfinite(total_loss):
        opt.zero_grad(set_to_none=True)
        print(f"  [step {step}] WARNING: non-finite loss ({total_loss.item():.4f}) — skipping batch.")
        return {k: 0.0 for k in breakdown}

    total_loss = total_loss / tc["accum_steps"]
    scaler.scale(total_loss).backward()

    if (step + 1) % tc["accum_steps"] == 0:
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        # Only step scheduler when the optimizer actually stepped.
        # scaler.step() is a no-op (skips opt.step) when gradients are inf/nan,
        # which happens on the first step and after any scaler reduction.
        if scaler.get_scale() >= scale_before:
            scheduler.step()

    return {k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in breakdown.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    cfg     = load_cfg(args.config)
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DrivingForward – Training")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(device)}")
        print(f"  {vram_summary(device)}")
    print(f"{'='*60}\n")

    model = DrivingForward(cfg).to(device)
    print("Model summary:")
    print_model_summary(model)

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader = get_dataloader(cfg, split="train")
    val_loader   = get_dataloader(cfg, split="val")

    # ── Optimisation ───────────────────────────────────────────────────────
    tc          = cfg["training"]
    total_steps = len(train_loader) * tc["epochs"]
    opt         = build_optim(model, cfg)

    # ── Resume (must happen before scheduler is built so we can correctly
    #    position the LR schedule using last_epoch) ─────────────────────────
    start_epoch = 0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        if "opt" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["opt"])
        start_epoch = resume_ckpt.get("epoch", 0)
        print(f"  Resumed from: {args.resume}  (starting at epoch {start_epoch + 1})")

    # Number of optimizer (scheduler) steps already completed.
    # scheduler.step() fires once per accum_steps data steps.
    sched_steps_done = (start_epoch * len(train_loader)) // tc["accum_steps"]
    scheduler = build_scheduler(opt, cfg, total_steps,
                                 last_epoch=sched_steps_done - 1)
    if resume_ckpt is not None and "sched" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["sched"])

    scaler    = build_scaler(tc["amp"])
    criterion = TotalLoss(cfg["loss"]).to(device)
    renderer  = GaussianRenderer(sh_degree=cfg["model"]["num_sh_degree"])

    # ── Logging ────────────────────────────────────────────────────────────
    lc       = cfg["logging"]
    log_dir  = lc["log_dir"]
    ckpt_dir = lc["checkpoint_dir"]
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer   = SummaryWriter(log_dir=log_dir)

    # ── Training loop ──────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            losses = train_step(
                model, batch, criterion, opt, scaler,
                scheduler, cfg, device, global_step, renderer
            )
            global_step += 1

            if global_step % lc["log_every"] == 0:
                lr = scheduler.get_last_lr()[-1]
                msg = (f"Ep {epoch+1}/{tc['epochs']}  "
                       f"Step {global_step}  "
                       f"Loss {losses.get('total', 0):.4f}  "
                       f"LR {lr:.2e}  "
                       f"Time {time.time()-t0:.1f}s")
                print(msg)
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("train/lr", lr, global_step)

            clear_cache()

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % lc["save_every"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "sched": scheduler.state_dict(),
                "cfg":   cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
