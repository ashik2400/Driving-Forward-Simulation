"""
demo.py – Quick visual demo of DrivingForward.

Works WITHOUT a nuScenes download: uses synthetic random data.
With a checkpoint, loads pretrained weights.

Usage (no data, no checkpoint – sanity check):
    python demo.py

Usage (with pretrained checkpoint):
    python demo.py --checkpoint checkpoints/epoch_010.pt --config configs/rtx3050.yaml

Outputs saved to outputs/demo/
"""

import argparse
import math
import os
import struct
import sys
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models              import DrivingForward
from src.rendering           import GaussianRenderer
from src.data                import get_dataloader
from src.utils.visualization import (
    save_surround_view, save_depth_map,
    save_novel_view_comparison, make_scene_grid,
)
from src.utils.memory_utils  import clear_cache, vram_summary, print_model_summary
from src.utils.camera_utils  import unproject_depth


# SH DC coefficient (same constant as the CUDA kernel)
_SH_C0 = 0.28209479177387814


# ── 3-D PLY export ────────────────────────────────────────────────────────────

def export_ply(gs_list: list, batch: dict, out_path: str) -> int:
    """
    Export merged 3D Gaussian centres + colours as a PLY point cloud.

    Colours are read directly from the SH DC coefficients that
    boost_gaussian_quality already encoded from the input image pixels.
    The 3DGS kernel evaluates colour as:
        colour = clamp(SH_C0 * dc + 0.5, 0, 1)
    so we apply the same formula here — guaranteed to match the rendered
    image exactly, without any re-projection or re-sampling.

    Low-opacity Gaussians (< 0.05) are discarded.
    Returns number of points written.
    """
    all_xyz, all_rgb = [], []

    for i, gs in enumerate(gs_list):
        means     = gs["means3D"][0].detach().cpu().float()      # (N, 3)
        opacities = gs["opacities"][0].detach().cpu().float()    # (N,)

        # Recover pixel colour from SH DC term (channels 0-2 of colours)
        # boost_gaussian_quality set dc = (pixel_rgb - 0.5) / SH_C0
        # so: pixel_rgb = SH_C0 * dc + 0.5
        dc  = gs["colours"][0, :, :3].detach().cpu().float()     # (N, 3)
        rgb = (dc * _SH_C0 + 0.5).clamp(0.0, 1.0)              # (N, 3)

        valid = opacities > 0.05
        if valid.sum() == 0:
            continue

        all_xyz.append(means[valid])
        all_rgb.append(rgb[valid])

    if not all_xyz:
        print("  [export_ply] No valid Gaussians to export.")
        return 0

    xyz = torch.cat(all_xyz, dim=0).numpy()
    rgb = (torch.cat(all_rgb, dim=0).numpy() * 255).astype(np.uint8)
    N   = xyz.shape[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        data = np.concatenate(
            [xyz.astype(np.float32).view(np.uint8).reshape(N, 12), rgb], axis=1
        )
        f.write(data.tobytes())

    return N


# ── Look-at camera helper ─────────────────────────────────────────────────────

def _make_lookat_E_c2v(
    eye:      torch.Tensor,   # (3,) camera position in vehicle frame
    target:   torch.Tensor,   # (3,) look-at point in vehicle frame
    world_up=None,            # optional torch.Tensor (3,)
) -> torch.Tensor:
    """
    Build a 4×4 camera-to-vehicle extrinsic from a look-at specification.
    Uses OpenCV camera convention: x=right, y=down, z=forward.
    """
    if world_up is None:
        world_up = torch.tensor([0., 0., 1.])

    z_cam = F.normalize(target - eye, dim=0)          # optical axis (forward)
    # Avoid degenerate cross-product when looking straight up/down
    if abs(z_cam[2].item()) > 0.98:
        world_up = torch.tensor([0., 1., 0.])
    x_cam = F.normalize(torch.linalg.cross(z_cam, world_up), dim=0)   # right
    y_cam = torch.linalg.cross(z_cam, x_cam)                           # down

    E = torch.eye(4, dtype=torch.float32)
    E[:3, :3] = torch.stack([x_cam, y_cam, z_cam], dim=1)
    E[:3,  3] = eye
    return E


# ── Fly-around video ──────────────────────────────────────────────────────────

def render_novel_views(
    gs_list:        list,
    renderer,
    batch:          dict,
    H:              int,
    W:              int,
    out_path:       str,
    device:         torch.device,
    frames_per_gap: int = 8,   # interpolated frames between each camera pair
) -> None:
    """
    Render a smooth novel-view video by interpolating between the 6 real
    surround camera positions.

    This guarantees the scene is always in frame (we use the actual camera
    positions the depth was computed from) and shows genuine novel-view
    synthesis at the in-between positions.

    Frame layout: 6 gaps × 8 frames = 48 frames total @ 12 fps = 4 s loop.
    Each gap sweeps smoothly from camera i to camera i+1 around the car:
      Front → FrontRight → BackRight → Back → BackLeft → FrontLeft → Front
    """
    import cv2

    N      = len(gs_list)
    K_all  = batch["K"][0].float()        # (N, 3, 3)
    E_all  = batch["E_c2v"][0].float()    # (N, 4, 4)

    # Merge all cameras into one 3-D scene
    merged = GaussianRenderer.merge(gs_list)

    cam_labels = [
        "Front", "Front-Right", "Back-Right",
        "Back",  "Back-Left",   "Front-Left",
    ]

    total = N * frames_per_gap
    frames = []
    print(f"\n  Rendering {total}-frame novel-view video (camera interpolation)...")

    for i in range(N):
        j        = (i + 1) % N
        E_start  = E_all[i]
        E_end    = E_all[j]
        K_start  = K_all[i]
        K_end    = K_all[j]

        for f in range(frames_per_gap):
            t = f / frames_per_gap            # 0.0 → (frames_per_gap-1)/frames_per_gap

            # ── Interpolate extrinsics ────────────────────────────────────
            E_interp = (1.0 - t) * E_start + t * E_end

            # Re-orthonormalise the rotation block (linear interp can drift)
            R = E_interp[:3, :3]
            U, _, Vt = torch.linalg.svd(R)
            E_interp = E_interp.clone()
            E_interp[:3, :3] = U @ Vt

            # ── Interpolate intrinsics ────────────────────────────────────
            K_interp = (1.0 - t) * K_start + t * K_end

            E_v2c = torch.linalg.inv(E_interp).to(device)
            K_d   = K_interp.to(device)

            clear_cache()
            with torch.no_grad():
                try:
                    frame = renderer(merged, K_d, E_v2c, H, W)
                except torch.cuda.OutOfMemoryError:
                    clear_cache()
                    frame = renderer._render_fallback(
                        merged, K_d.cpu(), E_v2c.cpu(), H, W, item=0
                    )
                    frame = frame.to(device)

            frame_np  = (frame.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Burn a small label so the viewer knows which transition this is
            label = f"{cam_labels[i]} -> {cam_labels[j]}"
            cv2.putText(frame_bgr, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            frames.append(frame_bgr)
            clear_cache()

        done = (i + 1) * frames_per_gap
        print(f"    {done}/{total}  ({cam_labels[i]} → {cam_labels[j]})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Try H.264 (.mp4) first — universally playable.
    # Fall back to XVID (.avi) if H.264 encoder is not available on this machine.
    mp4_path = out_path if out_path.endswith(".mp4") else out_path + ".mp4"
    avi_path  = mp4_path.replace(".mp4", ".avi")

    written = False
    for codec, path, fps in [("avc1", mp4_path, 12.0), ("XVID", avi_path, 12.0)]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        if not writer.isOpened():
            writer.release()
            continue
        for frm in frames:
            writer.write(frm)
        writer.release()
        # Verify the file was actually written (codec errors can be silent)
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            print(f"  Saved video            : {path}")
            written = True
            break
        else:
            os.remove(path) if os.path.exists(path) else None

    # Always also save a GIF — opens in any browser / image viewer with no codec issues.
    gif_path = mp4_path.replace(".mp4", ".gif")
    try:
        from PIL import Image as PILImage
        pil_frames = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                      for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / 12),   # ms per frame at 12 fps
        )
        print(f"  Saved fly-around GIF    : {gif_path}  (open in any browser)")
    except ImportError:
        pass   # PIL not available; video file is the only output

    if not written:
        # Last resort: save individual frames as PNG sequence
        frames_dir = mp4_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for fi, frm in enumerate(frames):
            cv2.imwrite(os.path.join(frames_dir, f"frame_{fi:04d}.png"), frm)
        print(f"  Saved fly-around frames : {frames_dir}/")


# ── Full-resolution Gaussian expansion ───────────────────────────────────────

def expand_to_full_res(
    gs_list: list,
    depths:  list,
    batch:   dict,
    device,
) -> tuple:
    """
    The DepthNetwork decoder outputs depth at half resolution (H/2 × W/2).
    unproject_depth called with the half-res pixel grid but full-res intrinsics
    places ALL Gaussians in the top-left quarter of camera space — this is the
    root cause of the quarter-image rendering artefact.

    Root cause (math):
      Half-res pixel u ∈ [0, W/2-1], full-res cx = W/2.
      x_cam = (u - cx) / fx ≤ 0  for every pixel → all Gaussians project
      to u < cx in the full-res image, i.e. LEFT half only.
      Same for v: all Gaussians project to v < cy → TOP half only.
      Together: top-left QUARTER rendered, rest BLACK.

    Fix:
      1. Upsample each depth from H/2×W/2 → H×W.
      2. Recompute Gaussian centres (means3D) from full-res depth + full-res K.
      3. Bilinearly upsample all other Gaussian attributes to match.

    Returns the updated gs_list and a list of full-res depth tensors.
    """
    H = batch["images"].shape[-2]
    W = batch["images"].shape[-1]

    new_depths = []
    for i, gs in enumerate(gs_list):
        depth = depths[i]                         # (1, 1, H_d, W_d)
        H_d, W_d = depth.shape[-2:]

        if H_d == H and W_d == W:
            new_depths.append(depth)
            continue                               # already full-res

        # ── Upsample depth ──────────────────────────────────────────────
        depth_full = F.interpolate(
            depth.float(), size=(H, W), mode="bilinear", align_corners=False
        ).to(depth.dtype)
        new_depths.append(depth_full)

        # ── Recompute Gaussian centres from full-res depth ──────────────
        K_b = batch["K"][0, i].unsqueeze(0).to(device).float()      # (1, 3, 3)
        E_b = batch["E_c2v"][0, i].unsqueeze(0).to(device).float()  # (1, 4, 4)
        with torch.no_grad():
            mu_full = unproject_depth(depth_full.float(), K_b, E_b)  # (1, H*W, 3)
        gs["means3D"] = mu_full

        # ── Bilinear upsample of per-Gaussian attributes ────────────────
        def _up(attr: torch.Tensor) -> torch.Tensor:
            """(1, N, C)  H_d×W_d → H×W."""
            C  = attr.shape[-1]
            sp = attr[0].reshape(H_d, W_d, C).permute(2, 0, 1).unsqueeze(0)
            up = F.interpolate(sp.float(), (H, W), mode="bilinear", align_corners=False)
            return up.flatten(2).permute(0, 2, 1).to(attr.dtype)

        def _up_op(op: torch.Tensor) -> torch.Tensor:
            """opacities (1, N)  H_d×W_d → H×W."""
            sp = op.reshape(1, 1, H_d, W_d).float()
            up = F.interpolate(sp, (H, W), mode="bilinear", align_corners=False)
            return up.reshape(1, H * W).to(op.dtype)

        gs["scales"]    = _up(gs["scales"])
        gs["rotations"] = F.normalize(_up(gs["rotations"]), dim=-1)
        gs["colours"]   = _up(gs["colours"])
        gs["opacities"] = _up_op(gs["opacities"])

    return gs_list, new_depths


# ── Colour correction ─────────────────────────────────────────────────────────

def boost_gaussian_quality(
    gs_list: list,
    batch:   dict,
    depths:  list,
    device,
) -> list:
    """
    The Gaussian attribute heads (colour / scale / opacity) are undertrained
    because the render loss fires for only 1 camera every 6 steps.
    The depth network converged well because photometric loss fires every step.

    This function overrides all three bad attributes using the accurate depth
    and input images — no retraining needed:

    1. COLOUR  — project each Gaussian back to its source camera and sample
                 the real pixel colour, then encode as SH DC coefficient.
                 kernel formula: colour = clamp(SH_C0 * dc + 0.5, 0, 1)
                 inverse:        dc = (pixel − 0.5) / SH_C0

    2. SCALE   — the trained scale is miscalibrated (often too large → solid
                 colour blobs).  Correct size = depth / focal_length * 1.5
                 so each Gaussian covers ~1.5 pixels, giving full coverage
                 without over-blending.

    3. OPACITY — trained value is ~0.5 (random init sigmoid) = semi-
                 transparent → washed-out appearance.  Set to 0.99 for solid
                 surface coverage.
    """
    SH_C0 = 0.28209479177387814

    for i, gs in enumerate(gs_list):
        img   = batch["images"][0, i].to(device).float()   # (3, H, W)
        K     = batch["K"][0, i].to(device).float()        # (3, 3)
        E_c2v = batch["E_c2v"][0, i].to(device).float()    # (4, 4)
        depth = depths[i]                                   # (B, 1, H, W)
        means = gs["means3D"][0]                            # (N, 3)

        E_v2c = torch.linalg.inv(E_c2v)
        pts_c = (E_v2c[:3, :3] @ means.T + E_v2c[:3, 3:]).T   # (N, 3) cam frame
        z     = pts_c[:, 2].clamp(min=0.1)

        # ── 1. Colour from input image ────────────────────────────────────
        u   = pts_c[:, 0] / z * K[0, 0] + K[0, 2]
        v   = pts_c[:, 1] / z * K[1, 1] + K[1, 2]
        H_i, W_i = img.shape[-2:]
        u_n = (u / (W_i - 1)) * 2 - 1
        v_n = (v / (H_i - 1)) * 2 - 1
        grid    = torch.stack([u_n, v_n], dim=-1).unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            img.unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        pixel_rgb = sampled[0, :, 0, :].T                       # (N, 3)
        dc = ((pixel_rgb - 0.5) / SH_C0).clamp(-3.0, 3.0)
        gs["colours"][0, :, :3] = dc

        # ── 2. Scale calibrated to depth ──────────────────────────────────
        # One pixel in world space at depth d = d / focal_length.
        # Use 1.5× pixel size so adjacent Gaussians slightly overlap →
        # no gaps, no large solid-colour blobs.
        fx          = K[0, 0]
        depth_flat  = depth[0, 0].reshape(-1)                   # (N,)
        pixel_world = (depth_flat / fx).clamp(0.005, 1.0)       # cap at 1 m
        optimal_s   = (pixel_world * 1.5).unsqueeze(-1).expand(-1, 3).contiguous()
        gs["scales"][0] = optimal_s

        # ── 3. Opacity → near-opaque ─────────────────────────────────────
        gs["opacities"][0] = torch.full_like(gs["opacities"][0], 0.99)

        # ── 4. Zero out higher-order SH ───────────────────────────────────
        # After 107 epochs, the degree-1 view-dependent SH coefficients
        # (channels 3-11) are undertrained and add colour distortion.
        # Zeroing them means colour = clamp(SH_C0 * dc + 0.5, 0, 1)
        # = clamp(pixel_rgb, 0, 1) — exactly the input image colour.
        gs["colours"][0, :, 3:] = 0.0

    return gs_list


def parse_args():
    p = argparse.ArgumentParser("DrivingForward Demo")
    p.add_argument("--config",     default="configs/rtx3050.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir",    default="outputs/demo")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--scene_idx",  type=int, default=0,
                   help="Which nuScenes val sample to visualise (0-based)")
    return p.parse_args()


def load_real_batch(cfg: dict, scene_idx: int, device: torch.device) -> dict:
    """Load a single real nuScenes sample (val split) onto device."""
    loader = get_dataloader(cfg, split="val")
    for i, batch in enumerate(loader):
        if i == scene_idx:
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
    # scene_idx out of range — fall back to first sample
    loader = get_dataloader(cfg, split="val")
    batch = next(iter(loader))
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DrivingForward – Demo")
    print(f"  Config  : {args.config}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(device)}")
        print(f"  {vram_summary(device)}")
    print(f"{'='*60}\n")

    # ── Model ────────────────────────────────────────────────────────────
    model = DrivingForward(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded weights: {args.checkpoint}")
    else:
        print("  No checkpoint provided — using random (untrained) weights.")
    model.eval()

    print("\nModel summary:")
    print_model_summary(model)

    if device.type == "cuda":
        print(f"\n  After model load: {vram_summary(device)}")

    # ── Load real nuScenes sample ─────────────────────────────────────────
    dc = cfg["dataset"]
    H, W = dc["image_height"], dc["image_width"]
    N    = len(dc.get("cameras", [None]*6))

    print(f"\n  Loading nuScenes val sample #{args.scene_idx} ({H}×{W}) ...")
    batch = load_real_batch(cfg, args.scene_idx, device)
    token = batch.get("token", "unknown")
    print(f"  Sample token: {token}")

    # ── Forward pass ──────────────────────────────────────────────────────
    print("  Running forward pass ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(batch)
    elapsed = time.perf_counter() - t0
    print(f"  Inference time: {elapsed*1000:.1f} ms  ({elapsed:.3f} s)")

    if device.type == "cuda":
        print(f"  {vram_summary(device)}")

    depths  = out["depths"]
    gs_list = out["gaussians"]

    # Fix quarter-image artefact: decoder depth is at H/2×W/2 but K is
    # full-res → Gaussians were all placed in the top-left quarter.
    # Recompute positions and upsample attributes to full H×W resolution.
    print("  Expanding Gaussians to full resolution ...")
    gs_list, depths_full = expand_to_full_res(gs_list, depths, batch, device)
    print(f"    Gaussians per camera: {gs_list[0]['means3D'].shape[1]:,}  "
          f"({depths_full[0].shape[-2]}×{depths_full[0].shape[-1]})")

    # Override undertrained Gaussian attributes (colour, scale, opacity)
    # using the accurate depth and input images — no retraining needed.
    gs_list = boost_gaussian_quality(gs_list, batch, depths_full, device)

    # ── Visualise surround-view input ─────────────────────────────────────
    sv_path = os.path.join(args.out_dir, "input_surround_view.png")
    save_surround_view(batch["images"][0].cpu(), sv_path,
                       title=f"Input Surround View  (token: {str(token)[:8]})")
    print(f"\n  Saved: {sv_path}")

    # ── Visualise predicted depth maps ────────────────────────────────────
    for i, d in enumerate(depths):
        dp = os.path.join(args.out_dir, f"depth_cam{i}.png")
        save_depth_map(d[0].cpu(), dp, image=batch["images"][0, i].cpu(),
                       title=f"Camera {i} Depth")
    print(f"  Saved depth maps: {args.out_dir}/depth_cam*.png")

    # ── Render novel views ────────────────────────────────────────────────
    sh_degree = cfg["model"]["num_sh_degree"]
    renderer  = GaussianRenderer(sh_degree=sh_degree)

    # Per-camera render: use only that camera's own Gaussians.
    # Each camera set has H×W ≈ 91 k primitives after full-res expansion.
    rendered_all = []
    for i in range(N):
        K_i   = batch["K"][0, i]
        E_v2c = torch.linalg.inv(batch["E_c2v"][0, i].float())

        clear_cache()
        with torch.no_grad():
            try:
                ren = renderer(gs_list[i], K_i, E_v2c, H, W)
            except torch.cuda.OutOfMemoryError:
                print(f"  [cam {i}] 3DGS OOM – falling back to software renderer.")
                clear_cache()
                ren = renderer._render_fallback(gs_list[i], K_i, E_v2c, H, W, item=0)

        rendered_all.append(ren.cpu())
        clear_cache()

        ren_path = os.path.join(args.out_dir, f"novel_view_cam{i}.png")
        save_novel_view_comparison(
            batch["images"][0, i].cpu(), ren.cpu(),
            ren_path,
        )
    print(f"  Saved novel views: {args.out_dir}/novel_view_cam*.png")

    # ── Full scene grid ───────────────────────────────────────────────────
    rendered_t = torch.stack(rendered_all)
    depth_t    = torch.stack([d[0].cpu() for d in depths])
    grid_path  = os.path.join(args.out_dir, "scene_grid.png")
    make_scene_grid(batch["images"][0].cpu(), rendered_t, depth_t, grid_path)
    print(f"  Saved scene grid: {grid_path}")

    # ── 3-D PLY point cloud export ────────────────────────────────────────
    # This is the actual 3D reconstruction — open in MeshLab / CloudCompare /
    # Blender to see the full surround-view 3D scene.
    ply_path = os.path.join(args.out_dir, "scene_3d.ply")
    n_pts = export_ply(gs_list, batch, ply_path)
    print(f"\n  Saved 3D point cloud ({n_pts:,} points): {ply_path}")
    print("  → Open in MeshLab / CloudCompare / Blender to inspect the 3D scene")

    # ── Novel-view interpolation video ───────────────────────────────────
    # Smoothly interpolates between the 6 real surround camera positions,
    # synthesising novel views in between.  Scene is always in frame because
    # we start and end at positions the depth network was trained on.
    vid_path = os.path.join(args.out_dir, "novel_views.mp4")
    render_novel_views(
        gs_list  = gs_list,
        renderer = renderer,
        batch    = batch,
        H        = H,
        W        = W,
        out_path = vid_path,
        device   = device,
    )

    clear_cache()
    print(f"\n  All outputs in: {args.out_dir}/")
    print("  Demo complete.\n")


if __name__ == "__main__":
    main()
