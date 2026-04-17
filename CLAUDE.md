# DrivingForward Simulation – Project Context

## Paper
"DrivingForward: Feed-forward 3D Gaussian Splatting for Driving Scene Reconstruction
from Flexible Surround-view Input" — AAAI 2025
Official repo: https://github.com/fangzhou2000/DrivingForward

## Hardware
- GPU : NVIDIA RTX 3050, 4 GB VRAM
- RAM : 16 GB
- CPU : AMD Ryzen 7 6000 series
- OS  : Windows 11 (commands executed via bash shell)

## Critical Hardware Notes
- Paper was trained on 8× A100 80 GB GPUs; we are limited to 4 GB VRAM.
- Always prefer `configs/rtx3050.yaml` over `configs/base.yaml`.
- Use SF (single-frame) mode; resolution 114×228.
- Enable AMP (FP16) and gradient checkpointing throughout.
- For full training, use gradient accumulation (accum_steps=8).
- Inference-only runs should set `torch.backends.cudnn.benchmark = True`.

## Architecture Summary
1. **Pose Network P**   – predicts relative camera motion T^{t'→t} from multi-frame pairs.
2. **Depth Network D**  – U-Net encoder-decoder; outputs depth map D and image features F.
3. **Gaussian Network G** – CNN; predicts {s, r, c, o} per pixel; position μ from unprojection.
4. **Scale-aware Localisation** – self-supervised photometric loss (SSIM + L1) across spatial/temporal contexts.
5. **Renderer** – diff-gaussian-rasterization (3DGS); renders novel views.

## Loss Functions
- L_repro : photometric warp loss (SSIM + L1)
- L_loc   : L_repro + λ_sp·L_sp + λ_sp-tm·L_sp-tm + λ_smooth·L_smooth
- L_render: L2 + LPIPS between rendered and target views
- Total   : L_loc + L_render

## Dataset
- nuScenes: 6 surround cameras (FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT)
- Use nuscenes-mini for quick experiments (10 scenes)
- nuscenes-devkit required; set NUSCENES_ROOT in .env

## Key Files
- train.py          – training entry point
- evaluate.py       – evaluation on nuScenes val split
- demo.py           – quick inference demo (single scene)
- configs/rtx3050.yaml – hardware-optimised config (use this)
- src/models/driving_forward.py – top-level model
