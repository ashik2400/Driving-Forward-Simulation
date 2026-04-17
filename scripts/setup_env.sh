#!/usr/bin/env bash
# =============================================================================
# setup_env.sh – One-shot environment setup for DrivingForward
# Tested on Windows (Git Bash / Conda) with RTX 3050 + CUDA 11.8
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================"
echo "  DrivingForward – Environment Setup"
echo "  Working dir: $REPO_ROOT"
echo "======================================================"

# 1. Create conda environment
echo ""
echo "[1/5] Creating conda environment..."
conda env create -f environment.yml || conda env update -f environment.yml
conda activate driving_forward

# 2. Build diff-gaussian-rasterization (requires CUDA & cmake)
echo ""
echo "[2/5] Building diff-gaussian-rasterization..."
GAUSS_DIR="$REPO_ROOT/third_party/gaussian-splatting"

if [ ! -d "$GAUSS_DIR" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git \
    "$GAUSS_DIR"
fi

cd "$GAUSS_DIR/submodules/diff-gaussian-rasterization"
pip install -e . --no-build-isolation

cd "$GAUSS_DIR/submodules/simple-knn"
pip install -e . --no-build-isolation

cd "$REPO_ROOT"

# 3. Install remaining pip deps
echo ""
echo "[3/5] Installing remaining dependencies..."
pip install -r requirements.txt

# 4. Verify CUDA availability
echo ""
echo "[4/5] Verifying CUDA..."
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name        : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM            : {mem:.1f} GB')
"

# 5. Quick model sanity check
echo ""
echo "[5/5] Sanity check – forward pass with synthetic data..."
python demo.py --device cpu
echo ""
echo "======================================================"
echo "  Setup complete! Run:"
echo "    conda activate driving_forward"
echo "    python demo.py                         # quick test"
echo "    python train.py --config configs/rtx3050.yaml"
echo "======================================================"
