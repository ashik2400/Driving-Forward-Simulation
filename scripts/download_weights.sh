#!/usr/bin/env bash
# =============================================================================
# download_weights.sh – Download pretrained DrivingForward checkpoint
#
# Official weights from: https://github.com/fangzhou2000/DrivingForward
# Check the README there for the latest download link.
# =============================================================================

set -euo pipefail

CKPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints"
mkdir -p "$CKPT_DIR"

echo "Checking for pretrained weights..."
echo "Please visit: https://github.com/fangzhou2000/DrivingForward"
echo "and download the checkpoint to: $CKPT_DIR/"
echo ""
echo "Once downloaded, run:"
echo "  python evaluate.py --checkpoint checkpoints/<name>.pt"
echo "  python demo.py     --checkpoint checkpoints/<name>.pt"
