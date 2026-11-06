#!/usr/bin/env bash
# Push and run the GPU validation notebook on Kaggle.
# Usage: scripts/kaggle-push.sh [gpu_type]
#   gpu_type: "t4" (default) or "p100"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NB_DIR="$REPO_DIR/notebooks"
KERNEL_DIR="$NB_DIR/kaggle-gpu-validation"

GPU_TYPE="${1:-t4}"
case "$GPU_TYPE" in
    t4)   ACCEL="gpu" ;;
    p100) ACCEL="gpu" ;;
    *)    echo "Usage: $0 [t4|p100]"; exit 1 ;;
esac

# Find kaggle.json (handle Windows paths with spaces)
KAGGLE_JSON="$USERPROFILE/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
fi
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "ERROR: Run scripts/kaggle-setup.sh first."
    exit 1
fi
USERNAME=$(grep -o '"username":"[^"]*"' "$KAGGLE_JSON" | cut -d'"' -f4)
KERNEL_SLUG="$USERNAME/redukt-gpu-validation"

echo "[kaggle-push] Preparing notebook..."

# Create kernel metadata
mkdir -p "$KERNEL_DIR"
cat > "$KERNEL_DIR/kernel-metadata.json" << METADATA
{
    "id": "$KERNEL_SLUG",
    "title": "Redukt GPU Validation",
    "code_file": "gpu-validation.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "competition_sources": [],
    "dataset_sources": [],
    "kernel_sources": []
}
METADATA

# Copy the notebook script
if [ ! -f "$NB_DIR/gpu-validation.py" ]; then
    echo "ERROR: $NB_DIR/gpu-validation.py not found."
    exit 1
fi
cp "$NB_DIR/gpu-validation.py" "$KERNEL_DIR/gpu-validation.py"

echo "[kaggle-push] Pushing to Kaggle ($GPU_TYPE)..."
python -m kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "[kaggle-push] Notebook submitted: $KERNEL_SLUG"
echo "[kaggle-push] Check status: scripts/kaggle-status.sh"
echo "[kaggle-push] Or visit: https://www.kaggle.com/code/$KERNEL_SLUG"
