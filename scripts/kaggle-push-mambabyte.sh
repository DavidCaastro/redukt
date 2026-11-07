#!/usr/bin/env bash
# Push MambaByte 50M validation training to Kaggle GPU T4x2.
# Usage: scripts/kaggle-push-mambabyte.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NB_DIR="$REPO_DIR/notebooks"
KERNEL_DIR="$NB_DIR/mambabyte-50m-validation"

# Find kaggle.json
KAGGLE_JSON="$USERPROFILE/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
fi
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "ERROR: Run scripts/kaggle-setup.sh first."
    exit 1
fi
USERNAME=$(grep -o '"username":"[^"]*"' "$KAGGLE_JSON" | cut -d'"' -f4)
KERNEL_SLUG="$USERNAME/mambabyte-50m-validation"

echo "[kaggle-push] Preparing MambaByte 50M validation notebook..."

# Copy notebook to kernel directory
mkdir -p "$KERNEL_DIR"
cp "$NB_DIR/mambabyte-50m-validate.py" "$KERNEL_DIR/mambabyte-50m-validation.py"

# kernel-metadata.json per official docs:
# https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
# - Boolean fields are STRINGS ("true"/"false")
# - machine_shape: "NvidiaTeslaT4" = GPU T4 x2 (30GB VRAM)
# - enable_gpu: "true" required alongside machine_shape
cat > "$KERNEL_DIR/kernel-metadata.json" << METADATA
{
    "id": "$KERNEL_SLUG",
    "title": "MambaByte 50M Validation",
    "code_file": "mambabyte-50m-validation.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "machine_shape": "NvidiaTeslaT4",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}
METADATA

echo "[kaggle-push] Pushing to Kaggle GPU T4x2 (30GB VRAM)..."
python -m kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "[kaggle-push] Notebook submitted: $KERNEL_SLUG"
echo "[kaggle-push] Check status: kaggle kernels status $KERNEL_SLUG"
echo "[kaggle-push] Or visit: https://www.kaggle.com/code/$KERNEL_SLUG"
echo ""
echo "[kaggle-push] GPU T4x2 session limit: 12 hours"
echo "[kaggle-push] Checkpoints saved every 2000 steps"
