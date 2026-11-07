#!/usr/bin/env bash
# Push MambaByte 50M validation training to Kaggle TPU v5e-8.
# Usage: scripts/kaggle-push-mambabyte.sh
#
# Metadata per official docs:
#   https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
# TPU workaround per #1197:
#   https://github.com/Kaggle/kaggle-cli/issues/1197
#   - docker_image_pinning_type: "latest" prevents stale CPU image override
#   - enable_tpu: "true" + machine_shape: "TpuV5E8"
#   - enable_gpu: "false" (must be explicit)
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

# kernel-metadata.json — TPU v5e-8 configuration
# Per official docs + issue #1197 workaround:
# - Boolean fields are STRINGS ("true"/"false")
# - docker_image_pinning_type: "latest" prevents stale CPU image pin
# - No docker_image field (would override TPU VM)
cat > "$KERNEL_DIR/kernel-metadata.json" << METADATA
{
    "id": "$KERNEL_SLUG",
    "title": "MambaByte 50M Validation",
    "code_file": "mambabyte-50m-validation.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "true",
    "machine_shape": "TpuV5E8",
    "docker_image_pinning_type": "latest",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}
METADATA

echo "[kaggle-push] Pushing to Kaggle TPU v5e-8..."
python -m kaggle kernels push -p "$KERNEL_DIR"

echo ""
echo "[kaggle-push] Notebook submitted: $KERNEL_SLUG"
echo "[kaggle-push] Check status: kaggle kernels status $KERNEL_SLUG"
echo "[kaggle-push] Or visit: https://www.kaggle.com/code/$KERNEL_SLUG"
echo ""
echo "[kaggle-push] TPU v5e-8 session limit: 9 hours"
echo "[kaggle-push] Checkpoints saved every 2000 steps"
echo "[kaggle-push] NOTE: 1 active TPU session per user (batch mode)"
