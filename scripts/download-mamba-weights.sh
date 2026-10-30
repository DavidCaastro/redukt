#!/usr/bin/env bash
set -euo pipefail

# Download Mamba-130m-hf weights for Redukt (R-035).
# Source: state-spaces/mamba-130m-hf (Apache-2.0 license)
# Requires: Python 3, pip install huggingface_hub

WEIGHTS_DIR="weights/mamba-130m"
MODEL_FILE="$WEIGHTS_DIR/model.safetensors"
TOKENIZER_FILE="$WEIGHTS_DIR/tokenizer.json"
REPO="state-spaces/mamba-130m-hf"

echo "=== Redukt Mamba weight downloader ==="
echo ""

# Check if already present
if [ -f "$MODEL_FILE" ] && [ -f "$TOKENIZER_FILE" ]; then
    echo "[ok] Mamba weights already present in $WEIGHTS_DIR"
    ls -lh "$WEIGHTS_DIR"
    exit 0
fi

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "error: Python 3 required (python3 or python)"
    echo "  Install: https://www.python.org/downloads/"
    exit 1
fi
PY=$(command -v python3 || command -v python)

# Check pip dependencies
$PY -c "from huggingface_hub import hf_hub_download" 2>/dev/null || {
    echo "error: missing Python dependency"
    echo "  Run: pip install huggingface_hub"
    exit 1
}

mkdir -p "$WEIGHTS_DIR"

# Step 1: Download model.safetensors
if [ ! -f "$MODEL_FILE" ]; then
    echo "[1/2] Downloading model.safetensors from $REPO..."
    $PY -c "
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id='$REPO', filename='model.safetensors')
shutil.copy(path, '$MODEL_FILE')
print('  Saved: $MODEL_FILE')
"
    echo "[1/2] Done."
else
    echo "[1/2] Model already exists, skipping."
fi

# Step 2: Download tokenizer.json
if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "[2/2] Downloading tokenizer.json from $REPO..."
    $PY -c "
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id='$REPO', filename='tokenizer.json')
shutil.copy(path, '$TOKENIZER_FILE')
print('  Saved: $TOKENIZER_FILE')
"
    echo "[2/2] Done."
else
    echo "[2/2] Tokenizer already exists, skipping."
fi

# Verify
echo ""
echo "=== Verification ==="
if [ -f "$MODEL_FILE" ] && [ -f "$TOKENIZER_FILE" ]; then
    ls -lh "$WEIGHTS_DIR"
    echo ""
    echo "[ok] Mamba weights ready. Run:"
    echo "  cargo run --release -- mamba-eval --input <FILE> --bytes 10000"
else
    echo "[error] Download incomplete — check errors above."
    exit 1
fi
