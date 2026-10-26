#!/usr/bin/env bash
set -euo pipefail

# Download RWKV-7 0.1B weights for Redukt.
# Requires: Python 3, pip install huggingface_hub safetensors torch

WEIGHTS_DIR="weights/rwkv7-0.1b"
MODEL_FILE="$WEIGHTS_DIR/model.safetensors"
VOCAB_FILE="$WEIGHTS_DIR/rwkv_vocab_v20230424.txt"
REPO="BlinkDL/rwkv-7-world"
PTH_NAME="RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"
VOCAB_URL="https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt"

echo "=== Redukt weight downloader ==="
echo ""

# Check if already present
if [ -f "$MODEL_FILE" ] && [ -f "$VOCAB_FILE" ]; then
    echo "[ok] Weights already present in $WEIGHTS_DIR"
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
$PY -c "import torch; import safetensors; from huggingface_hub import hf_hub_download" 2>/dev/null || {
    echo "error: missing Python dependencies"
    echo "  Run: pip install torch safetensors huggingface_hub"
    exit 1
}

mkdir -p "$WEIGHTS_DIR"

# Step 1: Download .pth and convert to safetensors
if [ ! -f "$MODEL_FILE" ]; then
    echo "[1/2] Downloading $PTH_NAME and converting to safetensors..."
    $PY -c "
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

print('  Downloading from HuggingFace ($REPO)...')
pth = hf_hub_download(repo_id='$REPO', filename='$PTH_NAME')
print('  Converting to safetensors...')
sd = torch.load(pth, map_location='cpu', weights_only=True)
save_file(sd, '$MODEL_FILE')
print('  Saved: $MODEL_FILE')
"
    echo "[1/2] Done."
else
    echo "[1/2] Model already exists, skipping."
fi

# Step 2: Download tokenizer
if [ ! -f "$VOCAB_FILE" ]; then
    echo "[2/2] Downloading tokenizer..."
    if command -v curl &> /dev/null; then
        curl -sL "$VOCAB_URL" -o "$VOCAB_FILE"
    elif command -v wget &> /dev/null; then
        wget -q "$VOCAB_URL" -O "$VOCAB_FILE"
    else
        echo "error: curl or wget required for tokenizer download"
        exit 1
    fi
    echo "[2/2] Done."
else
    echo "[2/2] Tokenizer already exists, skipping."
fi

# Verify
echo ""
echo "=== Verification ==="
if [ -f "$MODEL_FILE" ] && [ -f "$VOCAB_FILE" ]; then
    ls -lh "$WEIGHTS_DIR"
    echo ""
    echo "[ok] Weights ready. You can now run:"
    echo "  cargo build --release"
    echo "  ./target/release/redukt compress --input <FILE> --output <FILE>.azth"
else
    echo "[error] Download incomplete — check errors above."
    exit 1
fi
