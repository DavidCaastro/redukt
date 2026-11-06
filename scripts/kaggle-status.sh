#!/usr/bin/env bash
# Check Kaggle notebook status and download results when complete.
# Usage: scripts/kaggle-status.sh [--wait]
#   --wait: poll every 60s until complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_DIR/data/benchmark/results/kaggle"

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "ERROR: Run scripts/kaggle-setup.sh first."
    exit 1
fi
USERNAME=$(grep -o '"username":"[^"]*"' "$KAGGLE_JSON" | cut -d'"' -f4)
KERNEL_SLUG="$USERNAME/redukt-gpu-validation"

WAIT_MODE=false
if [ "${1:-}" = "--wait" ]; then
    WAIT_MODE=true
fi

check_status() {
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>/dev/null | tail -1)
    echo "[kaggle-status] $KERNEL_SLUG: $STATUS"
    echo "$STATUS"
}

download_results() {
    echo "[kaggle-status] Downloading results..."
    mkdir -p "$RESULTS_DIR"
    kaggle kernels output "$KERNEL_SLUG" -p "$RESULTS_DIR"
    echo "[kaggle-status] Results saved to: $RESULTS_DIR/"
    if [ -f "$RESULTS_DIR/gpu-benchmark.csv" ]; then
        echo ""
        echo "=== GPU Benchmark Results ==="
        cat "$RESULTS_DIR/gpu-benchmark.csv"
    fi
}

STATUS_LINE=$(check_status)

if echo "$STATUS_LINE" | grep -qi "complete"; then
    download_results
    exit 0
elif echo "$STATUS_LINE" | grep -qi "error\|cancel"; then
    echo "[kaggle-status] Notebook failed. Check logs at:"
    echo "  https://www.kaggle.com/code/$KERNEL_SLUG"
    exit 1
fi

if [ "$WAIT_MODE" = true ]; then
    echo "[kaggle-status] Waiting for completion (poll every 60s)..."
    while true; do
        sleep 60
        STATUS_LINE=$(check_status)
        if echo "$STATUS_LINE" | grep -qi "complete"; then
            download_results
            exit 0
        elif echo "$STATUS_LINE" | grep -qi "error\|cancel"; then
            echo "[kaggle-status] Notebook failed."
            exit 1
        fi
    done
else
    echo "[kaggle-status] Not complete yet. Run with --wait to poll."
fi
