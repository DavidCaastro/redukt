#!/usr/bin/env bash
# Redukt benchmark script — runs hybrid-eval and baseline on standard corpora.
# Usage: ./scripts/benchmark.sh [--bytes N] [--full]
#
# Default: 10KB per file (fast, ~2min/file).
# --full:  entire file (slow, hours per file).

set -euo pipefail

REDUKT="cargo run --release --"
BYTES=10000
RESULTS_DIR="data/benchmark/results"

for arg in "$@"; do
    case $arg in
        --full) BYTES=0 ;;
        --bytes) shift; BYTES="$1" ;;
        --bytes=*) BYTES="${arg#*=}" ;;
    esac
done

mkdir -p "$RESULTS_DIR"

run_eval() {
    local file="$1"
    local name="$2"
    local mode="$3"
    local extra="${4:-}"

    if [ "$mode" = "baseline" ]; then
        $REDUKT baseline --input "$file" --bytes "$BYTES" $extra 2>&1 | grep "BPB:" | awk '{print $2}'
    else
        $REDUKT hybrid-eval --input "$file" --bytes "$BYTES" $extra 2>&1 | grep "BPB:" | awk '{print $2}'
    fi
}

echo "=== Redukt Benchmark (bytes=$BYTES) ==="
echo ""

# enwik8
if [ -f "data/benchmark/enwik8" ]; then
    echo "--- enwik8 ---"
    BL=$(run_eval "data/benchmark/enwik8" "enwik8" "baseline")
    HY=$(run_eval "data/benchmark/enwik8" "enwik8" "hybrid")
    echo "  baseline: $BL BPB"
    echo "  hybrid:   $HY BPB"
    echo ""
fi

# Silesia corpus
if [ -d "data/benchmark/silesia" ]; then
    echo "--- Silesia corpus ---"
    echo "file,baseline_bpb,hybrid_bpb" > "$RESULTS_DIR/silesia-comparison-${BYTES}b.csv"
    for f in data/benchmark/silesia/*; do
        name=$(basename "$f")
        BL=$(run_eval "$f" "$name" "baseline")
        HY=$(run_eval "$f" "$name" "hybrid")
        echo "  $name: baseline=$BL  hybrid=$HY"
        echo "$name,$BL,$HY" >> "$RESULTS_DIR/silesia-comparison-${BYTES}b.csv"
    done
    echo ""
    echo "Results saved to $RESULTS_DIR/silesia-comparison-${BYTES}b.csv"
fi

echo ""
echo "=== Done ==="
