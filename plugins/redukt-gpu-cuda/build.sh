#!/usr/bin/env bash
# Build libredukt_gpu.so.  Uses the local nvcc if present, otherwise the official CUDA devel image
# (nvidia/cuda:12.4.1-devel-ubuntu22.04, https://hub.docker.com/r/nvidia/cuda) — no GPU needed to
# compile.  The full log (incl. ptxas register/smem stats) goes to build.log; the build fails if
# nvcc/ptxas/host compiler emit any warning.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"

start=$(date +%s)
if command -v nvcc >/dev/null 2>&1; then
    make clean all 2>&1 | tee build.log
else
    # MSYS_NO_PATHCONV: keep Git Bash on Windows from rewriting the container paths.
    # --entrypoint bash skips the image banner (it prints a "no driver" WARNING on GPU-less hosts).
    MSYS_NO_PATHCONV=1 docker run --rm --entrypoint bash -v "$(pwd):/w" -w /w "$IMAGE" \
        -c 'nvcc --version | tail -2 && make clean all' 2>&1 | tee build.log
fi
end=$(date +%s)
echo "[build.sh] elapsed: $((end - start)) s"

test -s build.log || { echo "[build.sh] FAIL: empty build.log" >&2; exit 1; }
if grep -qi "warning" build.log; then
    echo "[build.sh] FAIL: warnings found in build.log" >&2
    exit 1
fi
test -f libredukt_gpu.so
echo "[build.sh] OK: zero warnings"
