# SIMD Vectorization Benchmark (R-011)

**Date**: 2026-10-27
**CPU**: Intel i5-1235U (Alder Lake, 12th Gen)
**Features**: AVX2, FMA, AVX-VNNI
**Profile**: maxperf (opt-level=3, LTO=fat, codegen-units=1, panic=abort)
**Quantization**: Q8 (auto-selected)

## Micro-benchmark: matmul kernel (release mode, 200 iters, median)

### Q8 768×768 (layer matmul, fits L2/L3 cache)

| Kernel | Time (μs) | vs Scalar |
|---|---|---|
| Scalar (4-row ILP) | 97.2 | 1.00x |
| AVX2 (sign+maddubs) | 14.5 | **6.70x** |
| AVX-VNNI (dpbusd) | 6.7 | **14.51x** |
| Dispatch (auto) | 21.8 | **4.46x** |

### F32 768×768 (layer matmul)

| Kernel | Time (μs) | vs Scalar |
|---|---|---|
| Scalar (4-row ILP) | 80.5 | 1.00x |
| AVX2+FMA (8-wide) | 50.4 | **1.60x** |

Note: LLVM auto-vectorizes the scalar loop at -O3, limiting explicit SIMD gains for F32.

### Q8 65536×768 (head matmul, memory-bandwidth limited)

| Kernel | Time (ms) | vs Scalar |
|---|---|---|
| Scalar | 4.2 | 1.00x |
| Dispatch (VNNI) | 3.2 | **1.32x** |

Lower speedup because the 48MB Q8 head matrix exceeds L3 cache — becomes memory-bound.

## End-to-end throughput: hybrid-eval (enwik8 5KB, Q8)

### 3-run average

| Mode | tok/s | Time (s) | Speedup |
|---|---|---|---|
| REDUKT_NO_SIMD=1 (scalar) | 32.3 | 47.7 | 1.00x |
| Default (VNNI dispatch) | 37.9 | 40.7 | **1.17x** |

### compress operation

| Mode | tok/s | Compressed | Ratio |
|---|---|---|---|
| Scalar | 32.5 | 672 B | 0.1344 |
| SIMD | 36.9 | 676 B | 0.1352 |
| **Speedup** | **1.14x** | | |

BPB difference: 0.001 (0.1%) — expected from FMA rounding.
Compressed size difference: 4 bytes (0.6%) — acceptable numerical noise.

## Analysis

### Why end-to-end speedup (1.17x) is lower than kernel speedup (14.5x)

1. **Head matmul is memory-bound**: The largest matmul (65536×768, 48MB Q8) exceeds L3 cache,
   yielding only 1.32x speedup. This dominates total matmul time.
2. **Non-matmul overhead**: State updates, layer normalization, tokenization, arithmetic coder
   are not SIMD-optimized (~20% of total time).
3. **Laptop thermal constraints**: i5-1235U throttles under sustained load, capping both paths.

### Expected impact on different hardware

| CPU | SIMD tier | Expected matmul speedup | Expected end-to-end |
|---|---|---|---|
| Haswell/Broadwell (2013-15) | AVX2 | 6-7x | 1.5-2x |
| Skylake–Coffee Lake (2015-19) | AVX2 | 6-7x | 1.5-2x |
| Alder Lake+ (2021+) | AVX-VNNI | 14-15x | 1.1-1.5x (laptop thermal) |
| Server (Xeon, Ryzen 9) | AVX2/VNNI | 6-15x | **2-3x** (no throttle, more BW) |

### Numerical determinism

- **Same SIMD path** = deterministic (identical compress/decompress)
- **Scalar vs SIMD** = slightly different (FMA rounding, reduction order)
- Cross-platform determinism deferred to R-013

## Dispatch hierarchy

```
q8_mat_vec_mul_into:
  1. AVX-VNNI  (if avxvnni && cols%32==0)  ←  i5-12th+, Xeon 4th+
  2. AVX2      (if avx2)                    ←  Haswell (2013)+
  3. Scalar    (fallback)                   ←  any x86_64

mat_vec_mul_into (F32):
  1. AVX2+FMA  (if avx2 && fma)            ←  Haswell (2013)+
  2. Scalar    (fallback)                   ←  any x86_64

Override: REDUKT_NO_SIMD=1 forces scalar path for all operations.
```

## Reproducibility

```bash
# Micro-benchmark (matmul kernel)
cargo test --release domain::tensor::tests::bench_simd -- --ignored --nocapture

# End-to-end: SIMD
target/maxperf/redukt.exe hybrid-eval --input data/benchmark/enwik8 --weights weights/rwkv7-0.1b --bytes 5000

# End-to-end: scalar baseline
REDUKT_NO_SIMD=1 target/maxperf/redukt.exe hybrid-eval --input data/benchmark/enwik8 --weights weights/rwkv7-0.1b --bytes 5000
```
