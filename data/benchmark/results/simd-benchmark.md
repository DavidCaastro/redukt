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

### F32 768×768 (kernel only — NOT dispatched in production)

| Kernel | Time (μs) | vs Scalar |
|---|---|---|
| Scalar (4-row ILP) | 80.5 | 1.00x |
| AVX2+FMA (8-wide) | 50.4 | **1.60x** |

**DISABLED**: F32 FMA breaks cross-path determinism (733/768 rows differ).
LLVM auto-vectorizes the scalar loop at -O3 anyway. See "Determinism" section.

### Q8 65536×768 (head matmul, memory-bandwidth limited)

| Kernel | Time (ms) | vs Scalar |
|---|---|---|
| Scalar | 4.2 | 1.00x |
| Dispatch (VNNI) | 3.2 | **1.32x** |

Lower speedup: 48MB Q8 head matrix exceeds L3 cache — memory-bound.

## Bit-exact verification

| Kernel | Mismatches vs scalar (768 rows) | Status |
|---|---|---|
| Q8 AVX2 | **0/768** | Bit-exact (integer accumulation) |
| Q8 VNNI | **0/768** | Bit-exact (integer accumulation) |
| F32 AVX2+FMA | **733/768** | NOT bit-exact (FMA rounding) |

Q8 SIMD is safe because:
- i8*i8→i32 accumulation is exact (no overflow, no rounding)
- `maddubs_epi16` pairwise sums fit i16 (max 32258 < 32767)
- `madd_epi16` and `add_epi32` are exact integer operations
- Horizontal sum is exact (i32 addition is associative)
- Final `sum_i32 as f32 * (scale * v_scale)` is identical expression in all paths

F32 SIMD breaks because:
- FMA fuses mul+add into single operation (different rounding from separate ops)
- AVX2 horizontal sum reduces in different order (float addition is NOT associative)
- Tiny differences compound through 12 sequential RWKV layers

## Cross-path roundtrip validation

| Compress | Decompress | Result |
|---|---|---|
| SIMD | SIMD | **PASS** (byte-identical, CRC OK) |
| Scalar | Scalar | **PASS** (byte-identical, CRC OK) |
| SIMD | Scalar | **PASS** (byte-identical, CRC OK) |
| Scalar | SIMD | **PASS** (byte-identical, CRC OK) |

BPB: **1.281205** (identical for both paths).
Compressed output: **1681 bytes** (identical for both paths).

## End-to-end throughput (enwik8 5KB, Q8, post-fix)

| Mode | tok/s (run1) | tok/s (run2) | tok/s (run3) |
|---|---|---|---|
| Scalar | 24.6 | 23.9 | 29.6 |
| SIMD (Q8 only) | 32.8 | 35.5 | 36.5 |

Estimated speedup: **~1.3x** (varies with thermal throttling on laptop).
Previous measurement (with F32 FMA, before fix): ~1.17x — similar because F32 FMA
contributed little (LoRA matrices are small relative to Q8 weight matrices).

## Analysis

### Why end-to-end speedup (~1.3x) is lower than kernel speedup (14.5x)

1. **Head matmul is memory-bound**: The largest matmul (65536×768, 48MB Q8) exceeds L3 cache,
   yielding only 1.32x speedup. This dominates total matmul time.
2. **Non-matmul overhead**: State updates, layer normalization, tokenization, arithmetic coder
   are not SIMD-optimized (~20% of total time).
3. **LoRA F32 matrices not accelerated**: Kept scalar for determinism. Small impact.
4. **Laptop thermal constraints**: i5-1235U throttles under sustained load.

### Expected impact on different hardware

| CPU | SIMD tier | Expected matmul speedup | Expected end-to-end |
|---|---|---|---|
| Haswell/Broadwell (2013-15) | AVX2 | 6-7x | 1.3-2x |
| Skylake–Coffee Lake (2015-19) | AVX2 | 6-7x | 1.3-2x |
| Alder Lake+ (2021+) | AVX-VNNI | 14-15x | 1.3-1.5x (laptop) |
| Server (Xeon, Ryzen 9) | AVX2/VNNI | 6-15x | **2-3x** (no throttle) |

## Dispatch hierarchy (production)

```
q8_mat_vec_mul_into:
  1. AVX-VNNI  (if avxvnni && cols%32==0)  ←  i5-12th+, Xeon 4th+
  2. AVX2      (if avx2)                    ←  Haswell (2013)+
  3. Scalar    (fallback)                   ←  any x86_64

mat_vec_mul_into (F32):
  1. Scalar only (FMA disabled for determinism)

Override: REDUKT_NO_SIMD=1 forces scalar path for Q8 operations.
```

## Design decisions

1. **F32 FMA disabled**: Cross-path determinism > 1.6x F32 speedup. The Q8 weight
   matrices dominate computation; LoRA F32 matrices are small.
2. **REDUKT_NO_SIMD=1**: Environment variable for A/B benchmarking. Not intended for
   production use.
3. **Kernel retained as dead code**: F32 FMA kernel kept with `#[allow(dead_code)]` for
   future opt-in use (e.g., `--fast-math` flag that accepts non-determinism).

## Reproducibility

```bash
# Micro-benchmark (matmul kernel)
cargo test --release bench_simd -- --ignored --nocapture

# Bit-exact verification
cargo test --release bit_exact -- --nocapture

# Cross-path roundtrip
target/maxperf/redukt.exe compress --input FILE --output test.azth --weights weights/rwkv7-0.1b --bytes 10000
REDUKT_NO_SIMD=1 target/maxperf/redukt.exe decompress --input test.azth --output test_out.bin --weights weights/rwkv7-0.1b
diff FILE test_out.bin  # must be identical

# End-to-end throughput comparison
REDUKT_NO_SIMD=1 target/maxperf/redukt.exe hybrid-eval --input FILE --weights weights/rwkv7-0.1b --bytes 5000
target/maxperf/redukt.exe hybrid-eval --input FILE --weights weights/rwkv7-0.1b --bytes 5000
```
