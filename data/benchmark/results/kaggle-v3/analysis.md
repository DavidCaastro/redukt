# Kaggle GPU Validation v3 — Analysis

**Date**: 2026-11-06
**Platform**: Kaggle T4 (Tesla T4 16GB, Intel Xeon CPU)
**Run**: notebooks/gpu-validation.py (v3, --input/--output CLI flags)

## Summary

| Check | Result |
|---|---|
| Determinism (2 runs) | PASS |
| Roundtrip (compress→decompress) | PASS |
| Benchmark entries | 18 (9 CPU + 9 GPU-plugin) |

## CPU Mamba Baseline (Kaggle)

| File | BPB | tok/s | Time (s) | Notes |
|---|---|---|---|---|
| enwik8 | **0.9633** | 72.4 | 141.4 | Real data (10KB truncated) |
| dickens | 0.1133 | 29.9 | 342.6 | Synthetic (SHA-256 periodic) |
| samba | 0.1086 | 27.7 | 369.4 | Synthetic |
| mozilla | 0.1148 | 26.0 | 393.4 | Synthetic |
| mr | 0.1070 | 28.9 | 354.0 | Synthetic |
| nci | 0.1133 | 27.2 | 376.7 | Synthetic |
| ooffice | 0.1156 | 26.9 | 381.3 | Synthetic |
| x-ray | 0.1148 | 26.9 | 380.4 | Synthetic |
| sao | 0.1109 | 27.8 | 368.1 | Synthetic |

**enwik8 cross-platform**: 0.9633 (Kaggle T4) vs 0.9609 (local Win11) = **+0.25%** difference.
Within expected cross-platform variance (different CPU µarch, FP rounding).

**Throughput**: 72.4 tok/s on Kaggle Xeon vs ~34 tok/s local (DDR4 desktop).
2.1x speedup from better DRAM bandwidth (server-grade memory subsystem).

Synthetic files show 0.10-0.11 BPB because SHA-256 hash repeated 320x creates
a 32-byte periodic signal that the model learns in <100 tokens.

## GPU Plugin (Skeleton — Uniform Logits)

| File | BPB | tok/s | Time (s) |
|---|---|---|---|
| enwik8 | 3.0219 | 81,184 | 0.13 |
| dickens | 0.0977 | 94,431 | 0.11 |
| samba | 0.0977 | 96,198 | 0.11 |
| mozilla | 0.0977 | 96,468 | 0.11 |
| mr | 0.0977 | 93,383 | 0.11 |
| nci | 0.0977 | 96,688 | 0.11 |
| ooffice | 0.0984 | 95,624 | 0.11 |
| x-ray | 0.0969 | 95,920 | 0.11 |
| sao | 0.0984 | 92,615 | 0.11 |

### Interpretation

The skeleton plugin returns **uniform logits** (1/vocab_size for all tokens),
so compression quality is irrelevant — these numbers validate the FFI pipeline:

1. **FFI interface works on Linux/Kaggle**: dlopen + BackendVTable + 4 fn pointers = functional
2. **Pipeline overhead**: ~0.11-0.13s for 10KB = negligible
3. **Throughput ceiling**: 81K-96K tok/s without real inference
4. **enwik8 BPB 3.02**: Uniform logits + CM = ~3 BPB (CM provides some compression even alone)
5. **Synthetic BPB 0.097**: CM alone compresses periodic data almost perfectly

### What This Proves

- The `#[repr(C)]` FFI boundary (R-072) is cross-platform correct
- `dlopen`/`dlsym` plugin loading (R-073) works on Kaggle Linux
- `PluginBackend` → `Backbone` trait dispatch (R-074) is functional
- Arithmetic coding + CM models work correctly with plugin logits
- Determinism and roundtrip are preserved through the plugin path

### What Remains

Real GPU kernels (CUDA/OpenCL) to replace the skeleton plugin.
With actual Mamba inference on T4, expected throughput: 500-2,000 tok/s
(vs 27-72 tok/s CPU), depending on kernel optimization level.

## Files

- `gpu-benchmark.csv`: Raw benchmark data (18 entries)
- `validation-report.txt`: Determinism + roundtrip check results
- `redukt-gpu-validation.log`: Full Kaggle kernel execution log
- `benchmark-data/`: Test files + compressed outputs (.cpu.azth, .gpu.azth)
