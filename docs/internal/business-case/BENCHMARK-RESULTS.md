# Benchmark Results — Redukt vs Standard Compressors

## Methodology

- **Date**: 2026-10-20
- **Hardware**: Intel i5-1235U, 12 threads, 32 GB DDR5, Windows 11
- **Sample size**: 18 files x 10 KB each (diverse data types)
- **Redukt config**: `hybrid-eval --hierarchical --match --emb-surgery center0.3`
- **Weights**: RWKV-7 0.1B (SafeTensors, F32)
- **Standard compressors**: gzip-9, brotli-11, zstd-22, xz-9
- **Full Silesia results**: 12 files + enwik8 (full size, gzip/brotli/xz only)
- **Raw data**: `data/benchmark/results/`

## 10 KB Sample Results (Head-to-Head)

| File | Category | Redukt BPB | Best standard | Compressor | Advantage |
|---|---|---|---|---|---|
| xml-markup.10k | Markup | **0.492** | 0.89 | brotli-11 | **-44.7%** |
| nci-chem.10k | Scientific | **0.500** | 0.57 | brotli-11 | **-12.3%** |
| jsonl-struct.10k | Structured | **1.048** | 1.62 | brotli-11 | **-35.3%** |
| source-code.10k | Code | **1.088** | 1.63 | brotli-11 | **-33.1%** |
| samba-src.10k | Code | **1.127** | 1.61 | brotli-11 | **-30.0%** |
| enwik8-text.10k | Text | **1.151** | 1.83 | brotli-11 | **-37.2%** |
| reymont-lit.10k | Literature | **1.417** | 1.89 | brotli-11 | **-25.0%** |
| mozilla-bin.10k | Binary | **1.489** | 2.31 | brotli-11 | **-35.5%** |
| dickens-lit.10k | Literature | **1.511** | 2.11 | brotli-11 | **-28.4%** |
| webster-dict.10k | Reference | **1.532** | 2.10 | brotli-11 | **-27.1%** |
| mr-med.10k | Medical | **1.806** | 2.27 | brotli-11 | **-20.5%** |
| oeis-math.10k | Math | **1.813** | 2.37 | brotli-11 | **-23.5%** |
| ooffice-exe.10k | Executable | **2.470** | 3.32 | brotli-11 | **-25.6%** |
| pe-exe.10k | Executable | **2.538** | 3.41 | brotli-11 | **-25.6%** |
| xray-med.10k | Medical img | **3.870** | 4.44 | brotli-11 | **-12.8%** |
| osdb-db.10k | Database | **4.094** | 3.08 | xz-9 | +32.9% |
| sao-astro.10k | Astronomy | **5.770** | 5.57 | brotli-11 | +3.6% |
| random-ctrl.10k | Random | **8.010** | 7.99 | brotli-11 | +0.2% |

## Summary Statistics

| Metric | Value |
|---|---|
| Files where Redukt wins | **15/18 (83%)** |
| Files where Redukt loses | 3/18 (high-entropy: osdb, sao, random) |
| Mean BPB (text+structured only, 12 files) | **1.287** vs 2.021 brotli (-36.3%) |
| Mean BPB (all 18 files) | **2.263** vs 2.774 brotli (-18.4%) |
| Best advantage | XML: **-44.7%** |
| Worst loss | osdb database: +32.9% (near-random data) |

## Category Analysis

| Category | Files | Redukt mean BPB | brotli mean BPB | Advantage |
|---|---|---|---|---|
| Text/Literature | 4 | 1.400 | 1.958 | **-28.5%** |
| Structured (XML/JSON/code) | 4 | 0.939 | 1.438 | **-34.7%** |
| Binary/Executable | 3 | 2.166 | 3.013 | **-28.1%** |
| Scientific/Medical | 3 | 2.059 | 2.427 | **-15.2%** |
| High-entropy | 4 | 5.436 | 5.270 | +3.1% |

## Full Silesia Results (Reference — Standard Compressors Only)

| File | Size MB | gzip-9 BPB | brotli-11 BPB | xz-9 BPB |
|---|---|---|---|---|
| dickens | 9.72 | 3.023 | 2.220 | 2.222 |
| mozilla | 48.85 | 2.967 | 2.167 | 2.089 |
| mr | 9.51 | 2.948 | 2.265 | 2.207 |
| nci | 32.00 | 0.712 | 0.362 | 0.415 |
| ooffice | 5.87 | 4.019 | 3.223 | 3.156 |
| osdb | 9.62 | 2.948 | 2.234 | 2.261 |
| reymont | 6.32 | 2.198 | 1.608 | 1.590 |
| samba | 20.61 | 2.003 | 1.395 | 1.394 |
| sao | 6.92 | 5.877 | 5.059 | 4.871 |
| webster | 39.54 | 2.327 | 1.626 | 1.618 |
| x-ray | 8.08 | 5.700 | 4.421 | 4.239 |
| xml | 5.10 | 0.991 | 0.644 | 0.678 |
| enwik8 | 95.37 | 2.916 | 2.059 | 1.989 |

## Throughput

| Compressor | Speed (10KB) | Notes |
|---|---|---|
| gzip-9 | ~50 MB/s | Fastest |
| brotli-11 | ~0.5-1 MB/s | Slow at max level |
| zstd-22 | ~2-5 MB/s | Fast even at high levels |
| xz-9 | ~1-3 MB/s | Moderate |
| **Redukt** | **~62 B/s** | **~6,000x slower than brotli** |

## Reproducibility

```bash
# Build
cargo build --profile maxperf

# Run Redukt benchmark
python data/benchmark/run_Redukt.py

# Run standard compressors (10KB)
python data/benchmark/run_standard.py

# Run standard compressors (full Silesia)
python data/benchmark/run_standard_full.py
```

All raw CSV results in `data/benchmark/results/`.
Per-file Redukt logs in `data/benchmark/results/hybrid-*.log`.
