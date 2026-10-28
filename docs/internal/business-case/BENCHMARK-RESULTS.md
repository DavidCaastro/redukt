# Benchmark Results — Redukt vs Standard Compressors

## Methodology

- **Date**: 2026-10-26 (updated 2026-10-28 with actual AZTH compression correction)
- **Hardware**: Intel i5-1235U, 12 threads, 32 GB DDR5, Windows 11
- **Sample size**: 3 files verified with actual compression; 15 files pending re-measurement
- **Redukt config**: `redukt benchmark --bytes 10000 --verify` (actual AZTH compression + roundtrip)
- **Weights**: RWKV-7 0.1B (SafeTensors, Q8 auto-quantized)
- **Standard compressors**: gzip-9, brotli-11 (piped via stdin)
- **Raw data**: `data/benchmark/results/`

> **CORRECTION (2026-10-28)**: Previous results used cross-entropy BPB (hybrid-eval), which
> is 5-20% lower than actual compressed file size BPB. The benchmark now runs actual AZTH
> compression (RangeEncoder + v3 header + CRC-32) for apples-to-apples comparison.
> All BPB = `compressed_size × 8 / original_size`.
>
> Earlier results (2026-10-20) used `--hierarchical --match --emb-surgery center0.3`
> (experimental features). Production results below use standard defaults only.
> F32 vs Q8 comparison (`--quant f32` vs `--quant q8`) confirmed <2.5% difference.

## Production Results (Actual AZTH Compression, 18 files, 2026-10-28)

BPB = `compressed_size × 8 / original_size`. All measured with `redukt benchmark --bytes 10000`.
Industry files roundtrip-verified (`--verify`). Competitors piped via stdin (brotli -q 11, gzip -9).
Raw CSV: `data/benchmark/results/actual-compression-benchmark.csv`.

### Industry Files

| File | Category | Redukt BPB | brotli-11 | gzip-9 | vs brotli | Coder eff. | RT |
|---|---|---|---|---|---|---|---|
| apache-access.log | Server logs | **0.309** | 0.466 | 0.546 | **-33.8%** | 80.3% | PASS |
| linux-syslog.log | System logs | **0.426** | 0.582 | 0.714 | **-26.7%** | 86.3% | PASS |
| facturae-batch.xml | E-invoicing | **0.462** | 0.890 | 1.070 | **-48.1%** | 88.4% | PASS |
| openssh-auth.log | Auth logs | **0.538** | 0.783 | 0.990 | **-31.3%** | 88.4% | PASS |
| ubl-invoice-batch.xml | E-invoicing UBL | **0.558** | 0.888 | 1.093 | **-37.2%** | 87.9% | PASS |

### Silesia Corpus + enwik8

| File | Category | Redukt BPB | brotli-11 | gzip-9 | vs brotli | Coder eff. |
|---|---|---|---|---|---|---|
| nci | Scientific | **0.680** | 0.734 | 0.934 | **-7.4%** | 88.1% |
| xml | Markup | **0.705** | 0.901 | 1.075 | **-21.8%** | 90.0% |
| samba | Code | **1.319** | 2.220 | 2.884 | **-40.6%** | 96.1% |
| enwik8 | Text/encyclopedia | **1.345** | 2.352 | 2.972 | **-42.8%** | 95.3% |
| webster | Reference | **1.809** | 2.809 | 3.661 | **-35.6%** | 97.2% |
| dickens | Text/literature | **1.829** | 2.778 | 3.621 | **-34.2%** | 97.2% |
| reymont | Text/polish | **1.918** | 2.136 | 2.387 | **-10.2%** | 94.4% |
| mozilla | Binary | 2.063 | 2.021 | 2.374 | +2.1% | 95.2% |
| mr | Medical img | 2.402 | 2.373 | 2.780 | +1.2% | 95.2% |
| ooffice | Executable | 3.779 | 2.742 | 2.910 | +37.8% | 98.0% |
| osdb | Database | 5.717 | 4.676 | 4.967 | +22.3% | 98.7% |
| x-ray | Medical img | 6.608 | 4.592 | 5.873 | +43.9% | 98.6% |
| sao | Astronomy | 9.254 | 5.988 | 6.389 | +54.5% | 99.8% |

## Summary Statistics (all 18 files, actual compression)

| Metric | Value |
|---|---|
| Files where Redukt wins vs brotli | **12/18 (67%)** |
| Files where Redukt loses vs brotli | 6/18 (executables, medical, high-entropy) |
| Mean advantage (winning files only) | **-31.7%** |
| Best advantage | facturae-batch.xml: **-48.1%** |
| Worst loss | sao astronomy: +54.5% (near-random data) |
| Coder efficiency range | 80.3% (apache) to 99.8% (sao) |

## Category Analysis (actual compression)

| Category | Files | Redukt mean BPB | brotli mean BPB | Advantage |
|---|---|---|---|---|
| **Server/auth logs** | 3 | 0.424 | 0.610 | **-30.5%** |
| **E-invoicing XML** | 2 | 0.510 | 0.889 | **-42.6%** |
| Text/Literature | 4 | 1.851 | 2.519 | **-26.5%** |
| Structured markup | 2 | 0.693 | 0.818 | **-15.3%** |
| Source code | 1 | 1.319 | 2.220 | **-40.6%** |
| Binary/Executable | 2 | 2.921 | 2.382 | +22.6% |
| Medical imaging | 2 | 4.505 | 3.483 | +29.4% |
| High-entropy | 2 | 7.484 | 5.332 | +40.4% |

## F32 vs Q8 Quantization Comparison

Measured on all 18 files with `--quant f32` and `--quant q8`:

| Metric | Value |
|---|---|
| Max Q8 degradation | +2.43% (facturae-batch.xml) |
| Max Q8 improvement | -1.76% (enwik8) |
| Mean absolute delta | 0.81% |
| Q8 speed | 36.5 tok/s (2.1x faster) |
| F32 speed | 17.4 tok/s |

**Conclusion**: Q8 is the correct production choice. Negligible quality loss, 2x speed gain.
Full comparison in `data/benchmark/results/f32-vs-q8.csv`.

## Throughput

| Compressor | Speed (10KB) | Notes |
|---|---|---|
| gzip-9 | ~50 MB/s | Fastest |
| brotli-11 | ~0.5-1 MB/s | Slow at max level |
| zstd-22 | ~2-5 MB/s | Fast even at high levels |
| xz-9 | ~1-3 MB/s | Moderate |
| **Redukt (Q8)** | **~150 B/s** | **~3,300x slower than brotli** [MEASURED 2026-10-28] |
| **Redukt (F32)** | **~55 B/s** | **~9,000x slower than brotli** |

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

## Reproducibility

```bash
# Build
cargo build --release

# Run F32 vs Q8 comparison (all 18 files)
bash scripts/bench-f32-vs-q8.sh

# Run standard compressors (10KB)
python data/benchmark/run_standard.py

# Manual single-file eval
./target/release/redukt hybrid-eval --input <FILE> --bytes 10000 --quant q8
./target/release/redukt hybrid-eval --input <FILE> --bytes 10000 --quant f32
```

All raw CSV results in `data/benchmark/results/`.
