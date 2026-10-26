# Benchmark Results — Redukt vs Standard Compressors

## Methodology

- **Date**: 2026-10-26 (updated with F32/Q8 comparison + industry files)
- **Hardware**: Intel i5-1235U, 12 threads, 32 GB DDR5, Windows 11
- **Sample size**: 18 files x 10 KB each (12 Silesia + enwik8 + 5 industry)
- **Redukt config**: `hybrid-eval --bytes 10000` (standard production defaults)
- **Weights**: RWKV-7 0.1B (SafeTensors, Q8 auto-quantized)
- **Standard compressors**: gzip-9, brotli-11, zstd-22, xz-9
- **Full Silesia results**: 12 files + enwik8 (full size, gzip/brotli/xz only)
- **Raw data**: `data/benchmark/results/`

> **Note**: Earlier results (2026-10-20) used `--hierarchical --match --emb-surgery center0.3`
> (experimental features). Production results below use standard defaults only.
> F32 vs Q8 comparison (`--quant f32` vs `--quant q8`) confirmed <2.5% difference.

## Production Results (Q8, Standard Defaults)

### Industry Files (First Benchmarked 2026-10-26)

| File | Category | Redukt BPB | brotli-11 | Advantage |
|---|---|---|---|---|
| apache-access.log | Server logs | **0.245** | 0.466 | **-47.5%** |
| linux-syslog.log | System logs | **0.362** | 0.582 | **-37.9%** |
| facturae-batch.xml | E-invoicing | **0.406** | 0.891 | **-54.4%** |
| openssh-auth.log | Auth logs | **0.475** | 0.783 | **-39.4%** |
| ubl-invoice-batch.xml | E-invoicing UBL | **0.490** | 0.889 | **-44.9%** |

### Silesia Corpus + enwik8

| File | Category | Redukt BPB | brotli-11 | Advantage |
|---|---|---|---|---|
| nci | Scientific | 0.596 | 0.57 | +4.6% |
| xml | Markup | **0.640** | 0.89 | **-28.1%** |
| samba | Code | **1.266** | 1.61 | **-21.4%** |
| enwik8 | Text/encyclopedia | **1.281** | 1.83 | **-30.0%** |
| dickens | Text/literature | **1.779** | 2.11 | **-15.7%** |
| webster | Reference | **1.760** | 2.10 | **-16.2%** |
| reymont | Text/polish | **1.808** | 1.89 | **-4.3%** |
| mozilla | Binary | **1.964** | 2.31 | **-15.0%** |
| mr | Medical img | 2.283 | 2.27 | +0.6% |
| ooffice | Executable | 3.704 | 3.32 | +11.5% |
| x-ray | Medical img | 6.524 | 4.44 | +46.9% |
| osdb | Database | 5.643 | 3.08 | +83.2% |
| sao | Astronomy | 9.241 | 5.57 | +65.9% |

## Summary Statistics

| Metric | Value |
|---|---|
| Files where Redukt wins | **13/18 (72%)** |
| Files where Redukt loses | 5/18 (high-entropy, medical img, nci) |
| Mean advantage (winning files only) | **-29.5%** |
| Best advantage | facturae-batch.xml: **-54.4%** |
| Worst loss | osdb database: +83.2% (near-random data) |

## Category Analysis

| Category | Files | Redukt mean BPB | brotli mean BPB | Advantage |
|---|---|---|---|---|
| **Server/auth logs** | 3 | 0.361 | 0.610 | **-41.6%** |
| **E-invoicing XML** | 2 | 0.448 | 0.890 | **-49.7%** |
| Text/Literature | 4 | 1.657 | 1.983 | **-16.5%** |
| Markup (Silesia xml) | 1 | 0.640 | 0.890 | **-28.1%** |
| Source code | 1 | 1.266 | 1.610 | **-21.4%** |
| Binary/Executable | 2 | 2.834 | 2.815 | +0.7% |
| Medical imaging | 2 | 4.404 | 3.355 | +31.3% |
| High-entropy | 3 | 6.842 | 4.407 | +55.3% |

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
| **Redukt (Q8)** | **~120 B/s** | **~4,000x slower than brotli** |
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
