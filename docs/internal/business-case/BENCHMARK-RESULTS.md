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

## Verified Results (Actual AZTH Compression, Roundtrip-Verified, 2026-10-28)

BPB = `compressed_size × 8 / original_size`. Roundtrip = compress → decompress → byte-identical.

| File | Category | Redukt BPB [MEASURED] | brotli-11 [MEASURED] | gzip-9 [MEASURED] | vs brotli | Coder eff. | Roundtrip |
|---|---|---|---|---|---|---|---|
| apache-access.log | Server logs | **0.309** | 0.466 | 0.546 | **-33.8%** | 80.3% | PASS |
| facturae-batch.xml | E-invoicing | **0.462** | 0.890 | 1.070 | **-48.1%** | 88.4% | PASS |
| enwik8 | Text/encyclopedia | **1.345** | 2.352 | 2.972 | **-42.8%** | 95.3% | PASS |

### Cross-Entropy Results (hybrid-eval, pending re-measurement with actual compression)

> **WARNING**: These BPB values are cross-entropy estimates (ce_bpb), NOT actual compressed file
> size. Actual compressed BPB is typically 5-20% higher due to coder overhead. Do NOT compare
> directly with competitor BPB (which IS actual file size). Re-measurement pending.

| File | Category | Redukt ce_bpb | brotli-11 | Note |
|---|---|---|---|---|
| linux-syslog.log | System logs | 0.362 | 0.582 | needs actual compression |
| openssh-auth.log | Auth logs | 0.475 | 0.783 | needs actual compression |
| ubl-invoice-batch.xml | E-invoicing UBL | 0.490 | 0.889 | needs actual compression |
| nci | Scientific | 0.596 | 0.57 | needs actual compression |
| xml | Markup | 0.640 | 0.89 | needs actual compression |
| samba | Code | 1.266 | 1.61 | needs actual compression |
| dickens | Text/literature | 1.779 | 2.11 | needs actual compression |
| webster | Reference | 1.760 | 2.10 | needs actual compression |
| reymont | Text/polish | 1.808 | 1.89 | needs actual compression |
| mozilla | Binary | 1.964 | 2.31 | needs actual compression |
| mr | Medical img | 2.283 | 2.27 | needs actual compression |
| ooffice | Executable | 3.704 | 3.32 | needs actual compression |
| x-ray | Medical img | 6.524 | 4.44 | needs actual compression |
| osdb | Database | 5.643 | 3.08 | needs actual compression |
| sao | Astronomy | 9.241 | 5.57 | needs actual compression |

## Summary Statistics (based on 3 verified files)

| Metric | Value |
|---|---|
| Files verified with actual compression | **3/18** |
| All 3 verified files beat brotli-11 | **3/3 (100%)** |
| Best advantage (verified) | facturae-batch.xml: **-48.1%** |
| Weakest advantage (verified) | apache-access.log: **-33.8%** |
| Coder efficiency range | 80-95% (lower on more compressible files) |
| Pending re-measurement | 15 files |

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
