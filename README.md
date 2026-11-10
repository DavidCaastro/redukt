# Redukt

Neural compression engine that achieves state-of-the-art lossless compression
by combining a pretrained language model (RWKV-7) with online-adaptive Context
Mixing and arithmetic coding.

## Highlights

- **Best-in-class compression**: 48% smaller than brotli-11 on e-invoicing, 34% on server logs (actual compression)
- **18 domains validated**: Silesia corpus (12 files), enwik8, industry files (5)
- **Zero dependencies**: Pure Rust, single binary, CPU-only
- **Deterministic**: Bit-identical decompression guaranteed (CRC-32 verified)

## Quick Start

```bash
# 1. Download weights (~400 MB, requires Python 3 + torch + safetensors + huggingface_hub)
bash scripts/download-weights.sh

# 2. Build
cargo build --release

# 3. Compress
./target/release/redukt compress --input data.txt --output data.azth

# 4. Decompress (lossless — byte-identical output guaranteed)
./target/release/redukt decompress --input data.azth --output restored.txt
```

## How It Works

Redukt uses a neural network (RWKV-7, 0.1B parameters) as a probability
predictor for arithmetic coding. An ensemble of 19 Context Mixing models
provides online-adaptive statistical predictions. An LSTM mixer dynamically
weights all prediction sources per-byte.

```
Input → Tokenize → [RWKV-7 + CM ensemble + LSTM mixer] → Arithmetic Coder → .azth
```

## Performance (18 files, Q8, 10KB, actual AZTH compression)

All BPB = `compressed_size × 8 / original_size` (actual file output, not cross-entropy).

| File | Domain | Redukt BPB | brotli-11 | vs brotli |
|---|---|---|---|---|
| apache-access.log | Server logs | **0.31** | 0.47 | **-34%** |
| facturae-batch.xml | E-invoicing | **0.46** | 0.89 | **-48%** |
| ubl-invoice-batch.xml | E-invoicing UBL | **0.56** | 0.89 | **-37%** |
| xml | XML markup | **0.70** | 0.90 | **-22%** |
| samba | Source code | **1.32** | 2.22 | **-41%** |
| enwik8 | Text/encyclopedia | **1.34** | 2.35 | **-43%** |
| dickens | Text/literature | **1.83** | 2.78 | **-34%** |

Wins 12/18 files. Loses on high-entropy data (medical imaging, astronomy, databases, executables).
Throughput (Mamba backbone, i5-1235U laptop, measured 2026-11-10): 53.6 tok/s single-stream F32, 111-127 tok/s Q8
weight-only, and 315-322 tok/s aggregate with lockstep batching (N=32 streams) -- about 9x the 34 tok/s of the
previous release. Batch/archival workloads.
Full results: `data/benchmark/results/actual-compression-benchmark.csv`.

## Batch (lockstep)

`--jobs N` runs N files as lockstep streams: every step advances one token in each stream and shares a single fetch of
the model weights (GEMV becomes GEMM), so aggregate throughput grows with N until the CPU is compute-bound (N~8-32).
Each `.azth` is byte-identical to the single-file output regardless of N.

```bash
./target/release/redukt compress --input-dir logs/ --output-dir out/ --jobs 16
./target/release/redukt decompress --input-dir out/ --output-dir restored/ --jobs 16
```

Measured aggregate throughput (12 files x 10 KB, Q8, laptop): N=1 111, N=8 266, N=32 315 tok/s (F32: 58 -> 322 tok/s).
`--quant auto` (the default) selects the weight-only Q8 profile for Mamba (+0.6% to +1.2% BPB); `--quant f32` is
recommended for batches of 16 or more files. `REDUKT_THREADS=N` sets the
worker pool size.

## API

`redukt api` has byte-for-byte parity with the CLI (same backbone routing, preprocessing and profiles; verified
sha256 API == CLI). Flags: `--bind/--host`, `--port`, `--weights`, `--backbone`, `--quant`; `/v1/status` reports backbones,
profile, readable formats and version. It serves one request at a time.

## Deploy (GCP)

Reference deployment for Cloud Run (API), Cloud Batch (bulk compression) and GCE with GPU:
[`deploy/gcp/README.md`](deploy/gcp/README.md). Build the CPU image from the repo root with
`docker build -f deploy/gcp/Dockerfile -t redukt:cpu .` (the former root `Dockerfile` was removed; the canonical one is
`deploy/gcp/Dockerfile`). Verified locally (not yet on a live GCP project). The CUDA plugin batch interface (not yet run
on a real GPU) is documented in [`docs/api/PLUGIN-FFI-V2.md`](docs/api/PLUGIN-FFI-V2.md).

## Requirements

- Rust 1.89+
- RWKV-7 0.1B weights (~400 MB) — run `bash scripts/download-weights.sh`
- 4+ GB RAM

## Benchmark Your Data

```bash
# Evaluate Redukt vs gzip-9/brotli-11 on your files (JSON report)
./target/release/redukt benchmark --input /path/to/logs/ --output report.json

# Single file, custom window size
./target/release/redukt benchmark --input mydata.log --output report.json --bytes 50000

# Full benchmark suite (Python, includes zstd-19)
pip install brotli zstandard
python scripts/benchmark-suite.py --bytes 10240
```

The `benchmark` command runs actual AZTH compression on your data and compares against gzip-9
and brotli-11 (auto-detected). Add `--verify` for roundtrip decompression check. Output: JSON
with per-file BPB (actual compressed size), compression ratio, throughput, coder efficiency,
and storage savings projection. All results are deterministic and CRC-verified.

## Testing

```bash
# Unit tests (no weights needed) — 82 tests
cargo test

# End-to-end roundtrip: compress → decompress → verify byte-identical (requires weights)
cargo test --release --test roundtrip -- --ignored

# Verbose timing breakdown
./target/release/redukt compress --input data.txt --output data.azth --verbose
```

## Format

Output format: AZTH v5 (adds a PROFILE byte with the weight profile and the numeric contract; since R-104 the encoder
always writes v5, F32 included). The decoder accepts v2-v5.
See `docs/api/FORMAT-SPEC.md` for full specification.
Includes CRC-32 integrity verification, hyperparameter validation, and u32 overflow protection.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Built on [RWKV-7](https://github.com/BlinkDL/RWKV-LM) by Bo Peng (BlinkDL),
licensed under Apache 2.0. Context Mixing techniques inspired by
[PAQ](http://mattmahoney.net/dc/) (Matt Mahoney, public domain).
