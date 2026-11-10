# Redukt

Neural lossless compression engine: a pretrained state-space backbone (Mamba-130M by default, RWKV-7 0.1B for
generic binaries) feeds an online-adaptive context-mixing ensemble and an arithmetic coder. Pure Rust, zero
dependencies, single binary, CPU reference implementation with an optional CUDA plugin.

## Highlights

- **Strong ratio on text, logs and XML at small window sizes**: enwik8 0.964 bpb, apache log 0.216, facturae XML 0.389
  (10 KB windows, actual AZTH output). See "Competitive position" below for what those figures do and do not mean.
- **18 domains validated**: Silesia corpus (12 files), enwik8, industry files (5)
- **Zero dependencies**: Pure Rust, single binary, CPU-only reference; GPU is a plugin, never a requirement
- **Deterministic by contract**: byte-identical output batch == single file, Windows == Linux (x86-64-v3), API == CLI;
  weight profile and numeric contract are declared in the AZTH v5 header. CRC-32 verified on decompression.

## Quick Start

```bash
# 1. Download weights (~400 MB per backbone, requires Python 3 + torch + safetensors + huggingface_hub)
bash scripts/download-weights.sh          # RWKV-7 0.1B
bash scripts/download-mamba-weights.sh    # Mamba-130M (default backbone)

# 2. Build
cargo build --release

# 3. Compress
./target/release/redukt compress --input data.txt --output data.azth

# 4. Decompress (lossless — byte-identical output guaranteed)
./target/release/redukt decompress --input data.azth --output restored.txt
```

## How It Works

A pretrained sequence model (Mamba-130M, or RWKV-7 0.1B when the router classifies the input as generic binary)
produces per-token probabilities. A context-mixing ensemble of fast online models (order-N and token n-grams, a bias
head) is blended with the backbone by an adaptive SGD mixer, and the resulting distribution drives a 24-bit range coder.
Binary inputs can be preprocessed with E8E9, byte-plane shuffle and delta coding.

```
Input → Tokenize → [Mamba-130M | RWKV-7 + CM ensemble + adaptive mixer] → Arithmetic Coder → .azth
```

## Performance (12 files, 10 KB windows, actual AZTH compression, measured 2026-10-31)

All BPB = `compressed_size × 8 / original_size` (actual file output, not cross-entropy). **These are 10 KB
windows**: classical codecs improve far more than Redukt when they see the whole file (see the last column), so the
10 KB margin must not be quoted as a product-level advantage.

| File | Domain | Redukt BPB (10 KB) | brotli-11 (10 KB) | vs brotli (10 KB) | brotli-11 whole file |
|---|---|---|---|---|---|
| apache-access.log | Server logs | **0.216** | 0.466 | -54% | 0.313 (171 KB) |
| facturae-batch.xml | E-invoicing | **0.389** | 0.890 | -56% | 0.395 (39 KB) |
| ubl-invoice-batch.xml | E-invoicing UBL | **0.446** | 0.888 | -50% | 0.247 (83 KB) |
| samba | Source code | **0.271** | 2.220 | -88% | 1.395 (21 MB) |
| dickens | Text/literature | **0.652** | 2.778 | -77% | 2.220 (10 MB) |
| enwik8 | Text/encyclopedia | **0.964** | 2.352 | -59% | 2.061 (100 MB) |

Wins 8/12 files vs brotli-11 at 10 KB; loses on high-entropy binary (sao +45%, ooffice +37%, osdb +21%, x-ray +6%).
**Whole-file results (Kaggle T4, 2026-11-10, `data/benchmark/results/kaggle-fullsize/`)**: with the current Mamba
backbone the ratio collapses with length — enwik8 0.964 (10 KB) → 1.63 (100 KB) → 3.91 (2 MB), and all five whole
industry files lose to brotli-11/xz. The cause is Mamba-130M's recurrent state degenerating beyond ~16K tokens
(reproduced with the HF reference implementation). An experimental periodic state reset
(`REDUKT_MAMBA_RESET_EVERY=4096`, encoder and decoder must agree) restores 0.995 bpb on the 100 KB prefix; putting
the period in the format is roadmap item R-108. Until then, treat every figure above as a 10 KB-window figure.

Throughput (Mamba backbone, i5-1235U laptop, measured 2026-11-10): 53.6 tok/s single-stream F32, 111-127 tok/s Q8
weight-only, 315-322 tok/s aggregate with lockstep batching (N=32 streams), 366 tok/s with `--numeric fma`.
Kaggle T4 CUDA plugin: 151 tok/s single stream, 1,117 tok/s at batch 128 (FFI level). One BPE token ≈ 3.8 bytes of
English text. Batch/archival workloads only.
Full results: `data/benchmark/results/dual-backbone-benchmark.csv`, `competitors-fullsize-2026-11-01.csv`.

## Competitive position (2026-11-10)

Honest summary of `docs/internal/research/competitive-position-2026-11-10/REPORT.md`:

- No commercial product sells neural lossless compression for text/XML/logs. The category is empty because compute
  costs more than the bytes it saves (CIDR 2026 "Waiting to Decompress": ~2x ratio for >400x cost), not because the
  technique is unknown.
- On ratio Redukt is not the leader: Nacrith (SmolLM2-135M + CM, Apache-2.0, consumer GPU with CPU fallback) reaches
  0.9389 bpb on the full enwik8; the 2026 Hutter Prize lineage (cmix + a pretrained 6M-param transformer) reaches
  0.73-0.78 bpb on enwik9 on a single CPU core at 4.5-9 KB/s, 50-90x Redukt's per-core speed.
- On cost: cmix v21 gives 1.170 bpb for ~$3/GB of spot CPU; Redukt costs ~$50-60/GB on CPU and ~$9-17/GB on a T4 for
  0.96 bpb. xz/zstd cost $0.002-0.007/GB, and the bytes Redukt saves over them are worth $0.06-0.33/GB over 7 years in
  S3 Standard. Parser-based log compressors (CLP, LogReducer, LogShrink) beat Redukt's apache figure at MB/s.
- What is unique: the packaging (single Rust binary with a ≥100M pretrained backbone decodable on CPU), the numeric
  contract declared in the header, and accepting any file type. Redukt is deterministic within a tier; the GPU plugin
  stream differs from the CPU stream (R-104 F2/F3 pending).

Roadmap consequences: `docs/ROADMAP.md` Fase 19 (R-103 GPU kernels, R-105 fast tier, metered-link segments).

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
recommended for batches of 16 or more files. `REDUKT_THREADS=N` sets the worker pool size. With `--backbone plugin`
the lockstep falls back to sequential GPU forwards (batch GPU kernels are FFI-level only, R-103).

## API

`redukt api` has byte-for-byte parity with the CLI (same backbone routing, preprocessing and profiles; verified
sha256 API == CLI). Flags: `--bind/--host`, `--port`, `--weights`, `--backbone`, `--quant`; `/v1/status` reports backbones,
profile, readable formats and version. It serves one request at a time.

## Deploy (GCP)

Reference deployment for Cloud Run (API), Cloud Batch (bulk compression) and GCE with GPU:
[`deploy/gcp/README.md`](deploy/gcp/README.md). Build the CPU image from the repo root with
`docker build -f deploy/gcp/Dockerfile -t redukt:cpu .` (the former root `Dockerfile` was removed; the canonical one is
`deploy/gcp/Dockerfile`). Verified locally (not yet on a live GCP project). The CUDA plugin and its batch interface were
validated on a Kaggle T4 (roundtrip, BPB == CPU); see [`docs/api/PLUGIN-FFI-V2.md`](docs/api/PLUGIN-FFI-V2.md).

## Requirements

- Rust 1.89+
- Mamba-130M weights (default) and RWKV-7 0.1B weights (binary routing), ~400 MB each
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
and storage savings projection. All results are deterministic and CRC-verified. Compare against
whole-file classical output, not 10 KB windows, before drawing conclusions.

## Testing

```bash
# Unit tests (no weights needed) — 295 unit + 5 integration tests
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

Built on [Mamba](https://github.com/state-spaces/mamba) (Albert Gu, Tri Dao; Apache 2.0 weights
`state-spaces/mamba-130m`) and [RWKV-7](https://github.com/BlinkDL/RWKV-LM) by Bo Peng (BlinkDL), licensed under
Apache 2.0. Context Mixing techniques inspired by [PAQ](http://mattmahoney.net/dc/) (Matt Mahoney, public domain).
