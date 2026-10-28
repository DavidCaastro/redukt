# Redukt

Neural compression engine that achieves state-of-the-art lossless compression
by combining a pretrained language model (RWKV-7) with online-adaptive Context
Mixing and arithmetic coding.

## Highlights

- **Best-in-class compression**: 48% smaller than brotli-11 on e-invoicing, 34% on server logs
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

## Performance (Verified Q8, 10KB window, actual AZTH compression)

All BPB = `compressed_size × 8 / original_size` (actual file output, not cross-entropy estimate).

| File | Domain | Redukt BPB | brotli-11 | vs brotli |
|---|---|---|---|---|
| apache-access.log | Server logs | **0.31** | 0.47 | **-34%** |
| facturae-batch.xml | E-invoicing | **0.46** | 0.89 | **-48%** |
| enwik8 | Text/encyclopedia | **1.34** | 2.35 | **-43%** |

Roundtrip-verified (byte-identical decompression, CRC-32 checked).
Coder efficiency: 80-95% depending on file compressibility (header overhead dominates on very small compressed outputs).
Throughput: ~47 tok/s Q8 SIMD (~150 B/s). Batch/archival workloads.
F32 vs Q8: <2.5% BPB difference, Q8 is 2.1x faster. See `data/benchmark/results/`.

## Requirements

- Rust 1.70+
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

Output format: AZTH v3 (`.azth`). See `docs/api/FORMAT-SPEC.md` for full specification.
Includes CRC-32 integrity verification, hyperparameter validation, and u32 overflow protection.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Built on [RWKV-7](https://github.com/BlinkDL/RWKV-LM) by Bo Peng (BlinkDL),
licensed under Apache 2.0. Context Mixing techniques inspired by
[PAQ](http://mattmahoney.net/dc/) (Matt Mahoney, public domain).
