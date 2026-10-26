# Redukt

Neural compression engine that achieves state-of-the-art lossless compression
by combining a pretrained language model (RWKV-7) with online-adaptive Context
Mixing and arithmetic coding.

## Highlights

- **Best-in-class compression**: 43% smaller than brotli-11 on enwik8, 35% on logs
- **17 domains validated**: Silesia corpus (12 files), enwik8, industry files (5)
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

## Performance (Verified, Q8 quantization, 10KB window)

| File | Domain | Redukt BPB | brotli-11 | vs brotli |
|---|---|---|---|---|
| enwik8 | Text/encyclopedia | **1.34** | 2.36 | -43% |
| dickens | Text/literature | **1.82** | 2.77 | -34% |
| samba | Source code | **1.31** | 2.22 | -41% |
| xml | XML markup | **0.69** | 0.89 | -22% |
| apache.log | Server logs | **0.30** | 0.46 | -35% |
| ubl-invoice | E-invoicing | **0.55** | 0.87 | -37% |

Throughput: ~25 tok/s (~80 B/s text, ~20 B/s binary). Designed for batch/archival workloads.
Coder efficiency: 90–97% (overhead vs theoretical entropy).

## Requirements

- Rust 1.70+
- RWKV-7 0.1B weights (~400 MB) — run `bash scripts/download-weights.sh`
- 4+ GB RAM

## Testing

```bash
# Unit tests (no weights needed) — 61 tests
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
