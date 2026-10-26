# Redukt

Neural compression engine that achieves state-of-the-art lossless compression
by combining a pretrained language model (RWKV-7) with online-adaptive Context
Mixing and arithmetic coding.

## Highlights

- **Best-in-class compression**: 44.7% smaller than brotli-11 on XML/markup, 37.2% on text
- **25 domains validated**: Silesia corpus, enwik8, modern formats (JSON, Parquet, WASM)
- **Zero dependencies**: Pure Rust, single binary, CPU-only
- **Deterministic**: Bit-identical decompression guaranteed

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

## Performance

| Domain | BPB | vs brotli-11 |
|---|---|---|
| XML/markup | 0.31 | -44.7% |
| Text (English) | 1.13 | -37.2% |
| Source code | 1.04 | -28.5% |
| Binary/executables | 1.40 | -15.2% |
| Structured data | 1.82 | -22.1% |

Throughput: 27–72 B/s (text), 17–22 B/s (binary). Designed for batch/archival workloads.

## Requirements

- Rust 1.70+
- RWKV-7 0.1B weights (~400 MB) — run `bash scripts/download-weights.sh`
- 4+ GB RAM

## Testing

```bash
# Unit tests (no weights needed)
cargo test

# End-to-end roundtrip: compress → decompress → verify byte-identical (requires weights)
cargo test --release --test roundtrip -- --ignored
```

## Format

Output format: AZTH v2 (`.azth`). See `docs/api/FORMAT-SPEC.md` for full specification.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Built on [RWKV-7](https://github.com/BlinkDL/RWKV-LM) by Bo Peng (BlinkDL),
licensed under Apache 2.0. Context Mixing techniques inspired by
[PAQ](http://mattmahoney.net/dc/) (Matt Mahoney, public domain).
