# Migration Guide — Redukt → Redukt

Complete instructions for creating the `Redukt` repository from Redukt
without modifying the original codebase.

## 1. Dependency Map (Verified)

### codec.rs (compress/decompress) — Direct Dependencies

```
application/codec.rs
  ├── domain/error.rs        (parse_arg, next_arg, read_file, write_file)
  ├── domain/tensor.rs       (softmax, Tensor)
  ├── domain/ngram.rs        (TokenNgram)
  ├── domain/bias_head.rs    (BiasHead)
  ├── domain/mixer.rs        (AdaptiveMixer)
  ├── domain/coder.rs        (Cdf, RangeEncoder, RangeDecoder)
  ├── application/model_loader.rs
  │     ├── infrastructure/rwkv7/model.rs  (Rwkv7Model, load_auto)
  │     └── infrastructure/rwkv7/tokenizer.rs (WorldTokenizer)
  └── infrastructure/rwkv7/model.rs (Rwkv7State, Scratch)
        ├── domain/tensor.rs
        ├── domain/quant.rs (QuantMatrix, quantize_matrix)
        ├── domain/ram_monitor.rs (RAM guard)
        ├── infrastructure/rwkv7/safetensors.rs
        └── infrastructure/rwkv7/gguf.rs
```

### hybrid_eval.rs (BPB validation) — Direct Dependencies

```
application/hybrid_eval.rs
  ├── domain/error.rs
  ├── domain/cm.rs (ContextMixer — 19 hash models + LSTM mixer)
  │     ├── domain/lstm_mixer.rs
  │     └── (self-contained hash tables)
  ├── domain/bridge.rs (token→byte trie)
  │     └── infrastructure/rwkv7/tokenizer.rs
  ├── domain/backbone.rs (ByteBackbone trait, RwkvBackbone, Orchestrator)
  │     ├── domain/tensor.rs
  │     ├── domain/bridge.rs
  │     └── infrastructure/rwkv7/model.rs + tokenizer.rs
  ├── domain/match_model.rs (longest-match predictor)
  ├── domain/preprocess.rs (E8/E9 transform)
  └── application/telemetry.rs (progress + JSONL logging)
```

### baseline.rs (RWKV-only reference) — Direct Dependencies

```
application/baseline.rs
  ├── domain/error.rs, tensor.rs, ngram.rs, bias_head.rs, mixer.rs
  ├── application/telemetry.rs, model_loader.rs
  └── infrastructure/rwkv7/model.rs (Rwkv7State)
```

### cm_eval.rs (CM-only reference) — Direct Dependencies

```
application/cm_eval.rs
  ├── domain/error.rs, cm.rs, preprocess.rs
```

## 2. Files to COPY (Exact List)

### Application Layer (5 files)

| Source | Purpose |
|---|---|
| `src/application/codec.rs` | Core product: compress + decompress |
| `src/application/model_loader.rs` | Model + tokenizer loader |
| `src/application/hybrid_eval.rs` | BPB measurement (proves claims) |
| `src/application/baseline.rs` | RWKV-only baseline reference |
| `src/application/cm_eval.rs` | CM-only evaluation |
| `src/application/telemetry.rs` | Progress + JSONL logging |

### Domain Layer (15 files)

| Source | Purpose |
|---|---|
| `src/domain/error.rs` | Error types, CLI arg parsing, file I/O |
| `src/domain/tensor.rs` | Tensor, softmax, matmul, SIMD ops |
| `src/domain/ngram.rs` | Token N-gram predictor |
| `src/domain/bias_head.rs` | Online bias correction |
| `src/domain/mixer.rs` | Adaptive component mixer |
| `src/domain/coder.rs` | Range coder (arithmetic coding) |
| `src/domain/cm.rs` | Context Mixer (19 models + LSTM) |
| `src/domain/lstm_mixer.rs` | LSTM bit-level mixer |
| `src/domain/match_model.rs` | Longest-match predictor |
| `src/domain/bridge.rs` | Token-to-byte trie bridge |
| `src/domain/backbone.rs` | ByteBackbone trait + orchestrator |
| `src/domain/preprocess.rs` | E8/E9 preprocessing |
| `src/domain/state_io.rs` | CM state serialization |
| `src/domain/quant.rs` | Quantization (Q4/Q8/F32) |
| `src/domain/ram_monitor.rs` | RAM safety guard |

### Infrastructure Layer (4 files)

| Source | Purpose |
|---|---|
| `src/infrastructure/rwkv7/model.rs` | RWKV-7 forward pass, 6 loaders |
| `src/infrastructure/rwkv7/tokenizer.rs` | World tokenizer (65K vocab) |
| `src/infrastructure/rwkv7/safetensors.rs` | SafeTensors weight reader |
| `src/infrastructure/rwkv7/gguf.rs` | GGUF weight reader + quant |

### Total: 25 source files

## 3. Files to NOT COPY (Explicitly Excluded)

| File | Reason |
|---|---|
| `src/application/generate.rs` | Chat/generation only |
| `src/application/chat_cmd.rs` | Chat interface |
| `src/application/probe_cmd.rs` | Research probes |
| `src/application/steer_sweep.rs` | Research sweeps |
| `src/application/offset_sweep.rs` | Research sweeps |
| `src/application/state_sweep.rs` | Research sweeps |
| `src/application/extract_cmd.rs` | QA extraction |
| `src/application/qa_cmd.rs` | QA cascade |
| `src/application/benchmark_cmd.rs` | QA benchmark |
| `src/application/rwkv_test.rs` | Developer test harness |
| `src/domain/chat.rs` | Chat sampling/penalties |
| `src/domain/crg.rs` | Relevance gate (chat) |
| `src/domain/extraction.rs` | QA extraction engine |
| `src/domain/cascade.rs` | QA cascade protocol |
| `src/domain/probes.rs` | Research probes |
| `src/domain/spectral.rs` | Weight analysis (research) |
| `src/domain/intrinsic.rs` | Intrinsic telemetry (research) |

## 4. New Files to CREATE

### `src/main.rs` (new, minimal)

```rust
mod domain;
mod application;
mod infrastructure;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "compress" => crate::application::codec::cmd_compress(&args[2..]),
        "decompress" => crate::application::codec::cmd_decompress(&args[2..]),
        "hybrid-eval" => crate::application::hybrid_eval::cmd_hybrid_eval(&args[2..]),
        "baseline" => crate::application::baseline::cmd_baseline(&args[2..]),
        "cm-eval" => crate::application::cm_eval::cmd_cm_eval(&args[2..]),
        "version" => print_version(),
        _ => print_usage(),
    }
}

fn print_usage() {
    eprintln!("Redukt — neural compression engine");
    eprintln!("  https://github.com/<org>/Redukt");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  compress    --input PATH --output PATH [--weights DIR] [--bytes N]");
    eprintln!("  decompress  --input PATH --output PATH [--weights DIR]");
    eprintln!("  hybrid-eval --input PATH [--weights DIR] [--bytes N] [options]");
    eprintln!("  baseline    --input PATH [--weights DIR] [--bytes N] [--ensemble]");
    eprintln!("  cm-eval     --input PATH [--bytes N] [--e8e9]");
    eprintln!("  version     Show version and license info");
}

fn print_version() {
    eprintln!("Redukt {}", env!("CARGO_PKG_VERSION"));
    eprintln!("License: Apache-2.0");
    eprintln!("RWKV-7 weights: Apache-2.0 (BlinkDL)");
    eprintln!("Format: AZTH v1 (proprietary arithmetic stream)");
}

#[cfg(test)]
mod tests {
    #[test]
    fn skeleton_compiles() {
        assert!(true);
    }
}
```

### `src/application/mod.rs` (new, trimmed)

```rust
pub mod telemetry;
pub mod model_loader;
pub mod hybrid_eval;
pub mod codec;
pub mod baseline;
pub mod cm_eval;
```

### `src/domain/mod.rs` (new, trimmed)

```rust
pub mod error;
pub mod tensor;
pub mod quant;
pub mod ngram;
pub mod bias_head;
pub mod mixer;
pub mod coder;
pub mod cm;
pub mod bridge;
pub mod lstm_mixer;
pub mod match_model;
pub mod preprocess;
pub mod state_io;
pub mod backbone;
pub mod ram_monitor;
```

### `src/infrastructure/mod.rs` (unchanged)

```rust
pub mod rwkv7;
```

### `src/infrastructure/rwkv7/mod.rs` (unchanged)

```rust
pub mod safetensors;
pub mod gguf;
pub mod model;
pub mod tokenizer;
```

### `Cargo.toml`

```toml
[package]
name = "Redukt"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"
description = "Neural compression engine — hybrid CM + RWKV-7 arithmetic coding"
repository = "https://github.com/<org>/Redukt"
readme = "README.md"
keywords = ["compression", "neural", "arithmetic-coding", "rwkv"]
categories = ["compression", "command-line-utilities"]

[[bin]]
name = "Redukt"
path = "src/main.rs"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "unwind"

[profile.maxperf]
inherits = "release"
panic = "abort"
strip = true
```

### `.cargo/config.toml`

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

### `.gitignore`

```
/target
/weights/*.bin
/weights/*.gguf
/weights/*.pth
/weights/*.safetensors
/weights/rwkv7-*/
*.ckpt
*.jsonl
*.log
*.pdb
nul.*
*.rcgu.o
# IDE
.vscode/
.idea/
# OS
.DS_Store
Thumbs.db
```

## 5. Weights Setup

The model requires RWKV-7 0.1B weights (~400 MB). These are NOT included in the repo.

### Setup Script: `scripts/download-weights.sh`

```bash
#!/bin/bash
# Download RWKV-7 0.1B weights for Redukt
# Source: HuggingFace BlinkDL/rwkv-7-world (Apache 2.0)
set -euo pipefail

WEIGHTS_DIR="weights/rwkv7-0.1b"
mkdir -p "$WEIGHTS_DIR"

echo "Downloading RWKV-7 0.1B World weights (Apache 2.0)..."
echo "Source: https://huggingface.co/BlinkDL/rwkv-7-world"

# SafeTensors format (required for Redukt)
# Files needed:
#   model.safetensors (~400 MB)
#   rwkv_vocab_v20230424.txt (~2 MB)

if [ ! -f "$WEIGHTS_DIR/model.safetensors" ]; then
    echo "Download model.safetensors from HuggingFace and place in $WEIGHTS_DIR/"
    echo "  https://huggingface.co/BlinkDL/rwkv-7-world/tree/main"
    echo ""
    echo "Or use huggingface-cli:"
    echo "  pip install huggingface-hub"
    echo "  huggingface-cli download BlinkDL/rwkv-7-world --local-dir $WEIGHTS_DIR"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/rwkv_vocab_v20230424.txt" ]; then
    echo "ERROR: Missing tokenizer vocab file: $WEIGHTS_DIR/rwkv_vocab_v20230424.txt"
    exit 1
fi

echo "Weights ready in $WEIGHTS_DIR/"
echo "Verify: cargo run --release -- compress --input data/test.txt --output /dev/null --weights $WEIGHTS_DIR"
```

### `weights/MANIFEST.md`

```markdown
# Weights Manifest — Redukt

## Required Weights

| Model | Format | Size | Source | License |
|---|---|---|---|---|
| RWKV-7 0.1B World | SafeTensors | ~400 MB | HuggingFace BlinkDL/rwkv-7-world | Apache 2.0 |

## Files Expected

```
weights/rwkv7-0.1b/
  model.safetensors       (~400 MB)
  rwkv_vocab_v20230424.txt (~2 MB)
```

## Provenance

- Author: BlinkDL (Bo Peng)
- Paper: RWKV-7 "Goose" (2025)
- Vocab: 65,536 tokens (World tokenizer, multilingual)
- Head size: 64
- Layers: varies by model size

## NEVER delete these weights — all baselines depend on them.
```

## 6. Validation Procedure

After copying all files and creating the new repo:

### Step 1: Compile

```bash
cargo build --release
# Must compile with zero warnings, zero errors, zero dependencies
```

### Step 2: Roundtrip Test (Correctness)

```bash
# Create test file
echo "Hello, this is a compression roundtrip test for Redukt." > /tmp/test.txt

# Compress
cargo run --release -- compress --input /tmp/test.txt --output /tmp/test.azth --weights weights/rwkv7-0.1b

# Decompress
cargo run --release -- decompress --input /tmp/test.azth --output /tmp/test.decoded.txt --weights weights/rwkv7-0.1b

# Verify byte-identical
diff /tmp/test.txt /tmp/test.decoded.txt
# Must produce NO output (files identical)
```

### Step 3: BPB Validation (Performance)

```bash
# Must match Redukt's 1.15 BPB on enwik8 10KB
cargo run --release -- hybrid-eval \
  --input data/benchmark/samples/enwik8-text.10k \
  --weights weights/rwkv7-0.1b \
  --bytes 10240 \
  --hierarchical --match --emb-surgery center0.3

# Expected: BPB ~1.1535 (±0.01)
```

### Step 4: Unit Tests

```bash
cargo test
# domain/coder.rs has 5 roundtrip tests that MUST pass
```

## 7. Directory Structure of New Repo

```
Redukt/
├── CLAUDE.md                     # AI assistant instructions
├── Cargo.toml
├── LICENSE                       # Apache-2.0
├── NOTICE                        # Third-party attributions
├── README.md
├── .cargo/
│   └── config.toml
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md           # Compression pipeline description
│   ├── FORMAT.md                 # AZTH format specification
│   └── BENCHMARKS.md             # Benchmark methodology + results
├── data/
│   └── benchmark/
│       ├── samples/              # 18 x 10KB test files
│       ├── results/              # CSV results
│       ├── run_Redukt.py
│       ├── run_standard.py
│       └── run_standard_full.py
├── scripts/
│   └── download-weights.sh
├── weights/
│   ├── MANIFEST.md
│   └── rwkv7-0.1b/              # (gitignored, downloaded separately)
└── src/
    ├── main.rs                   # 5 commands: compress, decompress, hybrid-eval, baseline, cm-eval
    ├── application/
    │   ├── mod.rs                # 6 modules
    │   ├── codec.rs              # Core: compress + decompress
    │   ├── model_loader.rs
    │   ├── hybrid_eval.rs
    │   ├── baseline.rs
    │   ├── cm_eval.rs
    │   └── telemetry.rs
    ├── domain/
    │   ├── mod.rs                # 15 modules
    │   ├── error.rs
    │   ├── tensor.rs
    │   ├── coder.rs              # Range coder (arithmetic)
    │   ├── ngram.rs
    │   ├── bias_head.rs
    │   ├── mixer.rs
    │   ├── cm.rs                 # 19-model context mixer
    │   ├── lstm_mixer.rs
    │   ├── match_model.rs
    │   ├── bridge.rs
    │   ├── backbone.rs
    │   ├── preprocess.rs
    │   ├── state_io.rs
    │   ├── quant.rs
    │   └── ram_monitor.rs
    └── infrastructure/
        ├── mod.rs
        └── rwkv7/
            ├── mod.rs
            ├── model.rs
            ├── tokenizer.rs
            ├── safetensors.rs
            └── gguf.rs
```

## 8. Migration Checklist

```
[ ] 1. Create repo: git init Redukt
[ ] 2. Copy 25 source files (see Section 2)
[ ] 3. Create 5 new files: main.rs, 3x mod.rs, Cargo.toml (see Section 4)
[ ] 4. Create .cargo/config.toml, .gitignore (see Section 4)
[ ] 5. Copy LICENSE (Apache-2.0), create NOTICE
[ ] 6. Create CLAUDE.md (see CLAUDE-TEMPLATE.md)
[ ] 7. Copy data/benchmark/ (samples + scripts + results)
[ ] 8. Create weights/MANIFEST.md, scripts/download-weights.sh
[ ] 9. Download weights: scripts/download-weights.sh
[ ] 10. cargo build --release (zero warnings)
[ ] 11. cargo test (5 coder tests pass)
[ ] 12. Roundtrip test (compress → decompress → diff)
[ ] 13. BPB validation (hybrid-eval enwik8 10KB = 1.1535 ±0.01)
[ ] 14. git add, commit, push
[ ] 15. Verify Redukt is UNTOUCHED (git status shows no changes)
```
