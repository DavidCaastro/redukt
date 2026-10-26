# CLAUDE.md — Redukt

> Authority for engineering decisions in this repository.
> Maximum reasoning (ultrathink) for all analysis and deliverables.

---

## Identity

Redukt: neural compression engine.
Hybrid CM (19 hash-table models + LSTM mixer) + RWKV-7 (pretrained) + arithmetic coding.
CPU-only, zero external dependencies, single Rust binary.

Origin: extracted from original research project.
Purpose: production-grade compression product for batch/archival workloads.

## Product Scope

This repo is COMPRESSION ONLY. No chat, no generation, no QA, no probes.
Commands: compress, decompress, hybrid-eval, baseline, cm-eval, api.

## Architecture

`Domain <- Application <- Infrastructure` (dependency rule).

```
Bytes -> RWKV-7(pretrained) + TokenNgram + BiasHead -> AdaptiveMixer -> RangeEncoder -> AZTH
AZTH -> RangeDecoder + identical model -> original bytes (lossless, deterministic)
```

### AZTH Format (v3)
```
MAGIC:       8B  "AZTH\x03\x00\x00\x00"
TOTAL_BYTES: 4B  LE (original size, validated ≤ u32::MAX)
TOKEN_COUNT: 4B  LE (validated ≤ u32::MAX)
BIAS_LR:     4B  LE f32 (default 0.30, validated finite)
NGRAM_SCALE: 4B  LE f32 (default 0.50, validated finite)
MIX_ETA:     4B  LE f32 (default 0.01, validated finite)
STREAM:      var (arithmetic coded probability stream)
CRC32:       4B  LE (CRC-32 IEEE of original data)
```

Decompression requires IDENTICAL model + weights. Hyperparameters stored in header.
CRC-32 verified on decompression (v3). Decoder also accepts v2 (no CRC, backward compat).
Max file size: ~4 GB (u32 TOTAL_BYTES). Validated at compression time.

## Constraints

### Hard Rules
- NO external crate dependencies (compile-time). Only rustc + stdlib.
- This is a STANDALONE product repo.
- NO push without explicit authorization.
- NO destructive git operations without confirmation.
- NEVER delete weights/ directory or MANIFEST.md.
- NEVER use `run_in_background` — causes ghost tasks on Windows.

### Code Rules
- Edition 2021, zero-dep compilation
- Lines <= 120 chars, nesting <= 3 levels
- Zero dead code (warnings = errors)
- Conventional commits in English: `type(scope): description`
- Co-Authored-By on AI-assisted commits

### Build Profiles
```toml
[profile.release]
opt-level = 3, lto = "fat", codegen-units = 1, panic = "unwind"

[profile.maxperf]
inherits = "release", panic = "abort", strip = true
```

## Legal & Compliance

### License: Apache-2.0
- Compatible with RWKV-7 weights (Apache-2.0, BlinkDL)
- Permissive: allows commercial use, modification, distribution
- Patent grant included (protects users from patent claims)
- Compatible with: MIT, BSD, GPL (one-way), LGPL, MPL-2.0
- Compliant with EU, US, APAC jurisdictions

### NOTICE File Required
Apache-2.0 requires a NOTICE file with:
1. Project name and copyright holder
2. Third-party attributions (RWKV-7 weights)
3. Modifications disclaimer

### Export Control
- NO encryption in this product (compression only)
- No ITAR/EAR classification required
- No dual-use technology designation
- Safe for export to any non-embargoed jurisdiction

### Data Sovereignty
- Compression is a mathematical transform — no data leaves the process
- No telemetry phones home, no external API calls
- Weights are downloaded once, used locally
- Client data never touches our servers (in Transparent Proxy model,
  data transits through our proxy but is never stored)

### GDPR / Privacy
- The compressor processes bytes, not personal data
- No PII extraction, no profiling, no tracking
- If used on personal data: client is data controller, we are processor
- DPA (Data Processing Agreement) template should be provided for EU clients

### Patent Safety
- Arithmetic coding: IBM patents expired 2003-2006
- Range coding (Schindler): no active patents
- RWKV: Apache-2.0 with explicit patent grant
- Context mixing: public domain (Matt Mahoney, PAQ series)
- LSTM: patent-free (Hochreiter & Schmidhuber, 1997, academic)

### Regulatory Compliance by Region
| Region | Requirement | Status |
|---|---|---|
| EU | GDPR, AI Act (not high-risk) | Compliant |
| US | SOX (financial data retention) | Compatible |
| Spain | Ley Crea y Crece, RGPD | Compatible |
| UK | UK GDPR, Data Protection Act | Compliant |
| Japan | APPI | Compatible |
| Brazil | LGPD | Compatible |
| Australia | Privacy Act | Compatible |
| China | PIPL, DSL | Requires local deployment review |
| India | DPDP Act 2023 | Compatible |

### Certification Path (Future)
- SOC 2 Type II: for enterprise/financial clients
- ISO 27001: for healthcare/government
- HIPAA BAA: for US medical imaging (requires encryption layer)

## Weights Policy

Weights stored in `weights/` (gitignored).
Download via `scripts/download-weights.sh`.
Provenance documented in `weights/MANIFEST.md`.

**NEVER delete rwkv7-0.1b** — all baselines depend on it.

## Quality Gates

### Before Any Commit
1. `cargo build --release` — zero warnings
2. `cargo test` — all pass (especially coder roundtrip tests)
3. Roundtrip: compress -> decompress -> diff (byte-identical)
4. No new `unwrap()`/`panic!()` in codec or coder paths (see ROADMAP R-006)

### Before Any Release
1. All quality gates above
2. BPB validation: hybrid-eval enwik8 10KB ≈ 1.28 (Q8) or 1.15 (F32)
3. Full Silesia benchmark matches verified results (see below)
4. NOTICE file up to date
5. README badges current

## Benchmarks (Verified 2026-10-26)

Primary metric: BPB (bits per byte). Lower = better.
Quantization: auto-selected Q8 (0.163% relative error, 4x less memory than F32).

### Reference BPB (enwik8 10KB, Q8 quantization)
- **hybrid-eval**: 1.2812 BPB (full ensemble)
- **compress**: 1.3448 BPB (includes coder overhead)
- **coder efficiency**: 95.27% (overhead = 0.064 BPB)
- **baseline (RWKV-only)**: 1.4067 BPB

Note: Previous claim of 1.1535 BPB was measured with F32 weights + experimental
features (--hierarchical --match --emb-surgery). The production pipeline uses Q8
and default hyperparameters. Do NOT quote 1.1535 as a production number.

### Silesia Corpus BPB (10KB, Q8)
| File | BPB | File | BPB |
|---|---|---|---|
| xml | 0.640 | samba | 1.266 |
| nci | 0.596 | dickens | 1.779 |
| webster | 1.760 | mozilla | 1.965 |
| reymont | 1.808 | mr | 2.283 |
| ooffice | 3.704 | osdb | 5.643 |
| x-ray | 6.524 | sao | 9.241 |

### vs Competitors (10KB window, verified roundtrip)
| File | Redukt | brotli-11 | Δ |
|---|---|---|---|
| enwik8 | **1.34** | 2.36 | -43% |
| xml | **0.69** | 0.89 | -22% |
| samba | **1.31** | 2.22 | -41% |
| apache.log | **0.30** | 0.46 | -35% |

Benchmark data in `data/benchmark/`. Reproducible via Python scripts.
Results CSV in `data/benchmark/results/`.

## Process Safety (Windows)

- FOREGROUND terminal only — NEVER use run_in_background
- `taskkill //F //IM redukt.exe` kills all instances
- Use `--bytes N` not `--limit N` (silently ignored)
- `cargo build --profile maxperf` for benchmarks
