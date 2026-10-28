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
2. BPB validation: hybrid-eval enwik8 10KB ≈ 1.28 (Q8 or F32 — difference <2%)
3. Full benchmark matches verified results (see below)
4. NOTICE file up to date
5. README badges current

## Benchmarks (Verified 2026-10-28, actual AZTH compression)

Primary metric: BPB (bits per byte) = `compressed_size × 8 / original_size`. Lower = better.
All Redukt BPB are actual AZTH file output, NOT cross-entropy estimates.
Quantization: auto-selected Q8. F32 vs Q8 difference: <2.5% on all files.
Q8 is 2.1x faster — correct production choice.
Use `--quant f32|q8|q4|auto` to force quantization level.

### Reference BPB (enwik8 10KB)
- **compress (actual AZTH)**: 1.3448 BPB [MEASURED]
- **cross-entropy (hybrid-eval)**: 1.2812 BPB (Q8) | 1.3041 BPB (F32)
- **coder efficiency**: 95.3% on enwik8 (varies 80-95% by file)
- **baseline (RWKV-only)**: 1.4067 BPB

### Verified Results (actual compression, 10KB, roundtrip-verified)
| File | Redukt BPB | brotli-11 | Δ | Coder eff. |
|---|---|---|---|---|
| apache-access.log | **0.309** | 0.466 | **-34%** | 80.3% |
| facturae-batch.xml | **0.462** | 0.890 | **-48%** | 88.4% |
| enwik8 | **1.345** | 2.352 | **-43%** | 95.3% |

### Cross-Entropy BPB (hybrid-eval, 10KB, Q8) — pending re-measurement with actual compression
Note: ce_bpb is 5-20% lower than actual compressed BPB due to coder overhead.

| File | ce_bpb | File | ce_bpb |
|---|---|---|---|
| xml | 0.640 | samba | 1.266 |
| nci | 0.596 | dickens | 1.779 |
| webster | 1.760 | mozilla | 1.965 |
| reymont | 1.808 | mr | 2.283 |
| ooffice | 3.704 | x-ray | 6.524 |

Benchmark data in `data/benchmark/`. Raw CSV in `data/benchmark/results/`.
Reproducible: `redukt benchmark --input <FILE> --output report.json --verify`

## Process Safety (Windows)

- FOREGROUND terminal only — NEVER use run_in_background
- `taskkill //F //IM redukt.exe` kills all instances
- Use `--bytes N` not `--limit N` (silently ignored)
- `cargo build --profile maxperf` for benchmarks
