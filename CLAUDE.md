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

### AZTH Format (v4 — current)
```
MAGIC:       8B  "AZTH\x04\x00\x00\x00"
TOTAL_BYTES: 4B  LE (original size, validated ≤ u32::MAX)
FLAGS:       1B  (0x00=defaults, 0x01=custom HP, 0x02=emb-surgery)
[BIAS_LR]:   4B  LE f32 (only if FLAGS & 0x01)
[NGRAM_SCALE]:4B  LE f32 (only if FLAGS & 0x01)
[MIX_ETA]:   4B  LE f32 (only if FLAGS & 0x01)
[EMB_METHOD]:1B  (only if FLAGS & 0x02; 0=none,1=center,2=norm,3=spread)
[EMB_ALPHA]: 4B  LE f32 (only if FLAGS & 0x02)
STREAM:      var (arithmetic coded probability stream)
CRC32:       4B  LE (CRC-32 IEEE of original data)
```

Header: 13B (defaults) or 18B (emb-surgery) or 25B (custom HP) or 30B (both). v3 was 28B (fixed).
TOKEN_COUNT removed in v4 — decoder terminates by byte count.
Decompression requires IDENTICAL model + weights.
CRC-32 verified on decompression (v3+). Decoder accepts v2, v3, and v4.
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
- **compress (actual AZTH v4)**: 1.3328 BPB [MEASURED: R-018]
- **cross-entropy (hybrid-eval)**: 1.2812 BPB (Q8) | 1.3041 BPB (F32)
- **coder efficiency**: 96.1% on enwik8 (v4 header saves 15 bytes vs v3)
- **baseline (RWKV-only)**: 1.4067 BPB

### Strongest Results (actual AZTH compression, 10KB)
| File | Redukt BPB | brotli-11 | Δ | Coder eff. |
|---|---|---|---|---|
| apache-access.log | **0.309** | 0.466 | **-34%** | 80.3% |
| facturae-batch.xml | **0.462** | 0.890 | **-48%** | 88.4% |
| ubl-invoice-batch.xml | **0.558** | 0.888 | **-37%** | 87.9% |
| samba | **1.319** | 2.220 | **-41%** | 96.1% |
| enwik8 | **1.345** | 2.352 | **-43%** | 95.3% |

### Full Silesia + Industry (18 files, actual compression, 10KB Q8)
| File | BPB | vs brotli | File | BPB | vs brotli |
|---|---|---|---|---|---|
| apache-access.log | 0.309 | -34% | samba | 1.319 | -41% |
| linux-syslog.log | 0.426 | -27% | enwik8 | 1.345 | -43% |
| facturae-batch.xml | 0.462 | -48% | dickens | 1.829 | -34% |
| openssh-auth.log | 0.538 | -31% | webster | 1.809 | -36% |
| ubl-invoice-batch.xml | 0.558 | -37% | reymont | 1.918 | -10% |
| nci | 0.680 | -7% | mozilla | 2.063 | +2% |
| xml | 0.705 | -22% | mr | 2.402 | +1% |

Loses: ooffice +38%, osdb +22%, x-ray +44%, sao +55%.
Wins 12/18 files. Coder efficiency: 80-99% (lower on more compressible files).
Raw CSV: `data/benchmark/results/actual-compression-benchmark.csv`.
Reproducible: `redukt benchmark --input <FILE> --output report.json --verify`

## Process Safety (Windows)

- FOREGROUND terminal only — NEVER use run_in_background
- `taskkill //F //IM redukt.exe` kills all instances
- Use `--bytes N` not `--limit N` (silently ignored)
- `cargo build --profile maxperf` for benchmarks
