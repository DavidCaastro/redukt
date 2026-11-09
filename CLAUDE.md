# CLAUDE.md — Redukt

> Authority for engineering decisions in this repository.
> Maximum reasoning (ultrathink) for all analysis and deliverables.

---

## Identity

Redukt: neural compression engine.
Hybrid CM (5 models + adaptive SGD mixer) + RWKV-7/Mamba (pretrained) + arithmetic coding.
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
FLAGS:       1B  (0x01=HP, 0x02=emb, 0x04=E8E9, 0x08=shuffle, 0x10=delta, 0x20=skip, 0x40=backbone)
[BIAS_LR]:   4B  LE f32 (only if FLAGS & 0x01)
[NGRAM_SCALE]:4B  LE f32 (only if FLAGS & 0x01)
[MIX_ETA]:   4B  LE f32 (only if FLAGS & 0x01)
[EMB_METHOD]:1B  (only if FLAGS & 0x02; 0=none,1=center,2=norm,3=spread)
[EMB_ALPHA]: 4B  LE f32 (only if FLAGS & 0x02)
[BACKBONE]:  1B  (only if FLAGS & 0x40; 0=RWKV, 1=Mamba)
STREAM:      var (arithmetic coded probability stream)
CRC32:       4B  LE (CRC-32 IEEE of original data)
```

Header: 13B (defaults) up to 34B (all flags). v3 was 28B (fixed).
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

### BPB Budget for Optimizations
- Maximum allowed BPB regression from any optimization: **+5%**
- Measured on enwik8 10KB (reference) + at least 2 additional files
- Optimizations exceeding +5% BPB MUST be reverted or made opt-in
- Q4 quantization (+12.7%) remains NOT viable under this budget

## Benchmarks (Verified 2026-10-31, dual-backbone actual AZTH compression)

Primary metric: BPB (bits per byte) = `compressed_size × 8 / original_size`. Lower = better.
All Redukt BPB are actual AZTH file output, NOT cross-entropy estimates.
Dual backbone: Mamba-130M (default) + RWKV-7 0.1B (GenericBinary only).
Quantization: auto-selected Q8 applies to RWKV only (F32 vs Q8 difference: <2.5% on all files).
Mamba runs F32 in the hot path (verified 2026-11-09); weight-only Q8 for Mamba is in progress (R-098).
Use `--quant f32|q8|q4|auto` to force quantization, `--backbone rwkv|mamba|auto`.

### Reference BPB (enwik8 10KB)
- **compress (Mamba+CM, AZTH v4)**: 0.9609 BPB [MEASURED: 2026-10-31]
- **compress (RWKV+CM, AZTH v4)**: 1.3312 BPB [MEASURED: 2026-10-30]
- **cross-entropy (hybrid-eval, RWKV)**: 1.2813 BPB (Q8)
- **coder efficiency**: 94.5% on enwik8 Mamba, 96.0% RWKV

### Strongest Results (dual-backbone, actual AZTH, 10KB)
| File | Redukt BPB | brotli-11 | Δ | Backbone |
|---|---|---|---|---|
| apache-access.log | **0.216** | 0.466 | **-54%** | Mamba |
| samba | **0.271** | 2.220 | **-88%** | Mamba |
| facturae-batch.xml | **0.389** | 0.890 | **-56%** | Mamba |
| ubl-invoice-batch.xml | **0.446** | 0.888 | **-50%** | Mamba |
| dickens | **0.652** | 2.778 | **-77%** | Mamba |
| enwik8 | **0.961** | 2.352 | **-59%** | Mamba |

### Full Benchmark (12 files, dual-backbone, 10KB Q8) [MEASURED: 2026-10-31]
| File | BPB | Backbone | vs brotli | vs gzip |
|---|---|---|---|---|
| apache-access.log | 0.216 | Mamba | -54% | -60% |
| facturae-batch.xml | 0.389 | Mamba | -56% | -64% |
| ubl-invoice-batch.xml | 0.446 | Mamba | -50% | -59% |
| nci | 0.609 | Mamba | -17% | -35% |
| dickens | 0.652 | Mamba | -77% | -82% |
| enwik8 | 0.961 | Mamba | -59% | -68% |
| samba | 0.271 | Mamba | -88% | -91% |
| mozilla | 1.938 | Mamba | -4% | -18% |
| mr | 2.391 | RWKV | +1% | -14% |
| ooffice | 3.750 | RWKV | +37% | +29% |
| x-ray | 4.887 | Mamba | +6% | -17% |
| sao | 8.666 | Mamba | +45% | +36% |

Wins 8/12 vs brotli-11, 10/12 vs gzip-9. Average win margin: -50.5% vs brotli.
Loses: sao +45%, ooffice +37%, x-ray +6%, mr +1% (binary/high-entropy).
Full evidence: `data/benchmark/results/dual-backbone-benchmark.md`, `.csv`.

### RWKV-only Benchmark (19 files, 10KB Q8) [MEASURED: 2026-10-30]
Raw CSV: `data/benchmark/results/actual-compression-benchmark.csv`.

### Scale Test (batch mode, RWKV, model loaded once)
| File | 10KB | 50KB | 100KB | brotli 100KB | Δ 100KB |
|---|---|---|---|---|---|
| apache-access.log | 0.296 | 0.227 | 0.224 | 0.328 | -32% |
| dickens | 1.797 | 1.503 | 1.457 | 2.649 | -45% |
| samba | 1.298 | 1.356 | 1.206 | 1.944 | -38% |

Advantage grows with file size. RWKV throughput: 60-69 tok/s. Mamba: ~15-34 tok/s.
Scale CSV: `data/benchmark/results/scale-test-complete.csv`.

### Fase 19 — CPU Limit & Market Viability (2026-11-09)
Audit (`docs/internal/research/throughput-properties-audit.md`) refuted three premises behind "34 tok/s = DRAM ceiling":
1. Mamba runs F32 in the hot path, not Q8 (R-064 PTQ was reverted; weight-only Q8 in progress, R-098).
2. Dev laptop is 2×16 GB DDR4-2667 dual channel = 42.7 GB/s (~44 GB/s measured on identical laptops), not 21.3 GB/s;
   34 tok/s × 516 MB/token = 17.5 GB/s = 41% of the bus.
3. in_proj/out_proj (70% of bytes/token) run single-threaded (R-099 fixes this); the limit is per-core bandwidth.
Economics: ~$47/GB raw today (spot, 130 B/s); 7-year break-even is 18-100 KB/s per vCPU (enwik8 to apache, S3 Standard).
Open blocker: cross-machine determinism (R-104). Plan and gates: ROADMAP Fase 19 (R-098 → R-105).

## Process Safety (Windows)

- FOREGROUND terminal only — NEVER use run_in_background
- `taskkill //F //IM redukt.exe` kills all instances
- Use `--bytes N` not `--limit N` (silently ignored)
- `cargo build --profile maxperf` for benchmarks
