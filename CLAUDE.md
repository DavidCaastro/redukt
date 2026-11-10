# CLAUDE.md — Redukt

> Authority for engineering decisions in this repository.
> Maximum reasoning (ultrathink) for all analysis and deliverables.

---

## Identity

Redukt: neural compression engine.
Hybrid CM (5 models + adaptive SGD mixer) + RWKV-7/Mamba (pretrained) + arithmetic coding.
CPU-only, zero external dependencies, single Rust binary.

Origin: extracted from original research project.
Purpose: production-grade compression product for batch workloads where bytes are expensive
(hot object storage, metered links, regulated retention). Deep/cold archive is NOT a target.

## Commercial Target (2026-11-10)

Viability metric: compute cost per raw GB must be below the client's cumulative saving over its
retention horizon (model: `scripts/breakeven.py`). Reference thresholds (7 years, spot compute):
<= $0.33/GB for text in hot object storage (enwik8-like), <= $0.12/GB for e-invoice XML,
<= $0.06/GB for logs. Bytes saved in cold/deep archive ($0.001/GB-month) are worth ~0.
Measured status (2026-11-10): CPU lockstep ~1,180 B/s on 12 vCPU = $50-60/GB (spot) / ~$19 (dedicated);
GPU T4 batch N=128 ~4.2 KB/s = $9-17/GB. Gap to threshold: 30x-500x.
Kaggle re-measurement (2026-11-10, `data/benchmark/results/kaggle-fullsize/README.md`): 4-vCPU Xeon lockstep Q8 69 B/s per vCPU
= $56/GB (c7i spot); T4 through the CLI was single-stream only (405 B/s = $169/GB) until R-109 wired the FFI v2
slot batch into the lockstep (CLI aggregate on T4 pending measurement; FFI-level N=128 gave $9-17/GB).
Admitted paths to close it (every new phase must justify itself against one of them):
1. GPU kernels at cuBLAS/tensor-core efficiency. The binding T4 limit at N=128 is recurrent-state traffic (~4.8 MB per
   token and stream in F32, ~55k tok/s ceiling), not the 65 TFLOPS: the plugin's 1,117 tok/s is ~2% of that ceiling and
   the realistic headroom with kernels at 30-50% of roofline is 15-25x (not 60x). Reaches the text threshold, not logs.
2. Fast tier: 3-30M-param backbone + CM recipe (R-105). External references, correctly attributed: L3TC-3.2M 1.105 bpb
   enwik8 at 2.5 MB/s on an A100; L3TC-200K 1.404 bpb at 11-27 KB/s on an iPhone CPU; Hutter 2026 lineage (cmix + 6M
   pretrained transformer) 0.78 bpb enwik9 at 7.5 KB/s on one CPU core. Gate = KB/s per vCPU vs R* (`breakeven.py`).
3. Segments where bytes are expensive to move or retention is regulated (satellite/IoT links,
   egress, compliance), where the threshold is far lower than storage savings.
The metric is cost per GB (throughput per vCPU or per GPU-hour), NOT throughput per machine.
Competitive reference (2026-11-10, `docs/internal/research/competitive-position-2026-11-10/REPORT.md`): no
commercial neural lossless compressor exists for text/XML/logs (the category is empty for cost reasons, CIDR 2026
"Waiting to Decompress");
Nacrith 0.9389 bpb on full enwik8 (GPU, CPU fallback, binaries via LZMA); cmix v21 1.170 bpb at ~$3/GB; Hutter 2026
lineage 0.73-0.78 bpb enwik9 at 4.5-9 KB/s single core (~$0.65-1.07/GB); xz/zstd $0.002-0.007/GB. Redukt is unique only
in packaging (single zero-dep binary, CPU-decodable 130M backbone) and in the header-declared numeric contract.

## Portability Principle (hardware-agnostic by design)

The product must be demonstrable on CPU and on any accelerator (GPU today, TPU/NPU/other tomorrow)
through the same format and the same CLI/API. The goal is to find the best operating point per
deployment, never to discard a technology. Rules:
- CPU is the reference implementation: always functional, always able to decode anything produced
  anywhere. An accelerator is an addition (plugin, FFI v1/v2), never a replacement or a dependency.
- Any new architecture plugs in via the `Backbone` trait or the plugin ABI (docs/api/PLUGIN-FFI-V2.md)
  without touching the core; it must pass the same gates (roundtrip + CRC, BPB vs CPU reference,
  determinism within its own tier, batch == single).
- Numeric tiers are declared in the AZTH header (PROFILE: quant + numeric contract) so that
  encoder/decoder mismatches fail early; bit-identity across tiers is a goal (R-104), not a
  precondition for shipping a new tier.
- Optimizations are evaluated per architecture: a lever that is useless on one (e.g. Q6 on a
  compute-bound CPU) may be the right one on another. Document the operating point, keep the option.
- Deployment artifacts (deploy/) must stay buildable for every supported target (CPU image, CUDA
  image, future accelerators) from the same source tree.

## Product Scope

This repo is COMPRESSION ONLY. No chat, no generation, no QA, no probes.
Commands: compress, decompress, hybrid-eval, baseline, cm-eval, api.

## Architecture

`Domain <- Application <- Infrastructure` (dependency rule).

```
Bytes -> RWKV-7(pretrained) + TokenNgram + BiasHead -> AdaptiveMixer -> RangeEncoder -> AZTH
AZTH -> RangeDecoder + identical model -> original bytes (lossless, deterministic)
```

### AZTH Format (v4 — F32 profile; v5 adds PROFILE, see below)
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
CRC-32 verified on decompression (v3+). Decoder accepts v2 to v5 (the encoder always writes v5, see below).
Max file size: ~4 GB (u32 TOTAL_BYTES). Validated at compression time.

### AZTH Format v5 (weight profile + numeric contract — R-098, R-104; current)
v4 plus one PROFILE byte right after FLAGS. Authoritative layout: `docs/api/FORMAT-SPEC.md`.
```
MAGIC:       8B  "AZTH\x05\x00\x00\x00"
TOTAL_BYTES: 4B  LE
FLAGS:       1B  (same bits as v4)
PROFILE:     1B  (bits 0-3 QUANT: 0=F32, 1=Q8, 2=Q6 reserved;
                  bits 4-5 NUMERIC_CONTRACT: 0=legacy libm, 1=fmath-v1, 2=fma-v1, 3=error before model load;
                  bit 6 RESET_PERIOD -> 1 B RESET_LOG2 follows (R-108); bit 7 reserved)
[optional blocks as v4]  STREAM  CRC32 (4B)
```
Header: 14B up to 36B. R-108 (2026-11-10): PROFILE bit 6 = RESET_PERIOD, followed by 1 B RESET_LOG2 (8..=20); the
Mamba family writes it by default (12 = 4096 forwards: PROFILE 0x50 F32 / 0x51 Q8), `--reset-every 0` omits it,
RWKV/plugin never. Decoder applies the period from the header; pre-R-108 decoders reject it up front.
Since R-104 F1 (51dafd9) the encoder ALWAYS writes v5: F32 -> PROFILE 0x10, Q8 -> 0x11
(contract 1 = pure-Rust `fmath` exp/log, `src/domain/fmath.rs`). The coded stream is identical to v4/libm (6/6 cases);
only the header changes (+1 B for F32). v2-v4 and early v5 (bits 4-5 = 0) decode via the legacy libm branch.
The decoder rejects a conflicting `--quant`; `decompress` takes profile and contract from the header.

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
3b. If lockstep/GEMM/threading code changed: batch `.azth` sha256 == single-file `.azth` sha256 on all reference files
4. No new `unwrap()`/`panic!()` in codec or coder paths (see ROADMAP R-006)

### Before Any Release
1. All quality gates above
2. BPB validation: hybrid-eval enwik8 10KB ≈ 1.28 (Q8 or F32 — difference <2%)
2b. Cross-OS gate (R-104): same sha256 and byte-identical roundtrip Windows <-> Linux (x86-64-v3) on
    enwik8/facturae/apache, F32 and Q8
2c. API == CLI gate (R-102): `redukt api` compress output sha256 == CLI output sha256 (enwik8 10 KB, Q8 and F32)
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
Quantization: `--quant auto` = Q8 for Mamba since R-098 (ae7297a, `model_loader::AUTO_MAMBA_PROFILE`); RWKV keeps its
own auto Q8 (F32 vs Q8 difference: <2.5% on all files). `--quant f32` is recommended for batches >= 16 (~1% better BPB,
same aggregate throughput).
Use `--quant f32|q8|q4|auto` to force quantization, `--backbone rwkv|mamba|auto`.

### Reference BPB (enwik8 10KB)
- **compress (Mamba+CM F32, AZTH v5 + R-108 RESET_LOG2, `--bytes 10240`)**: 0.9648 BPB (1,235 B) [MEASURED: 2026-11-10]
  (current reference; 0.9641 before R-108 and 0.9633 under v4 — same coded stream, +1 B header each time)
- **compress (Mamba+CM Q8, AZTH v5 + R-108)**: 0.9695 BPB (1,241 B; 0.9688 before R-108) [MEASURED: 2026-11-10]
- **compress (Mamba+CM Q8, enwik8 100 KB, default reset 4096)**: 0.9951 BPB (12,737 B) [MEASURED: 2026-11-10];
  `--reset-every 0` reproduces the pre-R-108 1.6327 (20,899 B)
- compress (Mamba+CM, older binary): 0.9609 BPB [MEASURED: 2026-10-31] — outdated reference, not a platform
  divergence (R-104); the figures in the tables below were measured with that older binary
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

### Full Benchmark (12 files, 10KB, Mamba F32 / RWKV Q8) [MEASURED: 2026-10-31]
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

Wins 8/12 vs brotli-11, 10/12 vs gzip-9. Average win margin: -50.5% vs brotli — **10 KB windows only**. On whole files
brotli-11 gives UBL 0.247 (Redukt 0.446 at 10 KB), facturae 0.395 (0.389), apache 0.313 (0.216)
(`competitors-fullsize-2026-11-01.csv`): the XML margin is a window artifact and the log margin shrinks. Never quote
the -50.5% externally without the 10 KB label.
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

That scale test is RWKV-only. **With the Mamba backbone the ratio COLLAPSES with length** (Kaggle T4 + local CPU,
2026-11-10, `data/benchmark/results/kaggle-fullsize/README.md`): enwik8 0.964 (10 KB), 1.054 (50 KB), 1.630 (100 KB), 3.31 (500 KB),
3.71 (1 MB), 3.91 (2 MB) — worse than gzip from ~500 KB. Whole industry files lose to brotli/xz on all five
(apache 0.494 vs 0.313, UBL 0.481 vs 0.252, syslog 1.11 vs 0.37). Cause: Mamba-130M's recurrent state degenerates
to near-uniform logits beyond ~16-18K tokens; reproduced with the HF reference implementation, so it is the model,
not the runtime (`data/benchmark/results/kaggle-fullsize/mamba-length-collapse-2026-11-10.md`). RWKV-7 does not drift (1.29 bpb at 100 KB).
Mitigation shipped (R-108): the backbone state is reset every 4096 tokens by default for the Mamba family and the
period is in the AZTH header (`--reset-every N`, 0 = off); it gives 0.9950 bpb on enwik8 100 KB (vs 1.6327) and wins all five whole industry files
again (apache 0.185 vs brotli 0.313, facturae 0.248 vs 0.395, UBL 0.235 vs 0.247, syslog 0.245 vs 0.379, openssh
0.202 vs 0.359) at zero extra compute. Every headline Redukt figure is a 10 KB window inside the
regime where the backbone still works. RWKV throughput: 60-69 tok/s. Mamba: ~15-34 tok/s (2026-10-31 binary).
Scale CSV: `data/benchmark/results/scale-test-complete.csv`.

### Fase 19 — CPU Limit & Market Viability (2026-11-10, measured)
Audit (`docs/internal/research/throughput-properties-audit.md`) refuted three premises behind "34 tok/s = DRAM ceiling":
Mamba ran F32 in the hot path; the dev laptop is dual channel (42.7 GB/s, 41% used); in_proj/out_proj ran single-threaded.
Measured on i5-1235U, maxperf, 10 KB (thermally limited laptop):

| Step | Result | Status |
|---|---|---|
| Baseline (F32, 1 thread) | 34 tok/s | — |
| R-099 multithread F32 | 39.0 -> 53.6 tok/s single, `.azth` byte-identical (sha 384c3042) | done |
| R-098 Q8 weight-only (block-64) | 127 tok/s cold (111 warm); bytes/token 516.5 -> 153.6 MB | done |
| R-098 BPB F32 -> Q8 | enwik8 +0.57%, facturae +1.21%, apache +1.11% | within +5% budget |
| R-100 lockstep, Q8 aggregate N=1/2/4/8/16/32 | 111/134/236/266/292/315 tok/s | done |
| R-100 lockstep, F32 aggregate (same N) | 57.7/96.5/154.6/206.5/262.6/322.1 tok/s | done |
| R-100 re-measured, machine idle (`lockstep-throughput-2026-11-10.csv`) | Q8 N=8/16/32: 270.8/286.9/309.9; F32 N=16: 259.6 tok/s | confirms |
| R-104 F1 pure-Rust exp/log + numeric contract | 0 ulp on 43 golden values, <=1 ulp vs std over 1e6; Windows<->Linux OK | done |
| R-103 CUDA plugin on Kaggle T4 (`kaggle-cuda/`) | roundtrip OK, BPB 0.9641 == CPU F32; batch N=1/8/32/128 = 151/951/1,076/1,117 tok/s | validated |
| R-107 FMA numeric contract 2 (`--numeric fma`) | batch Q8 N=32 299 -> 366 tok/s (+22%), F32 N=32 303 -> 351; BPB identical; single unchanged | done, opt-in |
| Kaggle full-size run (T4 + 4 vCPU, `notebooks/redukt-gpu-fullsize-bench.py`) | Mamba length collapse: enwik8 0.964 -> 3.91 bpb from 10 KB to 2 MB; all 5 whole industry files lose to brotli/xz; per-vCPU 69 B/s; plugin `--jobs>1` corrupted predictions (1.949 vs 0.750 bpb) | measured, see `kaggle-fullsize/README.md` |
| R-108 periodic backbone state reset — in the format (PROFILE bit 6 + RESET_LOG2, default 4096 for Mamba) | enwik8 100 KB 1.6327 -> 0.9951 bpb; 5/5 whole industry files win again (-5% to -44% vs brotli); header-driven decompress roundtrip OK; `--reset-every 0` = old output; 10 KB references +1 B | done |
| R-109 plugin lockstep over FFI v2 slots | `prepare_streams` -> `batch_init(n)`, one slot per stream, `forward_multi` -> `forward_batch`; v1-only plugin with `--jobs > 1` rejected (it interleaved one state: 1.949 bpb) | implemented, T4 gate pending |

R-100 is compute-bound from N~8 (130-190 GFLOPS, no FMA), ~4.9 MB per stream; 192/192 batch `.azth` are identical
to single-file output. Per-row Q8 scales failed (+7%); R-064 failed due to W8A8 + per-row scale on the tied head.
Aggregate gain ~9x per machine over the 34 tok/s start, but per-vCPU efficiency only ~1.5-2x (the baseline used
1-2 cores, lockstep saturates 12 threads): cost per raw GB stays ~$50-60 on CPU spot; GPU T4 N=128 ~$9-17.
See "Commercial Target". Next lever: FMA/kernels (new numeric profile) and optimized GPU batch kernels.
The idle re-measurement (12 files = 75,398 tokens; 32 files = 198,844 tokens) and an independent review confirmed the BPB
gates exactly and batch-vs-single identity (N=1,2,3, both directions). `redukt api` has byte-for-byte parity with the CLI
(sha256 API == CLI on enwik8 10 KB Q8); it is single-threaded (one request at a time). Tests: 295 unit + 5 integration.
Status: R-098/R-099/R-100 done; R-102 GCP verified locally in a container (not deployed on real GCP); R-103 CUDA FFI v2
validated on Kaggle T4 (3a2e8be: roundtrip, BPB == CPU, batch 1,117 tok/s, ~2% of the state-bandwidth
ceiling, headroom 15-25x -> kernels are the next GPU lever); R-104 F1 done (libm replaced by `fmath`), F2 (scalar order = AVX2) and
F3 (multi-platform CI + libm lint) pending; R-101, R-105 pending. Plan and gates: ROADMAP Fase 19.

## Process Safety (Windows)

- FOREGROUND terminal only — NEVER use run_in_background
- `taskkill //F //IM redukt.exe` kills all instances
- Use `--bytes N` not `--limit N` (silently ignored)
- `cargo build --profile maxperf` for benchmarks
- `--numeric fma` on batches with N >= 8 (+12-22%); default stays contract 1 until api.rs supports contract 2
- Agents/sessions that build Docker images MUST clean up: `docker rm -f` test containers, `docker rmi` test tags,
  `docker builder prune -af` at the end. Keep only base images (rust, nvidia/cuda) if a rebuild is expected.
- Claude Code hooks (.claude/settings.json): SessionEnd runs .claude/hooks/docker-cleanup.sh (redukt:* images,
  containers, build cache); PostToolUse Edit|Write on *.rs runs .claude/hooks/cargo-check.sh and feeds
  warnings/errors back to the model. Review with /hooks.
- `--jobs N` = N lockstep streams sharing every weight fetch (default 8; N=16-32 for large hosts);
  `decompress --input-dir` is supported. The old independent-thread batch mode was removed.
- `REDUKT_THREADS=N` overrides the persistent worker pool size (default: available parallelism)
