# Strata backbone — 02. Evidence and constraints

> Evidence digest for the design of a new backbone (working name "Strata"). **No architecture is proposed here.**
> Date: 2026-11-12. Repo HEAD: 692d985 (Rust sources unchanged since e93b72e). Read-only audit; this file is
> the only artifact created.
> Method: primary sources read directly (research reports, raw CSV/logs, notebooks, `scripts/breakeven.py`,
> selected Rust sources); three reading agents extracted the older material (Fases 3-18, competitive notes).
> Citations from those agents that were not re-opened by the author are marked `†`; everything else was read or
> grep-verified at the cited line. A few gaps were closed with direct measurements on the dev laptop (§F.9).
> Companion files: `00-charter.md` (axioms, test-independence rules), `01-integration-map.md` (how a backbone
> plugs into the code). `src/` line numbers refer to commit 692d985; Rust sources are being edited concurrently
> by the implementation stage, so re-resolve them if a file has changed. The audit's runs used
> `target/maxperf/redukt.exe` as built on 2026-11-11 13:38.

## Labels and conventions

| Label | Meaning |
|---|---|
| **MEASURED** | The project ran it; a result file, log or report in the repo carries the number |
| **MEASURED (audit)** | Run during this audit on 2026-11-12 on the dev laptop; single runs, machine not idle |
| **DERIVED** | Arithmetic on MEASURED inputs; formula or inputs given |
| **EXTERNAL** | Third-party figure quoted by a repo document; not reproduced by the project |
| **CLAIMED** | Stated in the repo without a run: projection, estimate, or a diagnosis that was not ablated |

- bpb = compressed bytes x 8 / input bytes of a real `.azth`, unless marked **CE** (cross-entropy) or
  **backbone-only** (no CM, no mixer, no coder, no header).
- "HF tokens" = Hugging Face tokenizer of `state-spaces/mamba-130m-hf`; "Rust tokens" = the product tokenizer
  (`src/infrastructure/mamba/tokenizer.rs`). **They are not the same segmentation** (§A.5, §B.7).
- Source aliases (paths relative to the repo root):
  `RM` = `docs/ROADMAP.md` · `AUD` = `docs/internal/research/throughput-properties-audit.md` ·
  `SBP` = `docs/internal/research/small-backbone-probe-2026-11-11.md` ·
  `LOGS` = `docs/internal/research/log-specialist-2026-11-11.md` ·
  `P3` = `docs/internal/research/path3-expensive-bytes-2026-11-11.md` ·
  `RST` = `docs/internal/research/reset-period-sweep-2026-11-11.md` ·
  `CMP` = `docs/internal/research/competitive-position-2026-11-10/REPORT.md` ·
  `CN/` = `docs/internal/research/competitive-position-2026-11-10/notes/` ·
  `DX` = `docs/internal/research/determinism-cross-machine.md` ·
  `TR` = `docs/internal/research/token-reduction-strategies.md` ·
  `R/` = `data/benchmark/results/` · `KF` = `R/kaggle-fullsize/README.md` ·
  `COL` = `R/kaggle-fullsize/mamba-length-collapse-2026-11-10.md` · `NB/` = `notebooks/`.

## 0. What this audit found that no repo document states

1. **The product tokenizer is not the reference tokenizer.** On identical bytes the Rust BPE tokenizer emits 0% to
   +43.5% more tokens than the HF tokenizer (facturae +43.5%, UBL +18.2%, enwik8 first 10 KB +13.0%); on enwik8
   first 10 KB the Rust backbone-only CE is 0.9065 bpb vs 0.8623 for the HF reference (+5.1%). §A.5, §B.7.
2. **Fixed framing is 20 bytes, not 19**: a default Mamba Q8 archive has a 16-byte header + 4-byte CRC (§D.2).
3. **"~1,180 B/s on 12 vCPU" is tok/s x an assumed 3.8 B/token.** The wall rate of the same runs is 441-522 B/s
   (37-44 B/s per logical CPU) on the mixed 10 KB set (§A.3).
4. **The classical columns of `KF` table A do not match the run's own CSV** (e.g. xz at 2 MB: 2.17 vs 2.2885);
   the CSV reproduces locally. Log baselines differ between documents because of `xz -9` vs `xz -9e` (§I).
5. **The "Hutter 2026 lineage" reference is an in-sample number** (6M transformer pretrained on enwik9, scored
   on enwik9), the same trap the repo already flagged for L3TC (§B.2).
6. Process facts: a Q8 compress of one 256-byte message peaks at ~0.79 GB RSS and needs the 516.6 MB F32
   weights file (§D.3); `mamba-eval` ignores `--quant` (always F32).

---

## A. Measured figures relevant to a new backbone

### A.1 Ratio — full product (`.azth`: backbone + CM + mixer + coder + header)

| # | Quantity | Value | Conditions | Label | Source |
|---|---|---|---|---|---|
| 1 | enwik8 first 10,240 B, Mamba-130M F32 | 0.9648 bpb (1,235 B) | AZTH v5 + RESET_LOG2; Kaggle Xeon CPU; T4 plugin gives the same size | MEASURED | `R/kaggle-fullsize/run-a-r108/fullsize-results.csv:5`, `KF:114` |
| 2 | same, Q8 | 0.9695 bpb (1,241 B) | same | MEASURED | `.../run-a-r108/fullsize-results.csv:6` |
| 3 | enwik8 first 102,400 B | F32 0.9925 (12,704 B); Q8 0.9951 (12,737 B); Q8 without reset 1.6327 (20,899 B) | reset 4096 tokens, CPU 4 vCPU | MEASURED | `.../run-a-r108/fullsize-results.csv:7-8`, `KF:115-116` |
| 4 | Length curve with reset 4096 | 10 KB 0.9648 · 50 KB 1.0372 · 100 KB 0.9925 · 500 KB 0.9697 · 1 MB 0.9781 · 2 MB 0.9998 | T4 plugin F32, single stream. Largest input ever measured: 2 MB (4 MB skipped; full enwik8 never run) | MEASURED | `R/kaggle-fullsize/run-a-r108-gpu/fullsize-results.csv:7-12` |
| 5 | Length curve without reset | 0.9641 · 1.0539 · 1.6295 · 3.3099 · 3.7096 · 3.9056 | same prefixes, T4 F32 | MEASURED | `R/kaggle-fullsize/run-a/fullsize-results.csv:6-11` |
| 6 | Whole industry files, reset 4096 | facturae 0.2408 · UBL 0.2392 · apache 0.1849 · syslog 0.2449 · openssh 0.2014 | T4 F32, LF copies; best classical 0.4078 / 0.2519 / 0.3132 (brotli-11) / 0.3697 / 0.3460 (xz-9e) | MEASURED | `.../run-a-r108-gpu/fullsize-results.csv:2-6`, `.../classical-fullsize.csv:42-66` |
| 7 | Same files without reset | 0.4458 · 0.4810 · 0.4937 · 1.1145 · 1.1710 (loses 5/5) | T4 F32 | MEASURED | `.../run-a-r108/fullsize-results.csv:9-13` |
| 8 | 12-file table at 10 KB (2026-10-31 binary) | wins 8/12 vs brotli-11; losses sao +44.7%, ooffice +36.7%, x-ray +6.4%, mr +0.8% | Mamba values on 10,240 B, brotli/gzip/RWKV on 10,000 B (window mismatch: enwik8 brotli 2.3520 = 2,940 B of 10,000; Mamba 0.9609 = 1,230 B of 10,240); x-ray value is the corrupt one (corrected 5.8008 = +26.3%); "-50.5% average" is a 10 KB-window figure | MEASURED, with defects | `R/dual-backbone-benchmark.csv:2-13`, `RM:3151`, `CLAUDE.md:317-320` |
| 9 | Binary category vs brotli | +22.1% worse (6 files) | 10 KB, RWKV/Mamba era 2026-11-01 | MEASURED | `R/market-readiness-diagnostic-2026-11-01.md:51` |
| 10 | RWKV-7 0.1B + CM | enwik8 10 KB 1.3312 (2026-10-30) / 1.3258 (2026-11-10); 100 KB 1.2947 CE, no drift | Q8; vocab 65,536 | MEASURED | `CLAUDE.md:287`, `R/determinism-cross-machine-2026-11-10.tsv:7`, `COL:27` |
| 11 | RWKV scale test (10 / 50 / 100 KB) | apache 0.2960 / 0.2274 / 0.2235 · dickens 1.7968 / 1.5029 / 1.4574 · samba 1.2976 / 1.3555 / 1.2062 | batch, Q8 | MEASURED | `R/scale-test-complete.csv:3-12` † |
| 12 | Pretrained MambaByte-353M (vocab 256, trained on arXiv) | 1.3156 bpb; CE 1.1975; coder efficiency 91.02%; 4.8 tok/s; 1.7531 before the LayerNorm fix | enwik8 10 KB, i5 laptop | MEASURED | `RM:2311-2316` |
| 13 | CM ensemble alone, BPE-50K tokens | 3.0219 bpb | enwik8 10 KB; plugin returning uniform logits + CM | MEASURED | `R/kaggle-v3/analysis.md:42,60` |
| 14 | CM-only bypass, RWKV era | enwik8 2.687 vs 1.281 (+109.7%); sao +19.4%; osdb +13.7% | 10 KB, Q8 | MEASURED | `RM:590-595` |
| 15 | Classical codecs at 10 KB, for comparison with rows 13-14 | gzip-9 2.9883 · xz-9e 2.9312 · zstd-19 2.9406 · brotli-11 2.3625 | enwik8 first 10,240 B | MEASURED (reproduced in audit) | `.../run-a-r108-gpu/classical-fullsize.csv:2-6` |
| 16 | Short messages, mean bytes over 6 sources (brotli-11 / zstd-19+dict / `.azth` / payload) | 128 B: 97.3 / 52.4 / 57.8 / 38.8 · 256 B: 145.1 / 80.2 / 74.0 / 55.0 · 512 B: 211.0 / 124.3 / 103.4 / 84.4 · 1 KB: 319.4 / 216.9 / 150.1 / 131.1 · 2 KB: 510.7 / 389.7 / 243.7 / 224.7 | 480 messages, CPU path, Mamba Q8 | MEASURED | `R/short-message-bench/run-1/short-message-report.txt:83-87` |

### A.2 Ratio — backbone-only cross-entropy

PyTorch/HF rows use the **HF tokenizer**, independent chunks of C tokens (BOS + C-1), F32 (`SBP:68-81`).

| # | Model | enwik8 8x10K | enwik8 500K | enwik9 tail (unseen) | dickens 500K | facturae | UBL | apache | syslog | openssh | Label | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mamba-130M, C=4096 | 1.028 | 0.938 | 0.998 | 1.018 | 0.231 | 0.414 | 0.175 | 0.246 | 0.214 | MEASURED | `SBP:99` |
| 2 | SmolLM2-135M, C=4096 | 0.919 | 0.834 | 0.915 | 0.964 | 0.183 | 0.336 | 0.182 | 0.253 | 0.231 | MEASURED | `SBP:100` |
| 3 | Pythia-160M / 70M / 31M / 14M, C=2048 (same tokenizer and corpus as Mamba) | 1.149 / 1.313 / 1.420 / 1.593 | 1.026 / 1.193 / 1.283 / 1.448 | 1.094 / 1.269 / 1.368 / 1.541 | 1.112 / 1.292 / 1.377 / 1.551 | 0.292 / 0.369 / 0.470 / 0.629 | 0.599 / 0.740 / 0.832 / 0.985 | 0.236 / 0.263 / 0.289 / 0.356 | 0.308 / 0.362 / 0.412 / 0.502 | 0.302 / 0.402 / 0.441 / 0.529 | MEASURED | `SBP:101,104-106` |
| 4 | L3TC-12M / 3.2M / 800K / 200K (trained on enwik8) | 0.902 / 1.102 / 1.233 / 1.378 | 0.850 / 1.039 / 1.176 / 1.318 | 1.467 / 1.455 / 1.587 / 1.628 | 1.909 / 1.900 / 2.005 / 2.076 | 5.16 / 5.15 / 5.40 / 6.56 | 6.41 / 6.21 / 6.54 / 7.28 | 4.79 / 4.31 / 4.68 / 4.40 | 5.11 / 4.57 / 4.68 / 4.62 | 5.35 / 4.91 / 4.70 / 4.61 | MEASURED | `SBP:107-110` |
| 5 | best classical | 2.735 | 2.253 | 2.361 | 2.189 | 0.408 | 0.252 | 0.313 | 0.370 | 0.346 | MEASURED | `SBP:111` |

| # | Quantity | Value | Conditions | Label | Source |
|---|---|---|---|---|---|
| 6 | R-112 byte-level specialists, static, 5 unseen Loghub systems | mean 5.599 (lstm-xs 0.28M) · 4.979 (lstm-s 0.94M) · 4.791 (lstm-m 3.6M) · 7.397 (tf-s 1.0M); range 4.0-9.2 | trained on 11 systems (456.6 MB), 22-45 min on a T4 | MEASURED | `LOGS:86-89`, `R/log-specialist/run-1/specialist-local-eval.csv`, `.../specialist-results.csv:22-26` |
| 7 | Same models with online adaptation (1 Adam step per 256 B) | unseen mean 0.591 / 0.509 / 0.452; same procedure from random init 2.898 / 2.064 | best classical 0.295; Mamba-130M backbone-only 0.202 | MEASURED | `LOGS:90-94`, `LOGS:116-117` |
| 8 | Same models on the 11 seen systems (later slice of each) | tf-s 0.516 vs best classical 0.427 vs Mamba-130M 0.541; wins vs classical on 5/11 | 300 KB per system | MEASURED | `LOGS:98-117` |
| 9 | Held-out bpb during training | best at the first checkpoint, then worse: lstm-xs 4.856 -> 5.600; lstm-m 4.444 -> 4.822; tf-s 5.474 -> 7.219, while train bpb falls to 0.46-0.74 | static 4096, 100 KB of each unseen system | MEASURED | `R/log-specialist/run-1/specialist-train.csv:2-41` |
| 10 | Rust runtime, Mamba-130M backbone-only (`mamba-eval`, F32) | enwik8 10 KB 0.9074 · dickens 0.6149 · apache 0.1636 | 2026-10-30 binary, Rust tokens | MEASURED | `RM:996-998` |
| 11 | Same, today | 0.9065 bpb, 3,611 Rust tokens (2.84 B/token) | enwik8 first 10,240 B | MEASURED (audit) | §F.9 |
| 12 | HF reference on the same bytes | 0.8623 bpb, 3,196 HF tokens (3.20 B/token); 0.8635 with BOS | transformers 5.18, torch 2.14 CPU, F32, single chunk | MEASURED (audit) | §F.9 |
| 13 | CE of Mamba+CM vs Mamba-only on enwik8 10 KB | ~0.908 vs 0.9074: the CM adds ~0 in CE; the coder and framing then cost ~5.5% | 0.9609 bpb x 94.5% coder efficiency (older binary) | DERIVED | `CLAUDE.md:285,289`, `RM:996` |
| 14 | Real `.azth` vs HF backbone-only CE, whole files | UBL -42% · openssh -6% · syslog -0.5% · enwik8 500K +3% · facturae +4% · dickens 500K +4% · apache +6% | not an ablation: tokenizers differ (§A.5) and the `.azth` includes coder + header | DERIVED | `R/reset-period-sweep/run-1/reset-sweep-azth.csv:2-8`, `SBP:99` |
| 15 | Direct CM ablations that exist | match+run+word models: enwik8 -0.27%, apache -1.4%, facturae -2.3% CE (RWKV); entropy-adaptive mixer, segment model, bit-plane features: 0% to +/-0.7% at 10 KB (Mamba) | 10 KB | MEASURED | `RM:1057`, `RM:1293`, `RM:1317` |
| 16 | One short message, ensemble vs backbone | full ensemble Q8 CE 1.7611 bpb vs backbone-only F32 CE 1.6598 bpb (n=1) | 256-B Apache-log message | MEASURED (audit) | §F.9 |

### A.3 Throughput

| # | Quantity | Value | Conditions | Label | Source |
|---|---|---|---|---|---|
| 1 | i5-1235U, Mamba-130M single stream | 34 tok/s (F32, in/out_proj on 1 thread) -> 39.0 -> 53.6 (multithread) -> Q8 127 cold / 111 warm | 10 KB, maxperf, thermally limited laptop | MEASURED | `AUD:197-199`, `RM:2647`, `RM:2680` |
| 2 | i5-1235U lockstep, aggregate tok/s, N=1/2/4/8/16/32 | Q8 111 / 134 / 236 / 266 / 292 / 315 · F32 57.7 / 96.5 / 154.6 / 206.5 / 262.6 / 322.1 | 12-32 files x 10 KB, warm | MEASURED | `R/lockstep-throughput-2026-11-10.csv:6-17` |
| 3 | Same, idle machine | Q8 N=8 270.8 · N=16 286.9 · N=32 309.9; F32 N=16 259.6 | 75,398 / 89,966 / 198,844 tokens in 278.4 / 313.6-346.6 / 641.6 s | MEASURED | `.../lockstep-throughput-2026-11-10.csv:2-5` |
| 4 | Wall bytes/s of the idle runs | 441 (12 files) · 522 / 473 (16 files Q8 / F32) · 511 (32 files) = 37-44 B/s per logical CPU; 1.63-1.82 B/token | assumes 10,240 B per file | DERIVED | same rows |
| 5 | "~1,180 B/s on 12 vCPU, ~98 B/s per vCPU" | 315 tok/s x 3.8 B/token | conversion with an assumed B/token, not a wall measurement | CLAIMED | `AUD:213-215`, `CLAUDE.md:24` |
| 6 | FMA numeric contract (opt-in) | Q8 N=32 299.1 -> 365.9 tok/s (+22%); F32 N=32 303.1 -> 351.0; single stream unchanged | idle, maxperf | MEASURED | `RM:2871-2873` |
| 7 | Kaggle Xeon, 4 vCPU, lockstep | Q8 jobs=12 275.9 B/s = 69 B/s per vCPU; jobs=4 255.8; F32 jobs=12 238.4 | 7 enwik8 slices + 5 industry heads, 10 KB each, wall incl. model load | MEASURED | `R/kaggle-fullsize/run-a/fullsize-results.csv:2-4` |
| 8 | Kaggle Xeon, 4 vCPU, single stream | enwik8 10 KB: F32 121.1, Q8 178.4 B/s; 100 KB: F32 163.1, Q8 240.4 B/s (60 B/s per vCPU) | wall incl. load | MEASURED | `.../run-a-r108/fullsize-results.csv:5-8` |
| 9 | T4 via the bare FFI harness, N=1/8/32/128 | 150.6 / 951.3 / 1,076.1 / 1,116.9 tok/s (6.64 / 8.41 / 29.74 / 114.61 ms per step) | Mamba-130M F32 | MEASURED | `R/kaggle-cuda/cuda-plugin-results.csv:12-16` |
| 10 | T4 through the CLI, N=12/32/64/128 | 530.4 / 573.4 / 581.8 / 559.0 tok/s = 1,807 / 1,955 / 2,083 / 2,005 B/s; 128 x 50 KB 552.5 tok/s | lockstep, g4dn-class host with 4 vCPU; implies 3.41-3.59 B/token (Rust tokens, enwik8 windows) | MEASURED | `R/kaggle-fullsize/run-e-batch/fullsize-results.csv:2-6` |
| 11 | T4 through the CLI, single stream | 292-399 B/s | enwik8 prefixes | MEASURED | `.../run-a-r108-gpu/fullsize-results.csv:7-12` |
| 12 | Decompression time | equal to compression: 161 vs 159 s (facturae), 349 vs 352 s (UBL), 253 vs 257 s (enwik8 100 KB) | T4 single stream | MEASURED | `.../run-a-r108-gpu/fullsize-results.csv:2-3,9` |
| 13 | PyTorch CPU, 1 thread, cached decoding | Mamba-130M 14.0 · SmolLM2 13.3 · Pythia-160M 17.9 · 70M 46.5 · 31M 91.6 · 14M 154.6 · L3TC-12M 147.9 · 3.2M 289.8 · 800K 508.4 · 200K 560.6 tok/s (= 1.97 KB/s) | Kaggle Xeon; framework overhead dominates the small ones (`SBP:144-145`): not a runtime figure | MEASURED | `R/small-backbone-probe/run-1/small-backbone-speed.csv:2-9`, `.../run-2-l3tc/small-backbone-speed.csv:2-5` |
| 14 | Classical codecs, one Kaggle Xeon core, enwik8 100 MB | zstd-19 2.1564 bpb at 1.31-1.40 MB/s · zstd-22 2.0267 at 0.99-1.05 · xz-9e 1.9865 at 0.99-1.00 · brotli-11 2.0594 at 0.32-0.34 · gzip-9 2.9156 at 12.4-13 | decompression 57-358 MB/s | MEASURED | `.../run-a-r108-gpu/classical-fullsize.csv:37-41`, `KF:71-72` |
| 15 | T4 training throughput of tiny byte-level models | lstm-xs 0.94 MB/s · lstm-s 0.33 · lstm-m 0.107 · tf-s (AMP) 0.53 MB/s | bytes_seen / train_min; batch 128 or 64 x 1,024 B | DERIVED | `R/log-specialist/run-1/specialist-models.csv:2-5` |
| 16 | RWKV-7 Q8 | 94.8 tok/s after R-015; per token: backbone 10,600 us, six CM models 30 us, coder 0.2 us | enwik8 10 KB, i5; the per-token map is reported without a measurement label | MEASURED / reported | `R/r015-rank1-factorization-validation.csv:16-19` †, `docs/internal/research/execution-architecture-analysis.md:25-29` |
| 17 | This laptop today | `mamba-eval` F32 48.4 tok/s (3,611 tokens in 74.6 s); one 256-B message Q8: 1.7 s, 1.82 s wall incl. load | machine not idle | MEASURED (audit) | §F.9 |
| 18 | MKL on this laptop (torch 2.14) | SGEMM 87-121 GFLOP/s on 1 thread, 288-335 on 10-12 threads; SGEMV 1 thread 16-27 GFLOP/s for 0.26-9.4 MB matrices, 11.5 GFLOP/s (148 products/s = 22.9 GB/s of weights) for a 50,280 x 768 F32 matrix | quick probe, single pass | MEASURED (audit) | §G.3 |
| 19 | Effective rate of the product on the same laptop | 315 tok/s x 258.3 MFLOP = 81 GFLOP/s on 12 threads (94.5 with FMA) = 5.8-7.9 GFLOP/s per logical CPU | lockstep Q8 N=32 | DERIVED | rows 2, 6; `SBP:128` |

### A.4 Bytes, FLOPs and state per token

| # | Quantity | Value | Label | Source |
|---|---|---|---|---|
| 1 | Mamba-130M size and cost | 129,135,360 params; 258.3 MFLOP/token (2 x active params); 71.35 MFLOP/byte at 3.62 B/token (HF tokens) | MEASURED / DERIVED | `SBP:128` |
| 2 | Weight bytes read per token | 516.5 MB F32 -> 153.6 MB Q8. Loader print today: 492.6 MiB = layers 345.3 + head 147.3 (head = 29.9%) | MEASURED | `RM:2647`; §F.9 |
| 3 | FLOP shares per token | in_proj 44.1% · out_proj 22.0% · LM head 30.1% (38.6M) · SSM recurrence 1.9% · conv + dt + norms 2.0% | DERIVED (model dimensions) | `docs/internal/research/lottery-ticket-sparsity-protocol.md:14-19` |
| 4 | Vocabulary cost floor | Pythia-14M touches 7.6M params/token, 6.4M of them the 50,304 x 128 output matrix | MEASURED | `SBP:141-143` |
| 5 | Recurrent state per stream | 24 x 1,536 x (16 + 3) x 4 B = 2.80 MB F32 | DERIVED (code) | `src/infrastructure/mamba/model.rs:278-292`, `src/application/lockstep.rs:1009` |
| 6 | Host buffers per stream | 10 x vocab x 4 B = 2.01 MB at vocab 50,280 (logits, ensemble, probs, 5 CM bias buffers, bias head, segment counts) -> ~4.9 MiB per stream in total | DERIVED / MEASURED | `src/application/lockstep.rs:150-156`, `RM:2722` |
| 7 | CDF construction | one f64 `exp` per vocabulary entry per token (`cdf_from_logits_with`); `fmath` exp = 9.3 ns -> 50,280 x 9.3 ns = 0.47 ms per token on one thread before normalisation, CDF build, CM mix and coding | DERIVED | `src/application/codec.rs:737-750`, `DX:256` |
| 8 | Host cost around the device call (GPU tier) | 1.72-1.89 ms per token-stream through the CLI vs 0.90 ms in the FFI harness: ~0.85 ms for the copy of 50,280 F32 logits (25.7 MB per step at N=128) + CDF + CM + coder on 4 vCPU. **The split is not measured** | MEASURED totals / CLAIMED attribution | `KF:146-154` |
| 9 | R-112 specialists | 0.529 / 1.844 / 7.095 / 1.653 MFLOP/byte (lstm-xs / s / m / tf-s) = 0.7% / 2.6% / 9.9% / 2.3% of Mamba-130M | DERIVED | `R/log-specialist/run-1/specialist-models.csv:2-5` |
| 10 | Coder | CDF total 2^24, minimum frequency 1; efficiency (CE / file bits) 94.5% (Mamba, enwik8 10 KB), 96.0% (RWKV), 91.0% (MambaByte-353M), 67.9% on one 256-B message (56.4 B of CE + 20 B framing + ~2 B uniform first token + ~5 B flush = 83 B) | MEASURED | `src/domain/coder.rs:12,184-197`, `CLAUDE.md:289`, `RM:2314`, §F.9, §D.2 |

### A.5 Tokens per byte — product tokenizer vs reference tokenizer (same local bytes)

| Input | Bytes | HF tokens | Rust tokens | Rust vs HF | Rust B/token |
|---|---|---|---|---|---|
| enwik8 first 10,240 B | 10,240 | 3,196 | 3,611 | +13.0% | 2.84 |
| enwik8 first 102,400 B | 102,400 | 26,001 | 26,969 | +3.7% | 3.80 |
| dickens first 102,400 B | 102,400 | 26,919 | 28,749 | +6.8% | 3.56 |
| apache-access.log | 171,239 | 61,101 | 61,101 | 0% | 2.80 |
| linux-syslog.log | 216,485 | 85,232 | 86,311 | +1.3% | 2.51 |
| openssh-auth.log | 225,216 | 85,384 | 85,505 | +0.1% | 2.63 |
| facturae-batch.xml (local CRLF copy) | 39,384 | 12,642 | 18,143 | +43.5% | 2.17 |
| ubl-invoice-batch.xml (local CRLF copy) | 83,260 | 33,032 | 39,052 | +18.2% | 2.13 |

All rows MEASURED (audit), §F.9. Mechanism visible in the code: the Rust pre-tokenizer starts a new "word" at
every whitespace byte (`src/infrastructure/mamba/tokenizer.rs:232-254`), so every space of a run (XML indentation)
becomes its own token, and merges run across letter/digit/punctuation boundaries; the code calls it "simplified
GPT-2 pre-tokenization" (`tokenizer.rs:108`). Consequences that are certain: (a) every "Mamba-130M backbone-only"
figure computed in PyTorch (`SBP`, `LOGS`, `RST` §3) describes the HF token stream, which differs from the
product's on 7 of the 8 inputs checked (only apache has the same count); (b)
forwards per byte are up to 43.5% higher than the HF count implies; (c) a 4,096-token reset window covers
8.7-8.9 KB of these XML files, not the 10.3-12.8 KB the HF count implies. Whether the 5.1% CE gap of §A.2
rows 11-12 is entirely due to tokenization is **not isolated** (the forward itself matched an independent NumPy
implementation to 2.4e-6 on the CUDA tier: `R/kaggle-cuda/cuda-plugin-results.csv:7`).

### A.6 Cost per raw GB

| # | Quantity | Value | Conditions | Label | Source |
|---|---|---|---|---|---|
| 1 | CPU, Kaggle Xeon | $56.4/GB | 69 B/s per vCPU at c7i spot $0.014/vCPU-h | MEASURED rate, DERIVED cost | `R/kaggle-fullsize/run-a/gate-table.csv:2` |
| 2 | CPU, laptop, as documented | $50-60/GB (also "~$62" when recomputed) | 1,180 B/s (claimed conversion) on 12 vCPU at $0.022 | CLAIMED | `AUD:215-216`, `CN/posicion_interna_redukt.md:91` † |
| 3 | CPU, laptop, from wall rates | $144-166/GB at $0.022; $114-132 at $0.0175; $91-106 at $0.014 | 441-511 B/s on 12 logical CPUs, mixed 10 KB set | DERIVED | §A.3 row 4; `CMP:52` says $120-140 |
| 4 | T4 through the CLI | $32.9-38.0/GB spot; $70.2-80.9 on-demand | g4dn.xlarge $0.247 / $0.526 per hour | MEASURED rate, DERIVED cost | `KF:134-138` |
| 5 | T4, FFI harness only | ~$9-17/GB; **not reachable through the product** | N=128, 1,117 tok/s | DERIVED | `KF:155-157` |
| 6 | Classical codecs | zstd-19 $0.003 · xz-9e $0.004 · brotli-11 $0.011 per GB | one core, same machine | DERIVED | `KF:71-72` |
| 7 | Value of the saving, S3 Standard, 7 years | text $0.300; facturae $0.040; apache / syslog / openssh $0.031 / $0.030 / $0.035; UBL $0.003 per raw GB | whole files, reset 4096, vs best classical | DERIVED | `R/kaggle-fullsize/run-a-r108-gpu/gate-table.csv:2-17` |
| 8 | Thresholds quoted in `CLAUDE.md` | <= $0.33 text, <= $0.12 e-invoice XML, <= $0.06 logs | from 10 KB-window deltas vs brotli (b and S of `scripts/breakeven.py:56`) | DERIVED from window figures | `CLAUDE.md:22-23` |
| 9 | Projected cost of smaller published backbones | Pythia-14M $3.32; L3TC-200K $0.77 per GB vs $0.18-0.20 of value | assumes the per-FLOP efficiency of Mamba-130M in the current runtime; unmeasured | CLAIMED | `SBP:171-184`, `SBP:90-91` |

### A.7 Quantisation

| # | Quantity | Value | Label | Source |
|---|---|---|---|---|
| 1 | Mamba Q8, weight-only, one f32 scale per 64 columns (int8 in_proj, out_proj, tied embedding/head; F32 activations, A_log, D, dt_proj, x_proj, conv1d, norms) | enwik8 +0.57% (0.9633 -> 0.9688), facturae +1.21%, apache +1.11% at 10 KB; 3.3x single-stream speed | MEASURED | `RM:2645-2649`, `R/determinism-cross-machine-2026-11-10.tsv:1-6` |
| 2 | Same with one scale per row | +6.6-7.0% on enwik8/facturae ("outlier columns of the tied embedding dominate each row's max") | MEASURED | `src/infrastructure/mamba/model.rs:50-56`, `RM:2650` |
| 3 | R-064 (2026-11-02), Q8 at load | all-Q8 +16.6%; in/out_proj only +16.2%; head-only +14.5%; 31-34 tok/s (no gain); reverted | MEASURED | `RM:1762-1768` |
| 4 | Cause of R-064 | "W8A8 (quantised activations) and per-row scale on the tied head" — forensic reading after the code was reverted; only the per-row part was reproduced (row 2) | CLAIMED | `RM:2650-2651` |
| 5 | RWKV-7 F32 -> Q8 (int8 weights x int8 activations) | -1.76% to +2.43% (mean abs 0.81%) over 18 files | MEASURED | `R/f32-vs-q8-analysis.md:45` |
| 6 | RWKV-7 Q4 | +5.1% to +25.7% over 6 files (mean +13.3%), and 4-10x slower; enwik8 +12.7% | MEASURED | `R/q4-evaluation.csv:2-7`, `RM:842` |
| 7 | Mamba Q6 / Q4 / int8 activations done properly / QAT | never measured (Q6 "+1-3%" is an estimate) | CLAIMED | `RM:2744-2746` |
| 8 | Q8 vs F32 in lockstep | at N >= 16 F32 matches Q8 in aggregate tok/s (compute-bound, not bytes-bound) | MEASURED | `RM:2719-2721` |
| 9 | Numeric tiers | CPU F32 and GPU F32 give the same size (1,234 B) and different sha256; GPU run-to-run identical | MEASURED | `R/kaggle-cuda/cuda-plugin-results.csv:20-22` |

### A.8 Length collapse and state reset

| # | Quantity | Value | Label | Source |
|---|---|---|---|---|
| 1 | Backbone-only marginal cost by token window, enwik8 100 KB (HF reference) | 2.24 (0-2K) · 3.75-4.48 (2K-16K) · 3.99 (16K-18K) · 5.35 (18K-20K) · 5.70 · 7.66 · 12.02 bits/token (24K-26K, max 165 bits on one token) | MEASURED | `COL:12-20`; first window reproduced in audit (2.238) |
| 2 | Same, Rust `mamba-eval` | 9.9 (18K-20K) · 18.9 · 18.2 · 16.3 bits/token: at or above uniform (15.6). Window values were derived by the source from a cumulative counter with an assumed 3.8 B/token, and the Rust stream has 26,969 tokens, not 26,001: positions are approximate | MEASURED (derived by the source) | `COL:10,17-20,24`; §A.5 |
| 3 | Content dependence | dickens 80 KB (23,078 tokens) stays at ~3.8 bits/token; with reset 32768 dickens 500K still loses 16% (1.2304 vs 1.0620) | MEASURED | `COL:26`, `R/reset-period-sweep/run-1/reset-sweep-azth.csv:3,31` |
| 4 | RWKV-7 + CM on the same 100 KB | 1.2947 cumulative, windowed 1.10-1.56, no drift. Not measured beyond 100 KB | MEASURED | `COL:27` |
| 5 | Period sweep, enwik8 100 KB, Q8 | 1024: 1.0259 · 2048: 1.0070 · **4096: 0.9951** · 8192: 0.9969 · none: 1.6327 | MEASURED | `COL:48-50` |
| 6 | Period sweep, 5 whole industry files, total bytes | 2048: 21,895 · **4096: 19,850** · 8192: 20,050 · 16384: 30,637 · 32768: 46,747 | MEASURED | `RST:29` |
| 7 | Context sensitivity, backbone-only, C=4096 -> 2048 | text +1.4% to +3.0%; XML and logs +18% to +75% | MEASURED | `SBP:151-159` |
| 8 | Same step in the real `.azth` (with CM) | industry +2.7% to +15.1% (total +10.3%); text +1.1% to +1.5% | MEASURED | `RST:22-29` |
| 9 | Re-warming the state with the last K tokens at each reset (backbone-only, PyTorch, **not implemented in Rust**) | K=512: -20.2% bits on the 5 industry files at 1.12x forwards; K=1024: -22.0% at 1.24x; text -1.4% | MEASURED estimate | `RST:48-61`, `R/reset-period-sweep/run-1/reset-sweep-overlap.csv` |
| 10 | Where the 4,096-token reset fires (Rust tokens) | ~15 KB of text, 10.3-11.5 KB of these logs, 8.7-8.9 KB of these XML; never inside a message <= 2 KB | DERIVED | §A.5 |
| 11 | Training context of the pretrained models | Mamba-130M: 2,048 tokens (stated, not verified); RWKV-7 World 0.1B: ctx 4096 | CLAIMED / EXTERNAL | `COL:33-34`, `weights/MANIFEST.md:17` |

---

## B. Failure modes the new backbone must not repeat

Each item: evidence, then cause **as far as it is established**.

**B.1 Collapse with input length (Mamba-130M).** Without reset: enwik8 0.964 (10 KB) -> 3.91 bpb (2 MB), worse than
gzip-9 from ~500 KB; all five whole industry files lose to brotli/xz (§A.1 rows 5, 7). The HF reference degrades at
the same position (§A.8 row 1), so it is the pretrained model, not the runtime. Cause: attributed to training with
2,048-token contexts and to the known length-extrapolation weakness of Mamba-1 (`COL:31-34`) — **CLAIMED**, no
ablation. Established facts: the onset is content-dependent (§A.8 row 3); the product degrades harder than the
reference in the collapse zone (18.9 vs 5.7 bits/token; the explanation "the CM/mixer cannot recover from
near-uniform logits" is marked "likely" in `COL:32`); the mitigation is a blind reset whose cold start costs
16-33% on XML/logs backbone-only (§A.8 row 9). Every headline figure before 2026-11-10 was a 10 KB window inside
the working regime (`COL:37`).

**B.2 Training-data leakage presented as capability.**
- L3TC: 1.04-1.10 bpb (3.2M) on enwik8, its training file; 1.455 on unseen Wikipedia; 4.3-6.2 on XML/logs; the 12M
  model reaches 0.850 on enwik8 and is no better than the 3.2M one on unseen data (§A.2 row 4, `SBP:119-121`). The
  repo used "~1.10 bpb at 27 KB/s" as "the decisive external datum" on 2026-11-09 (`AUD:125`); that sentence also
  mixed two models (`CMP:75`).
- Hutter-2026 lineage (still cited in `CLAUDE.md:41-42` and `RM:2621-2622`): the 6M transformer was "pre-trained on
  enwik9 on 8 RTX 5090" for 26 h and is scored on enwik9; the only enwik8 figure (fractal-zip 0.984) is also
  in-sample because enwik8 is the first 10^8 bytes of enwik9 (`CN/estado_arte_neural.md:33,144` — **EXTERNAL**).
  No out-of-sample figure for that lineage exists in the repo.
- Pretrained generalists: The Pile contains Wikipedia (`SBP:87`), so Mamba-130M / Pythia on enwik8 are seen-source;
  whether dickens (Gutenberg) or the Loghub samples are in The Pile was never checked.
- Latent leaks in recipes that never ran: `NB/mambabyte-train.py` validates on enwik8 first 10 KB (`:553-566`) while
  its fallback training source is the HF `wikipedia` dump (`:405-414`); both data and evaluation fall back silently
  to random bytes (`:421-435`, `:543-549`). The fallback fired in the only logged attempt:
  `R/mambabyte-50m-v1/mambabyte-50m-validation.log:25-27`.
- Baseline side: the zstd dictionary of R-113 was trained on the first half of the same file as the test messages
  (`NB/short-message-bench.py:257`, `P3:107-108`): best case for zstd, by design.

**B.3 Small static models do not generalise to unseen sources (R-112).** Four byte-level models (0.28M-3.6M)
trained on 11 Loghub systems give 4.0-9.2 bpb on 5 unseen systems, 11-38x worse than the best classical codec; the
transformer on Proxifier (9.23 bpb) is worse than not compressing. More training made it worse (§A.2 row 9). Longer
context does not help (1,024 B -> whole file: < 1% on the LSTMs, `specialist-local-eval.csv:2-5`). On seen systems
the 1M transformer matches Mamba-130M's backbone (0.516 vs 0.541) and loses to the classical best (0.427). Online
adaptation rescues the unseen systems (0.452 mean) but still loses to xz (0.295) and to Mamba-130M (0.202), costs
~3x compute and is not bit-reproducible run to run (0.3292 vs 0.3293 on apache, `LOGS:164-165`). Cause stated: "the
model learns the formats of the 11 systems" (`LOGS:75-76`) — consistent with the curves, not ablated. Limits of the
experiment (`LOGS:154-165`): 11 sources, 22-45 min per model, no CM, LSTM/transformer only, LSTMs evaluated on 8 of
16 sets. **Not measured: a small backbone mixed with the online CM ensemble on unseen sources.**

**B.4 Quantisation that touches the wrong thing.** R-064: +14.5% to +16.6% bpb and no speed gain (§A.7 row 3).
Per-row scale on the tied embedding/head: +6.6-7.0% (row 2). The working recipe is weight-only with block scales
and F32 activations (row 1). The original write-up concluded "Mamba-130M does not tolerate Q8 PTQ" (`RM:1769`);
eight days later Q8 cost +0.6-1.2%: a wrong generalisation from one implementation.

**B.5 The BPE-50K vocabulary starves the CM and sets a cost floor.**
- CM-only entropy over tokens: 15.49-15.62 bits (uniform = 15.62); CM-only top probability never above 5%
  (`RM:2101`, `RM:1587`). Gate at 15.5 bits: 31.5% skip, +50.5% bpb; at <= 14.0 bits: 0% skip (`RM:2102-2104`).
- Every attempt to skip or replace the backbone at token level failed: confidence skip +35.7% (enwik8) / +81%
  (apache) / +101% (`RM:1589-1590`); prediction cache +17.4% at 16.7% hits (`RM:1444-1447`); CM-only bypass +109.7%
  (`RM:595`). The CM alone codes enwik8 10 KB at 3.02 bpb, about gzip level (§A.1 rows 13-15).
- Cost floor: the output matrix is 30% of FLOPs and bytes per token at 130M and 84% of the active parameters at 14M
  (§A.4 rows 3-4); host buffers are 10 x vocab x 4 B per stream and the CDF needs one `exp` per vocabulary entry per
  token (§A.4 rows 6-7).
- One CM component is blind under BPE by construction: the word model marks token ids 9, 10, 13 and 32 as
  separators, which are byte values, and its own comment says that for a subword vocabulary "we can't know which
  tokens are separators" (`src/domain/word_model.rs:40-51`; also `01-integration-map.md` §tokenizers). Its effect on
  bpb under BPE vs bytes was never measured.
- The stated cause ("CM biases are negligible relative to a uniform distribution over 50K classes", `RM:2106-2107`)
  and the remedy ("viable only with vocab 256 or files > 100 KB", `RM:2110`; "CM entropy ~5-6 bits", "40-70% skip")
  are **CLAIMED**: nothing was ever measured at vocab 256 except the pretrained MambaByte-353M (§A.1 row 12).

**B.6 Host pipeline as the real limit of an accelerator tier (R-110).** Through the CLI the T4 is flat at
530-582 tok/s from N=12 to N=128, half the FFI harness (§A.3 rows 9-10); cost per token-stream is constant at
~1.75 ms. Even an infinitely fast forward leaves ~1,150 tok/s (~$17/GB) on a 4-vCPU host (`KF:162-166`). The
attribution to the logits copy vs CDF/CM/coder is an inference from two totals (`KF:153-154`).

**B.7 Runtime tokenizer that differs from the training tokenizer (new).** §A.5. No parity test between the Rust
tokenizer and the HF tokenizer exists in the repo; `COL:22-23` states "same `tokenizer.json` (26,001 tokens)" for
both, but 26,001 is the HF count and the Rust count for those bytes is 26,969. The planned PyTorch-vs-Rust logit
check (R-083, `RM:2194`) was never ticked.

**B.8 Shrinking a trained model after the fact.** Pruning one d_inner channel of Mamba-130M: +20.4%; 10%: +431%
(`RM:1683-1686`). Layer skip: +3.86% at 0.02% skips, +713% at 5.5% (`RM:1702-1704`). SVD of RWKV matrices: rank 750
of 768 already +3.47% (`R/r016-svd-pruning-evaluation.csv:2-5` †). Activation sparsity: 1.7-1.9% (`RM:1642-1643`).
Smaller pretrained generalists lose ratio continuously (§A.2 row 3).

**B.9 A pretrained model from another domain, and a silent architecture mismatch.** MambaByte-353M trained on arXiv:
1.3156 bpb on enwik8, +37% vs Mamba-130M, at 4.8 B/s; with the wrong norm type (RMSNorm applied to LayerNorm
weights) it gave 1.7531 without any error (`RM:2313-2317`). "Domain mismatch" is the stated diagnosis, not ablated.
The figure "0.396 bpb CE on enwik8" quoted for the same weights beforehand (`RM:2294`) was off by 3x.

**B.10 Projections that missed by 2-10x.** Q8 single stream 250-300 tok/s projected vs 111-127; lockstep
700-1,500 vs 315 (`AUD:195-201`); prediction cache "0-3%" vs +17.4% (`RM:1447`); column pruning "+1-3%"
(`docs/internal/research/lottery-ticket-sparsity-protocol.md:189` †) vs +20% to +431%; BPE entropy gate "15-35% skip
at -2..+3% bpb" (`TR:665-666` †) vs 31.5% at +50.5%. Consequence already adopted by the repo: decisions on
measurements only (`AUD:165-169`).

**B.11 Window artefacts.** "-50.5% vs brotli" is a 10 KB-window figure; on whole files brotli reaches 0.247 on UBL
(`CLAUDE.md:317-320`, `CMP:21`). "Advantage grows with size" was true only for the RWKV scale test.
With reset, whole-file margins vs the best classical are -41% facturae, -5% UBL, -41% apache, -34% syslog,
-42% openssh (`KF:99-103`); both XML files are synthetic (§E).

**B.12 Determinism traps.** An AVX2-encoded archive decoded on the scalar path fails ("decoded 10242 bytes,
expected 10240", `DX:117`): F32 SSM recurrence and F32 GEMV accumulate in a different order (`DX:52`); the Q8 GEMV
is tier-invariant by construction (`DX:54`). `REDUKT_NO_SIMD=1` "must match between compress and decompress"
(`src/main.rs:102`). libm differs across platforms (f32 tanh: 4.77% of inputs, `DX:154`). FFI v1 kept one global
state: interleaved streams gave 1.949 vs 0.750 bpb (`KF:191-197`). Verified so far: Windows <-> Linux x86-64 only
(9/9 sha, `DX:14`); ARM64 untested (`docs/internal/research/determinism-audit.md:18`).

**B.13 Training infrastructure that never trained.** The three MambaByte-50M attempts ran on CPU ("CPU only"),
were killed after 44-56 s and printed no step (`R/mambabyte-50m-v1/...log:1,28`, `v2:1,20`, `v4:1,20`). In R-112
the in-kernel LSTM evaluation died on a cuDNN error and was redone on CPU on 8 of 16 sets (`LOGS:38-42`). No model
designed by this project has ever been trained to completion except the four R-112 specialists (22-45 min each).

**B.14 Accounting errors that hid the gap.** Per-machine throughput compared with a per-vCPU threshold (corrected
in `AUD:213-217`); tok/s converted to B/s with an assumed 3.8 B/token (§A.3 rows 4-5; binaries are ~1 B/token, so
their cost per byte is ~3.8x that of text).

---

## C. Economic thresholds as engineering budgets

### C.1 Model (`scripts/breakeven.py:9-13`) and price points

```
C  = P_cpu * 1e9 / (3600 * R)                 compute $ per raw GB; R = bytes/s per vCPU
V  = (bpb_classical - bpb_model) / 8 * P_st * months     value $ per raw GB
R* = P_cpu * 1e9 / (3600 * V)                 break-even throughput per vCPU
```

With S3 Standard $0.023/GB-month x 84 months, one stored GB is worth $1.932, so
`C = k / R[KB/s]` and `min delta bpb = 4.141 * C`, with k = 3.889 ($0.014), 4.861 ($0.0175), 6.111 ($0.022),
10.07 ($0.03625), 12.39 ($0.0446).

| $/vCPU-hour | What it is | Used by |
|---|---|---|
| 0.014 | c7i spot | `NB/redukt-gpu-fullsize-bench.py:68`; every `gate-table.csv`; the ">55 KB/s" of `LOGS:146` |
| 0.0175 | c7a spot | `CMP:25` |
| 0.022 | "c7g/c7a/m7i" spot | `scripts/breakeven.py:32`; the 18 / 51 / 101 KB/s of `AUD:110-114` and `RM:2611-2615` |
| 0.03625 / 0.0446 | on-demand c7g / c7i | `scripts/breakeven.py:32`; `NB/redukt-gpu-fullsize-bench.py:68` |

Not in the model: decompression (same compute again per full read, §A.3 row 12), the classical codec's own cost
($0.003-0.011/GB), egress, discounts on storage. Other storage classes scale V linearly: B2 x0.30, on-prem
$50/TB-year x0.18, Deep Archive x0.043.

### C.2 Output of `scripts/breakeven.py` as shipped (run 2026-11-12)

Inputs are **10 KB-window** ratios (`breakeven.py:56`): spot $0.022, S3 Standard, 7 years.

| Data | b (brotli fraction) | S (extra saving) | Value $/raw GB | R* KB/s per vCPU | Deep Archive R* | On-prem R* |
|---|---|---|---|---|---|---|
| enwik8 text | 0.294 | 59% | 0.3351 | 18.2 | 423.6 | 100.7 |
| facturae XML | 0.111 | 56% | 0.1204 | 50.8 | 1,179.6 | 280.3 |
| apache log | 0.058 | 54% | 0.0608 | 100.6 | 2,336.2 | 555.1 |
| dickens | 0.347 | 77% | 0.5166 | 11.8 | 274.8 | 65.3 |

### C.3 Same formula with whole-file measurements (DERIVED; inputs §A.1 rows 4, 6; §A.2 rows 1, 5-7)

"Model bpb" is what Mamba-130M achieves today. The $0.014 column reproduces
`R/kaggle-fullsize/run-a-r108-gpu/gate-table.csv` exactly.

| Segment | Model bpb | Best classical | Delta bpb | Value $/raw GB | R* @ $0.014 | R* @ $0.022 | R* @ $0.0446 | R* if the model reached 0 bpb ($0.014 / $0.022) |
|---|---|---|---|---|---|---|---|---|
| Text: enwik8 2 MB | 0.9998 | brotli-11 2.2420 | 1.2422 | 0.3000 | 13.0 | 20.4 | 41.3 | 7.2 / 11.3 |
| Text: unseen Wikipedia 500 KB (backbone-only) | 0.998 | brotli-11 2.3608 | 1.3628 | 0.3291 | 11.8 | 18.6 | 37.6 | 6.8 / 10.7 |
| Prose: dickens 500 KB | 1.0620 | bz2-9 2.1892 | 1.1272 | 0.2722 | 14.3 | 22.4 | 45.5 | 7.4 / 11.6 |
| Logs: apache | 0.1849 | brotli-11 0.3132 | 0.1283 | 0.0310 | 125.5 | 197.2 | 399.8 | 51.4 / 80.8 |
| Logs: syslog | 0.2449 | xz-9e 0.3697 | 0.1248 | 0.0301 | 129.0 | 202.8 | 411.1 | 43.6 / 68.4 |
| Logs: openssh | 0.2014 | xz-9e 0.3460 | 0.1446 | 0.0349 | 111.4 | 175.0 | 354.8 | 46.5 / 73.1 |
| Logs: 5 unseen Loghub systems, mean (backbone-only) | 0.202 | 0.295 | 0.093 | 0.0225 | 173.2 | 272.1 | 551.6 | 54.6 / 85.8 |
| XML: facturae (synthetic) | 0.2408 | brotli-11 0.4078 | 0.1670 | 0.0403 | 96.4 | 151.5 | 307.2 | 39.5 / 62.1 |
| XML: UBL (synthetic) | 0.2392 | brotli-11 0.2519 | 0.0127 | 0.0031 | 1,268 | 1,992 | 4,039 | 63.9 / 100.5 |

**Budgets in KB/s per vCPU (spot $0.014 - $0.022):** text **13-20** at today's ratio (never below 7-11); logs
**110-200** at today's ratio (never below 44-86); invoice XML **96-150** on the one synthetic facturae file and
~1,300-2,000 on the synthetic UBL file. Measured with Mamba-130M: **0.069** (Xeon) and 0.037-0.044 (laptop, mixed
set). Gap: 190-300x on text, 1,600-2,900x on logs.

Accelerator form of the same budget (g4dn.xlarge spot $0.247/h, on-demand $0.526/h): text 229 / 487 KB/s per T4;
logs ~2,000-2,300 / ~4,200-4,900 KB/s per T4. Measured through the CLI: 1.8-2.1 KB/s per T4.

### C.4 Minimum ratio at a given speed

`bpb_model < bpb_codec - 4.141 * k / R`. Largest admissible model bpb (spot $0.014 / $0.022); "none" = no ratio
suffices at that speed.

| R, KB/s per vCPU | $/GB ($0.014 / $0.022) | Min delta bpb | Text (codec 2.242) | Apache (0.3132) | Syslog (0.3697) | Facturae (0.4078) | UBL (0.2519) |
|---|---|---|---|---|---|---|---|
| 0.069 (today) | 56.4 / 88.6 | 233 / 367 | none | none | none | none | none |
| 10 | 0.389 / 0.611 | 1.610 / 2.530 | 0.632 / none | none | none | none | none |
| 25 | 0.156 / 0.244 | 0.644 / 1.012 | 1.598 / 1.230 | none | none | none | none |
| 50 | 0.078 / 0.122 | 0.322 / 0.506 | 1.920 / 1.736 | none | 0.048 / none | 0.086 / none | none |
| 100 | 0.039 / 0.061 | 0.161 / 0.253 | 2.081 / 1.989 | 0.152 / 0.060 | 0.209 / 0.117 | 0.247 / 0.155 | 0.091 / none |
| 250 | 0.016 / 0.024 | 0.064 / 0.101 | 2.178 / 2.141 | 0.249 / 0.212 | 0.305 / 0.268 | 0.343 / 0.307 | 0.187 / 0.151 |
| 1,000 | 0.004 / 0.006 | 0.016 / 0.025 | 2.226 / 2.217 | 0.297 / 0.288 | 0.354 / 0.344 | 0.392 / 0.382 | 0.236 / 0.227 |

Per-codec baselines on the same bytes (MEASURED; reproduced locally in the audit with the Python bindings):

| Bytes | brotli-11 (w24) | xz-9e | zstd-19 | zstd-22 | bz2-9 | gzip-9 |
|---|---|---|---|---|---|---|
| enwik8 2 MB | 2.2420 | 2.2885 | 2.3578 | 2.3577 | 2.3056 | 2.8928 |
| enwik8 100 MB | 2.0594 | 1.9865 | 2.1564 | 2.0267 | — | 2.9156 |
| apache-access.log | 0.3132 | 0.3151 | 0.3345-0.3370 | 0.3349-0.3375 | 0.3326 | 0.4284-0.4298 |
| linux-syslog.log | 0.3789 | 0.3697 | 0.4185-0.4204 | 0.4185-0.4204 | 0.5261 | 0.5421-0.5444 |
| openssh-auth.log | 0.3592 | 0.3460 | 0.3855-0.3857 | 0.3855-0.3857 | 0.4269 | 0.5522-0.5532 |
| facturae-batch.xml (LF, 38,371 B) | 0.4078 | 0.4428 | 0.4731-0.4739 | 0.4670-0.4679 | 0.5283 | 0.5631-0.5696 |
| ubl-invoice-batch.xml (LF, 81,558 B) | 0.2519 | 0.2809 | 0.2934-0.2938 | 0.2823-0.2827 | 0.3506 | 0.3850-0.3883 |

(Ranges: Python bindings vs Kaggle CLI.) Sources: `R/kaggle-fullsize/run-a-r108-gpu/classical-fullsize.csv:27-66`;
bz2 from `R/log-specialist/run-1/specialist-classical.csv:58,63,68`, `SBP:198-199` and, for enwik8 2 MB, the audit.
`xz -9` without `-e` is 7-18% worse on the three logs (0.3381 / 0.4318 / 0.4206), which is what
`R/competitors-fullsize-2026-11-01.csv:14-17` contains.

### C.5 Compute per byte implied by a budget (DERIVED; one assumption stated)

If the runtime delivered the same effective rate per vCPU as it does today with Mamba-130M — 5.8-7.9 GFLOP/s per
logical CPU on the laptop (tokens logged, §A.3 row 19); on the Kaggle Xeon the tokens of the 12-slice run were not
logged: 69 B/s per vCPU at the 2.5-3.5 B/token the Rust tokenizer gives on such data is 20-28 tok/s per vCPU, i.e.
roughly 5-7 GFLOP/s per vCPU — then:

| Budget R, KB/s per vCPU | at 5 GFLOP/s | at 10 GFLOP/s | at 20 GFLOP/s |
|---|---|---|---|
| 13 | 0.385 MFLOP/byte | 0.769 | 1.538 |
| 18.2 | 0.275 | 0.549 | 1.099 |
| 50 | 0.100 | 0.200 | 0.400 |
| 100 | 0.050 | 0.100 | 0.200 |
| 200 | 0.025 | 0.050 | 0.100 |

Reference points: Mamba-130M 71.35 MFLOP/byte; R-112 lstm-xs 0.529; L3TC-200K 0.98 (`SBP:139`). **The assumption is
unmeasured for small models** (`SBP:90-91`): with a small backbone the CM, the CDF and the coder dominate, and at
vocab 50,280 the CDF alone is ~0.47 ms per token (§A.4 row 7), i.e. at most ~7 KB/s per thread at 3.5 B/token.
The laptop's BLAS peak is far above the product's effective rate (§A.3 rows 18-19), so the effective rate is a
property of the current pipeline, not of the silicon.

---

## D. Path 3 — narrowband satellite messaging: measured constraints

### D.1 Value and traffic shape
- Price per transmitted GB (**EXTERNAL**, reseller prices 2025-2026, not official tariffs, `P3:17-35`): Iridium SBD
  $0.79-3.15M; Iridium IMT $300-500K; BGAN M2M ~$30K; Certus 100 $3.5-15K; BGAN IP $5.8-7K. Not viable: cellular IoT
  $28-210, Starlink maritime $1.15-6, cloud egress $0.05-0.12.
- Compute cost is not the binding term there ($33-60/GB vs >= $3.5K/GB); what binds is whether the **sending
  terminal** can run the model (`P3:131-132`).
- Message sizes tested: 128 / 256 / 512 / 1,024 / 2,048 B; 16 messages per source and size; 6 sources (Wikipedia,
  dickens, Loghub Apache / Linux / OpenSSH, GitHub-event JSON); 480 messages; roundtrip 24/24 sampled
  (`NB/short-message-bench.py:49-50,199`, `short-message-report.txt:26-27`).

### D.2 Results and framing
- vs brotli-11 (built-in dictionary): `.azth` smaller in 30/30 cells, -41% to -53% on the all-source means.
- vs zstd-19 with a 110 KB dictionary trained on the first half of the **same source**: `.azth` wins 16/30 cells;
  all text cells (-13% to -52%) and all six sources at 2 KB. It loses on logs/JSON at 128-512 B: apache +57% / +59% /
  +53% (128 / 256 / 512 B), openssh +42% / +36% / +31%, JSON +41% at 128 B (`P3:72-91`).
- Payload alone (file minus the fixed framing) wins 24/30; still loses on apache at 128-1,024 B (+3%, +15%, +21%,
  +1%) and openssh at 256-512 B (+4%, +8%) (`P3:72-86`).
- **Fixed framing: 20 B, not 19.** A default Mamba Q8 archive starts
  `41 5A 54 48 05 00 00 00 | 00 01 00 00 | C0 | 51 | 0C | 01` = MAGIC 8 + TOTAL_BYTES 4 + FLAGS + PROFILE +
  RESET_LOG2 + BACKBONE = 16 B, plus CRC-32 4 B (MEASURED (audit); agrees with `docs/api/FORMAT-SPEC.md:189`). The
  bench assumed 19 (`NB/short-message-bench.py:51`), so its "payload" figures are 1 B too high. The same message
  (log-apache, 256 B, index 0) gives 83 B on the Kaggle Xeon batch and on this laptop single-file
  (`R/short-message-bench/run-1/short-message-results.csv:3234`).
- Anatomy of that message: CE 56.4 B -> 83 B on the wire = 20 B framing + ~6.6 B of coder cost outside the CE;
  zstd+dict: 42 B. Two fixed terms explain the 6.6 B: the first token is coded with a uniform CDF over the
  vocabulary (log2 50,280 = 15.6 bits, about 2 B; `src/application/codec.rs:1197-1198`,
  `src/application/lockstep.rs:310-311`) and the range coder flushes 5 bytes at the end
  (`src/domain/coder.rs:184-197`). Fixed cost per message today: 16 + 4 + ~2 + ~5 = ~27 B. At 128 B the
  framing alone is 35% of the mean `.azth` (20 / 57.8).
- The 4,096-token reset never fires at these sizes; the CM has nothing to learn from within a message (one data
  point: ensemble CE worse than backbone-only CE, §A.2 row 16).

### D.3 Terminal footprint, speed, determinism
- Model artefact: 516,567,560 B F32 safetensors + 2.1 MB tokenizer; quantisation happens at load. Weight bytes
  touched per token in Q8: 153.6 MB. Peak working set of one Q8 `compress` of a 256-B message: ~791 MB (MEASURED
  (audit), Windows, 50 ms polling). `P3:142` and `CLAUDE.md:65` say "~130 MB model": that is neither the file nor
  the resident size.
- Speed: 480 messages in 1,082.5 s = 124.4 tok/s aggregate on 4 vCPU, 2.3 s per message wall
  (`short-message-report.txt:26`); "1-4 s per 256 B-1 KB message" single stream on a 4-vCPU server (`P3:143`,
  partly inferred); 1.7 s for a 256-B message on this laptop (audit). **Nothing measured on ARM; no energy figure.**
- Determinism: bit-identical only x86-64 Windows <-> Linux, any thread count, `target-cpu` native or x86-64-v3
  (`DX:14,115`). A scalar-path decoder cannot read an AVX2-path archive (§B.12). ARM64 untested; R-104 F2 (scalar
  order = AVX2 lanes) pending and promoted to a requirement by R-115 (`RM:3120-3125`).

### D.4 What this asks of a backbone (requirements as measured gaps)
1. Useful prior from byte 0, evaluated at 128-2,048 B: the benchmark of §F.3 is the gate (R-114 criterion:
   beat zstd-19 + dictionary in >= 24 of 30 cells, `RM:3119`).
2. Total fixed cost per message (header + CRC + uniform first token + coder flush) is today ~27 B against 35-66 B
   of dictionary-coded output for logs/JSON at 128-256 B; the first-token term scales with log2 of the vocabulary.
   The model-side gap on the worst cells is 15-21% of payload.
3. Must run on the sender: file size, resident memory and seconds per message need an ARM measurement; today
   0.5 GB / 0.8 GB / 1-4 s on x86.
4. Encoder (ARM) and decoder (x86) must produce the same CDF bit for bit: any floating-point accumulation whose
   order depends on the ISA breaks decoding.
5. Evidence covers text, logs and JSON only; packed binary telemetry is unmeasured (`P3:153-154`); market size
   unverified (`P3:148-152`).

---

## E. Local data inventory and a leak-free split by source

### E.1 What is on disk (`data/`)

| File | Bytes | Content (type verified with `file`) | Provenance | Used as evaluation data in |
|---|---|---|---|---|
| `benchmark/enwik8` | 100,000,000 | English Wikipedia XML dump, UTF-8 | LTCB | every text reference: prefixes 10 KB-2 MB; windows at i x 1 MB (i=0..6), i x 700 KB (i=0..127), i x 12 MB (i=0..7); first 4 MB for messages (`NB/redukt-gpu-fullsize-bench.py:55,219,228`, `NB/small-backbone-probe.py:158,162`, `NB/short-message-bench.py:94`) |
| `benchmark/silesia/dickens` | 10,192,446 | English prose, ASCII CRLF | Silesia | 12-file table; probe and reset sweep (first 500 KB); messages (first 4 MB); collapse control (80 KB); RWKV scale test |
| `benchmark/silesia/samba` | 21,606,400 | tar of source code | Silesia | 12-file table; RWKV scale test |
| `benchmark/silesia/nci` | 33,553,445 | chemical database, ASCII | Silesia | 12-file table |
| `benchmark/silesia/mozilla` | 51,220,480 | tar of executables | Silesia | 12-file table |
| `benchmark/silesia/mr`, `ooffice`, `sao`, `x-ray` | 9,970,564 / 6,152,192 / 7,251,944 / 8,474,240 | medical image / PE32 DLL / star catalogue / X-ray image | Silesia | 12-file table (the four losses) |
| `benchmark/silesia/webster` | 41,458,703 | dictionary text, ASCII CRLF | Silesia | RWKV-era 19-file table only |
| `benchmark/silesia/xml` | 5,345,280 | tar of XML documents | Silesia | RWKV-era 19-file table; Q4 evaluation |
| `benchmark/silesia/reymont` | 6,627,202 | PDF 1.3 | Silesia | RWKV-era 19-file table only |
| `benchmark/silesia/osdb` | 10,085,684 | database, binary | Silesia | RWKV-era 19-file table only |
| `benchmark/industry/apache-access.log` | 171,239 | Loghub `Apache_2k.log` (1,999 lines; an Apache **error** log) | `scripts/prepare-benchmark-data.py:22` | whole-file and 10 KB tables; messages; R-112 held-out |
| `benchmark/industry/linux-syslog.log` | 216,485 | Loghub `Linux_2k.log` | `...:24` | same |
| `benchmark/industry/openssh-auth.log` | 225,216 | Loghub `OpenSSH_2k.log` | `...:25` | same |
| `benchmark/industry/facturae-batch.xml` | 39,384 (CRLF; 38,371 as LF on Kaggle) | **synthetic** FacturaE 3.2.2: one template x 20 invoices | `...:148-269` | whole-file and 10 KB tables |
| `benchmark/industry/ubl-invoice-batch.xml` | 83,260 (CRLF; 81,558 as LF) | **synthetic** UBL 2.1: one template x 20 invoices | `...:45-145` | same |
| `benchmark/industry/sample-invoice.pdf` | 13,264 | PDF 1.4 | — | PDF benchmark |
| `benchmark/enwik8.zip`, `silesia.zip` | 36,445,475 / 68,182,744 | archives of the above | — | — |
| `test-fixtures/golden.azth`, `golden-input.bin` | 35 / 39 | format golden (v4) | — | unit test |

Totals: 100 MB Wikipedia + 211.9 MB Silesia + 0.75 MB industry. **Every file has been used as an evaluation file;
there is no training-only corpus on disk.** The two XML files are not valid as a generalisation gate (`SBP:89`,
`RM:3095`). The Kaggle and local copies of the XML differ in line endings (bpb differs by 1-4%, `KF:105-106`).

### E.2 Not available locally

| Corpus | Needed for | Status |
|---|---|---|
| Loghub full logs (Zenodo 3227177): BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack, HealthApp (train), Zookeeper, Proxifier (held-out) | R-112 protocol | not local; only the three 2k samples are (`HDFS_2k` is listed in the script but absent) |
| enwik9 | "unseen Wikipedia" set (512,000 B at offset 600 MB) | not local |
| GH Archive `2015-01-01-15.json.gz` | JSON cells of the short-message bench | not local |
| PG19, Wikipedia 20220301.en, SlimPajama, OpenWebText | planned MambaByte training data (HF streaming) | not local; never successfully streamed |
| The four R-112 checkpoints (`models/*.pt`) | re-evaluating the specialists | not in the repo |
| L3TC checkpoints + rebuilt tokenizer, Pythia, SmolLM2 | re-running the probe | not local (HF cache holds only mamba-130m-hf and rwkv-7-world) |
| Real e-invoice XML; real satellite traffic (weather reports, shipboard mail, JSON telemetry) | XML gate; path 3 (R-116) | do not exist in the project |

### E.3 Proposed split (by source; no file contributes to more than one role)

| Role | Sources | Rule |
|---|---|---|
| **Held-out test** (frozen, report-only) | `enwik8` (whole), `dickens` (whole), the three Loghub samples, both synthetic XML, `samba`, `nci`, `mozilla`, `mr`, `ooffice`, `sao`, `x-ray`, `sample-invoice.pdf` | Never used for training, early stopping, hyper-parameter choice, vocabulary or dictionary building. They carry every Mamba-130M reference number, so a new backbone is compared on the same bytes |
| **Validation** (model selection) | `webster` (text), `xml` (markup), `reymont` (PDF container), `osdb` (binary records) | Distinct sources from both other roles; selection decisions use these only |
| **Train** | none on disk | Must come from outside `data/` (§E.2). For a pipeline smoke test only: train on `webster` + `osdb`, validate on `xml` + `reymont`; such a run says nothing about generalisation |

Exclusions that follow from "by source":
1. Systems, not files: the full Loghub Apache, Linux and OpenSSH logs stay out of training (the 2k samples are
   their subsets). `Linux_2k` and `OpenSSH_2k` both consist largely of sshd lines; treat them as related sources.
2. enwik8 is the first 10^8 bytes of enwik9; any Wikipedia dump (any date) is the same source as both.
3. Anything produced by `scripts/prepare-benchmark-data.py` is the same source as the two XML test files.
4. A pretrained initialisation or a tokenizer/vocabulary built on a corpus counts as training on it (§B.2).
5. Report "seen-source" results (later slice of a training source, as R-112 did) separately; never average them
   with held-out results.
6. Keep a second validation set of sources if the held-out sources are monitored during training:
   `NB/log-specialist-train.py:463-473` monitored the held-out systems (no selection was done there).

---

## F. Existing evaluation protocols reusable as independent gates

Local binary used by the audit: `target/maxperf/redukt.exe` as built on 2026-11-11 from the sources at 692d985
(nothing was compiled during the audit). All Kaggle protocols are launched with
`cp notebooks/<name>.py notebooks/<name>/ && python -m kaggle kernels push -p notebooks/<name>`; pushes are subject
to `00-charter.md` §5 and §5b.

**F.1 Correctness gates (`CLAUDE.md` "Quality Gates").**
```
cargo build --release            # zero warnings
cargo test                       # unit + tests/roundtrip.rs + tests/api_parity.rs + tests/libm_lint.rs
redukt compress   --input F --output F.azth --backbone mamba --quant q8|f32 --weights weights/mamba-130m --no-governor [--bytes N]
redukt decompress --input F.azth --output F.dec --weights weights/mamba-130m --no-governor
redukt compress   --input-dir D --output-dir O --jobs N ...     # batch .azth sha256 must equal single-file sha256
```
Reference bytes: enwik8 `--bytes 10240` -> 1,235 B (F32) / 1,241 B (Q8, sha256 d550671b...); 100 KB -> 12,704 /
12,737 B (sha d907ffc0...) (`KF:112-116`). Cross-OS table: `R/determinism-cross-machine-2026-11-10.tsv`.

**F.2 Length curve 10 KB -> 2 MB (R-108).** `NB/redukt-gpu-fullsize-bench.py`: prefixes
`[10_240, 51_200, 102_400, 512_000, 1_048_576, 2_097_152, 4_194_304]` of enwik8 (`:55`), single stream, roundtrip
at 10 KB and 100 KB (`:325-346`); whole industry files (`:367-385`); 12 x 10 KB lockstep for per-vCPU rate
(`:388-404`); 128 windows x 10/50 KB for batch scaling (`:223-228`). Reference: §A.1 rows 4-7 with classical codecs
in `run-a-r108-gpu/classical-fullsize.csv`. Acceptance criterion already written: whole-file bpb <= 10 KB bpb + 3%
(`RM:2941`). Local form: the F.1 commands with `--bytes N` on `data/benchmark/enwik8` (at ~0.2-0.4 KB/s single
stream, 2 MB takes 1.5-3 h on CPU).

**F.3 Short messages (R-113).** `NB/short-message-bench.py`: `cut_messages` takes `count` messages of `size` bytes
from the second half of each source, each starting at a line start (`:117-125`); classical codecs with minimal
framing and a 110 KB zstd dictionary trained on the first half (`:130-158`); Redukt through
`compress --input-dir ... --jobs 32 --backbone mamba --quant q8` (`:190-191`); roundtrip of 24 sampled messages.
Outputs: `short-message-results.csv` (per message), `short-message-summary.csv`. Reproducible offline for
Wikipedia, prose and the three logs; the JSON source needs GH Archive. Fix before reuse: `AZTH_OVERHEAD = 19`
should be 20 (§D.2). Add a dictionary trained on a **different** source than the test messages (§H.12).

**F.4 Unseen-source split (R-112).** `NB/log-specialist-train.py`: split by system (`:54-58`), 48 MB cap per
training system, 300 KB "seen" slice 1.5 MB past the training range (`:159-164`); static evaluation in independent
chunks from a zero state, first byte of each piece charged 8 bits (`:326-357`); dynamic evaluation, one Adam step
(lr 5e-4) per 256 B with carried state (`:360-381`); Mamba-130M reference in 4,096-token chunks (`:402-426`).
Training recipe used: AdamW, peak lr 2e-3 (LSTM) / 1e-3 (transformer), weight decay 0.01, 200 warm-up steps, cosine
to 5% over a wall-clock budget, clip 1.0, batch 128 / 64 x 1,024 B, uniform sampling over systems, seed 1204
(`:270-321`). Local re-evaluation:
```
python data/benchmark/results/log-specialist/run-1/local_eval.py notebooks/log-specialist-train.py <dir> out.csv \
    lstm-s static-4096,dynamic-256,dynamic-256-random-init        # needs <dir>/models/*.pt and <dir>/data/*.log
```
Needs the Loghub download and the checkpoints (§E.2). The three benchmark logs it holds out are local.

**F.5 Backbone-only cross-entropy with reset chunking (R-105).** `NB/small-backbone-probe.py`: datasets (`:149-194`:
8 x 10,240 B at i x 12 MB; first 512,000 B; enwik9 at 600 MB; dickens first 512,000 B; five industry files),
`chunk_ids` (BOS + C-1 tokens, `:225-228`), `hf_bits` (`:231-252`), analytic cost = 2 x parameters touched per token
(`:255-263`), one-thread speed probe (`:266-285`). Switches: `PROBE_ONLY=<labels>`, `PROBE_SMOKE=1`. Caveat: it
measures HF tokenisation, not the product's (§A.5).

**F.6 Reset period and overlap (R-111).** `NB/reset-period-sweep.py`: real archives with
`compress --input-dir ... --backbone plugin --reset-every P` for P in 2048..32768 (`:124-137`), batch roundtrip of
the 8192 archives, and `overlap_bits` (state re-primed with the last K tokens, `:167-183`). CPU form:
`--backbone mamba --reset-every P`.

**F.7 Economic gate.** `python scripts/breakeven.py` (no arguments); `gate()` in
`NB/redukt-gpu-fullsize-bench.py:504-584` writes `gate-table.csv` (value per raw GB, R* per vCPU and per T4).
State the $/vCPU-hour used (§C.1).

**F.8 Training recipe on file that never ran.** `NB/mambabyte-train.py`: vocab 256, d_model 768, d_inner 1,536,
d_state 16, d_conv 4, dt_rank 48, 24 layers, tied embeddings (~96M); 8,192-byte sequences, batch 48, AdamW lr 6e-4
-> 6e-5 cosine, 2,000 warm-up steps, betas (0.9, 0.95), weight decay 0.1, clip 1.0, 76,000 steps ~ 30 B bytes, F32
(`:42-67`, `:775-780`); the SSM scan is an explicit Python loop over the sequence (`:225`). Hazards: §B.2, §B.13.

**F.9 Probes run in this audit (scripts kept outside the repo; method reproducible as below).**
```
redukt mamba-eval --input data/benchmark/enwik8 --bytes 10240 --weights weights/mamba-130m --no-governor
    -> "tokenized: 3611 tokens (2.84 bytes/token)", BPB 0.906514, 48.4 tok/s        (token counts of §A.5: same command per file)
HF reference, offline (HF_HUB_OFFLINE=1; model in the local HF cache):
    tok/model = AutoTokenizer/AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", local_files_only=True)
    ids = tok(text, add_special_tokens=False).input_ids; bits = sum CE(model(ids[:-1]).logits, ids[1:]) / ln 2
    -> 3,196 tokens; 8,830.4 bits = 0.862343 bpb; 2.238 bits/token over the first 2,000 predicted tokens
redukt compress --input msg256.bin --output msg256.azth --backbone mamba --quant q8 --weights weights/mamba-130m --no-governor
    msg256.bin = 256 B from the first line start after the midpoint of apache-access.log
    -> 83 B; CE 1.7611 bpb; coder efficiency 67.90%; 1.7 s; header bytes as in §D.2
Classical codecs: Python zlib / bz2 / lzma (preset 9 and 9|EXTREME) / brotli (quality 11, lgwin 24) / zstandard (19, 22)
```

---

## G. Hardware and tools of this machine

### G.1 Hardware (queried 2026-11-12: `Win32_Processor`, `Win32_CacheMemory`, `GetLogicalProcessorInformationEx`)

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i5-1235U; 10 cores / 12 logical: 2 P-cores with SMT (efficiency class 1) + 8 E-cores (class 0) |
| P-core caches (each) | L1d 48 KB (12-way), L1i 32 KB (8-way), L2 1,280 KB private (10-way) |
| E-core caches | L1d 32 KB, L1i 64 KB per core; L2 2,048 KB **shared by each cluster of 4 E-cores** (2 clusters, 16-way) |
| L3 | 12,288 KB shared by all cores (12-way); line size 64 B everywhere |
| ISA (rustc `target-cpu=native`) | avx, avx2, fma, f16c, bmi1, bmi2, avxvnni, aes, sha, gfni, vaes, vpclmulqdq; **no AVX-512**; torch reports capability AVX2 |
| RAM | 2 x 16 GB DDR4 (SMBIOS type 26), rated 3200, configured 2667 MT/s, two DIMMs; 31.7 GB visible, 16.9 GB free during the audit |
| GPU | Intel Iris Xe (integrated); `torch.cuda.is_available()` = False; no `nvcc` |
| OS / power | Windows 11 Pro 10.0.26300; on AC, "Balanced" plan |
| Disk | 137.7 GB free on C: |
| Remote hardware used by the project | Kaggle: T4 16 GB + 4-vCPU Xeon (all GPU and "per vCPU cloud" figures); Kaggle TPU never started. GPU quota stated by the owner: 18.5 h remaining, 12 h planned for this effort (`00-charter.md:112`) |

### G.2 Tools

| Tool | Version / status |
|---|---|
| Rust | rustc 1.98.1 (2026-09-01), cargo 1.98.1, LLVM 22.1.8, host `x86_64-pc-windows-msvc`; installed targets: windows-msvc and windows-gnu only (no aarch64, no Linux); `.cargo/config.toml` sets `-C target-cpu=native` |
| Built binaries | `target/maxperf/redukt.exe` (1.20 MB), `target/release/redukt.exe` (1.71 MB), both 2026-11-11 as found at the start of the audit; the implementation stage may have rebuilt them since |
| Python | 3.13.5 (CPython, AMD64) |
| numpy / torch | 2.3.5 / 2.14.0+cpu (MKL 2026.1, OpenMP, oneDNN 3.12; 10 threads by default) |
| Other Python packages present | transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, sentencepiece 0.2.3, huggingface_hub 1.33.0, scipy 1.16.3, pandas 3.0.3, kaggle 2.2.5, tqdm, psutil, brotli 1.2.1, zstandard 0.25.0 (zstd 1.5.7); stdlib lzma, bz2, zlib |
| Python packages absent | datasets, mamba_ssm, causal_conv1d, onnx, onnxruntime, numba, matplotlib, lz4 |
| `xz`, `zstd`, `brotli` on PATH | PowerShell PATH: **none** (only `tar.exe`). Git Bash PATH: `xz` 5.8.3, `brotli` 1.2.0, `gzip` 1.14, `bzip2` 1.0.8, `sha256sum`; **`zstd` CLI absent everywhere** -> use the Python `zstandard` binding (as the notebooks do) |
| Other | git, docker CLI, kaggle CLI, wsl.exe present (not exercised) |
| Weights on disk | `weights/mamba-130m` (516,567,560 B + tokenizer), `weights/rwkv7-0.1b` (382,105,536 B + vocab), `weights/mambabyte-353m` (1.50 GB safetensors + 1.50 GB .bin; no entry in `weights/MANIFEST.md`) |
| Local HF cache | `state-spaces/mamba-130m-hf` (518 MB), `BlinkDL/rwkv-7-world` (382 MB) |

### G.3 Quick compute probes (MEASURED (audit); single pass, other processes running)

| Probe (torch + MKL, float32) | Result |
|---|---|
| SGEMM n=512 / n=1,536, 1 thread | 121.0 / 86.9 GFLOP/s |
| SGEMM n=512 / n=1,536, 10 threads | 287.6 / 310.7 GFLOP/s; 12 threads 294.9 / 335.3 |
| SGEMV 1 thread: 256x256 (0.26 MB) / 768x768 (2.4 MB) / 1,536x768 (4.7 MB) / 3,072x768 (9.4 MB) | 16.2 / 27.5 / 17.3 / 23.4 GFLOP/s (123,648 / 23,276 / 7,338 / 4,960 products per second) |
| SGEMV 1 thread: 50,280x768 (154 MB, the LM-head shape) | 11.5 GFLOP/s = 148 products/s = 22.9 GB/s of weights read |
| Product, same machine | F32 single stream 48.4 tok/s today; documented 53.6 (F32) / 111-127 (Q8) / 315 aggregate (Q8, N=32) |

---

## H. Open questions with no evidence either way

1. **Small backbone + online CM on unseen sources.** R-112 measured backbone-only models; the product's adaptive
   ensemble was never combined with a small backbone (`LOGS:159-160`, `SBP:84-86`, `SBP:209`).
2. **Does source diversity fix B.3?** 11 systems failed; hundreds of sources were not tried (`LOGS:156-158`).
3. **Runtime speed of a small model in Rust.** Only PyTorch one-thread numbers exist (§A.3 row 13). The KB/s per vCPU
   of the CM + CDF + coder path at a small vocabulary has never been measured.
4. **Anything at vocab 256 besides one mismatched pretrained model:** CM entropy, CM confidence, skip rates, coder
   efficiency with a byte model trained in-domain. All figures in the repo are projections (§B.5).
5. **Length behaviour beyond the training context** of any architecture other than Mamba-130M; RWKV-7 is measured to
   100 KB only (§A.8 row 4).
6. **How much ratio the tokenizer mismatch costs** beyond the one 10 KB measurement (+5.1%), and whether the
   collapse onset and the optimal reset period move with the reference tokenisation.
7. **Logit-level parity** between the Rust CPU forward and the PyTorch reference (never run; the CUDA plugin was
   checked against NumPy on 8 steps only).
8. **Re-warmed reset inside the product** (-20% bits backbone-only on industry files; with CM unknown, `RST:62-65`).
9. **ARM64:** speed, memory, energy per message, and whether any archive roundtrips ARM <-> x86.
10. **Cost of making the scalar path match the AVX2 order** (R-104 F2) in tok/s.
11. **CM contribution on short messages** (one data point suggests negative) and the effect of a 2-4 B message
    framing (R-114).
12. **Dictionary baselines that are not best-case:** zstd dictionary trained on a different source than the test
    data; per-format dictionaries for invoice XML (never run, `CN/compresores_clasicos_coste.md:182` †).
13. **Real invoice XML and real satellite traffic:** none exists in the project; XML conclusions rest on two
    synthetic files of 20 templated invoices each.
14. **Full enwik8 (100 MB) and the 4 MB point** with any Redukt configuration (never run).
15. **Quantisation below 8 bits, int8 activations with block scales, or QAT** for a model of this class.
16. **Training budget on available hardware:** only tiny models on a T4 are measured (§A.3 row 15); no local CPU
    training measurement; the Kaggle TPU never started; no project-designed model beyond 3.6M has been trained.
17. **Whether the reference test files are inside Mamba-130M's pretraining data** (Wikipedia: yes per `SBP:87`;
    dickens, Loghub samples: unchecked).
18. **Decompression economics:** the break-even model ignores reads although decode costs the same as encode.
19. **A deterministic online-trained backbone:** cost and feasibility of bit-exact training on both ends.
20. **Split of the GPU host overhead** (logits copy vs CDF vs CM vs coder); R-110 was deprecated before measuring.
21. **Binary / high-entropy data with reset and whole files** for the Mamba tier (only 10 KB windows exist).
22. **Peak memory and load time of the runtime** as engineering targets (one audit data point: 791 MB RSS).

---

## I. Contradictions between documents (both sides, and which is newer)

| # | Quantity | Version A | Version B | Newer / resolution |
|---|---|---|---|---|
| 1 | Fixed framing of a default Mamba archive | 19 B = 15 + 4 (`P3:51-52`, `NB/short-message-bench.py:51`, `CLAUDE.md:64`; 2026-11-11) | 16-B header (`docs/api/FORMAT-SPEC.md:189`) + 4-B CRC = 20 B | B; confirmed by hexdump in the audit |
| 2 | Classical codecs in `KF` table A | 10 KB gzip 3.666; 50 KB brotli 2.35; 2 MB zstd-19 2.28, xz 2.17 (`KF:13-18`) | 2.9883; 2.4528; 2.3578, 2.2885 (`run-a/classical-fullsize.csv:2,11,28,30`) | CSV (primary data); 10 KB and 2 MB reproduced locally. Brotli matches except at 50 KB; the 100 MB row matches |
| 3 | xz on the three logs | 0.3384 / 0.4321 / 0.4209 (`R/competitors-fullsize-2026-11-01.csv:14,16,17`) | 0.3151 / 0.3697 / 0.3460 (`classical-fullsize.csv:55,60,65`) | Both correct: `xz -9` vs `xz -9e`. Best classical for syslog/openssh is xz-9e; `KF:51-52` and `COL:66-67` compare against brotli (-35% / -44%), `KF:102-103` against xz (-34% / -42%) |
| 4 | Laptop bytes/s | ~1,180 B/s, ~98 B/s per vCPU (`AUD:213-215`, `CLAUDE.md:24`) | 441-522 B/s wall on the mixed set (`R/lockstep-throughput-2026-11-10.csv:2-5`); $120-140/GB (`CMP:52`) | A is tok/s x 3.8 B/token (text-like); B is the wall rate of the actual files (2026-11-10/11) |
| 5 | Spot price per vCPU-hour | $0.022 (`scripts/breakeven.py:32`, 2026-11-09) | $0.014 (`NB/redukt-gpu-fullsize-bench.py:68`), $0.0175 (`CMP:25`), $0.015 (`CN/coste_inferencia_modelos_pequenos.md:90` †) | No single value; text R* is 18.2 / 13.0 / 14.7 KB/s accordingly. State the price with every R* |
| 6 | Value thresholds | $0.33 / $0.12 / $0.06 per GB (`CLAUDE.md:22-23`; 10 KB windows vs brotli) | $0.300 / $0.040 / $0.030-0.035, UBL $0.003 (`run-a-r108-gpu/gate-table.csv`, 2026-11-11) | B is newer and uses whole files and the best classical codec |
| 7 | Tokens in enwik8 first 100 KB for `redukt mamba-eval` | 26,001 (`COL:8`, `COL:22-23`) | 26,969 Rust tokens; 26,001 is the HF count (audit) | Audit |
| 8 | Bytes per token | 3.8 (`AUD:214`, `RM:3016`); 3.5 / 4.0 elsewhere † | Rust tokenizer: 2.84 (enwik8 first 10 KB), 3.80 (first 100 KB), 3.41-3.59 (128 windows, `KF:134-138`), 2.1-2.8 (industry files) | Measured values; `TR:13` already said 2.84 |
| 9 | First row of the collapse table (tokens 0-2000) | Redukt ~3.4 vs HF 2.24 bits/token (`COL:14`) | HF 2.238 reproduced; Rust cumulative 0.3303 bpb x 10,240 B = 3,382 bits at 2,000 Rust tokens (1.69 bits/token) | The table was "derived from the cumulative counter (3.8 B/token)" (`COL:10`); per-index comparisons across two tokenisations are not like-for-like. The collapse conclusion does not depend on this row |
| 10 | Mamba state per stream | "~48 KB" (`AUD:74`, `RM:2696`); "700 KB" (`TR:436` †) | 2.80 MB recurrent (code) / ~4.9 MiB with buffers (`RM:2722`); "~4.8 MB of state traffic per token" = 2 x 2.36 MB, SSM only (`CN/coste_inferencia_modelos_pequenos.md:8-9`) | Code |
| 11 | Mamba-130M and Q8 | "+16.6%, does not tolerate Q8 PTQ" (`RM:1762-1769`, 2026-11-02) | +0.57% to +1.21% (`RM:2648`, 2026-11-10) | B |
| 12 | Why RWKV tolerates Q8 | "GGUF weights quantised during training (QAT)" (`RM:1772`) | SafeTensors, quantised at load (`R/f32-vs-q8-analysis.md:7` †) | B is the measurement report |
| 13 | enwik8 10 KB, Mamba+CM | 0.9609 (older binary) · 0.9633 (v4) · 0.9641 (v5) · 0.9648 (v5 + RESET_LOG2) | — | Same coded stream, +1 B of header each time (`CLAUDE.md:280-286`); "FP rounding" (`RM:2118`) was superseded by "stale reference" (`DX:17` †) |
| 14 | L3TC | "~1.10 bpb at 27 KB/s on an iPhone CPU" (`AUD:125`, 2026-11-09) | 3.2M: 1.105 bpb at 2.5 MB/s on an A100; 200K: 1.404 bpb at 11-27 KB/s on an iPhone CPU; enwik8 is the training file (`RM:2619-2622`, `SBP:231-233`) | B |
| 15 | x-ray, Mamba, 10 KB | 4.8867, "+6%" (`R/dual-backbone-benchmark.csv:10`, `CLAUDE.md` tables) | 5.8008 after the tokenizer fix (`RM:3151`) = +26.3% vs brotli | B |
| 16 | Column pruning 10% | +431% (`RM:1686`) | "+2-5% (R-061)" (`RM:1925` †) | A is the measurement |
| 17 | MambaByte-353M CE on enwik8 | 0.396 (`RM:2294`, `docs/internal/research/strata-competitive-landscape.md:80`) | 1.1975 measured (`RM:2314`) | B |
| 18 | Size of the model on a terminal | "~130 MB" (`P3:142`, `CLAUDE.md:65`) | 153.6 MB touched per token in Q8; 516.6 MB file; ~791 MB peak RSS (audit) | B |
| 19 | RWKV-7 tokenisation | "RWKV-7 (byte-level)" (`src/domain/backbone.rs:4`) | World tokenizer, vocab 65,536; 3,005 tokens per 10,000 B of enwik8 (`R/f32-vs-q8.csv:26` †) | B |
| 20 | Dev-laptop memory | "32 GB DDR5" (`R/f32-vs-q8-analysis.md:4` †); single channel 21.3 GB/s (`RM:1824` †) | 2 x 16 GB DDR4 at 2667 MT/s, two DIMMs (§G.1; `AUD:25`) | B |
| 21 | External CPU reference for a small-model tier | "0.78 bpb enwik9 at 7.5 KB/s on one core" as a general reference (`CLAUDE.md:41-42`) | transformer pretrained on enwik9, scored on enwik9 (`CN/estado_arte_neural.md:33,144`) | Same in-sample caveat as row 14 |
| 22 | Laptop CPU cost | $50-60/GB (`CLAUDE.md:24`) | "~$62" recomputed from the same inputs (`CN/posicion_interna_redukt.md:91` †) | Minor; see row 4 for the larger issue |
