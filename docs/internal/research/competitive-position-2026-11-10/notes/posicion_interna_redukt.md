# Redukt — Internal (repo-measured) competitive position, as of 2026-11-10

Scope: LOCAL repository files only (root `C:\Users\Josue David Peñuela\Documents\David\redukt`). All links are
`path:line` inside that root. No web research. Dates are the dates the repo itself stamps on each measurement.
"BPB" = bits per byte of the ACTUAL `.azth` output (`compressed_size*8/original_size`) unless marked CE (cross-entropy).
Spanish quotations are left verbatim where the wording matters.

Reading guide for the report writer: the repo contains THREE generations of numbers that must not be mixed:
(1) RWKV-only era, Q8, 2026-10-26..10-30 (README, business-case docs, `actual-compression-benchmark.csv`);
(2) dual-backbone Mamba-130M era, 2026-10-31..11-01 (`dual-backbone-benchmark.*`, `market-readiness-diagnostic`);
(3) Fase 19 era, 2026-11-10 (CLAUDE.md, ROADMAP Fase 19, `lockstep-throughput-2026-11-10.csv`, `kaggle-cuda/`).
The business-case folder was written in era (1) and never updated for eras (2)-(3).

---

## KQ1. What does Redukt measurably achieve today (Nov 2026) on CPU and GPU — BPB, bytes/s, and cost per GB under the repo's own model?

### Takeaway
Compression ratio is world-class on text/structured data (enwik8 10 KB = 0.9641 BPB F32 / 0.9688 Q8, -59% vs brotli-11; logs -54%, e-invoice XML -56%, C source -88%) but loses by +6% to +54% on binary/high-entropy files. Throughput after Fase 19 is ~315-366 tok/s aggregate on a 12-thread laptop (≈1.2 KB/s on text, ≈0.5 KB/s on the mixed 12-file set) and 1,117 tok/s (≈4.2 KB/s) on a Kaggle T4; the repo's own cost model puts that at ≈$50-60/GB raw on CPU spot and ≈$9-17/GB on a T4, versus break-even thresholds of $0.06-0.33/GB — a 30x-500x gap the repo itself states.

### Cited Findings

**Current reference BPB (era 3, 2026-11-10)**
- enwik8 10 KB, Mamba+CM F32, AZTH v5, `--bytes 10240`: **0.9641 BPB** [MEASURED 2026-11-10]; "0.9633 under v4 until R-104 — same stream, +1 B header" — [CLAUDE.md:L231-232](CLAUDE.md#L231)
- enwik8 Q8 (AZTH v5): **0.9688 BPB (+0.57% vs F32)** — [CLAUDE.md:L233](CLAUDE.md#L233)
- The older 0.9609 figure (2026-10-31) is "an outdated reference, not a platform divergence" — [CLAUDE.md:L234-235](CLAUDE.md#L234); ROADMAP: "La divergencia 0.9609 vs 0.9633 era una **referencia antigua**" — [docs/ROADMAP.md:L2786-2787](docs/ROADMAP.md#L2786)
- RWKV+CM enwik8 (AZTH v4): 1.3312 BPB [2026-10-30]; RWKV cross-entropy 1.2813 (Q8); coder efficiency 94.5% enwik8 Mamba, 96.0% RWKV — [CLAUDE.md:L236-238](CLAUDE.md#L236)
- Q8 vs F32 deltas (R-098 gate): enwik8 0.9633→0.9688 (+0.57%), facturae 0.3883→0.3930 (+1.21%), apache 0.2164→0.2188 (+1.11%); "Todos < +2.5% (gate) y < +5% (presupuesto)" — [docs/ROADMAP.md:L2610-2611](docs/ROADMAP.md#L2610)
- Cross-OS determinism evidence (Windows↔Linux identical sha256, 9/9): enwik8 Q8 0.9688, facturae Q8 0.3930, apache Q8 0.2188, facturae F32 0.3883, apache F32 0.2164, enwik8 F32 0.9633 (v4), RWKV enwik8 1.3258 / facturae 0.4406 / apache 0.2930 — [data/benchmark/results/determinism-cross-machine-2026-11-10.tsv:L1-9](data/benchmark/results/determinism-cross-machine-2026-11-10.tsv#L1)

**Full 12-file dual-backbone table (era 2, 2026-10-31, 10 KB, Mamba F32 default / RWKV Q8 for GenericBinary)** — [data/benchmark/results/dual-backbone-benchmark.csv:L2-13](data/benchmark/results/dual-backbone-benchmark.csv#L2); narrative in [data/benchmark/results/dual-backbone-benchmark.md:L81-105](data/benchmark/results/dual-backbone-benchmark.md#L81)

| File | type | RWKV+CM | Mamba+CM | best | brotli-11 | gzip-9 | vs brotli | vs gzip |
|---|---|---|---|---|---|---|---|---|
| apache-access.log | StructuredText | 0.2960 | 0.2164 | 0.2164 | 0.4664 | 0.5464 | -53.6% | -60.4% |
| facturae-batch.xml | StructuredText | 0.4472 | 0.3891 | 0.3891 | 0.8904 | 1.0704 | -56.3% | -63.6% |
| ubl-invoice-batch.xml | StructuredText | 0.5288 | 0.4461 | 0.4461 | 0.8880 | 1.0928 | -49.8% | -59.2% |
| nci | Text | 0.6656 | 0.6094 | 0.6094 | 0.7344 | 0.9344 | -17.0% | -34.8% |
| dickens | Text | 1.7952 | 0.6523 | 0.6523 | 2.7784 | 3.6208 | -76.5% | -82.0% |
| enwik8 | Text | 1.3312 | 0.9609 | 0.9609 | 2.3520 | 2.9720 | -59.1% | -67.7% |
| samba | Executable | 1.2968 | 0.2711 | 0.2711 | 2.2200 | 2.8840 | -87.8% | -90.6% |
| mozilla | Executable | 2.0208 | 1.9383 | 1.9383 | 2.0208 | 2.3728 | -4.1% | -18.3% |
| x-ray | FloatArray | 5.8544 | 4.8867 | 4.8867 | 4.5920 | 5.8728 | +6.4% | -16.8% |
| sao | HighEntropy | 9.2360 | 8.6656 | 8.6656 | 5.9880 | 6.3888 | +44.7% | +35.6% |
| ooffice | GenericBinary | 3.7504 | 3.8539 | 3.7504 (RWKV) | 2.7424 | 2.9104 | +36.7% | +28.9% |
| mr | GenericBinary | 2.3912 | 2.4430 | 2.3912 (RWKV) | 2.3728 | 2.7800 | +0.8% | -14.0% |

- Score: "Wins 8/12 vs brotli-11, 10/12 vs gzip-9. Average win margin: -50.5% vs brotli. Loses: sao +45%, ooffice +37%, x-ray +6%, mr +1% (binary/high-entropy)" — [CLAUDE.md:L266-267](CLAUDE.md#L266); same in [dual-backbone-benchmark.md:L96-105](data/benchmark/results/dual-backbone-benchmark.md#L96)
- Mamba beats RWKV on 10/12 files; "Mamba ~3-4x slower (BPE tokenization + larger model + no SIMD optimization yet)" [2026-10-31] — [dual-backbone-benchmark.md:L41-44](data/benchmark/results/dual-backbone-benchmark.md#L41)
- Per-file compression wall time at 10 KB, 2026-10-31 (pre-Fase-19): enwik8 Mamba ~109 s, apache ~140 s, sao ~363 s, ooffice ~385 s; RWKV enwik8 ~34 s — [dual-backbone-benchmark.md:L24-35](data/benchmark/results/dual-backbone-benchmark.md#L24)

**18-file diagnostic (era 2, 2026-11-01)** — [data/benchmark/results/market-readiness-diagnostic-2026-11-01.md:L15-42](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L15)
- Adds linux-syslog 0.414 (-28.7% vs brotli), openssh-auth 0.521 (-33.5%), xml 0.674 (-25.2%), webster 1.801 (-35.9%), reymont 1.896 (-11.2%), osdb 5.675 (+21.4%, RWKV); enwik8 0.963. Scorecard: wins 13/18 vs brotli (72%), 15/18 vs gzip (83%); average delta -23.7% vs brotli, median -27.0%.
- By category: Structured (logs/XML, 5 files) 0.397 vs brotli 0.722 = -45.0%; Text (6) 1.099 vs 1.952 = -43.7%; Executable (samba) -87.8%; Binary (images/db/HE, 6 files) 4.558 vs 3.732 = **+22.1%** — [market-readiness-diagnostic-2026-11-01.md:L46-54](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L46)
- PDF (R-V03): Redukt **11.11 BPB vs brotli 7.90 (+40.7% WORSE)**; "PDF binary content is near-incompressible"; caveat: W3C dummy.pdf 13 KB — [docs/internal/business-case/BUSINESS-CASES.md:L47-53](docs/internal/business-case/BUSINESS-CASES.md#L47); raw row sample-invoice.pdf 11.1152 vs brotli 7.8968 — [data/benchmark/results/actual-compression-benchmark.csv:L20](data/benchmark/results/actual-compression-benchmark.csv#L20)

**RWKV-only 19-file CSV (era 1, 2026-10-30, Q8, 10 KB)** — [data/benchmark/results/actual-compression-benchmark.csv:L2-20](data/benchmark/results/actual-compression-benchmark.csv#L2): apache 0.2960 (-36.5% vs brotli), syslog 0.4144, facturae 0.4472 (-49.8%), openssh 0.5208, ubl 0.5288, nci 0.6656, xml 0.6736, samba 1.2968, enwik8 1.3312 (-43.4%), dickens 1.7952, webster 1.8008, reymont 1.8960, mozilla 2.0208 (+0.0%), mr 2.3912, ooffice 3.7504, osdb 5.6752, x-ray 5.8544, sao 9.2360 (+54.2%), pdf 11.1152 (+40.8%). All roundtrip PASS. Coder efficiency 83.8% (apache) to 99.9% (sao).

**Scale behaviour (size dependence)**
- RWKV batch scale test (Q8): apache 10 KB 0.2960 → 50 KB 0.2274 → 100 KB 0.2235 (brotli 100 KB 0.3276, -31.7%); dickens 1.7968 → 1.5029 → 1.4574 (brotli 2.6493, -45.0%); samba 1.2976 → 1.3555 → 1.2062 (brotli 1.9444, -38.0%) — [data/benchmark/results/scale-test-complete.csv:L2-12](data/benchmark/results/scale-test-complete.csv#L2); "Advantage grows with file size" — [CLAUDE.md:L280](CLAUDE.md#L280)
- BUT Mamba enwik8 10 KB → 50 KB got WORSE: 0.963 → 1.054 (+9.4%), tok/s 15.2 → 12.6; "This is unexpected ... Throughput degrades ~17% at 50KB" [2026-11-01] — [market-readiness-diagnostic-2026-11-01.md:L178-184](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L178)
- Classical compressors at FULL file size (2026-11-01): brotli-11 ubl-invoice 0.247 (vs Redukt 0.446 at 10 KB), nci 0.362 (vs 0.609), samba 1.3945, dickens 2.2195, enwik8 full 95.37 MB: gzip 2.916 / brotli 2.059 / xz-9 1.989 — [data/benchmark/results/competitors-fullsize-2026-11-01.csv:L2-18](data/benchmark/results/competitors-fullsize-2026-11-01.csv#L2); [docs/internal/business-case/BENCHMARK-RESULTS.md:L108-122](docs/internal/business-case/BENCHMARK-RESULTS.md#L108); note "At full-size, brotli-11 reaches 0.247 BPB on ubl-invoice (vs Redukt 0.446 at 10KB)" — [market-readiness-diagnostic-2026-11-01.md:L87-89](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L87)
- Classical speeds measured locally at full size: gzip-9 3.4-20.4 MB/s compress, brotli-11 0.2-0.6 MB/s, xz-9 0.5-3.4 MB/s; decompress 41-143 MB/s (brotli) — [competitors-fullsize-2026-11-01.csv:L2-18](data/benchmark/results/competitors-fullsize-2026-11-01.csv#L2)
- Max file size ~4 GB (u32); "Max tested file size: 100 MB (enwik8)" — [CLAUDE.md:L84](CLAUDE.md#L84); [docs/internal/business-case/LIMITATIONS.md:L141](docs/internal/business-case/LIMITATIONS.md#L141)

**Throughput, CPU (era 3, i5-1235U 2P+8E, DDR4-2667 dual, maxperf, 10 KB, "thermally limited laptop")**
- Baseline F32 single-thread: 34 tok/s; R-099 multithread F32: 39.0 → 53.6 tok/s single; R-098 Q8 block-64: 127 tok/s cold / 111 warm; bytes/token 516.5 → 153.6 MB — [CLAUDE.md:L289-292](CLAUDE.md#L289)
- Lockstep R-100 aggregate, warm laptop, Q8 N=1/2/4/8/16/32 = 111/134.2/236.3/265.9/292.1/314.7 tok/s; F32 = 57.7/96.5/154.6/206.5/262.6/322.1 — [data/benchmark/results/lockstep-throughput-2026-11-10.csv:L6-17](data/benchmark/results/lockstep-throughput-2026-11-10.csv#L6)
- Idle re-measurement: Q8 N=8 270.8 (12 files, 75,398 tokens, 278.4 s), N=16 286.9 (16 files, 89,966 tok, 313.6 s), N=32 309.9 (32 files, 198,844 tok, 641.6 s); F32 N=16 259.6 — [lockstep-throughput-2026-11-10.csv:L2-5](data/benchmark/results/lockstep-throughput-2026-11-10.csv#L2)
- "Compute-bound desde N~8 (130-190 GFLOPS, sin FMA)"; "a N=32 F32 iguala a Q8 en throughput, con mejor BPB"; "~4.9 MB de estado por stream. 192/192 `.azth` de lote idénticos (sha256) al single-file" — [docs/ROADMAP.md:L2678-2681](docs/ROADMAP.md#L2678)
- R-107 FMA (numeric contract 2, opt-in `--numeric fma`): single unchanged (Q8 130.3→130.5); batch Q8 N=16 279.4→335.1 (+20%), F32 N=16 257.5→288.6 (+12%), Q8 N=32 299.1→**365.9** (+22%), F32 N=32 303.1→351.0 (+16%); BPB identical — [docs/ROADMAP.md:L2823-2825](docs/ROADMAP.md#L2823); [CLAUDE.md:L299](CLAUDE.md#L299)
- Kaggle Xeon CPU (2026-11-06, Mamba F32 single): enwik8 0.9633 BPB at **72.4 tok/s** ("2.1x local") — [data/benchmark/results/kaggle-v3/analysis.md:L19,L32](data/benchmark/results/kaggle-v3/analysis.md#L19)
- Era-2 single-stream bytes/s (2026-11-01): "Redukt Mamba 43 B/s (15.2 tok/s)", "Redukt RWKV 60 B/s (60 tok/s)"; gzip-9 11 MB/s, brotli-11 0.3 MB/s, xz-9 1.2 MB/s compress — [market-readiness-diagnostic-2026-11-01.md:L92-98](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L92); "brotli-11 / Redukt ~7,300x compress, ~2,000,000x decompress" — [L102-106](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L102)
- Compression is symmetric: "el CM es simétrico: el cliente descomprime a la misma velocidad. Sin decode rápido no hay caso CDN" — [docs/internal/research/throughput-properties-audit.md:L122](docs/internal/research/throughput-properties-audit.md#L122)
- Bytes per token vary by file type (RWKV Q8 CSV, 2026-10): dickens 4.07, webster 3.97, samba 3.51, facturae 3.48, apache 2.80, enwik8 n/a, xml 2.22, nci 1.71, mozilla 1.24, mr 1.07, sao 1.05, ooffice 1.03, x-ray 1.00 B/token — [data/benchmark/results/gzip9-brotli11-comparison.csv:L2-18](data/benchmark/results/gzip9-brotli11-comparison.csv#L2)

**Throughput, GPU (Kaggle T4, 2026-11-10, commit 3a2e8be, CUDA plugin FFI v2)**
- v1 single stream 151.1 tok/s; batch N=1/8/32/128 = 150.6 / 951.3 / 1,076.1 / 1,116.9 tok/s (6.64 / 8.41 / 29.74 / 114.61 ms per step) — [data/benchmark/results/kaggle-cuda/cuda-plugin-results.csv:L12-16](data/benchmark/results/kaggle-cuda/cuda-plugin-results.csv#L12)
- BPB plugin 0.9641 == CPU 0.9641 same machine; compress 10 KB enwik8: plugin 33.0 s vs CPU 84.9 s; determinism 2 runs PASS; roundtrip PASS; GPU `.azth` sha differs from CPU F32 sha ("expected DIFFERENT") — [cuda-plugin-results.csv:L17-23](data/benchmark/results/kaggle-cuda/cuda-plugin-results.csv#L17)
- "meseta ~1.1k, kernel-bound: ~4% del pico de la T4; siguiente paso kernels a eficiencia cuBLAS/tensor cores" — [docs/ROADMAP.md:L2750-2751](docs/ROADMAP.md#L2750)
- Earlier GPU skeleton (uniform logits) 81K-96K tok/s = FFI overhead ceiling only, BPB 3.02 — [kaggle-v3/analysis.md:L38-45,L59](data/benchmark/results/kaggle-v3/analysis.md#L38)

**Cost per GB (repo's own figures)**
- "Measured status (2026-11-10): CPU lockstep ~1,180 B/s on 12 vCPU = $50-60/GB (spot) / ~$19 (dedicated); GPU T4 batch N=128 ~4.2 KB/s = $9-17/GB. Gap to threshold: 30x-500x." — [CLAUDE.md:L24-25](CLAUDE.md#L24)
- Derivation: "El lockstep satura 12 hilos para dar ~1,180 B/s (3.8 B/token), es decir ~100 B/s por vCPU frente a ~65-130 B/s por vCPU de la línea base ... la eficiencia mejora ~1.5-2x, no 9x. Con C = P * 1e9 / (3600 * R): 12 vCPU spot ($0.26/h) ≈ $50-60/GB raw; Hetzner dedicado ≈ $19/GB; GPU T4 N=128 (4.2 KB/s, $0.13-0.25/h) ≈ $9-17/GB. La estimación previa de '≈$5/GB' ... queda retirada." — [throughput-properties-audit.md:L213-217](docs/internal/research/throughput-properties-audit.md#L213); same in [docs/ROADMAP.md:L2873-2879](docs/ROADMAP.md#L2873)
- Pre-Fase-19 cost: "Hoy: 130 B/s → $47 por GB de entrada (raw) comprimido (spot), $14.5 en Hetzner dedicado" — [throughput-properties-audit.md:L116](docs/internal/research/throughput-properties-audit.md#L116); script output "Redukt today 130 B/s: hours/GB=2137 {spot: 47.0, on-demand c7g: 77.5, on-demand m7i: 107.7, Hetzner: 14.5}" — [scripts/breakeven.py:L39](scripts/breakeven.py#L39) (executed 2026-11-10)
- Nov-01 framing (era 2): "Time to compress 1 TB (1 thread) ~780 years; EC2 c5.xlarge ($0.17/hr) ~$1,161,000; Break-even vs brotli savings ~47,000 years" — [market-readiness-diagnostic-2026-11-01.md:L145-149](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L145)

### Inferences
- Effective aggregate bytes/s on the repo's OWN 12-file mix is lower than the headline 1,180 B/s: the idle run processed 12×10 KB (≈122,880 B) in 278.4 s → ≈440 B/s (N=8), and 32×10 KB in 641.6 s → ≈510 B/s (N=32), i.e. ≈1.6 B/token on the mix vs the 3.8 B/token used in the $50-60/GB estimate (which is a text-like ratio). On mixed data the CPU spot cost would be roughly 2.3x higher (≈$120-140/GB); the GPU 4.2 KB/s figure is enwik8-only and inherits the same caveat for binary inputs (≈1 B/token).
- Re-running the CLAUDE.md formula with its own inputs (1,180 B/s, 12 vCPU × $0.022) gives ≈$62/GB, slightly above the stated "$50-60"; the dedicated figure ($19) and GPU figures ($8.6-16.5) reproduce exactly.
- Single-stream end-to-end on the T4 (33 s per 10 KB enwik8 ≈ 310 B/s) shows the per-token CM+coder on the host is now a visible fraction; batch N=128 is 7.4x the single stream, not 128x.
- Classical compressors' ratios improve far more with file size than Redukt's documented scale tests (brotli ubl 0.888→0.247 from 10 KB to full; Redukt measured only to 100 KB on RWKV and 50 KB on Mamba, with Mamba enwik8 degrading). The "-50.5% vs brotli" headline is a 10 KB-window result and should be labelled as such in any external comparison.

### Gaps
- No Redukt measurement exists at full Silesia file sizes (the full-size CSV only has gzip/brotli/xz); the repo itself flags this ("Redukt at larger sizes is [UNMEASURED] — see R-V04", [MARKET-ANALYSIS.md:L23](docs/internal/business-case/MARKET-ANALYSIS.md#L23)). R-100 "Archivos grandes" (chunking) is still an open subtask — [docs/ROADMAP.md:L2664-2665](docs/ROADMAP.md#L2664).
- No zstd measurement of Redukt vs zstd exists in any CSV (zstd appears only as a target baseline in `docs/BENCHMARKS.md:L21,L167,L181` and as an option in `scripts/benchmark-suite.py` per README:L106-108). xz-9 exists only at full size for classical codecs.
- No real-GCP $/GB measurement: "Pendiente (requiere proyecto GCP real): $/GB medido en Cloud Run / Cloud Batch" — [docs/ROADMAP.md:L2734](docs/ROADMAP.md#L2734).
- Decompression throughput is asserted symmetric but no separate decompress tok/s table exists for Fase 19.

---

## KQ2. What thresholds does `scripts/breakeven.py` say a buyer would tolerate, under which assumptions?

### Takeaway
The model compares compute cost per raw GB (`P_cpu·1e9/(3600·R)`) with storage saving over T years (`S·b·P_st·12·T`); with spot $0.022/vCPU-h, S3 Standard $0.023/GB-mo and 7-year retention it yields break-even throughputs of ≈18 KB/s per vCPU for enwik8-like text, ≈51 KB/s for Facturae XML and ≈101 KB/s for Apache logs (equivalently ≤$0.33, ≤$0.12, ≤$0.06 per raw GB). Deep Archive ($0.00099) makes savings ~23x smaller, so the repo now explicitly excludes cold archive as a target.

### Cited Findings
- Model: "C_comp = P_cpu * 1e9 / (3600 * R)  coste de cómputo; R en bytes/s por vCPU / A_stor = S * b * P_st * 12 * T / R* = P_cpu * 1e9 / (3600 * S * b * P_st * 12 * T) ... b = fracción que deja el baseline (bpb de brotli / 8) y S = ahorro adicional de Redukt frente a ese baseline" — [scripts/breakeven.py:L8-13](scripts/breakeven.py#L8)
- Price inputs (2026-11, instances.vantage.sh): spot c7g/c7a/m7i $0.022/vCPU-h; on-demand c7g $0.03625; on-demand m7i $0.0504; "dedicated Hetzner AX42 ~EUR0.0068/thread-h" (EUR treated as USD) — [scripts/breakeven.py:L25,L32](scripts/breakeven.py#L25)
- Storage inputs: S3 Standard $0.023/GB-mo; S3 Deep Archive $0.00099; On-prem $50/TB/yr — [scripts/breakeven.py:L33](scripts/breakeven.py#L33)
- Redukt case inputs (brotli BPB, S): enwik8 2.352/0.59; apache 0.466/0.54; facturae 0.890/0.56; dickens 2.778/0.77 — [scripts/breakeven.py:L56](scripts/breakeven.py#L56)
- Executed output (2026-11-10), 7-year S3 Standard, spot: enwik8 b=0.294 S=59% save7y=$0.3351/GB raw, R*=18.2 KB/s (x140 vs 130 B/s); apache b=0.058 S=54% save7y=$0.0608, R*=100.6 KB/s (x774); facturae b=0.111 S=56% save7y=$0.1204, R*=50.8 KB/s (x391); dickens b=0.347 S=77% save7y=$0.5166, R*=11.8 KB/s (x91) — [scripts/breakeven.py:L55-61](scripts/breakeven.py#L55) (run output)
- Deep Archive, 7y, spot: enwik8 R*=423.6 KB/s (save $0.0144/GB); apache 2,336 KB/s ($0.0026); facturae 1,180 KB/s ($0.0052); dickens 275 KB/s — same run; On-prem $50/TB/yr: enwik8 100.7 KB/s, apache 555 KB/s, facturae 280 KB/s, dickens 65 KB/s.
- Generic R* table (b=1, spot): S3 Standard S=50%: 1y 44.3 / 3y 14.8 / 7y 6.3 KB/s; Deep Archive S=50%: 1,028.8 / 342.9 / 147.0 KB/s; On-prem S=50%: 244.4 / 81.5 / 34.9 KB/s. On-demand $0.036 raises every R* by 1.65x (e.g. S3 Std S=50% 7y = 10.4 KB/s) — run output of [scripts/breakeven.py:L44-53](scripts/breakeven.py#L44)
- Compute cost by throughput (run output): R=0.1 KB/s → $61.1/GB spot, $18.9 dedicated; 1 KB/s → $6.11 / $1.89; 10 KB/s → $0.611 / $0.189; 100 KB/s → $0.061 / $0.019; 1 MB/s → $0.006 — [scripts/breakeven.py:L35-38](scripts/breakeven.py#L35)
- Egress: "saving per download per GB raw = S*b*0.09; months of S3 Std equivalent = 0.09/0.023 = 3.91" — [scripts/breakeven.py:L54](scripts/breakeven.py#L54)
- CLAUDE.md codifies the thresholds: "Reference thresholds (7 years, spot compute): <= $0.33/GB for text in hot object storage (enwik8-like), <= $0.12/GB for e-invoice XML, <= $0.06/GB for logs. Bytes saved in cold/deep archive ($0.001/GB-month) are worth ~0." — [CLAUDE.md:L20-23](CLAUDE.md#L20); "Deep/cold archive is NOT a target" — [CLAUDE.md:L16](CLAUDE.md#L16)
- "The metric is cost per GB (throughput per vCPU or per GPU-hour), NOT throughput per machine." — [CLAUDE.md:L31](CLAUDE.md#L31)
- Thesis-reordering facts: "Cuanto mejor comprime brotli (logs, XML), menos bytes absolutos quedan por ahorrar: el archivo profundo ($0.001/GB-mes) es el peor caso de negocio, no el mejor." — [throughput-properties-audit.md:L119](docs/internal/research/throughput-properties-audit.md#L119); "Todo el mercado español de factura electrónica B2B (~1,200 M facturas × 15 KB ≈ 18 TB/año raw, ~2 TB tras brotli) ahorraría ≈1.1 TB/cohorte = $300/año en S3 Standard" — [L120](docs/internal/research/throughput-properties-audit.md#L120); "No existe precedente comercial de un compresor <1 MB/s con clientes de pago. PetaGene cobra % del ahorro a 290 MB/s; AtomBeam monetiza ancho de banda satelital, no storage; ts_zip y Nacrith no son productos." — [L121](docs/internal/research/throughput-properties-audit.md#L121)
- Alternative price points elsewhere in the repo (not in the script): S3 Std $23/TB-mo, S3 IA $12.50, Glacier Instant $4, Glacier Flexible $3.60, Deep Archive $1, Azure Archive $1, GCS Archive $1.20; bandwidth Starlink Maritime $1.08-5.00/GB, OneWeb $1.75-3.25, VSAT $10-225, Datadog ingestion $0.10/GB, Splunk ~$150/GB/day; "Hot storage x long retention ($23/TB x 60 months = $1,380/TB lifetime)" — [docs/internal/research/execution-architecture-analysis.md:L381-426](docs/internal/research/execution-architecture-analysis.md#L381)
- External economics cited by the repo: Kipf et al. "Waiting to Decompress" (VLDB/CIDR 2026): LLM compression takes "10 años" to amortize compute vs storing uncompressed, "120 años" vs traditional compression — [docs/internal/research/throughput-paradigm-shift.md:L732-738](docs/internal/research/throughput-paradigm-shift.md#L732)

### Inferences
- Against the per-vCPU thresholds, today's CPU figure (~100 B/s per vCPU) is ≈180x short for text, ≈500x for XML and ≈1,000x for logs; the T4 (4.2 KB/s per GPU, $0.13-0.25/h) is ≈27x short for text and ≈280x for logs. This matches CLAUDE.md's "30x-500x" range only if one takes the most favourable pairings.
- The audit's own sentence "El umbral económico (18-100 KB/s por vCPU) sigue ~15-80x por encima del throughput actual (1.2 KB/s por máquina)" ([throughput-properties-audit.md:L218](docs/internal/research/throughput-properties-audit.md#L218)) compares a per-machine number with a per-vCPU threshold and so understates the gap by the 12-vCPU factor; the $/GB framing in the same file (L213-217) is the consistent one.
- The model credits the client with the FULL 7-year saving; any revenue share to the vendor, or retention shorter than 7 years, lowers the tolerable cost proportionally (R* scales as 1/T).

### Gaps
- The script has no GPU cost branch (GPU $/GB is computed by hand in ROADMAP/audit), no egress-driven case beyond the one-liner, and no revenue-share parameter.
- Storage prices are hard-coded 2026-11 snapshots with a single source (instances.vantage.sh); Hetzner EUR→USD is an admitted approximation.

---

## KQ3. What does the repo itself say are the differentiators, the admitted gaps/weaknesses, and the admitted paths to close the cost gap?

### Takeaway
The repo positions Redukt as a CPU-first, zero-dependency single Rust binary that compresses ANY file type with a deterministic, header-declared numeric contract and dual backbone, and it is unusually candid about weaknesses: binary/high-entropy losses, throughput 3-4 orders of magnitude below classical codecs, a $50-60/GB cost vs $0.06-0.33 thresholds, format lock-in to exact weights, and an unproven large-file path. It admits exactly three ways to close the gap: tensor-core GPU kernels, a 3-30M-parameter fast tier (L3TC-class), or segments where bytes are expensive to move.

### Cited Findings

**Self-declared differentiators**
- "Hybrid CM (5 models + adaptive SGD mixer) + RWKV-7/Mamba (pretrained) + arithmetic coding. CPU-only, zero external dependencies, single Rust binary." — [CLAUDE.md:L10-11](CLAUDE.md#L10)
- "Purpose: production-grade compression product for batch workloads where bytes are expensive (hot object storage, metered links, regulated retention)." — [CLAUDE.md:L15-16](CLAUDE.md#L15)
- "NO external crate dependencies (compile-time). Only rustc + stdlib." — [CLAUDE.md:L105](CLAUDE.md#L105)
- "This repo is COMPRESSION ONLY. No chat, no generation, no QA, no probes. Commands: compress, decompress, hybrid-eval, baseline, cm-eval, api." — [CLAUDE.md:L53-54](CLAUDE.md#L53)
- Portability principle: "CPU is the reference implementation: always functional, always able to decode anything produced anywhere. An accelerator is an addition (plugin, FFI v1/v2), never a replacement" ; "Numeric tiers are declared in the AZTH header (PROFILE: quant + numeric contract) so that encoder/decoder mismatches fail early" — [CLAUDE.md:L33-49](CLAUDE.md#L33)
- README highlights: "Best-in-class compression: 48% smaller than brotli-11 on e-invoicing, 34% on server logs"; "18 domains validated"; "Zero dependencies: Pure Rust, single binary, CPU-only"; "Deterministic: Bit-identical decompression guaranteed (CRC-32 verified)" — [README.md:L9-12](README.md#L9)
- Determinism/format: AZTH v5 with PROFILE byte (QUANT bits 0-3; NUMERIC_CONTRACT bits 4-5: 0 legacy libm, 1 fmath-v1 default, 2 fma-v1 opt-in, 3 rejected); decoder accepts v2-v5 — [CLAUDE.md:L86-101](CLAUDE.md#L86); [docs/api/FORMAT-SPEC.md:L193,L201-220](docs/api/FORMAT-SPEC.md#L193)
- Cross-OS determinism proven 2026-11-10: "roundtrip Windows (MSVC) <-> Linux (glibc, x86-64-v3, contenedor) idéntico 9/9 ... el `enwik8.cpu.azth` de Kaggle (sha 384c3042) coincide bit a bit con el local" — [docs/ROADMAP.md:L2783-2788](docs/ROADMAP.md#L2783); F1 pure-Rust exp/log "0 ulp en 43 valores dorados; <=1 ulp vs std en barrido de 1e6" — [L2790-2792](docs/ROADMAP.md#L2790)
- Batch == single identity: "Each `.azth` is byte-identical to the single-file output regardless of N" — [README.md:L65](README.md#L65); "192/192 batch `.azth` are identical to single-file output" — [CLAUDE.md:L301-302](CLAUDE.md#L301)
- API parity: "`redukt api` has byte-for-byte parity with the CLI ... verified sha256 API == CLI ... It serves one request at a time." — [README.md:L79-81](README.md#L79)
- Any-file-type: dual backbone, "Mamba-130M (default) + RWKV-7 0.1B (GenericBinary only)"; binary preprocessing flags E8E9/shuffle/delta in header — [CLAUDE.md:L223-224,L65](CLAUDE.md#L223)
- Legal: Apache-2.0, no encryption/export control, no telemetry, patent-safe arithmetic coding/CM — [CLAUDE.md:L121-160](CLAUDE.md#L121)
- Industry-reference claims: "Determinismo es ventaja competitiva — paper de feasibility lo identifica como blocker"; "CPU-only es nicho legítimo — deployment sin GPU, data sovereignty, edge" — [docs/internal/research/backbone-scalability-analysis.md:L235-237](docs/internal/research/backbone-scalability-analysis.md#L235)

**Admitted weaknesses / gaps**
- Binary losses: +45% sao, +37% ooffice, +6% x-ray, +1% mr (10 KB) — [CLAUDE.md:L267](CLAUDE.md#L267); +21% osdb, +40.7% PDF — [market-readiness-diagnostic-2026-11-01.md:L33](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L33), [BUSINESS-CASES.md:L47](docs/internal/business-case/BUSINESS-CASES.md#L47)
- Throughput: "Throughput: NOT VIABLE for production bulk workloads ... 43 B/s (Mamba) = 3.6 MB/day = fundamentally insufficient ... 7,300x slower than brotli-11" [2026-11-01] — [market-readiness-diagnostic-2026-11-01.md:L243-247](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L243); after Fase 19: "Aggregate gain ~9x per machine over the 34 tok/s start, but per-vCPU efficiency only ~1.5-2x" — [CLAUDE.md:L303-304](CLAUDE.md#L303)
- Cost gap: "Gap to threshold: 30x-500x" — [CLAUDE.md:L25](CLAUDE.md#L25); "Con ≈$50-60/GB en CPU y ≈$9-17/GB en GPU T4 el sistema sigue 30-500x por encima del umbral económico ($0.06-0.33/GB)" — [docs/ROADMAP.md:L2891-2893](docs/ROADMAP.md#L2891)
- Projections missed: "R-099 quedó en 53.6 (proyectado ~70), R-098 en 111-127 single (proyectado 250-300) y R-100 en 315 agregados (proyectado 700-1,500): el compute sin FMA, no el ancho de banda, es el límite" — [docs/ROADMAP.md:L2890-2891](docs/ROADMAP.md#L2890)
- Model-size dependence: "Decompression requires IDENTICAL model + weights" — [CLAUDE.md:L82](CLAUDE.md#L82); "AZTH format requires our exact model + weights for decompression" with escrow/open-spec mitigations — [LIMITATIONS.md:L111-117](docs/internal/business-case/LIMITATIONS.md#L111); weights ~400 MB download, 4+ GB RAM — [README.md:L17,L94-95](README.md#L17)
- Mamba-130M bytes/token 516.5 MB F32 / 153.6 MB Q8 — [CLAUDE.md:L292](CLAUDE.md#L292); Q4 rejected (+12.7% BPB; "+13.3% BPB regression AND 7.6x slower") — [CLAUDE.md:L218](CLAUDE.md#L218), [LIMITATIONS.md:L88](docs/internal/business-case/LIMITATIONS.md#L88)
- Large files unsupported in lockstep: "Límite: un archivo grande solo se beneficia si se trocea en chunks independientes (BOA Constrictor, +1-3% BPB)" — [throughput-properties-audit.md:L75](docs/internal/research/throughput-properties-audit.md#L75); "No streaming API (full file must be buffered)"; "Single-file processing (no multi-file archive format yet)" — [LIMITATIONS.md:L138-139](docs/internal/business-case/LIMITATIONS.md#L138)
- Not a transport protocol: "Redukt is a storage/archival product, NOT a transport protocol." — [LIMITATIONS.md:L109](docs/internal/business-case/LIMITATIONS.md#L109)
- Determinism still partially open: R-104 F2 (scalar == AVX2 lane order) and F3 (multi-platform CI + libm lint) pending; only x86 AVX2 allowed in deploy ("Restringir a x86 con AVX2 (determinismo); excluir ARM") — [docs/ROADMAP.md:L2797-2800,L2730](docs/ROADMAP.md#L2797)
- GPU plugin uses "~4% of T4 peak" — [CLAUDE.md:L27](CLAUDE.md#L27); GCP deployment "verified locally (not yet on a live GCP project)" — [README.md:L88](README.md#L88)
- API single-threaded, "containerConcurrency: 1", no liveness probe while compressing — [deploy/gcp/README.md:L29](deploy/gcp/README.md#L29)
- Pretrained MambaByte 353M path failed: 1.3156 BPB enwik8 10 KB (+37% vs 0.9609), 4.8 tok/s, "Domain mismatch"; "Pesos pre-entrenados NO reemplazan training propio" — [docs/ROADMAP.md:L2278-2283](docs/ROADMAP.md#L2278)
- Mamba 10 KB → 50 KB enwik8 regression +9.4% (see KQ1) — [market-readiness-diagnostic-2026-11-01.md:L180-184](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L180)
- 2026-11-01 recommendation: "**DO NOT launch as a production compression product at current throughput.** ... Launch as open-source research tool ... Target initial market: academic/research archival" — [market-readiness-diagnostic-2026-11-01.md:L281-287](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L281)

**Admitted paths to close the gap**
- "Admitted paths to close it (every new phase must justify itself against one of them): 1. GPU kernels at cuBLAS/tensor-core efficiency (current CUDA plugin uses ~4% of T4 peak). 2. Fast tier: 3-30M-param backbone + CM recipe (R-105; L3TC reference ~1.10 bpb at 27 KB/s mobile CPU). 3. Segments where bytes are expensive to move or retention is regulated (satellite/IoT links, egress, compliance), where the threshold is far lower than storage savings." — [CLAUDE.md:L26-30](CLAUDE.md#L26)
- R-105 (fast tier) is "única vía CPU-only que cruza el umbral económico"; decision gate "GO/NO-GO frente al umbral R* (18-100 KB/s por vCPU)" — [docs/ROADMAP.md:L2828-2846](docs/ROADMAP.md#L2828)
- Earlier strategic alternative (2026-11-09, pre-Fase 19): two-stage zstd-inline + Redukt-batch, "per-TB-saved pricing", model-as-consultant, speculative decoding; "No perseguir GPU como unica via: GPU lleva el throughput a ~10 KB/s (A100), que sigue sin cerrar el gap" — [throughput-paradigm-shift.md:L385-492,L852-863](docs/internal/research/throughput-paradigm-shift.md#L852)
- Projection that the then-pending CPU roadmap would reach only ~1.5 KB/s: "las optimizaciones del roadmap actual no cierran el gap de 4 ordenes de magnitud" — [throughput-paradigm-shift.md:L76-84](docs/internal/research/throughput-paradigm-shift.md#L76)

### Inferences
- The strongest externally defensible differentiators, per the repo's own evidence, are (a) byte-identical output across Windows/Linux/Kaggle and across batch sizes with a header-declared numeric contract, (b) a single zero-dependency binary, and (c) measured wins on logs/XML/code at 10 KB. Ratio alone is NOT a clear differentiator vs Nacrith (0.9389 vs 0.9641).
- Three self-contradictions a report should surface: README still describes the RWKV-7 architecture ("19 Context Mixing models", "LSTM mixer", `README.md:L32-35`) and quotes era-1 BPB (enwik8 1.34, facturae 0.46, `README.md:L45-53`) while CLAUDE.md describes 5 CM models + SGD mixer and Mamba 0.9641; README:L119 says "82 tests" vs CLAUDE.md "295 unit + 5 integration" (`CLAUDE.md:L308`); BENCHMARK-RESULTS.md:L6 says "32 GB DDR5" while the audit measured DDR4-2667.

### Gaps
- No decompress-side throughput or memory (RAM peak) figures are recorded for the Fase-19 build; `docs/BENCHMARKS.md:L169-170` sets targets (<16 GB RAM, decompress >1 MB/s) that are never reported against.
- No measurement exists for the fast-tier path (R-105 is PENDIENTE); the L3TC reference is literature, not a Redukt measurement.

---

## KQ4. Which competitor numbers already exist in the repo (with dates), for reconciliation with external research?

### Takeaway
The repo holds four competitor snapshots (2026-10-01, 2026-10-31/11-01, 2026-11-09 ×2) that largely agree on Nacrith 0.939 BPB (GPU, text-only) as the nearest neural competitor and ts_zip ≈1.11, NNCP 1.19, cmix 1.1-1.25 as classical-neural references, but they disagree internally on cmix (1.1 / 1.17 / 1.25 / 1.63), on Nacrith throughput (21-70 vs 50-90 tok/s), and on L3TC throughput (27 KB/s CPU / 633 KB/s ANE / 1.30 MB/s ANE / 4.35 MB/s), so external reconciliation is needed.

### Cited Findings

**Snapshot A — 2026-10-01, "Sub-1.0 BPB ecosystem"** — [docs/research/competitive-landscape.md:L1-25](docs/research/competitive-landscape.md#L1)
| System | enwik8 BPB | Arch / params | Level |
|---|---|---|---|
| Nacrith | 0.939 | SmolLM2-135M + N-gram + bias head, 135M | Token |
| fx2-cmix-transformer | ~0.97* (extrapolated from enwik9 0.0969 ratio, Hutter Prize) | 6M Transformer + cmix 2000+ CM | Bit |
| SHA-RNN | 1.07 | 63M RNN | Char |
| ts_zip (Bellard) | 1.106 | RWKV-169M v4 Q8, no CM | Token |
| cmix v22 | 1.17 | 2077 models + LSTM mixer | Bit |
| NNCP v3 | 1.19 | 199M Transformer-XL online | Byte |
| PAQ8px | 1.27 | 200+ models | Bit |
| Redukt (smoke test, 10 KB) | 1.42 | RWKV-7 0.1B Q8 | Token |
| Midicoth | 1.753 | PPM + match + Tweedie, 0 params | Byte |
| StateSMix | 2.13 | Mamba SSM + sparse N-gram, ~120K | Token |
- Decomposition claim: ts_zip (1.106) → Nacrith (0.939) gap attributed to "CDF-24 precision: -0.517 BPB ... N-gram ensemble ~-0.02 to -0.05 ... bias head ~-0.03 ... Confidence skip ~-0.39" — [docs/research/competitive-landscape.md:L27-33](docs/research/competitive-landscape.md#L27)

**Snapshot B — 2026-10-31 / 2026-11-01**
- "NNCP (Bellard) ~1.2 LSTM ensemble; cmix v19 ~1.1 PAQ-based CM [SOURCED: literature, full file]"; "Note: NNCP and cmix numbers are for full enwik8 (100MB), not 10KB slices" — [dual-backbone-benchmark.md:L128-140](data/benchmark/results/dual-backbone-benchmark.md#L128)
- "cmix v19 ~1.1 (full) ~1 KB/s Research; NNCP (Bellard) ~1.2 (full) ~10 KB/s Research; brotli-11 2.352 (10KB) 300 KB/s; gzip-9 2.972 11 MB/s" — [market-readiness-diagnostic-2026-11-01.md:L214-222](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L214)
- Minimum viable throughput by segment (repo's own view, 2026-11-01): Individual archival 1 KB/s; Small business backup 100 KB/s; Enterprise archival 1 MB/s; Cloud storage tier 10 MB/s; CDN/streaming 100 MB/s "Not feasible with neural" — [market-readiness-diagnostic-2026-11-01.md:L260-266](data/benchmark/results/market-readiness-diagnostic-2026-11-01.md#L260)

**Snapshot C — 2026-11-09, R-087 "Strata & competitive landscape"** — [docs/internal/research/strata-competitive-landscape.md:L1-4](docs/internal/research/strata-competitive-landscape.md#L1)
- Nacrith (github robtacconelli/Nacrith-GPU, arXiv:2602.19626): "SmolLM2-135M + context mixing ensemble + arithmetic coder 32-bit. BPB enwik8: 0.9389 (Redukt: 0.9609, diferencia: 2.3%). Throughput: 21-70 tok/s en GTX 1050 Ti (con llama.cpp). Limitaciones: Solo texto, requiere GPU, Python." — [L91-108](docs/internal/research/strata-competitive-landscape.md#L91)
- L3TC (Alipay, AAAI 2025, arXiv:2412.16642): "RWKV 3.2-12M params + outlier-aware tokenizer + arithmetic coding. Throughput: 4.35 MB/s decompression, 1.30 MB/s en iPhone 12 ANE." — [L110-116](docs/internal/research/strata-competitive-landscape.md#L110)
- Table: ts_zip ~1.11 (RWKV-4 169M, GPU, text only); NNCP v2 1.19 (Transformer, GPU, closed binary); cmix-neif-2.0 ~1.25 (CM neural, CPU, C++, GPL, "muy lento"); FineZip 1.024 (LLaMA-7B, GPU, "4h por corpus"); LLMZip ~0.709* (LLaMA-7B, "*Bound asintótico, 9.5 días"); DZip ">gzip" (dual RNN, online); NeuralZip (GRU PoC) — [L120-128](docs/internal/research/strata-competitive-landscape.md#L120)
- MambaByte pretrained (HuggingFace JunxiongWang): 353M arXiv-trained 0.396 CE BPB (on arXiv-like data), Books ~0.40, Code ~0.40, PG19 972M ~0.35 — [L72-78](docs/internal/research/strata-competitive-landscape.md#L72); repo later MEASURED it at 1.3156 BPB on enwik8 (R-088) — [docs/ROADMAP.md:L2278](docs/ROADMAP.md#L2278)
- strata-inference CPU 0.1-6.9 tok/s vs "Redukt 34 tok/s"; Metal 33-365 tok/s — [L33,L37-38](docs/internal/research/strata-competitive-landscape.md#L33)
- NNLCB benchmark "17 compresores × 28 datasets × 4600h CPU/GPU" suggested for formal positioning — [L132-133](docs/internal/research/strata-competitive-landscape.md#L132)
- R-089 "Formal benchmark vs Nacrith" is BLOQUEADO; planned datasets alice29.txt, enwik8 full — [docs/ROADMAP.md:L2288-2296](docs/ROADMAP.md#L2288)

**Snapshot D — 2026-11-09, throughput-paradigm-shift §8** — [docs/internal/research/throughput-paradigm-shift.md:L687-745](docs/internal/research/throughput-paradigm-shift.md#L687)
| System | enwik8 BPB | Throughput | Hardware | Model | Year |
|---|---|---|---|---|---|
| Redukt | 0.961 | 43 B/s | CPU i5-1235U | Mamba-130M + CM | 2026 |
| Nacrith | 0.939 | 50-90 tok/s | GTX 1050 Ti | SmolLM2-135M + CM | 2026 |
| High-throughput LLM (arXiv:2610.17586) | N/A (7.17x ratio) | 181 KB/s | A100 | TensorRT-LLM | 2026 |
| EDPC | N/A | 170.23 KB/s | GPU+CPU | pipeline decoupled | 2025 |
| BOA Constrictor | N/A (HEP data) | 3.5-45 MB/s | GPU | Mamba + parallel RC | 2025 |
| StateSMix (arXiv:2605.02904) | 2.123 | 1,300 tok/s | CPU AVX2 | Mamba-120K + CM | 2025 |
| OmniZip | 1.107 | 200 KB/s | CPU | MoE 3-expert | 2026 |
| cmix v21 | ~1.63 | 0.67 KB/s | CPU | 2077 models + LSTM | 2024 |
| NNCP v3 | 1.19 | ~KB/s | CPU multi-core | Transformer-XL | 2023 |
- Commercial neural-compression companies listed: The Compression Company (UK, satellite EO codecs, EUR 2.8M pre-seed, lossy); Neurovia AI/Robo.ai NeuroStream (video); Multiverse Computing CompactifAI (EUR 189M Series B, model compression); ZipNN (IBM, model weights); PetaGene PetaSuite (genomics). "NO existe ningun producto comercial de compresion neuronal lossless de proposito general." — [L703-715](docs/internal/research/throughput-paradigm-shift.md#L703)
- GPU classical reference: nvCOMP gANS ~530 GB/s, Cascaded up to 500 GB/s, LZ4 100 GB/s — [L717-728](docs/internal/research/throughput-paradigm-shift.md#L717)
- PetaGene model: "Per-TB-saved (fraccion del ahorro)", decompression free (PetaLink), clients recover licence in ~2 months, 200K+ files/24h — [L439-453](docs/internal/research/throughput-paradigm-shift.md#L439)

**Snapshot E — 2026-11-09, throughput-properties-audit**
- "L3TC (AAAI 2025): RWKV de 3.2M parámetros + vocab 16K logra ≈1.10 bpb en enwik8 (Redukt 0.96) a 27 KB/s en CPU de iPhone con batch 1, y 633 KB/s batcheado en ANE. Un modelo 40× más pequeño que Mamba-130M pierde 15% de ratio y gana 200× en throughput." — [throughput-properties-audit.md:L125](docs/internal/research/throughput-properties-audit.md#L125)
- "PetaGene cobra % del ahorro a 290 MB/s; AtomBeam monetiza ancho de banda satelital" — [L121](docs/internal/research/throughput-properties-audit.md#L121)
- "Neural compression es ~2600x más lento que Zstd [SOURCED: arXiv:2603.25526]"; "Nadie ha resuelto el throughput — el mejor reportado es 181 KB/s en A100"; "NNCP v3 ~0.1 KB/s" — [docs/internal/research/backbone-scalability-analysis.md:L225-234](docs/internal/research/backbone-scalability-analysis.md#L225)
- GTM-era competitive table (2026-10-27): Microsoft MLVC (video), NVIDIA NTC (textures), cmix/NNCP (academic, "Low" threat), PetaGene, Brotli/zstd incumbent; "No direct competitor in neural compression for text/structured data archival"; risk "Competitor ships neural compressor: Low" — [docs/internal/business-case/GO-TO-MARKET-STRATEGY.md:L345-357,L341](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L345)
- MARKET-ANALYSIS competitor table: "NNCP/cmix — Academic — High compression — Slower than Redukt on CPU" — [MARKET-ANALYSIS.md:L74-78](docs/internal/business-case/MARKET-ANALYSIS.md#L74)

### Inferences
- Internal inconsistencies to resolve externally: cmix enwik8 BPB appears as ~1.1 (v19), 1.17 (v22), ~1.25 (cmix-neif-2.0) and ~1.63 (v21, almost certainly an error since v21 should be ≈1.1-1.2); Nacrith throughput 21-70 vs 50-90 tok/s; L3TC throughput quoted four ways (27 KB/s mobile CPU batch-1, 633 KB/s ANE batched, 1.30 MB/s iPhone-12 ANE, 4.35 MB/s decompression); NNCP "~10 KB/s" (Nov-01) vs "~0.1 KB/s" (backbone-scalability) vs "~KB/s".
- The claim "NNCP/cmix slower than Redukt on CPU" (MARKET-ANALYSIS:L78) is contradicted by the repo's own later numbers (cmix ~0.67-1 KB/s, NNCP ~KB/s vs Redukt 43 B/s single-stream in Nov-01 and ≈0.5-1.2 KB/s aggregate in Nov-10).
- StateSMix (1,300 tok/s CPU, 2.123 BPB) and OmniZip (200 KB/s CPU, 1.107 BPB) are the most relevant CPU-only comparators for the "fast tier" argument; neither is benchmarked against Redukt on the same files.
- All competitor enwik8 figures are full-100 MB results, while Redukt's are 10 KB windows — the repo acknowledges this is not directly comparable ([dual-backbone-benchmark.md:L137-140](data/benchmark/results/dual-backbone-benchmark.md#L137)).

### Gaps
- No competitor was ever run locally on Redukt's 12/18-file set (only gzip/brotli/xz). R-089 is blocked; the NNLCB benchmark was suggested but not executed.
- No zstd-19 or lzma2/7z numbers exist despite being named baselines in `docs/BENCHMARKS.md:L178-184`.

---

## KQ5. Do the business-case documents make market/pricing/segment claims that the measured numbers contradict?

### Takeaway
Yes. The business-case folder (written 2026-10-20..10-28, RWKV era) proposes EUR 5-12 per TB pricing, a "≥2 KB/s makes the product viable" threshold, log-archival savings of EUR 220K over 5 years for 1 PB/yr, a cold/archive vertical and "no direct competitor" — each of which is contradicted by the repo's own later measurements and break-even model ($50-60/GB ≈ $50,000-60,000 per TB of compute; thresholds 18-100 KB/s per vCPU; deep archive "worth ~0"; Nacrith 0.9389). The docs are self-aware about PDF (+40.7%) and medical imaging (dead) but were never updated for Fase 19 economics.

### Cited Findings

**Claims made (business-case folder)**
- Pricing (GTM, "PetaGene-validated"): "fixed price per TB of input processed, regardless of savings achieved": Starter <10 TB/mo EUR 12/TB (EUR 1,440/yr); Growth 10-100 TB/mo EUR 8/TB (EUR 9,600-96,000); Enterprise >100 TB/mo EUR 5/TB — [GO-TO-MARKET-STRATEGY.md:L308-319](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L308). Restated as "Precio/TB ahorrado" with free decompression in [throughput-paradigm-shift.md:L479-489](docs/internal/research/throughput-paradigm-shift.md#L479)
- Target segments & TAM: Enterprise Log Archival TAM $4.0-5.0B (log MANAGEMENT market, admitted narrower); E-Invoicing Spain $1.1B by 2034 "[SOURCED: Vocal Media — low authority]"; Medical Imaging/PACS "INVALIDATED"; Cold/Archive Storage "TAM: [UNSOURCED — $49B figure could not be verified]"; Satellite/maritime niche ($1-225/GB bandwidth, unsourced); Genomics "Not yet benchmarked" — [MARKET-ANALYSIS.md:L13-70](docs/internal/business-case/MARKET-ANALYSIS.md#L13)
- Target clients named: "Banks (Bankinter, Abanca, Kutxabank), telcos (Telefonica, MasMovil), utilities" — [MARKET-ANALYSIS.md:L24](docs/internal/business-case/MARKET-ANALYSIS.md#L24); "Target Customer Profile: Log volume >1 TB/month, Retention 5+ years, Storage S3/GCS/Azure Blob (compliance tier)" — [GO-TO-MARKET-STRATEGY.md:L297-306](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L297)
- Log archival case: mean log BPB 0.424 vs gzip 0.750 → 1 PB/yr raw stores 53.0 TB vs 93.8 TB; "Total storage savings 5 years: 220,320 [EUR, CALCULATED]" at EUR 30/TB-mo [ESTIMATE]; "Cores needed: 240,000 ... This is not viable at current throughput"; "Required for viable case: throughput ≥2 KB/s" — [BUSINESS-CASES.md:L123-194](docs/internal/business-case/BUSINESS-CASES.md#L123)
- Viability threshold claimed: "R-018 + R-015 + R-012 (8 cores) = ~2-3 KB/s en logs, 0% BPB loss. Esto es suficiente para un PoC de log archival a escala modesta" — [GO-TO-MARKET-STRATEGY.md:L275-278](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L275); "Current target: ~2-3 KB/s ... would make the product viable for mid-tier e-invoicing and small-scale log archival" — [LIMITATIONS.md:L98-99](docs/internal/business-case/LIMITATIONS.md#L98)
- Satellite case: "Fleet of 100 vessels × 10 GB/day × $6/GB × 365 = $21.9M/year bandwidth; With Redukt (-34%) saves $7.4M/year"; "Blocker: Throughput (~69 B/s on logs) insufficient for 10 GB/day"; "Need ~1 KB/s per core + 116 cores" — [MARKET-ANALYSIS.md:L59-63](docs/internal/business-case/MARKET-ANALYSIS.md#L59); [GO-TO-MARKET-STRATEGY.md:L82-92](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L82)
- Competitive claims: "No direct competitor in neural compression for text/structured data archival"; risk "Competitor ships neural compressor: Low probability" — [GO-TO-MARKET-STRATEGY.md:L356,L341](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L356)
- Positioning numbers used in those docs: "-34% vs brotli on logs, -48% on XML", "Wins 12/18", "~150 B/s (Q8), ~55 B/s (F32)", "~3,300x slower than brotli" — [GO-TO-MARKET-STRATEGY.md:L13-16,L48](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L13); [BENCHMARK-RESULTS.md:L103-104](docs/internal/business-case/BENCHMARK-RESULTS.md#L103)
- Self-corrections already inside the docs: PACS "Previous BPB numbers were fabricated" (x-ray claimed 3.87, actual 6.524) → "DEAD vertical" — [BUSINESS-CASES.md:L87-111](docs/internal/business-case/BUSINESS-CASES.md#L87); E-invoicing "BLOCKED ... PDF never benchmarked (67% of mix)" then R-V03 PDF +40.7% → "only viable with selective XML-only compression" — [L40-63](docs/internal/business-case/BUSINESS-CASES.md#L40); "Previous version used cross-entropy BPB (5-20% lower than actual)" — [GO-TO-MARKET-STRATEGY.md:L20-22](docs/internal/business-case/GO-TO-MARKET-STRATEGY.md#L20)
- Audit note: "Multiple TAM figures in this document are UNSOURCED ... Do not use unsourced figures in external communications" — [MARKET-ANALYSIS.md:L8-9](docs/internal/business-case/MARKET-ANALYSIS.md#L8)

**Measured facts that contradict them**
- Compute cost $50-60/GB raw (CPU spot) and $9-17/GB (T4) — [CLAUDE.md:L24](CLAUDE.md#L24); thresholds $0.06-0.33/GB — [CLAUDE.md:L22-23](CLAUDE.md#L22)
- Break-even throughput 18-100 KB/s PER vCPU — [scripts/breakeven.py](scripts/breakeven.py) run output; [ROADMAP.md:L2572-2576](docs/ROADMAP.md#L2572)
- Current per-vCPU throughput ~100 B/s — [throughput-properties-audit.md:L214](docs/internal/research/throughput-properties-audit.md#L214)
- "Deep/cold archive is NOT a target" — [CLAUDE.md:L16](CLAUDE.md#L16); "el archivo profundo ($0.001/GB-mes) es el peor caso de negocio" — [throughput-properties-audit.md:L119](docs/internal/research/throughput-properties-audit.md#L119)
- Spanish e-invoice market total saving ≈ $300/yr in S3 Standard — [throughput-properties-audit.md:L120](docs/internal/research/throughput-properties-audit.md#L120)
- Nacrith 0.9389 "Competidor Directo" (2026-11-09) — [strata-competitive-landscape.md:L91-95](docs/internal/research/strata-competitive-landscape.md#L91); 2025-2026 neural systems table — [throughput-paradigm-shift.md:L691-701](docs/internal/research/throughput-paradigm-shift.md#L691)
- No commercial precedent <1 MB/s — [throughput-properties-audit.md:L121](docs/internal/research/throughput-properties-audit.md#L121)

### Inferences (explicit contradiction list)
1. **Pricing vs cost**: EUR 5-12 per TB of input vs a measured compute cost of ≈$50,000-60,000 per TB (CPU spot) or ≈$9,000-17,000 per TB (T4). Revenue would cover ≈0.01-0.1% of compute. The per-TB-saved variant (paradigm-shift §5.5) is worse, since saved TB < input TB. Neither doc re-prices after Fase 19.
2. **"≥2 KB/s makes it viable"** (GTM, LIMITATIONS, BUSINESS-CASES) vs the break-even model requiring 18-100 KB/s per vCPU (≈10-50x higher), and vs the fact that Fase 19 already reached ≈1.2 KB/s per machine (≈0.1 KB/s per vCPU) while the repo still judges the gap 30x-500x.
3. **Log archival EUR 220K / 5 years savings on 1 PB/yr** vs compute cost of ≈$50M per PB at $50/GB; the doc itself computes 240,000 cores needed. The case's "DIRECTION CORRECT" verdict rests on a ratio advantage that is real (-54% apache at 10 KB, better than the -34% it quotes) but economically immaterial at measured throughput.
4. **Cold/Archive Storage vertical** (MARKET-ANALYSIS §4) is directly contradicted by CLAUDE.md L16 and the audit's "worst business case" finding; Glacier Deep Archive at $1/TB-mo makes 7-year savings $0.0026-0.0222/GB (script output).
5. **E-invoicing as a $1.1B-TAM segment**: the audit's own arithmetic says the ENTIRE Spanish B2B invoice flow would save ≈$300/yr of S3 Standard; XML is only 10% of the storage mix and PDF (67%) loses +40.7%. The repo's residual framing is "cumplimiento/inmutabilidad", not storage savings.
6. **Medical imaging / PACS**: already declared dead inside the docs (x-ray +42%/+6.4%, mr -3.8%/+0.8%); any external claim about DICOM must be treated as refuted by the repo.
7. **"No direct competitor" / "competitor ships neural compressor: Low probability"** vs Nacrith (0.9389, 2.6% better than 0.9641), OmniZip (1.107 at 200 KB/s CPU), StateSMix (1,300 tok/s CPU), FineZip, LLMZip, L3TC. The narrower claim that no COMMERCIAL general-purpose neural lossless product exists (paradigm-shift §8.2) is not contradicted inside the repo.
8. **"NNCP/cmix slower than Redukt on CPU"** (MARKET-ANALYSIS:L78) is contradicted by the repo's later figures (cmix ≈0.67-1 KB/s; Redukt 43 B/s single in Nov-01).
9. **Satellite/maritime niche**: the only segment both the business-case docs and CLAUDE.md path 3 keep alive; but its $6/GB bandwidth price is "[UNSOURCED]" and the throughput need (10 GB/day/vessel ≈ 116 KB/s) is ≈100x the current per-machine rate. Not contradicted by measurement, but unsupported.
10. **Headline ratios in the business docs are stale and understated**: they quote RWKV-era -34% logs / -48% XML / wins 12/18, while era-2 Mamba data shows -54% / -56% / 13/18 (18 files) or 8/12 (12 files). This is a favourable update, not a contradiction, but numbers must be aligned before external use.
11. **README vs internal diagnostic**: README positions "Best-in-class", "production", while the 2026-11-01 diagnostic says "DO NOT launch as a production compression product at current throughput" and recommends an open-source research tool positioning; Fase 19 did not revisit that verdict.

### Gaps
- No business-case document has been revised after 2026-10-28; none incorporates `scripts/breakeven.py`, Fase 19 throughput, the T4 results, or the per-vCPU cost metric. The report should treat the folder as superseded by CLAUDE.md "Commercial Target" + ROADMAP Fase 19.
- No customer, PoC, letter of intent or price test is recorded anywhere in the repo; pricing tiers are analogies to PetaGene, not market evidence.
- Bandwidth-niche economics (satellite, Datadog/Splunk ingestion $0.10-150/GB in `execution-architecture-analysis.md:L383,L425-426`) are listed but never run through the break-even model.
