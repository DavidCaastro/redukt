# Kaggle full-size benchmark — 2026-11-10 (T4 + 4 vCPU Xeon)

Kernels: `redukt-gpu-fullsize-bench` (run-a: phases C lockstep + A GPU scale curve) and
`redukt-gpu-fullsize-bench-b` (run-b: phases A' CPU points + B whole industry files). Script:
`notebooks/redukt-gpu-fullsize-bench.py`, source at commit 48dbb06 (code identical to the 2026-11-10 HEAD).
Raw: `run-a/`, `run-b/` (`fullsize-results.csv`, `classical-fullsize.csv`, `gate-table.csv`, `fullsize-report.txt`).
Diagnosis of the length collapse and its mitigation: `mamba-length-collapse-2026-11-10.md`.

## A. Scale curve — enwik8 prefixes, Mamba-130M + CM, F32, single stream (T4 CUDA plugin via CLI)

| Prefix | Redukt bpb | gzip -9 | zstd -19 | zstd -22 --ultra --long | xz -9e | brotli -11 w24 | Redukt B/s (T4) |
|---|---|---|---|---|---|---|---|
| 10 KB | **0.9641** | 3.666 | 2.901 | 2.941 | 2.931 | 2.363 | 297 |
| 50 KB | 1.0539 | 3.47 | 2.70 | 2.75 | 2.72 | 2.35 | 396 |
| 100 KB | 1.6295 | 3.37 | 2.64 | 2.70 | 2.64 | 2.351 | 405 |
| 500 KB | **3.3099** | ~3.2 | ~2.4 | ~2.4 | ~2.3 | ~2.26 | 370 |
| 1 MB | **3.7096** | ~3.1 | ~2.3 | ~2.3 | ~2.2 | ~2.24 | 373 |
| 2 MB | **3.9056** | 3.05 | 2.28 | 2.26 | 2.17 | 2.242 | 375 |
| 100 MB (classical only) | — | 2.9156 | 2.1564 | 2.0267 | 1.9865 | 2.0594 | — |

(Classical values for the 50 KB-1 MB rows: see `run-a/classical-fullsize.csv` for the exact figures; the table rounds.)
The 4 MB point was skipped by the deadline guard. CPU on the same machine (run-b, 4 vCPU): 10 KB F32 0.9641 / Q8
0.9688; 100 KB F32 1.6299 / Q8 1.6327 — identical curve to the GPU plugin, so this is the model, not the device.
Roundtrips (10 KB, 100 KB) OK; `.azth` sha256 at 10 KB equals the 2026-11-10 CUDA validation (302d399d…).

**Redukt is worse than gzip -9 from ~500 KB onwards and worse than xz/brotli from ~300 KB.** Cause: Mamba-130M's
recurrent state degenerates beyond ~16-18K tokens (reproduced with the HF reference implementation; see the
diagnosis file). Mitigation measured locally: resetting the backbone state every 4096 tokens gives **0.9950 bpb on the
100 KB prefix** (vs 1.6327 without), roundtrip verified.

## B. Whole industry files (run-b, T4 plugin, F32, single stream; LF line endings on Kaggle)

| File | Size | Redukt | gzip -9 | zstd -19 | zstd -22 | xz -9e | brotli -11 | best classical | Redukt B/s |
|---|---|---|---|---|---|---|---|---|---|
| facturae-batch.xml | 38,371 | 0.4458 | 0.5696 | 0.4739 | 0.4679 | 0.4428 | **0.4078** | brotli | 252 |
| ubl-invoice-batch.xml | 81,558 | 0.4810 | 0.3883 | 0.2938 | 0.2827 | 0.2809 | **0.2519** | brotli | 238 |
| apache-access.log | 171,239 | 0.4937 | 0.4298 | 0.3370 | 0.3375 | 0.3151 | **0.3132** | brotli | 305 |
| linux-syslog.log | 216,485 | 1.1145 | 0.5421 | 0.4204 | 0.4204 | **0.3697** | 0.3789 | xz | 269 |
| openssh-auth.log | 225,216 | 1.1710 | 0.5532 | 0.3857 | 0.3857 | **0.3460** | 0.3592 | xz | 273 |

Redukt loses on all five whole files (it won all five at 10 KB). Roundtrip OK on both XML files (decompress ≈ compress
time).

### Same files with the experimental state reset every 4096 tokens (local CPU, Q8)

| File (local CRLF copy) | Size | no reset (Kaggle, LF copy) | **reset 4096** | best classical on the same bytes | Δ vs best classical |
|---|---|---|---|---|---|
| apache-access.log | 171,239 | 0.4937 | **0.1850** | brotli-11 0.3132 | **-41%** |
| facturae-batch.xml | 39,384 | 0.4458 | **0.2484** | brotli-11 0.3951 | **-37%** |
| ubl-invoice-batch.xml | 83,260 | 0.4810 | **0.2348** | brotli-11 0.2470 | **-5%** |
| linux-syslog.log | 216,485 | 1.1145 | **0.2453** | brotli-11 0.3789 | **-35%** |
| openssh-auth.log | 225,216 | 1.1710 | **0.2015** | brotli-11 0.3592 | **-44%** |

Classical figures from `competitors-fullsize-2026-11-01.csv` (same CRLF bytes). Lockstep `--jobs 5`, Q8, local i5; the
aggregate tok/s of this run (85.6) is NOT a throughput figure (it ran alongside builds and tests). LogShrink reports
0.143 bpb and LogReducer 0.186 on the Loghub Apache_2k log that `apache-access.log` is: with the reset Redukt ties
LogReducer and still trails LogShrink, at 3-4 orders of magnitude lower speed.


## C. Throughput and economic gate (run-a)

| System | Config | Aggregate B/s | Per vCPU / per T4 | $/GB (spot, Nov 2026) |
|---|---|---|---|---|
| CPU 4 vCPU Xeon (Kaggle) | lockstep Q8 jobs=4, 12 × 10 KB | 255.8 | 64 B/s per vCPU | ~$61 (c7i spot $0.014/vCPU-h) |
| CPU 4 vCPU Xeon | lockstep Q8 jobs=12 | 275.9 | 69 B/s per vCPU | **$56** |
| CPU 4 vCPU Xeon | lockstep F32 jobs=12 | 238.4 | 60 B/s per vCPU | $65 |
| T4 via CLI | single stream F32, enwik8 | 297-405 | 405 B/s per T4 | **$169** (g4dn spot $0.247/h) |
| T4 via FFI (2026-11-10 validation) | batch N=128 | ~4,200 (1,117 tok/s) | — | ~$16 |
| Laptop i5-1235U, 12 threads (reference) | lockstep Q8 N=32 | ~1,180 | ~98 B/s per vCPU | $50-60 |

Classical compute cost on the same machine, enwik8 100 MB, one core: zstd -19 1.40 MB/s ($0.003/GB), xz -9e
1.00 MB/s ($0.004/GB), brotli -11 0.34 MB/s ($0.011/GB), gzip -9 13 MB/s.

Gate: with the 2 MB figure the saving vs brotli is negative, so R* is undefined; with the reset mitigation and the
10 KB-like ratios, R* stays 18-100 KB/s per vCPU (`scripts/breakeven.py`), i.e. 260-1,450x the measured 69 B/s.

## Final: GPU scale curve and whole industry files WITH the R-108 reset (run-a-r108-gpu, kernel A v3, 56570a1)

T4 CUDA plugin via the CLI, F32, single stream, header RESET_LOG2 = 12 (4096 tokens), roundtrips OK at 10 KB,
100 KB and both XML files. The 4 MB point was skipped by the budget guard (needed 3.65 h with 4.94 h left and a
1.5 h margin); 2 MB is the largest measured prefix.

| Prefix | Redukt bpb, no reset (run-a) | **Redukt bpb, reset 4096 (run-a-r108-gpu)** | brotli -11 w24 | xz -9e | T4 B/s |
|---|---|---|---|---|---|
| 10 KB | 0.9641 | **0.9648** (+1 B header) | 2.363 | 2.931 | 292 |
| 50 KB | 1.0539 | **1.0372** | 2.35 | 2.72 | 391 |
| 100 KB | 1.6295 | **0.9925** | 2.351 | 2.64 | 399 |
| 500 KB | 3.3099 | **0.9697** | ~2.26 | ~2.3 | 365 |
| 1 MB | 3.7096 | **0.9781** | ~2.24 | ~2.2 | 368 |
| 2 MB | 3.9056 | **0.9998** | 2.242 | 2.17 | 367 |
| 100 MB (classical only) | — | — | 2.0594 | 1.9865 | — |

The curve is flat at 0.97-1.04 bpb from 10 KB to 2 MB instead of collapsing to 3.9; at 2 MB Redukt is 2.2x smaller
than xz -9e and brotli -11. Still 6% behind Nacrith's 0.9389 on the full 100 MB (not measured here: ~75 h on a T4 at
this single-stream rate).

| File (LF copy) | Size | no reset (run-b) | **reset 4096 (T4)** | best classical | Δ |
|---|---|---|---|---|---|
| facturae-batch.xml | 38,371 | 0.4458 | **0.2408** | brotli 0.4078 | **-41%** |
| ubl-invoice-batch.xml | 81,558 | 0.4810 | **0.2392** | brotli 0.2519 | **-5%** |
| apache-access.log | 171,239 | 0.4937 | **0.1849** | brotli 0.3132 | **-41%** |
| linux-syslog.log | 216,485 | 1.1145 | **0.2449** | xz 0.3697 | **-34%** |
| openssh-auth.log | 225,216 | 1.1710 | **0.2014** | xz 0.3460 | **-42%** |

All five whole industry files win again on the GPU tier too, matching the local CPU reset run within 1-4% (CPU
figures were measured on the CRLF copies).

## R-108 re-run (run-a-r108, kernel A v2, commit 61a9027, T4 + 4 vCPU)

CPU points with the header reset period (default 4096), Kaggle Xeon:

| Input | CPU F32 | CPU Q8 | sha256 Q8 (Kaggle Linux) | sha256 Q8 (local Windows) |
|---|---|---|---|---|
| enwik8 10 KB | 0.9648 (1,235 B) | 0.9695 (1,241 B) | d550671b4258c908 | d550671b4258c908 |
| enwik8 100 KB | **0.9925** (12,704 B) | **0.9951** (12,737 B) | d907ffc04c1d179b | d907ffc04c1d179b |
| enwik8 100 KB, `--reset-every 0` | — | 1.6327 (20,899 B) | 77c7e44692bcdbee (run-b) | 77c7e44692bcdbee |

Cross-OS gate (R-104 2b) holds for R-108 archives: byte-identical Windows <-> Linux. CPU lockstep 12 × 10 KB:
Q8 0.7551 at 251-269 B/s, F32 0.7531 at 231 B/s (4 vCPU). **The GPU rows of this run repeat the collapsed curve
and the industry losses** (0.9641 -> 3.9056; apache 0.4937 …) because at 61a9027 the plugin still ignored the
header period (fixed in 56570a1); the plugin lockstep was rejected by the pre-R-109 guard. Kernel A v3 (56570a1,
phases A+B only so the 4 MB point fits the budget) re-measures the GPU curve with the reset.

## E. Plugin batch scaling through the CLI (run-e-batch, kernel B v4, 2026-11-11, T4 + 4 vCPU)

Open item of R-109: the bare FFI harness gave 1,117 tok/s at N=128 (2026-11-10), but the product path
(`redukt compress --input-dir … --jobs N --backbone plugin`) had only been measured at N=12. Phase E of
`notebooks/redukt-gpu-fullsize-bench.py` measures N = 12/32/64/128 lockstep streams on 128 distinct enwik8 windows
(700 KB apart, 10 KB each) and one long pass of 128 × 50 KB (6.55 MB, 3 state resets per stream). Rust code at
56570a1 (unchanged); raw data in `run-e-batch/fullsize-results.csv` and `fullsize-report.txt`.

| Streams (`--jobs`) | Input | bpb | Wall s | Wall B/s (incl. model load) | Engine tok/s | Engine B/s | $/GB spot | $/GB on-demand | Gate |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 12 × 10 KB | 1.1215 | 69.0 | 1,781 | 530.4 | 1,807 | $38.0 | $80.9 | batch == single sha256 3/3 PASS |
| 32 | 32 × 10 KB | 1.1068 | 168.4 | 1,946 | 573.4 | 1,955 | $35.1 | $74.7 | batch decompress roundtrip 32/32 PASS (157 s) |
| 64 | 64 × 10 KB | 1.0736 | 315.5 | **2,077** | **581.8** | 2,083 | **$32.9** | $70.2 | — |
| 128 | 128 × 10 KB | 1.0861 | 654.8 | 2,002 | 559.0 | 2,005 | $34.2 | $72.9 | batch == single sha256 3/3 PASS |
| 128 | 128 × 50 KB | 1.0493 | 3,344.7 | 1,959 | 552.5 | 1,960 | $35.0 | $74.6 | — |

"Engine" is the CLI's own summary line (`N tokens in X s of lockstep`), i.e. the lockstep without model load;
$/GB = price_h × 1e9 / (B/s × 3600) with g4dn.xlarge at $0.247/h spot and $0.526/h on-demand (engine B/s).
Model load is ~1 s of wall here, so wall and engine rates agree within 1.5%.

Findings:

1. **The product path saturates at ~530-580 tok/s (~2.0 KB/s) from N=12 on.** Going from 12 to 128 streams changes
   the aggregate rate by +5% (peak at N=64, slightly lower at 128). Cost per token-stream is constant at ~1.72-1.89 ms,
   so the batch is not buying anything beyond N≈12: every stream adds a fixed cost.
2. **It is half the bare FFI figure.** The harness that only calls `forward_batch` reached 1,117 tok/s at N=128
   (0.90 ms per token-stream). The CLI adds ~0.85-0.9 ms per token-stream on top. That increment is whatever the
   lockstep does around the device call: for every stream and every token the device copies 50,280 F32 logits to
   the host (25.7 MB per step at N=128, pageable memory) and the 4 host vCPUs build the CDF (one `exp` per vocab
   entry, 50,280 per stream-token), run the CM ensemble + mixer and the range coder. The split between the logits
   copy and the host CPU work is NOT measured yet; it is an inference from the two totals.
3. **Real GPU-tier cost through the CLI: ~$33-35/GB spot, $70-75/GB on-demand** — not the $9-17/GB extrapolated
   from the FFI harness. That is ~100x the text threshold ($0.33/GB) and ~1.7x better than the 4-vCPU CPU tier
   on the same machine ($56-61/GB). Earlier figure for 12 mixed slices (run-c-r109b): 1,376 B/s = $49.9/GB.
4. **Correctness holds at every N**: batch == single-file sha256 at N=12 and N=128 (3/3 each), batch decompress
   roundtrip 32/32 byte-identical, and the 128 × 50 KB pass (12-14K tokens per stream, R-108 resets inside the
   batch) stays at 1.049 bpb. The bpb differences between rows are the windows themselves (more windows = other
   regions of enwik8), not the batch size; 50 KB streams compress better than 10 KB ones as expected.
5. **Consequence for the plan**: "GPU kernels at cuBLAS efficiency" (admitted path 1) cannot by itself deliver the
   15-25x headroom through the product: even an infinitely fast device forward leaves the ~0.85 ms per token-stream
   of copy + host work, i.e. a ceiling near ~1,150 tok/s (~4 KB/s, ~$17/GB) on this 4-vCPU host. The next GPU
   lever is the host pipeline (R-110): measure the split, then move softmax/CDF construction to the device
   (return a quantized CDF or only what the coder needs instead of 50,280 F32 logits) and/or pin the transfer.

Phase D of the script was also fixed in this run: it used to print "T4 single-stream best 0 B/s -> $inf/GB"
whenever phase A was not selected, and "measured 0.000 KB/s" when phase C was not; it now takes the best T4 row of
any phase (single stream, lockstep or batch scaling), reports spot and on-demand $/GB for wall and engine rates,
adds per-T4 R* columns to `gate-table.csv`, and says "n/a" for tiers that were not measured.

## R-109 validated (run-c-r109, kernel B v2, commit b219870, T4 + 4 vCPU)

Phase C re-run with the FFI v2 slot lockstep (`run-c-r109/fullsize-results.csv`):

| System | Config | bpb (12 × 10 KB) | Aggregate B/s (incl. model load) | Gate |
|---|---|---|---|---|
| T4 plugin, before R-109 (run-a) | jobs=12, sequential v1 fallback | 1.9492 | 336.6 | streams interleaved one state |
| **T4 plugin, R-109** | jobs=12, FFI v2 slots | **0.7481** | **1,401.5** (87.7 s) | **batch == single sha256 3/3 PASS** |
| **T4 plugin, R-109 + header reset (run-c-r109b, 56570a1)** | jobs=12, FFI v2 slots, RESET_LOG2 12 | **0.7529** | **1,375.7** (89.3 s) | **batch == single 3/3 PASS** |
| CPU 4 vCPU, F32 | jobs=12 | 0.7531 | 245.6 | — |
| CPU 4 vCPU, Q8 | jobs=12 | 0.7551 | 283.1 | — |

The plugin lockstep is now ~5x the 4-vCPU CPU aggregate on the same machine and ~4.2x its own single-stream
CLI rate (336 B/s). CPU bpb moved 0.7503 -> 0.7551 (Q8) because the R-108 default reset (4096 tokens) fires inside
the 10 KB log slices (~2 B/token -> ~5K tokens); enwik8 10 KB (2.7K tokens) is unaffected. In run-c-r109 the plugin
ignored the period (archives written with reset 0, bpb 0.7481); with 56570a1 it follows the header and lands at
0.7529, i.e. the CPU F32 figure (0.7531) within 3 bytes of GPU numerics (run-c-r109b).

## Defect found (run-a): `--jobs N --backbone plugin` corrupted predictions

run-a lockstep with the plugin (jobs=12, 12 slices) gave **1.9492 bpb vs 0.7503 on CPU** for the same files. The
FFI v1 plugin keeps ONE process-global recurrent state (`PluginState` holds only a reset fn), so interleaved streams
overwrite each other's state. Output is still decodable only if the decoder replays the identical interleaving; a
single-file decompress of such an archive would fail. Guard added: the CLI rejects `--jobs > 1` with the plugin until
the lockstep uses the FFI v2 slot API.
