# Strata backbone — 08. Viability frontier and capacity ladder (gate V7)

> Opened 2026-11-14 on the owner's instruction. The rule of gate V7 is normative in `05-gate-amendments.md`
> (amendment 4); this file holds the reasoning, the position at registration, and the plan.
> Labels: MEASURED (with its command), DERIVED (arithmetic on stated inputs), HYPOTHESIS, INFERENCE.
> Sections 6.2 and 6.3 were registered on 2026-11-14 (charter ledger) before any ladder model was evaluated on a
> decision file.

## 0. Result first

1. **The storage saving does not grow exponentially with time; it grows linearly, and the 7-year horizon is already
   in every threshold.** At today's cost ($33-60 per raw GB) the saving on text ($0.335 per GB over 7 years) pays
   back in 690-1,250 years. Volume does not help: cost and saving both scale with the bytes. (Section 1.)
2. **The only variable that can close the gap is the compute per byte, i.e. the size of the model.** That is the
   owner's objective for Strata, and it is now gate V7. (Section 2.)
3. **Weak point found while writing this (MEASURED): with PPMd in the classical set, Strata no longer beats the best
   classical codec on text.** `dickens` 500 KB: PPMd 1.968 bpb, Strata 2.04-2.13 (+3.7 % to +8.1 %). The bake-off's
   classical set had only the LZ and BWT families. Strata still wins on the three held-out logs (-15 % to -22 % in
   total bytes) and loses on markup. Projected gate V1: FAIL for both v1 tiers. (Section 3.)
4. **Position at registration.** Logs: the width-0 tier is projected viable (value $0.012 per GB over 7 years
   against a projected cost of $0.004) and is 29 % above Mamba-130M's bytes; width 88 is viable only if the kernels
   reach 10 G MAC/s. Text: not viable at any trained size, and 1.9-2.0x Mamba's bytes. Nothing meets the objective.
5. **Ceilings (DERIVED).** Storage economics cannot pay for a model above 14K-115K MAC/byte on logs or 138K-1.1M on
   text even if it matched Mamba-130M's ratio (Mamba itself: about 35.7M). The trained Strata rungs are inside that
   range; what they lack is the ratio. (Section 4.)
6. **Slope so far (MEASURED, frozen test set): 4x more compute (width 88 -> 208) bought 0.3-2.3 %; 133x (width 0 ->
   88) bought 2.9-6.2 %; on the selection set both bought less than 1 %.** The ladder asks whether that slope
   changes with more capacity and far more training, and whether
   knowledge held in tables (arm P) does what the trunk does not. (Sections 5 and 6.)

## 1. Why "100x too expensive" does not fade with time

Model of `scripts/breakeven.py` (hot object storage at $0.023 per GB-month, 7 years = 84 months), per raw GB of
enwik8-like text; brotli -11 leaves 29.4 % (2.352 bpb) and Redukt saves 59 % of that (DERIVED from the script's
inputs; `python scripts/breakeven.py`):

| | Bytes kept | Storage paid over 7 years | Compute to compress |
|---|---|---|---|
| Uncompressed | 100 % | $1.932 | — |
| brotli -11 | 29.4 % | $0.568 | about $0.005 |
| Redukt (Mamba-130M) | 12.1 % | $0.233 | **$33-60** (measured 2026-11-10 / 11) |

- The value of Redukt is the difference against the classical codec the client already has: **$0.335 per raw GB
  over 7 years**, i.e. $0.048 per GB and year. It is linear in time: 14 years are worth $0.67, 30 years $1.44.
- Break-even at today's cost: 33 / 0.048 = 690 years to 60 / 0.048 = 1,250 years (logs, at $0.06 per GB over
  7 years: 3,850-7,000 years).
- Volume: 1,000 GB cost 1,000x more to compress and save 1,000x more. The ratio does not move.
- Time works against the saving, not for it: storage prices fall, and the codec is symmetric, so every full read
  pays the compression cost again (the model above counts no read at all).
- The one place where a byte saved is worth far more than the compute is a link billed per byte sent (path 3).

So the saving is a fixed number per GB and the cost must come down to it: 100-170x on text, more on logs. Cost is
operations per byte. That is the case for a backbone designed for the codec, and the reason V7 exists.

## 2. The frontier

Same inequality as gate V4 and `02-evidence-and-constraints.md` C.4, per raw byte:

```
value   = (bpb_classical - bpb_model) / 8 * 1.932            $ per raw GB over the horizon
seconds = 1 us + MAC / R_mac                                  specification 7.2 (HYPOTHESIS)
cost    = P_cpu * 1e9 * seconds / 3600                        $ per raw GB, compression only
viable <=> bpb_classical - bpb_model >= 16.10 / R[KB/s]       at $0.014 per vCPU-hour (25.30 / R at $0.022)
```

| Constant | Value | Origin | Status |
|---|---|---|---|
| storage price, horizon | $0.023 per GB-month, 7 years | `scripts/breakeven.py` | registered before |
| compute price | $0.014 per vCPU-hour (and $0.022 reported) | gate V4; `breakeven.py` | registered before |
| table and bit work | 1 us per byte | specification 7.2 | HYPOTHESIS, unmeasured |
| MAC rate | 10 and 20 G MAC/s (7.2); 2.5 G = today's effective rate | 7.2; `02-evidence` C.5 (5-7 GFLOP/s) | HYPOTHESIS |
| materiality | saving >= 10 % of the best classical bytes | amendment 4 | new, a judgement |
| parity | bytes <= 1.10 x Mamba-130M on identical bytes | amendment 4 | new, a judgement |

Every verdict below is a projection until the Rust backbone exists and gate V4 measures bytes per second.
Tool: `python -I tools/strata/viability/viability.py` (standard library only; `--self-test` checks the constants
against V4's 16.1 and against `breakeven.py`'s $0.335).

## 3. Position at registration (MEASURED inputs, projected verdicts)

### 3.1 PPMd baseline

`python -I tools/strata/viability/ppmd_baseline.py` (7-Zip 26.03, PPMd var.H, orders 4-32, 256 MB, packed stream
without the container header, archive tested after writing; run 2026-11-14). bpb on exactly the bytes of the
bake-off's sets:

| Source | Role | Bytes | Best LZ / BWT (bake-off) | Best PPMd (order) | Best classical now | Strata `cm` / 88 / 208 |
|---|---|---|---|---|---|---|
| `webster` | selection | 300,000 | 1.8547 (bz2) | **1.6681** (8) | 1.6681 | 1.7388 / 1.7293 / 1.7213 |
| `xml` | selection | 300,000 | 0.3219 (bz2) | **0.2778** (32) | 0.2778 | 0.3872 / 0.4047 / 0.4057 |
| Zookeeper | selection | 300,000 | 0.2043 (xz) | **0.1856** (16) | 0.1856 | 0.1490 / 0.1382 / 0.1356 |
| Proxifier | selection | 300,000 | 0.2416 (bz2) | **0.2201** (32) | 0.2201 | 0.2023 / 0.1990 / 0.1971 |
| `dickens` | test | 500,000 | 2.1960 | **1.9681** (6) | 1.9681 | 2.1269 / 2.0653 / 2.0408 |
| Apache | test | 171,239 | 0.3132 | **0.2576** (12) | 0.2576 | 0.2054 / 0.1948 / 0.1861 |
| Linux | test | 216,485 | **0.3697** | 0.4039 (24) | 0.3697 | 0.3495 / 0.3203 / 0.3136 |
| OpenSSH | test | 225,216 | 0.3460 | **0.3222** (8) | 0.3222 | 0.2506 / 0.2408 / 0.2381 |
| `facturae` (synthetic) | test | 39,384 | 0.3953 | **0.3845** (24) | 0.3845 | 0.4785 / 0.4718 / 0.4876 |
| UBL (synthetic) | test | 83,260 | **0.2470** | 0.2490 (32) | 0.2470 | 0.3150 / 0.3184 / 0.3278 |
| `enwik8` | seen source | 500,000 | 2.2532 | **2.0326** (8) | 2.0326 | 2.1051 / 1.9437 / 1.8853 |

Strata columns: mean of seeds 1204 and 1205, ideal code lengths of the integer reference (`03-bakeoff.md` 12 and
12.2). Reading:
- Selection macro of the best classical codec: 0.6556 -> **0.5879**. The G0 tiers (0.614-0.619) are 4.4-5.3 % above
  it; the statement "`cm` is 5.5 % below the best classical codec" of `03-bakeoff.md` 12 holds only against the LZ
  and BWT families.
- Four real held-out test sources, total bytes: classical 163,699 -> 147,595; `cm` +4.2 %, width 88 +0.75 %, width
  208 -0.6 %. All tiers lose on `dickens` and win on the three logs.
- PPMd is a 2000s context-model codec that runs at megabytes per second on one core: the same cost class as `cm`.

### 3.2 Frontier for the three trained rungs (frozen test set, classical set with PPMd, $0.014 per vCPU-hour)

Projected throughput per vCPU at 2.5 / 10 / 20 G MAC/s: `cm` 923 / 980 / 990 KB/s; width 88 83 / 265 / 419; width
208 21 / 81 / 149.

| Class | Rung | MAC / byte | bpb | vs best classical | Value, $ / GB | Cost at 2.5 / 10 / 20 G | Cost <= value | Material | Bytes vs Mamba-130M |
|---|---|---|---|---|---|---|---|---|---|
| text (`dickens`) | R0 `cm` | 208 | 2.1269 | +8.1 % | -0.038 | 0.0042 / 0.0040 / 0.0039 | loses | — | x2.00 |
| | R1 width 88 | 27,752 | 2.0653 | +4.9 % | -0.024 | 0.047 / 0.015 / 0.009 | loses | — | x1.94 |
| | R2 width 208 | 113,912 | 2.0408 | +3.7 % | -0.018 | 0.181 / 0.048 / 0.026 | loses | — | x1.92 |
| logs (3 systems) | R0 `cm` | 208 | 0.2729 | **-15.0 %** | +0.0116 | 0.0042 / 0.0040 / 0.0039 | yes / yes / yes | yes | x1.29 |
| | R1 width 88 | 27,752 | 0.2560 | **-20.2 %** | +0.0157 | 0.047 / 0.015 / 0.009 | no / yes / yes | yes | x1.21 |
| | R2 width 208 | 113,912 | 0.2502 | **-22.0 %** | +0.0171 | 0.181 / 0.048 / 0.026 | no / no / no | yes | x1.18 |
| synthetic XML (no vote) | R0-R2 | | 0.368-0.379 | +26 % to +30 % | negative | | loses | — | x1.53-1.58 |

Mamba-130M figures are the repo's (real `.azth`, other conditions: `03-bakeoff.md` 12.2); V7 re-runs them on the
identical bytes. On the selection set the logs class gives the same picture with smaller margins (R0 -13.4 %, value
$0.0066; R1 and R2 fail the cost at every rate).

Verdict at registration, by the V7 rule: `logs` — "CM only" passes; R1 is O2 at the decision point (10 G MAC/s) and
fails at today's effective rate; `text` — O3; nothing is O1.

What the logs result is and is not: the width-0 tier is a context mixer with 1,199 parameters. It is a classical-class
codec that beats PPMd and xz by 15 % on three small log files (171-225 KB), worth about one cent per raw GB over
seven years. It has not been compared with stronger context mixers (zpaq, the PAQ family) or with log-specific
tools, and classical codecs improve with file size. It does not carry the product by itself.

## 4. Ceilings (DERIVED from 3.1 and section 2)

Largest model, in MAC per byte, whose compute the storage value can pay for, at 2.5 / 10 / 20 G MAC/s:

| Class | Price | If it coded to zero bytes | If it matched Mamba-130M | If it saved the material 10 % |
|---|---|---|---|---|
| text | $0.014 | 303K / 1,212K / 2,424K | **138K / 553K / 1,105K** | 28K / 112K / 224K |
| text | $0.022 | 192K / 768K / 1,536K | 87K / 348K / 696K | 17K / 68K / 136K |
| logs | $0.014 | 47K / 189K / 379K | **14K / 58K / 115K** | 2K / 10K / 20K |
| logs | $0.022 | 29K / 117K / 234K | 8K / 33K / 66K | 1K / 3K / 5K |
| synthetic XML | $0.014 | 43K / 171K / 342K | 5K / 22K / 44K | 2K / 8K / 16K |

- Mamba-130M spends about 35.7M MAC per byte (71.35 MFLOP, `02-evidence` C.5): 32-260x above the text ceiling at its
  own ratio. This is the Fase 20 verdict in another unit.
- Strata's rungs (208; 27,752; 113,912) are inside or near the ceilings. The cost side of the objective is reachable
  on paper. The missing half is the ratio: the text ceiling "at Mamba's ratio" requires 1.06 bpb on `dickens`, and
  the widest trained rung gives 2.04.
- No rung above about 0.4M MAC/byte can pay for itself on logs, and none above about 2.4M on text, whatever it
  compresses to. R4 of the ladder is therefore a slope probe, not a candidate.

## 5. What is already measured about capacity

| From -> to | Compute | Selection macro | Test, 6 held-out | Text (`dickens`) | Logs (test) |
|---|---|---|---|---|---|
| width 0 -> 88 | x133 | -0.02 % to -0.5 % | -3.1 % | -2.9 % | -6.2 % |
| width 88 -> 208 | x4.1 | -0.1 % to -0.8 % | -0.3 % to -0.6 % | -1.2 % | -2.3 % |

(`03-bakeoff.md` 12 and 12.2.) Other measured facts that bound the expectation:
- At equal budget the original trunk ties with GRU and LSTM controls (D2, D3, G0a): at this scale the architecture
  of the cell does not move the ratio; the online part does (static models alone: 4.7-5.1 bpb).
- Before the bit cells existed, 70-79 % of the bits went to the 4-30 % of bytes that no online table predicted
  (`03-bakeoff.md` 8). Only knowledge in the weights helps there, and a 73K-220K parameter trunk trained on
  33.55 MB seen once holds very little.
- Published small static models stay far from Mamba on unseen text (R-105: L3TC-3.2M 1.46 bpb on unseen enwik9;
  Pythia-14M 1.45 on enwik8) and our own static specialists failed on unseen systems (R-112). None of them had an
  online component, so they bound a trunk alone, not trunk + context mixer.
- Limits of the measurement itself: one pass over 33.55 MB, training curves still falling at the end, no XML in
  training, widest rung 114K MAC/byte, no general primer ever tried on long inputs.

INFERENCE (prior, stated before measuring): reaching parity with Mamba-130M on text inside the ceiling is unlikely;
a clear gain on logs with a better-trained R1 and a material gain from a general primer on text and markup are the
plausible positives. V7 is built so that either answer is reportable.

## 6. The ladder

### 6.1 Fixed by amendment 4 (not repeated here in full)

Arms W (compute rungs R0-R4 by MAC cap) and P (general primer sizes on R0 and R1); decision on the frozen test set
and on an extended set, the worse of the two; cost at 10 G MAC/s and $0.014, materiality 10 %, parity 1.10;
outcomes O1 / O2 / O3 per class; stop rules; GPU order and 12 h ceiling of charter 5b.

### 6.2 Arm W — configuration registered 2026-11-14, before the long run was launched

Trainer: `tools/strata/train/` (stage 4c; report in `09b-trainer-port.md`). It keeps the prototype's arithmetic and
runs its three sequential loops on the device. Exactness, MEASURED: new model == prototype on CPU, 0 mismatches
(`check_exact.py --test A`); cuda == cpu on a Tesla T4, 0 mismatches of 884,736 / 1,179,648 probabilities, trunk and
mixer state identical, for random and for trained weights (`--test B`, smoke run); trained export == sequential
int64 reference, 0 of 96,000 probabilities per tier (`export.py --check`); features == prototype's, 0 mismatches.

| Item | Registered value |
|---|---|
| Model | one nested Strata model `ladder3`: L = 3, geared, bit cells on, widths (88, 208, 456) = rungs R1, R2, R3 |
| MAC / byte, parameters | 27,752 / 73,392; 113,912 / 220,437; 499,552 / 731,906 (456 is the largest legal width under 512,000) |
| R0 | the existing G0 `cm` tier (no retraining: it has 1,199 parameters and no trunk) |
| R4 (only under stop rule 2) | `r4`: width 760 alone, 1,961,768 MAC / byte, 2,348,364 parameters |
| Training bytes | text 0.45: enwik9 bytes [1e8, 1e9) (900 MB); logs 0.45, uniform over the 11 training systems of the bake-off, first 64 MB of each (about 560 MB); misc 0.10: first 64 MB of the public DBLP XML snapshot of 2026-01-01 and the four non-literary Canterbury files. About 1.5 GB against the bake-off's 33.55 MB |
| Excluded by source | every selection and decision source (`corpus.check_training_list`), and the two Canterbury files that are Project Gutenberg etexts (`world192.txt`, `lcet10.txt`: the bake-off trained on them; PG text also occurs in `webster` and `dickens`) |
| Chunks | 8,192 bytes as 32 windows of 256, state carried inside a chunk, as in G0 (not the 32,768 of specification 8.3: kept equal to G0 so that the only changes are data, updates and width) |
| Optimiser | AdamW, lr 2e-3 with the bake-off's warm-up and cosine, clip 1.0, as G0 |
| Batch, updates | 256 streams x 256 bytes per update, 22,000 updates = 1.44 GB seen (G0: 64 x 256 x 2,048 = 33.55 MB): 43x the bytes, 10.7x the updates |
| Seeds | 1204 and 1205, one per GPU of the same session |
| Hyper-parameter search | none. The smoke run probed lr 1e-3 / 2e-3 / 4e-3 for 256 updates at batch 512 (train cost of the widest tier 1.608 / 1.531 / 1.465): too short to decide, so the G0 value is kept. Batch 256 instead of 512 because the T4 is saturated (118 vs 130 KB/s) and 256 gives 1.8x the updates per hour |
| Evaluation | `tools/strata/train/run_eval.py` on CPU, integer model, one stream per file, no reset; selection first, decision files only after this file's sha256 is in the ledger |

Smoke run (MEASURED, Kaggle T4 x2, 2026-11-14, 17.8 minutes = 0.30 GPU h, kernel version 1): every step passed;
`ladder3` 118 / 130 / 134 KB/s at batch 256 / 512 / 1,024 (1.80 / 0.99 / 0.51 updates/s), `ladder4` 65 / 61 KB/s,
`r4` 88 KB/s at batch 256; peak memory 2.7 / 5.4 / 10.7 GB; pause + resume reproduced the uninterrupted run's logged
costs exactly. The projection of `09b-trainer-port.md` (240 KB/s at batch 512) was 1.8x too optimistic.

Expected cost of the long run: 22,000 updates at 1.80 updates/s = 3.4 h plus downloads and export; the script stops
training at 260 minutes and itself at 300. Budget after it: at most 0.3 + 5.0 = 5.3 of the 12 GPU hours.

### 6.3 Arm P, arm P-s and the decision files — registered 2026-11-14 (amendment 5), before any decision evaluation

Selection evidence and harness: `09a-arm-p-selection.md`, `tools/strata/viability/primer_long.py`.

| Item | Registered value |
|---|---|
| Arm P | specification 3.4 unchanged; primer `mixI` (4,096-byte line-aligned blocks of the Wikipedia training slice, the 11 training log systems and the Canterbury training files, 45 / 45 / 10; sizes are prefixes); sizes 0, 65,536, 1,048,576, 16,777,216; **primary point 1,048,576**; tables `L` for primed rungs (successor 8/16/22x6, bit cells 8/16/24/24: 235.4 MB of state per stream), `D` and `L` at size 0; rungs R0 and R1 with the G0 weights, seeds 1204 and 1205 |
| Arm P-s (labelled, not specification 3.4) | the snapshot keeps only the successor tables; same primer, tables and rungs; sizes 1,048,576 and 16,777,216; **primary point 16,777,216**. Picked among six variants on the four selection sources |
| Verdict of a primed rung | at its primary point only. Classical side = min(best classical without prior, zstd -19 / -22 with the primer as dictionary); conditional PPMd / xz reported as a stricter reference |
| Extended decision set (`tools/strata/viability/sources.py`), 1,048,576 bytes each | text: `dickens` [500,000, 1,548,576), `bible.txt` [0, 1 MiB); logs: full `Apache.log` [0, 1 MiB), `Linux.log` and `OpenSSH.log` [951,424, 2,000,000); structured: GitHub event archive hour 2015-01-01-15 [0, 1 MiB), Silesia `nci` [0, 1 MiB). None is a training or selection source; the Linux and OpenSSH slices share no line with the frozen-test samples |
| PPMd | standard input with 1 GB (`baselines_ext.py`); best of both modes per file |
| Mamba-130M on identical bytes | `redukt compress --quant q8` on each decision slice (real `.azth`), run after the Strata evaluations |

Selection result that the registration is based on (MEASURED, macro of the four sources, bpb, mean of two seeds):
no primer 0.6193 (R0, default tables) / 0.6150 (large tables); arm P at 1 MiB with large tables 0.5995 (R0), 0.6048
(R1); arm P-s 0.5912 at 1 MiB and 0.5891 at 16 MiB (R0); best classical without prior 0.5879. Expectation written
before the measurement: O3 on text for both primed arms.

### 6.4 Order of work

1. Arm P on CPU with the existing G0 weights (no training): it is the cheapest experiment and tests the cheapest
   kind of capacity. Needs the long-input primer harness and the same-prior baselines.
2. Extended decision set and Mamba head-to-head on identical bytes.
3. Trainer port (stage 4c), bit-exactness check, CPU end-to-end run, 30-minute GPU smoke run.
4. Arm W: R1 and R2 retrained on the large set, then R3; R4 only under stop rule 2.
5. Verdict by the V7 rule in a new `09-capacity-ladder-results.md`; the charter ledger records it.

Stage 4a (Rust) is not required for the V7 verdict: it is required to turn an O1 or O2 into a measured V4.

## 7. Commands

```
python -I tools/strata/viability/viability.py --self-test     # constants vs gate V4 and scripts/breakeven.py
python -I tools/strata/viability/viability.py                  # sections 3.2 and 4
python -I tools/strata/viability/ppmd_baseline.py              # section 3.1 (needs 7-Zip and data/strata/, gitignored)
python scripts/breakeven.py                                    # section 1
```
