# Strata backbone — 03. Bake-off (stage 2 of `00-charter.md`)

> Date: 2026-11-12 / 13. Local CPU only (i5-1235U, shared with another process); nothing was sent to Kaggle.
> Code: `tools/strata/bakeoff/`. Data: `data/strata/` (git-ignored). Criterion: `tools/strata/bakeoff/CRITERION.md`,
> written before the runs, sha256 `c29fa87015d06fc2bb2311c26b8cb949ef8da2af4b7a5b55b90e4348d84a023f`.
> Every number in this file is **MEASURED** by a command of section 11 unless it carries another label.
> Bits are ideal code lengths of the probabilities handed to the coder (16-bit integer probabilities for the
> integer models); real range-coder bytes are in sections 6 and 7.6.

## 0. Result first

**Weak points first.**

1. **No model of this bake-off beats the best classical codec on the selection set** (macro 0.748-0.771 bpb
   against 0.656): all win on the two log sources and lose on `webster` and `xml`. On the frozen test set the
   candidate wins on the three Loghub samples (-5 % to -20 %) and loses on `dickens` (+2.5 %) and on the two
   synthetic invoice files (+61 %, +70 %); its total over the six test sources is 1.9 % above the best classical
   codec of each file. Against Mamba-130M (repo figures, other conditions) it is 35-164 % worse per file.
2. **The trunk is a minor term on long unseen streams.** Online components off: cost x6.3-6.6. Trunk zeroed:
   +11-14 %. With adaptive bit statistics added post hoc (section 8) the trained trunk is worth 0-15 % on three
   unseen sources and costs 4-8 % on `xml`; it is worth 14-18 % on the seen source (Wikipedia).
3. **On the frozen test set (report-only) the candidate is 1.9 % worse than the GRU control** in the macro of the
   six held-out sources (0.7016 against 0.6888; corrected on 2026-11-13 after review 04 m-5: the first version
   averaged the seen-source `enwik8` in and said 3.5 %), all of it on text (`dickens` +4.6 %; seen `enwik8`
   +7.1 %); the fixed-time-constant `strata_f` (0.6904) is better than the chosen `strata_g` on that set; it is equal or better on
   the logs. The float gated cells also fit the training distribution better (final train cost 1.60 against
   1.44 for the GRU, 1.50 for the LSTM).
4. **The prototype's online part had a gap** (no adaptive low-order bit statistics), found after the decision.
   The specification adds them (its section 3.3) and is **not frozen** until this bake-off is repeated with them
   in the loop (its gate G0).
5. Process: the runs were interrupted twice by agent usage limits and resumed from intact checkpoints (section
   2); the first draft of the tight coder flush was wrong and was caught by the random coder test (section 6);
   D3 was written after the test set had become visible (its rule is mechanical; disclosed in 7.2).

**Decision by the pre-registered criterion (selection set only).**

| | Candidate | Control | Ratio | Verdict |
|---|---|---|---|---|
| D1 | `strata_g` 0.7480 | `strata_f` 0.7708 | 0.970 | geared trunk chosen |
| D2, tier B | `strata_g` 0.7480 (integer, 113,848 MAC/byte) | `gru` 0.7518 (float, 127,164 MAC/byte) | 0.995 | **TIE**: trunk frozen as designed |
| D3, tier A | `strata_g` tier A 0.7511 (integer, nested, 27,688 MAC/byte) | `lstm_a` 0.7484 (float, trained alone, 29,824 MAC/byte) | 1.004 | **TIE**, but **written after the frozen test set was visible; not used for any later decision** |

**What held up.**

- Integer arithmetic costs +0.04 % to +0.35 % against the same architecture trained in float (budget +5 %).
- Integer simulation == sequential int64 reference on both trained candidates: 0 mismatches over 96,000
  probabilities per tier; real range-coder round trips byte-identical.
- No length collapse without any reset: from 10 KB to 2 MiB the late/early window ratio is 1.043 on `dickens` and
  1.033 on `enwik8`, the same as the float LSTM (1.043 / 1.032). Mamba-130M without reset: 0.96 -> 3.91 bpb.
- Nesting is free here: tier A, at 24 % of tier B's MACs and trained in the same run, is 0.4 % worse on the
  selection set and ties with separately trained controls.
- **Short messages (path 3)**, the 480 messages of R-113 (byte-identical to that run: 2,399 of 2,400 classical
  sizes match): with a 64 KB primer taken from the same half of each source that trains the zstd dictionary, the
  payload + 4 B (the R-114 target) beats zstd-19 + dictionary in **30 of 30 cells**; with the real framing of a
  primed archive (24 B compact / 28 B default) in **22 / 20 of 30**, and **17 / 15 of the 25 cells** without the
  seen-source Wikipedia (corrected after review 04 G-1: the first version counted 20 B and said 24 of 30);
  without the primer 10 of 30 (payload). With the primer the trunk is worth 4-33 % per cell. Mamba-130M (no
  primer) gives 26 of 30 (payload) and stays far better on English text (prose, 2 KB: 349 against 662 bytes).

---

## 1. What was compared

All trunks run inside the same coder-domain harness (binary-tree head, online successor tables, online
calibration, online mixer; `BACKBONE-STRATA-SPEC.md` sections 2.5 and 3.2), so the comparison isolates the trunk.
One extra control shows the published cell in its textbook form.

| Name | Trunk | Arithmetic evaluated | Tiers |
|---|---|---|---|
| `strata_f` | original: strata layers with fixed power-of-two time constants | integer (exact simulation) | A and B nested, one training run |
| `strata_g` | original: same with geared time constants (input-driven choice between two shifts) | integer (exact simulation) | A and B nested, one training run |
| `lstm`, `gru` | PyTorch `nn.LSTM` / `nn.GRU`, one layer, byte embedding 64 | float (their best case) | B |
| `lstm_a`, `gru_a` | same, embedding 32 | float | A |
| `lstm_sm` | LSTM + 256-way softmax, static, no online component (the R-112 form) | float | B |
| `strata_f_float`, `strata_g_float` | the candidates trained and evaluated without the integer simulation | float | A and B |

Budget caps (same for all): tier A <= 32,000 MAC/byte and <= 96,000 parameters; tier B <= 128,000 MAC/byte and
<= 256,000 parameters. MACs are counted analytically (`models.py`: `macs()`), never timed. Same data in the same
order (seed 1204), 2,048 steps of 64 streams x 256 bytes (33.55 MB seen once), AdamW 2e-3 with warm-up and cosine
decay, one seed.

## 2. Pre-registration and what happened to the schedule

- `CRITERION.md` was written at 22:03 on 2026-11-12, its sha256 was printed at 22:04:13, the first full run
  started at 22:06. Smoke runs of 64 steps had been executed before, to debug the pipeline; their selection-set
  numbers are disclosed in the criterion file.
- Decision rule (verbatim in the criterion): **D1** the candidate with the lower selection macro bpb at tier B
  (if within 2 %: `strata_f`); **D2** candidate (integer) vs the better of `lstm` / `gru` (float) at tier B:
  WIN below 0.98x, TIE within +-2 %, LOSS above 1.02x; **D3** the same at tier A. The frozen test set is read
  only after the decision is written (section 7.2 was written before section 7.3 was computed).
- The runs were interrupted at about 22:46 by a usage limit of the agent harness (not by a code error) and
  resumed at 02:02 from the checkpoints on disk: `lstm` was complete; `strata_f` was at step 1,088 and
  `strata_g` at step 1,024. Both checkpoints loaded and every tensor was finite. The trainer resumes from
  `(model, optimiser, step)` and the data of a step is a pure function of the seed and the step, so the resumed
  runs see the same data; the resumed part used 4 BLAS threads instead of 3, which changes floating-point
  summation order in the gradients (not the data, not the integer forward pass).
- Source files used by the runs did not change after the launch: the sha256 printed at 22:04 on 2026-11-12 and
  re-checked at 08:35 on 2026-11-13 are identical (`common.py` 29fc2638..., `models.py` bef57807..., `train.py`
  6e77a3e5..., `evaluate.py` 5e1d9ed6..., `CRITERION.md` c29fa870...). Changed afterwards, none of them used by
  training or by the selection evaluation: `ref_int.py` (the tight-flush defect of section 6), `fetch_data.py`
  (GitHub events), `shortmsg.py` (ablation and alignment options); added: `baselines.py`, `curve.py`,
  `shortmsg.py`, `explore_bitcells.py`, `report.py`.
- After the review (04, m-18): the four measured sources are kept unchanged in
  `tools/strata/bakeoff/frozen-bakeoff-2026-11-12/` (sha256 29fc2638…, bef57807…, 6e77a3e5…, 5e1d9ed6…, as above).
  The working copies were extended for G0 at 08:47 on 2026-11-13 (bit cells as an option); with the option off they
  reproduce the bake-off's selection results byte for byte: `lstm`, `gru` and `strata_g` `eval-select-on.csv`
  re-computed and compared with `cmp` (MEASURED, 08:48 and 09:28-09:30).

## 3. Data and split (by source)

Download: `python -I tools/strata/bakeoff/fetch_data.py` (plain GET with a generic agent; nothing else is sent).

| Role | Source | Bytes used | Origin / note |
|---|---|---|---|
| train | Wikipedia | 200,000,000 | bytes [1e8, 3e8) of enwik9. The script hashes the first 1e8 bytes while extracting and refuses to write the slice unless the hash equals that of the local `enwik8`; the slice exists, so the check passed (its two printed hash lines were cut from the captured output) |
| train | Loghub: BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack, HealthApp | 16,000,000 each | first 16 MB of each full log (Zenodo record 3227177); the Hadoop slice starts with the archive's readme text |
| train | Canterbury, non-literary: world192.txt, lcet10.txt, cp.html, fields.c, grammar.lsp, xargs.1 | 2,943,855 in total | breadth only |
| selection | Silesia `webster`, Silesia `xml`, Loghub Zookeeper, Loghub Proxifier | first 300,000 each | never trained on; the only data used for the decision |
| frozen test | `dickens` (first 500,000), Loghub 2k samples Apache / Linux / OpenSSH (whole), synthetic `facturae` and `ubl` (whole, local CRLF copies) | | report-only |
| seen-source | `enwik8` first 500,000 | | Wikipedia is a training source: **not held-out**, reported separately |
| length curves | `dickens` and `enwik8`, first 2,097,152 | | `enwik8` is seen-source |
| short messages | R-113 protocol: `enwik8` (seen), `dickens`, the three logs, one hour of GitHub events (JSON) | | section 7.6 |

Sampling weights of the training chunks: Wikipedia 0.45, logs 0.45 (uniform over systems), Canterbury 0.10.
No XML was trained on. Project Gutenberg text was, in small amounts: Canterbury's `world192.txt` and
`lcet10.txt` are PG etexts, and PG boilerplate also occurs in `webster` (selection) and `dickens` (test) (review
04, m-8; the first version of this sentence said no PG text was trained on). Sources that share a source with a test file were
excluded as in `02-evidence-and-constraints.md` E.3.

## 4. Models and measured budgets

| Model | Configuration | Tier | MAC / byte | Parameters | Cap (MAC / params) |
|---|---|---|---|---|---|
| `strata_f` | widths (96, 224), L = 3, fixed shifts | A | 29,328 | 78,767 | 32,000 / 96,000 |
| | | B | 117,392 | 231,727 | 128,000 / 256,000 |
| `strata_g` | widths (88, 208), L = 3, geared (one gate per 8 channels) | A | 27,688 | 73,104 | 32,000 / 96,000 |
| | | B | 113,848 | 220,149 | 128,000 / 256,000 |
| `lstm` | embedding 64, hidden 147 | B | 126,564 | 181,264 | 128,000 / 256,000 |
| `gru` | embedding 64, hidden 174 | B | 127,164 | 188,401 | 128,000 / 256,000 |
| `lstm_a` | embedding 32, hidden 70 | A | 29,824 | 56,665 | 32,000 / 96,000 |
| `gru_a` | embedding 32, hidden 82 | A | 29,500 | 59,237 | 32,000 / 96,000 |
| `lstm_sm` | embedding 64, hidden 125, softmax 256 | B | 126,500 | 144,073 | 128,000 / 256,000 |

MACs include the head (8 node dot products per byte), the trunk-driven gains and the online mixer (dot product
and update) for every model that has them; `lstm_sm` counts its 256-way output layer instead.

The controls reach the MAC cap first and stay below the parameter cap: every weight of a dense recurrent cell is
touched once per byte, while the candidates keep part of their parameters in tables that are read sparsely
(byte embedding, 255 node vectors of which 8 are used per byte).

## 5. Procedure

- Training: chunks of 8,192 consecutive bytes (start moved to the next line start when one is within 512
  bytes), fed as 32 windows of 256 bytes with the recurrent state, the online tables, the calibration cells and
  the online mixer carried across windows and reset at chunk start. Loss = mean over tiers of the mean coding
  cost per byte. The online tables and cells are computed exactly in the data pipeline (they are functions of
  the bytes only); the online mixer is simulated exactly inside the forward pass.
- Evaluation: each source is one stream coded from its first byte with the boot state, state carried to the
  end, **no reset**. Ablations on the same weights: `online off` (tables off, mixer frozen at its boot weights)
  and `trunk off` (trunk output zeroed: what remains is node biases + tables + calibration + mixer).
- Classical codecs on the identical bytes: xz -9e, zstd -19 and -22, brotli -11 (lgwin 24), bz2 -9, gzip -9
  (`baselines.py`, Python bindings).

## 6. Checks of the integer contract and of the coder (design stage)

| Check | Result | Command |
|---|---|---|
| PyTorch integer simulation (vectorised over time, vectorised table emulation) vs sequential int64 reference (bit by bit, sequential tables and mixer), smoke models | `strata_f` 64 steps: 48,000 probabilities per tier, **0 mismatches**, both tiers; `strata_g` 64 steps: 40,000 per tier, **0 mismatches**, both tiers | `ref_int.py --run strata_f_smoke --source sel-webster --bytes 6000 --roundtrip 1500`; `ref_int.py --run strata_g_smoke --source sel-zookeeper --bytes 5000 --roundtrip 1200` |
| Same on the trained candidates | `strata_g`: 96,000 probabilities per tier, **0 mismatches**, both tiers; 2,000-byte round trips OK (tier B: ideal 678.24 B, coded 678 B tight / 682 B legacy). `strata_f`: 96,000 per tier, **0 mismatches**, round trips OK (tier B: ideal 685.74 B, coded 686 / 690 B) | `ref_int.py --run strata_f|strata_g --source sel-webster --bytes 12000 --roundtrip 2000` |
| Real compress -> decompress through a Python port of the AZTH range coder (binary decisions, total 2^16), decoder driven only by the integer reference | byte-identical on every run above; e.g. smoke `strata_f` tier B, 1,500 bytes: ideal 671.03 B, coded 672 B with the tight flush, 676 B with the legacy flush | same commands |
| Tight flush alone, random streams | 120,000 random bit streams (1-300 decisions, extreme and moderate probabilities): 0 decode failures with either flush; the tight flush is exactly 4 bytes shorter. **The first draft of the rule was wrong** (it wrote the encoder's dummy first cache byte): 8 failures in 30,000 streams, all on streams short enough that no byte had been emitted yet; fixed by the `first` condition now in the specification | inline test, section 11 |

The reference of this section is a design-stage check written by the architect; the oracle required by rule T2
is a separate implementation written from the specification alone.

## 7. Results

### 7.1 Selection set (the only table that decides)

`online` = successor tables + calibration + online mixer; `trunk off` = trunk output zeroed on the same weights.

| model | arith | tier | MAC/byte | params | online | trunk | sel-webster | sel-xml | sel-zookeeper | sel-proxifier | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|
| strata_f | int | B | 117,392 | 231,727 | on | on | 2.1346 | 0.5287 | **0.1662** | 0.2537 | 0.7708 |
| **strata_g** | int | B | 113,848 | 220,149 | on | on | 2.0426 | **0.5216** | 0.1689 | 0.2589 | **0.7480** |
| strata_f_float | float | B | 117,392 | 231,727 | on | on | 2.1135 | 0.5309 | 0.1683 | 0.2596 | 0.7681 |
| strata_g_float | float | B | 113,848 | 220,149 | on | on | 2.0376 | 0.5220 | 0.1699 | 0.2587 | 0.7470 |
| lstm | float | B | 126,564 | 181,264 | on | on | **2.0283** | 0.5422 | 0.1713 | 0.2774 | 0.7548 |
| gru | float | B | 127,164 | 188,401 | on | on | 2.0350 | 0.5331 | 0.1696 | 0.2694 | 0.7518 |
| lstm_sm | float | B | 126,500 | 144,073 | none | on | 4.5159 | 5.9370 | 4.1586 | 4.8671 | 4.8697 |
| strata_f | int | A | 29,328 | 78,767 | on | on | 2.1039 | 0.5178 | 0.1705 | 0.2579 | 0.7625 |
| **strata_g** | int | A | 27,688 | 73,104 | on | on | 2.0524 | 0.5221 | 0.1754 | **0.2546** | 0.7511 |
| strata_f_float | float | A | 29,328 | 78,767 | on | on | 2.1043 | 0.5203 | 0.1735 | 0.2575 | 0.7639 |
| strata_g_float | float | A | 27,688 | 73,104 | on | on | 2.0442 | 0.5193 | 0.1761 | 0.2637 | 0.7508 |
| lstm_a | float | A | 29,824 | 56,665 | on | on | **2.0221** | 0.5256 | 0.1769 | 0.2689 | **0.7484** |
| gru_a | float | A | 29,500 | 59,237 | on | on | 2.0430 | **0.5185** | 0.1796 | 0.2611 | 0.7506 |
| strata_f | int | B | | | on | off | 2.3401 | 0.5372 | 0.2254 | 0.3075 | 0.8526 |
| strata_g | int | B | | | on | off | 2.3505 | 0.5385 | 0.2258 | 0.3080 | 0.8557 |
| lstm | float | B | | | on | off | 2.3674 | 0.5376 | 0.2252 | 0.3088 | 0.8598 |
| gru | float | B | | | on | off | 2.3682 | 0.5384 | 0.2254 | 0.3089 | 0.8602 |
| strata_f | int | A | | | on | off | 2.3431 | 0.5357 | 0.2249 | 0.3092 | 0.8532 |
| strata_g | int | A | | | on | off | 2.3503 | 0.5369 | 0.2262 | 0.3089 | 0.8556 |
| strata_f | int | B | | | off | on | 5.3353 | 6.2341 | 4.1877 | 4.5682 | 5.0813 |
| strata_g | int | B | | | off | on | 4.8723 | 6.0339 | 4.2946 | 4.6053 | 4.9515 |
| strata_f_float | float | B | | | off | on | 5.2097 | 6.2893 | 4.1887 | 4.4299 | 5.0294 |
| strata_g_float | float | B | | | off | on | 4.9923 | 5.9865 | 4.2632 | 4.9196 | 5.0404 |
| lstm | float | B | | | off | on | 4.6603 | 6.1320 | 3.9984 | 4.3309 | 4.7804 |
| gru | float | B | | | off | on | 4.8391 | 5.9574 | 3.8952 | 4.2651 | 4.7392 |
| strata_f | int | A | | | off | on | 5.2235 | 5.8954 | 4.3948 | 4.6984 | 5.0530 |
| strata_g | int | A | | | off | on | 5.0582 | 5.8960 | 4.6490 | 4.6895 | 5.0732 |
| strata_f_float | float | A | | | off | on | 5.1614 | 6.1361 | 4.4288 | 4.6854 | 5.1029 |
| strata_g_float | float | A | | | off | on | 5.0885 | 5.9296 | 4.5783 | 4.6783 | 5.0687 |
| lstm_a | float | A | | | off | on | 4.8629 | 6.2723 | 4.3557 | 4.7340 | 5.0562 |
| gru_a | float | A | | | off | on | 4.8262 | 5.8704 | 4.3531 | 4.3480 | 4.8494 |
| **best classical** | | | | | | | 1.8547 (bz2-9) | 0.3219 (bz2-9) | 0.2043 (xz-9e) | 0.2416 (bz2-9) | **0.6556** |

Reading: with the online components on, the six tier-B trunks (float twins included) span 0.7470-0.7708 (3 %);
with them off,
4.74-5.08; the textbook static LSTM (`lstm_sm`) 4.87. The static binary-tree head and the static softmax head are
equivalent (`lstm` online off 4.78 against `lstm_sm` 4.87): what changes the picture is the online part.

### 7.2 Decision (written before the frozen test set was read)

Written at 03:07 on 2026-11-13 with the tier-B rows of section 7.1 complete, before
`data/strata/runs/post-decision.log` (frozen test set, curves, short messages) was opened.

- **D1: `strata_g`.** Selection macro 0.7480 against 0.7708 for `strata_f`: 3.0 % lower, outside the 2 % band.
  The geared trunk is the candidate; the fixed-time-constant trunk is dropped as the v1 default.
  Post-hoc caveat (review 04, m-10): the difference comes mostly from `webster` (ratio 0.957); `strata_f` is
  better on Zookeeper and Proxifier, the geometric mean of the per-source ratios differs by 0.5 % (inside the
  band) and on the held-out test set `strata_f` is 1.6 % better. D1 is procedurally valid (arithmetic macro, as
  registered) but weak; `GEARED` stays a file option.
- **D2: TIE.** `S` = 0.7480 (`strata_g`, integer arithmetic, 113,848 MAC/byte). `C` = 0.7518 (`gru`, float,
  127,164 MAC/byte; `lstm` 0.7548). `S / C` = 0.995, inside the +-2 % band. By the rule the trunk stands as
  designed.
- **D3 (tier A):** **TIE.** `strata_g` tier A (integer, nested, 27,688 MAC/byte) 0.7511 against the better tier-A control `lstm_a` (float, trained alone, 29,824 MAC/byte) 0.7484; `gru_a` 0.7506; ratio 1.004, inside the band. Disclosure: this line was written at 08:30 on 2026-11-13, after the frozen test set of 7.3 had been printed together with the tier-A selection rows; the rule is a fixed ratio and involves no judgement. Following the coordinator's instruction of 2026-11-13, D3 is excluded from every decision taken after the bake-off (G0 pre-registers its own tier-A comparison).

What this tie is and is not. The three tier-B trunks are within 0.9 % of each other, while switching the online
components off multiplies the cost by 6.3-6.6 and zeroing the trunk raises it by 11-14 %: on these sources the
trunk is a minor term, so a tie between trunks is the expected outcome and says little about which cell is
better. It does say that, at equal budget and with 12 % fewer MACs, the integer, bounded, nested design is not
worse than the float controls here. It does **not** say the design beats classical codecs: every model of the
table loses to the best classical codec on the selection macro (0.748-0.771 against 0.656), winning on Zookeeper
only (section 8 explains where the bits go).

### 7.3 Frozen test set and seen-source slice (report-only)

| model | tier | trunk | test-dickens | test-apache | test-linux | test-openssh | test-facturae | test-ubl | seen-enwik8 | held-out macro (6) |
|---|---|---|---|---|---|---|---|---|---|---|
| strata_f | B | on | 2.2703 | 0.2506 | 0.3498 | 0.2995 | **0.5761** | **0.3960** | 1.9683 | 0.6904 |
| strata_g | B | on | 2.2502 | **0.2494** | 0.3511 | 0.3020 | 0.6367 | 0.4205 | 1.9277 | 0.7016 |
| lstm | B | on | 2.1870 | 0.2546 | **0.3353** | **0.2958** | 0.6335 | 0.4309 | 1.8375 | 0.6895 |
| gru | B | on | **2.1516** | 0.2599 | 0.3358 | 0.2966 | 0.6566 | 0.4321 | **1.7996** | **0.6888** |
| strata_f | A | on | 2.3151 | 0.2513 | 0.3635 | 0.3027 | 0.5563 | 0.3894 | 2.0413 | 0.6964 |
| strata_g | A | on | 2.3093 | 0.2574 | 0.3515 | 0.3017 | 0.6143 | 0.4040 | 2.0218 | 0.7064 |
| strata_g | B | off | 2.9231 | 0.3114 | 0.4673 | 0.3727 | 0.6713 | 0.4563 | 2.7140 | 0.8670 |
| strata_g | A | off | 2.9302 | 0.3116 | 0.4671 | 0.3737 | 0.6639 | 0.4530 | 2.7240 | 0.8666 |
| best classical | | | 2.1960 (bz2-9) | 0.3132 (brotli-11) | 0.3697 (xz-9e) | 0.3460 (xz-9e) | 0.3953 (brotli-11) | 0.2470 (brotli-11) | 2.2532 (brotli-11) | 0.6445 |
| Mamba-130M, repo (real `.azth`, reset 4096, BPE; XML on LF copies) | | | 1.0620 | 0.1849 | 0.2449 | 0.2014 | 0.2408 | 0.2392 | 0.9697 (512,000 B) | |

- `strata_g` tier B against the best classical codec: wins on Apache (-20.4 %), Linux (-5.0 %), OpenSSH
  (-12.7 %); loses on `dickens` (+2.5 %), `facturae` (+61 %), `ubl` (+70 %). Bytes over the six test sources:
  171,490 against 168,216 (+1.9 %). Seen-source `enwik8`: -14.4 % (not held-out).
- Against the GRU: +4.6 % on `dickens`, +7.1 % on `enwik8`; -4.0 % on Apache, +4.6 % on Linux, +1.8 % on
  OpenSSH; -3.0 % and -2.7 % on the two XML files. On text the gated float cells are better.
- With the trunk zeroed (same weights) the held-out macro rises by 23.6 % (0.8670 against 0.7016); on the three
  logs by 23-33 %. (The first version said 29 %, with the seen-source column averaged in.) These are upper bounds of the trunk's value (boot weights were trained with the trunk present).
- The two invoice files are synthetic (20 copies of one template); long exact repeats favour LZ codecs and no
  XML was in the training data.

### 7.4 Integer arithmetic vs float, and the cost of nesting

| Candidate | Tier | Integer (evaluated) | Float twin (trained and evaluated without the simulation) | Integer / float |
|---|---|---|---|---|
| `strata_g` | B | 0.7480 | 0.7470 | +0.13 % |
| `strata_g` | A | 0.7511 | 0.7508 | +0.04 % |
| `strata_f` | B | 0.7708 | 0.7681 | +0.35 % |
| `strata_f` | A | 0.7625 | 0.7639 | -0.18 % |

The integer contract costs less than the repo's +5 % budget by more than an order of magnitude (gate V5).

Nesting: tier A of `strata_g` uses 24 % of tier B's MACs and comes from the same training run. Selection macro
0.7511 against 0.7480 (+0.4 %); frozen test macro 0.8943 against 0.8768 (+2.0 %); `dickens` +2.6 %; seen
`enwik8` +4.9 %. Against separately trained controls of the same size it ties (D3). The cost of nesting itself
(the same tier trained alone) was not measured.

### 7.5 Length curves (no reset exists)

Single streams from the boot state, no reset of any kind; cumulative bpb at each prefix and bpb of two windows.

| model | tier | source | 10,240 | 51,200 | 102,400 | 512,000 | 1 MiB | 1.5 MiB | 2 MiB | window [0.5, 1) MiB | window [1.5, 2) MiB | late / early |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strata_g | A | dickens | 2.8206 | 2.6072 | 2.5260 | 2.3047 | 2.2642 | 2.2926 | 2.2974 | 2.2272 | 2.3119 | 1.038 |
| strata_g | B | dickens | 2.6832 | 2.5024 | 2.4382 | 2.2459 | 2.2108 | 2.2403 | 2.2484 | 2.1788 | 2.2729 | 1.043 |
| lstm | B | dickens | 2.5618 | 2.4162 | 2.3628 | 2.1828 | 2.1505 | 2.1781 | 2.1866 | 2.1209 | 2.2121 | 1.043 |
| strata_g | A | enwik8 (seen) | 2.1332 | 2.2673 | 2.1557 | 2.0233 | 2.0136 | 2.0314 | 2.0403 | 2.0063 | 2.0668 | 1.030 |
| strata_g | B | enwik8 (seen) | 1.9419 | 2.1233 | 2.0331 | 1.9295 | 1.9279 | 1.9485 | 1.9591 | 1.9279 | 1.9911 | 1.033 |
| lstm | B | enwik8 (seen) | 1.8343 | 2.0078 | 1.9304 | 1.8395 | 1.8418 | 1.8613 | 1.8720 | 1.8451 | 1.9041 | 1.032 |
| best classical (independent compressions of each prefix) | | dickens | 2.7742 (brotli) | 2.7092 (brotli) | 2.6213 (bz2) | 2.1890 (bz2) | 2.1790 (bz2) | | 2.2208 (bz2) | | | |
| best classical | | enwik8 | 2.3625 (brotli) | 2.4528 (brotli) | 2.3511 (brotli) | 2.2532 (brotli) | 2.2358 (brotli) | | 2.2420 (brotli) | | | |
| Mamba-130M + CM, reset 4096 (repo, T4) | | enwik8 | 0.9648 | 1.0372 | 0.9925 | 0.9697 | 0.9781 | | 0.9998 | | | |
| Mamba-130M + CM, no reset (repo, T4) | | enwik8 | 0.9641 | 1.0539 | 1.6295 | 3.3099 | 3.7096 | | 3.9056 | | | |

- The candidate passes the length gate C4a of the specification on both sources and both tiers (2 MiB within 3 %
  of 10 KB or better; late window within 25 % of the early window: worst ratio 1.043).
- The late/early ratios of the integer candidate equal those of the float LSTM to the third decimal, so the small
  rise of the late window is a property of the text, not of the state.
- The absolute level is the weak point again: on `dickens` the candidate ends 1.2 % above bz2 at 2 MiB.

### 7.6 Short messages (path 3), real coded bytes

Protocol of R-113 (`notebooks/short-message-bench.py`, reproduced in `shortmsg.py`): 16 messages per
(source, size), cut at line starts from the second half of each source; zstd-19 with a 110 KB dictionary trained
on the first half of the **same** source (best case for zstd); minimal framing for every classical codec. The
messages are the same bytes as in R-113: 2,399 of the 2,400 classical sizes per message are identical to the R-113
file. Strata = `strata_g` tier B, real range-coder bytes from the integer reference with the tight flush, each
message coded from the boot state; **payload** = coded bytes without header and CRC. Primer = the stream machine
first reads the last 65,536 bytes of the first half of the same source (the half that trains the dictionary) and
every message starts from that snapshot (tables, cells and mixer kept; trunk, delay taps and history reset).
Mamba-130M = R-113 archives (Q8, CPU) minus the 20-byte framing. Wikipedia is a seen source for Strata.

Mean bytes per message:

| source | size | zstd-19 + dict | brotli-11 | Mamba-130M payload | Strata payload | Strata + primer | Strata + primer, trunk zeroed |
|---|---|---|---|---|---|---|---|
| json-events | 128 | 38.3 | 105.7 | 33.9 | 88.8 | **26.9** | 28.1 |
| json-events | 256 | 65.8 | 162.8 | **48.4** | 143.1 | 54.6 | 57.6 |
| json-events | 512 | 109.3 | 258.2 | **92.2** | 246.6 | 104.9 | 113.7 |
| json-events | 1024 | 194.9 | 382.7 | **149.2** | 387.9 | 188.1 | 206.4 |
| json-events | 2048 | 357.5 | 600.1 | **242.7** | 637.9 | 349.5 | 384.9 |
| log-apache | 128 | 34.8 | 98.9 | 34.8 | 58.4 | **17.6** | 23.3 |
| log-apache | 256 | 44.1 | 125.1 | 49.9 | 87.8 | **22.8** | 28.9 |
| log-apache | 512 | 58.7 | 157.9 | 70.0 | 120.9 | **32.0** | 39.2 |
| log-apache | 1024 | 87.2 | 192.2 | 87.1 | 151.8 | **50.2** | 61.4 |
| log-apache | 2048 | 135.2 | 248.9 | 114.1 | 200.2 | **82.6** | 101.1 |
| log-linux | 128 | 57.2 | 89.7 | 39.3 | 49.2 | **27.3** | 36.1 |
| log-linux | 256 | 62.9 | 108.9 | 48.5 | 62.8 | **31.5** | 42.1 |
| log-linux | 512 | 74.9 | 124.2 | 56.1 | 77.9 | **41.0** | 56.4 |
| log-linux | 1024 | 117.2 | 171.1 | 79.8 | 119.3 | **73.4** | 99.2 |
| log-linux | 2048 | 163.1 | 240.3 | **114.3** | 183.2 | 114.6 | 154.1 |
| log-openssh | 128 | 41.6 | 99.4 | 39.0 | 61.4 | **18.8** | 23.4 |
| log-openssh | 256 | 60.5 | 158.1 | 62.1 | 107.1 | **30.8** | 38.0 |
| log-openssh | 512 | 81.8 | 204.6 | 87.6 | 152.1 | **45.2** | 55.5 |
| log-openssh | 1024 | 111.7 | 245.8 | 107.4 | 190.0 | **67.6** | 83.1 |
| log-openssh | 2048 | 165.5 | 301.6 | 129.1 | 242.7 | **104.9** | 128.9 |
| prose | 128 | 68.6 | 93.4 | **35.6** | 53.9 | 43.2 | 54.9 |
| prose | 256 | 120.7 | 160.4 | **59.6** | 107.4 | 86.2 | 108.0 |
| prose | 512 | 211.1 | 275.6 | **103.5** | 210.6 | 171.6 | 214.5 |
| prose | 1024 | 398.2 | 485.9 | **188.9** | 408.2 | 339.1 | 426.6 |
| prose | 2048 | 769.4 | 878.6 | **349.1** | 782.0 | 662.3 | 839.9 |
| wikipedia (seen) | 128 | 73.8 | 96.8 | **44.1** | 48.8 | 44.6 | 64.1 |
| wikipedia (seen) | 256 | 127.1 | 155.4 | **55.8** | 90.8 | 84.7 | 122.9 |
| wikipedia (seen) | 512 | 210.2 | 245.4 | **91.0** | 162.1 | 152.9 | 227.3 |
| wikipedia (seen) | 1024 | 391.9 | 438.9 | **168.3** | 310.9 | 292.4 | 430.8 |
| wikipedia (seen) | 2048 | 747.4 | 794.3 | **392.8** | 610.3 | 575.8 | 843.7 |

Cells (of 30) where the mean is smaller than zstd-19 + dictionary, adding a framing of 0 / 4 / 20 bytes to the
payload (20 B = the full Strata v5 header in compact form + CRC-32):

| | +0 B | +4 B (R-114 target) | real framing: +20 B without primer, +24 B with (compact) | +28 B (primer, default header) | held-out 25 cells, real framing |
|---|---|---|---|---|---|
| Strata payload, no primer | 10 | 8 | 5 | — | 0 |
| Strata + primer | **30** | **30** | **22** | 20 | **17** (15 with the default header) |
| Strata + primer, trunk zeroed | 21 | 21 | 5 | 3 | 5 |
| Mamba-130M (no primer; its archive has 20 B of framing) | 26 (payload) | 23 | 16 (real archive) | — | 11 |

Corrected after review 04 G-1: the first version of this table counted the primed archives with 20 B of
framing; a primed archive also carries the 4-byte `PRIMER_ID`.

- The primer is the decisive lever for path 3: 10 -> 30 cells at payload + 4 B, 5 -> 22 with the real framing. It is hook H2 of the draft specification, now
  promoted into v1 (specification section 3.4).
- With the primer the trunk is worth 4-9 % on JSON, 18-27 % on the logs and prose, 30-33 % on the seen
  Wikipedia: on short messages the online statistics have little to learn from, and the trained prior matters.
- Strata + primer is smaller than Mamba-130M (which has no primer) in 15 of 30 cells: all log cells but one,
  JSON at 128 B; Mamba is far better on prose, Wikipedia and JSON from 256 B.
- Limits: the primer and the dictionary come from the same half of the same source (best case for both; a
  primer from a different source was not measured); 16 messages per cell; one model.

## 8. Where the bits go, and a post-hoc exploration (decides nothing)

Cost breakdown of `lstm` (tier B, trained) on the first 100,000 bytes of each selection source, by what the
successor tables offered for the byte (`diag` command of section 11):

| Source | bpb | highest live order right | another order right | predictions exist, all wrong | no prediction |
|---|---|---|---|---|---|
| sel-xml | 0.6296 | 90.4 % of bytes, 0.118 bits/byte, 17.0 % of the bits | 1.4 %, 3.43, 7.7 % | **8.1 %, 5.80, 74.5 %** | 0.1 %, 6.81, 0.8 % |
| sel-webster | 2.3136 | 61.3 %, 0.654, 17.3 % | 9.0 %, 3.10, 12.1 % | **29.6 %, 5.51, 70.4 %** | 0.1 %, 4.38, 0.2 % |
| sel-proxifier | 0.3057 | 94.7 %, 0.049, 15.1 % | 0.6 %, 3.31, 6.5 % | **4.6 %, 5.14, 77.7 %** | 0.1 %, 4.06, 0.7 % |
| sel-zookeeper | 0.2480 | 95.3 %, 0.044, 16.9 % | 0.3 %, 2.64, 3.5 % | **4.3 %, 4.58, 78.6 %** | 0.1 %, 3.66, 1.1 % |

70-79 % of the bits are spent on the 4-30 % of bytes that no table predicted, at 4.6-5.8 bits per byte: for
those bytes the only predictor is the static trunk, which on a source it was not trained on is at the level of
the `online off` rows of section 7.1. The prototype has no adaptive statistics for such bytes; every classical
context mixer has them.

**Post-hoc exploration** (`explore_bitcells.py`; float arithmetic; nothing retrained; on the selection set and on
the seen-source slice, never on the frozen test set): adaptive bit cells `P(bit | previous 0..3 bytes, tree node)`
added as extra mixer inputs with boot weight 0.3, to the trained `lstm` (tier B). First 100,000 bytes, bpb:

| Variant | sel-xml | sel-webster | sel-zookeeper | sel-proxifier | seen-enwik8 |
|---|---|---|---|---|---|
| as trained (re-computed by the script; matches `evaluate.py` to 0.0003) | 0.6293 | 2.3134 | 0.2480 | 0.3057 | 1.9307 |
| + bit cells, order 0 | 0.5905 | 2.2472 | 0.2467 | 0.2922 | 1.9345 |
| + orders 0-1 | 0.5412 | 2.1795 | 0.2353 | 0.2670 | 1.9429 |
| + orders 0-2 | 0.5167 | 2.1315 | 0.2246 | 0.2456 | 1.9439 |
| + orders 0-3 | 0.5014 | 2.0899 | 0.2213 | 0.2416 | 1.9394 |
| trunk off, as trained | 0.6573 | 2.7899 | 0.3395 | 0.3488 | 3.0683 |
| trunk off + orders 0-3 | 0.4626 | 2.1021 | 0.2587 | 0.2418 | 2.3580 |

Same exploration on the selected candidate (`strata_g` tier B; its integer head re-implemented in float by
the script, so the first row is a float approximation of the integer model):

| Variant | sel-xml | sel-webster | sel-zookeeper | sel-proxifier | seen-enwik8 |
|---|---|---|---|---|---|
| as trained | 0.6068 | 2.3155 | 0.2512 | 0.2869 | 2.0340 |
| + bit cells, order 0 | 0.5736 | 2.2454 | 0.2477 | 0.2730 | 2.0351 |
| + orders 0-1 | 0.5135 | 2.1435 | 0.2362 | 0.2564 | 2.0387 |
| + orders 0-2 | 0.4897 | 2.0801 | 0.2247 | 0.2353 | 2.0305 |
| + orders 0-3 | 0.4770 | 2.0371 | 0.2210 | 0.2294 | 2.0186 |
| trunk off, as trained | 0.6623 | 2.7778 | 0.3407 | 0.3478 | 3.0491 |
| trunk off + orders 0-3 | 0.4589 | 2.0974 | 0.2555 | 0.2363 | 2.3541 |

Reading (exploratory, one model, 100 KB, cells not trained in the loop):

1. The cells lower the cost by 9.7 % (webster), 20.3 % (xml), 10.8 % (zookeeper) and 21.0 % (proxifier) on
   unseen sources, and by nothing on the seen source (where the trunk already predicts those bytes).
2. With the cells present, **the trained trunk is worth -0.6 % on webster, 0.0 % on proxifier, -14.5 % on
   zookeeper, and it costs +8.4 % on xml**; on the seen source it is worth -17.8 % (2.358 -> 1.939).
   In other words: on a source that resembles nothing in the training set the trunk buys nothing; on a source
   that resembles the training set (Java service logs, Wikipedia) it buys 15-18 %.
3. This is the R-105 / R-112 conclusion again (a small pretrained model is a domain specialist), now measured
   with the online component that those experiments lacked.

This is why the specification adds section 3.3 (bit cells) and gate G0 (repeat this bake-off with them in the
loop) and marks itself as not frozen.

## 9. Wall-clock (information only; another process shared the machine)

| Run | Steps | Training wall time (min) | Final train cost, mean of the last 50 logged steps (tier A / tier B) |
|---|---|---|---|
| strata_f | 2,048 | 84.4 | 1.7931 / 1.6779 |
| strata_g | 2,048 | 83.3 | 1.7592 / 1.5979 |
| strata_f_float | 2,048 | 78.9 | 1.7839 / 1.6648 |
| strata_g_float | 2,048 | 80.7 | 1.7498 / 1.5876 |
| lstm | 2,048 | 28.0 | 1.4981 |
| gru | 2,048 | 51.6 | 1.4401 |
| lstm_a | 2,048 | 26.7 | 1.6883 |
| gru_a | 2,048 | 23.1 | 1.6399 |
| lstm_sm | 2,048 | 8.8 | 2.4560 |

Train cost of the widest tier at common steps: `strata_g` 1.929 / 1.841 / 1.751 / 1.649 / 1.598 at steps
250 / 500 / 1,000 / 1,500 / 2,000; `gru` 1.784 / 1.714 / 1.593 / 1.486 / 1.440; `lstm` 1.842 / 1.792 / 1.658 /
1.549 / 1.498. All curves are still falling at the end. The candidates' wall time is about 3x the LSTM's, because
their exact sequential parts (accumulator scans, online mixers of two tiers) run as Python loops over numpy; the
controls use the fused PyTorch cells.

## 10. Limitations

- One seed. The 2 % band of the criterion is a judgement, not a measured variance.
- 33.55 MB of training data seen once; every model is far from converged (train cost still falling at the end).
- Controls in float, candidates in integer arithmetic: biased against the candidates on purpose.
- All trunks share the head, tables and mixer; the trunk differences are therefore small next to the effect of
  the online components, and the selection sources are far from the training distribution.
- The online component of the prototype lacks low-order adaptive bit statistics (section 8); every absolute
  number against classical codecs in this file is for that prototype.
- Ideal code lengths, not container bytes, except in sections 6 and 7.6.
- The successor tables use 2^16 / 2^18 slots; table sizes were not swept.
- The Mamba-130M numbers quoted for comparison come from the repo's documents (other tokenisation, real `.azth`
  with header); they are context, not a measured head-to-head.

## 11. Commands

```
python -I tools/strata/bakeoff/fetch_data.py                                   # corpora -> data/strata/raw
python -I tools/strata/bakeoff/precompute.py --first 0 --last 31               # optional: training feature cache
python -I tools/strata/bakeoff/train.py --model M --steps 2048 --minutes 7.5 --threads 4     # repeat until DONE
        M in strata_f strata_g strata_f_float strata_g_float lstm gru lstm_a gru_a lstm_sm
python -I tools/strata/bakeoff/evaluate.py --run M --set select --online on|off [--trunk off]
python -I tools/strata/bakeoff/evaluate.py --run M --set test --online on [--trunk off]
python -I tools/strata/bakeoff/curve.py --prepare curve-dickens|curve-enwik8    # once
python -I tools/strata/bakeoff/curve.py --run M --minutes 7                    # repeat until DONE
python -I tools/strata/bakeoff/ref_int.py --run strata_g --source sel-webster --bytes 12000 --roundtrip 2000
python -I tools/strata/bakeoff/shortmsg.py --run strata_g --tier 1                       # no primer
python -I tools/strata/bakeoff/shortmsg.py --run strata_g --tier 1 --primer 65536
python -I tools/strata/bakeoff/shortmsg.py --run strata_g --tier 1 --primer 65536 --trunk off         --sources wikipedia,prose --out shortmsg-nt1.csv           # (and two more processes for the other sources)
python -I tools/strata/bakeoff/shortmsg.py --summary strata_g         --mamba data/benchmark/results/short-message-bench/run-1/short-message-results.csv
python -I tools/strata/bakeoff/explore_bitcells.py --run lstm|strata_g --bytes 100000         --sources sel-webster,sel-xml,sel-zookeeper,sel-proxifier,seen-enwik8
python -I tools/strata/bakeoff/baselines.py                                    # classical codecs
python -I tools/strata/bakeoff/report.py all                                   # the tables of section 7
```

The cost breakdown of section 8 (`diag`), the random coder test of section 6 and the spot values of the
specification's appendix A were produced by short inline scripts that import `common.py`, `models.py` and
`ref_int.py`; the coder test is: 3 seeds x 40,000 streams, each 1-300 binary decisions with probabilities drawn
from {1, 2, 32768, 65534, 65535}, uniform, or near-certain, encoded with `RangeEncoder.encode_bit`, finished with
`finish()` and with `finish_tight()`, decoded with `RangeDecoder.decode_bit`.

---

## 12. Gate G0 (bit cells in the loop) — decision

> Rule: `05-gate-amendments.md`, pre-registration of 08:44 and amendment 1 of 09:16 (sha256 of the file at
> registration `30a8cb51…`, logged in `00-charter.md` at 09:16, before any G0 selection or test result was read).
> Decision computed at 11:05 with `python -I tools/strata/bakeoff/g0_decide.py --g0b pass` from selection data only
> (output kept in `data/strata/runs/g0-decision.txt`); this section was written before any test-set run of a G0
> model was started. Same data, steps, optimiser, budgets and chunk length as the bake-off; bit cells of
> specification 3.3 with 8 / 16 / 22 / 22 table bits; seeds 1204 and 1205 for the candidate, 1204 for the controls.

**G0b, integer contract (both seeds):** trainer simulation == sequential int64 reference, bit cells included: 0
mismatches over 96,000 probabilities per tier for the three tiers (`cm`, A, B) of each seed; 2,000-byte round trips
byte-identical (e.g. seed 1204 tier B: ideal 664.70 B, coded 665 B). A same-author consistency check, not T2
evidence (review 04, m-9). PASS.

**Metric L, selection macro bpb (long streams, no reset):**

| model | seed | tier | MAC/byte | webster | xml | Zookeeper | Proxifier | macro |
|---|---|---|---|---|---|---|---|---|
| strata_g_bc | 1204 | cm | 208 | 1.7387 | **0.3872** | 0.1491 | 0.2024 | 0.6193 |
| strata_g_bc | 1204 | A | 27,752 | 1.7279 | 0.4086 | 0.1390 | 0.2014 | 0.6192 |
| strata_g_bc | 1204 | B | 113,912 | 1.7171 | 0.4061 | 0.1340 | 0.1988 | 0.6140 |
| strata_g_bc | 1205 | cm | 208 | 1.7388 | **0.3872** | 0.1489 | 0.2022 | 0.6193 |
| strata_g_bc | 1205 | A | 27,752 | 1.7308 | 0.4008 | 0.1373 | 0.1966 | 0.6164 |
| strata_g_bc | 1205 | B | 113,912 | 1.7255 | 0.4054 | 0.1372 | **0.1953** | 0.6159 |
| gru_bc | 1204 | B | 127,228 | **1.6975** | 0.4252 | **0.1298** | 0.2001 | **0.6131** |
| lstm_bc | 1204 | B | 126,628 | 1.7102 | 0.4244 | 0.1347 | 0.1997 | 0.6172 |
| gru_a_bc | 1204 | A | 29,564 | 1.7153 | 0.4185 | 0.1310 | 0.1983 | 0.6158 |
| lstm_a_bc | 1204 | A | 29,888 | 1.7127 | 0.4094 | 0.1339 | 0.1986 | 0.6136 |
| best classical | | | | 1.8547 (bz2) | 0.3219 (bz2) | 0.2043 (xz) | 0.2416 (bz2) | 0.6556 |

**Metric S, selection short messages** (4 sources x 5 sizes x 2 primer cases, 16 messages per cell, integer
reference, real range-coder bytes; saving per cell against the `cm` tier of the same file):

| seed | tier | S saving vs `cm` | e (% per 100K MAC) | marginal e over A | logs only | text + markup only | payload bytes (same / foreign primer) |
|---|---|---|---|---|---|---|---|
| 1204 | cm | — | — | — | — | — | 38,932 / 75,940 |
| 1204 | A | 9.93 % | 36.04 | — | 16.27 % | 3.10 % | 37,924 / 67,481 |
| 1204 | B | 12.59 % | 11.08 | 3.43 | 19.88 % | 4.65 % | 37,514 / 65,435 |
| 1205 | cm | — | — | — | — | — | 38,952 / 75,897 |
| 1205 | A | 10.20 % | 37.04 | — | 17.22 % | 2.59 % | 37,840 / 67,320 |
| 1205 | B | 13.05 % | 11.48 | 3.68 | 20.96 % | 4.35 % | 37,389 / 65,156 |

On `L` the trunk saves 0.02-0.86 % over `cm` (e 0.07-1.70 % per 100K MAC/byte): no tier qualifies on long streams.
Seed spread: 0.27-0.46 points.

**Decision (by the registered rule):**

- **G0c:** tier A (width 88) is kept on `S` on both seeds (saving 9.9 / 10.2 %, e 36-37 % per 100K MAC/byte). Tier B
  is not kept: its saving over `cm` qualifies (12.6 / 13.1 %, e 11.1 / 11.5), but its marginal efficiency over tier A
  is 3.4 / 3.7 % per 100K MAC/byte, below 5. No tier qualifies on `L`. Consequence fixed by the rule: **v1 = the
  width-0 tier `cm` for long inputs, and the width-88 trunk tier for messaging weight files (path 3).**
- **G0a** (widest kept tier = A, on `L`): seed 1204 0.6192 vs best tier-A control `lstm_a_bc` 0.6136 -> ratio
  1.0091, TIE; seed 1205 0.6164 -> 1.0044, TIE. Worse of the two: **TIE**. Controls trained with one seed only
  (their seed noise is unknown).
- **G0b:** PASS. **G0: PASS.**

**G0d (reported, decides nothing):**

- Long streams: with bit cells in the loop the `cm` tier alone (208 MAC/byte) reaches 0.6193, **5.5 % below the best
  classical codec per source** (wins on webster -6.3 %, Zookeeper -27 %, Proxifier -16 %; loses on xml +20 %). The
  bake-off's harness without bit cells gave 0.748-0.771 with any trunk.
- Short messages, cells (of 20) smaller than zstd-19 + dictionary, tier A, both seeds:

| Strata primer | dictionary | payload + 4 B (R-114 target) | real framing +24 B (compact) | +28 B (default header) |
|---|---|---|---|---|
| same source | same source | 17 | 8 | 5 / 6 |
| **foreign** (the unfavourable case) | **same source** | 3 | **0** | 0 |
| foreign | foreign (same source as the primer) | 20 | 12 | 9 / 10 |

  Totals over the 320 messages: zstd-19 + same-source dictionary 45,348 B; Strata tier A with a same-source primer
  37,924 B of payload (about 45,600 B with 24 B of framing per message); with a foreign primer 67,481 B. **On these
  selection sources, with real framing, the primed archives do not beat a same-source zstd dictionary**, and with
  a foreign primer they lose in every cell. The trunk's 10 % comes mostly from the logs (16-17 %).

**Commands (G0).**

```
python -I tools/strata/bakeoff/precompute.py --first 0 --last 31 --bc [--seed 1205]          # features + bit cells
python -I tools/strata/bakeoff/train.py --model strata_g_bc --steps 2048 --minutes 7.5 --threads 4   # repeat to DONE
python -I tools/strata/bakeoff/train.py --model strata_g_bc --tag _s1205 --seed 1205 --steps 2048 --minutes 7.5
python -I tools/strata/bakeoff/train.py --model gru_bc|lstm_bc|gru_a_bc|lstm_a_bc --steps 2048 --minutes 7.5
python -I tools/strata/bakeoff/evaluate.py --run <run> --set select --online on
python -I tools/strata/bakeoff/ref_int.py --run strata_g_bc[_s1205] --source sel-webster --bytes 12000 --roundtrip 2000
python -I tools/strata/bakeoff/shortmsg.py --run strata_g_bc[_s1205] --tier 0|1|2 --primer 65536 \
        --primer-src same|foreign --sources sel-webster|sel-xml|sel-zookeeper|sel-proxifier \
        --out shortmsg-sel-t<tier>-<case>-<source>.csv                # 24 jobs per seed (data/strata/runs/g0-shortmsg.sh)
python -I tools/strata/bakeoff/g0_decide.py --g0b pass
```

### 12.2 Report-only, after the decision (frozen test set; R-113 messages)

Run from 11:07 to 13:12 (`data/strata/runs/g0-test.sh`); nothing here changes the decision above.

**Frozen test set**, bpb (the macro is over the six held-out sources; `enwik8` is seen-source and shown apart):

| model | seed | tier | dickens | Apache | Linux | OpenSSH | facturae | UBL | held-out macro | seen `enwik8` |
|---|---|---|---|---|---|---|---|---|---|---|
| strata_g_bc | 1204 | cm | 2.1268 | 0.2055 | 0.3496 | 0.2507 | 0.4785 | **0.3150** | 0.6210 | 2.1049 |
| strata_g_bc | 1204 | A (88) | 2.0662 | 0.1964 | 0.3212 | 0.2394 | **0.4713** | 0.3212 | 0.6026 | 1.9422 |
| strata_g_bc | 1204 | B (208) | 2.0398 | 0.1882 | 0.3200 | 0.2396 | 0.4846 | 0.3313 | 0.6006 | 1.8846 |
| strata_g_bc | 1205 | cm | 2.1270 | 0.2052 | 0.3494 | 0.2504 | 0.4785 | 0.3149 | 0.6209 | 2.1052 |
| strata_g_bc | 1205 | A (88) | 2.0644 | 0.1932 | 0.3194 | 0.2422 | 0.4723 | 0.3156 | 0.6012 | 1.9452 |
| strata_g_bc | 1205 | B (208) | 2.0418 | **0.1839** | 0.3072 | 0.2366 | 0.4906 | 0.3243 | 0.5974 | 1.8860 |
| gru_bc | 1204 | B | **1.9872** | 0.1908 | 0.3000 | 0.2348 | 0.5183 | 0.3411 | 0.5954 | **1.7846** |
| lstm_bc | 1204 | B | 1.9998 | 0.2045 | **0.2997** | 0.2387 | 0.5214 | 0.3326 | 0.5995 | 1.7925 |
| gru_a_bc | 1204 | A | 2.0316 | 0.1888 | 0.3084 | **0.2275** | 0.4777 | 0.3225 | **0.5928** | 1.8869 |
| lstm_a_bc | 1204 | A | 2.0364 | 0.1950 | 0.3143 | 0.2356 | 0.4946 | 0.3279 | 0.6006 | 1.8992 |
| best classical | | | 2.1960 | 0.3132 | 0.3697 | 0.3460 | 0.3953 | 0.2470 | 0.6445 | 2.2532 |
| Mamba-130M (repo, real `.azth`, other conditions) | | | 1.0620 | 0.1849 | 0.2449 | 0.2014 | 0.2408 | 0.2392 | | |

- Bit cells in the loop moved the candidate's held-out macro from 0.7016 (bake-off, tier B) to 0.5974-0.6026; the
  width-0 tier alone reaches 0.6210, 3.6 % below the best classical codec per file.
- On the four real held-out sources (`dickens` and the three logs) `cm` beats the best classical codec on all four
  and by 6.0 % in total bytes; tier A by 9.1 %. It still loses on the two synthetic invoice files.
- On this set the width-88 trunk is worth 3.1 % over `cm` (2.9 % on `dickens`, 4-8 % on the logs): more than on the
  selection set (0.0-0.5 %), still below the 5 % rule. The tier-A controls stay ahead: `gru_a_bc` 0.5928 against
  0.6012 / 0.6026 (+1.4 to +1.7 %).
- Against Mamba-130M the gap narrowed on logs (Apache +4.5 % / +6.2 %) and stays large on text (`dickens` +94 %) and
  XML (+32 % to +96 %).

**R-113 short messages** (the same 480 messages as R-113; width-88 tier = `t1`, seed 1204; real framing = 20 B
without primer, 24 B with a primer, compact header):

| configuration | vs zstd-19 + same-source dict, payload + 4 B | same, real framing | same, held-out 25 cells | vs a dictionary from the primer's source, real framing |
|---|---|---|---|---|
| `t1`, no primer | 7 / 30 | 4 / 30 | 0 / 25 | |
| `t1`, same-source primer | 30 / 30 | **24 / 30** | **19 / 25** | |
| `t1`, foreign primer (the unfavourable case) | 13 / 30 | **2 / 30** | **0 / 25** | 25 / 30 |
| `cm`, same-source primer | 30 / 30 | 22 / 30 | 18 / 25 | |
| `cm`, foreign primer | 7 / 30 | 0 / 30 | 0 / 25 | 8 / 30 |
| Mamba-130M archive (no primer, 20 B framing) | | 16 / 30 | 11 / 25 | |

Foreign primers: Wikipedia and prose <- `webster`; the three logs <- Zookeeper; JSON <- another hour of the same
event archive (same source under the by-source rule, labelled so). Mean bytes per message with a same-source primer
(`t1`): Apache 37.1 against 72.0 for zstd + dictionary, Linux 55.1 / 95.1, OpenSSH 47.1 / 92.2, prose 235.6 / 313.6,
JSON 131.6 / 153.2 (Mamba-130M: 91.2, 87.6, 105.0, 167.3, 133.3). With real framing and a primer from the message's
own source the width-88 file wins 19 of the 25 held-out cells (gate V3 asks 20); with a primer from another source
it wins none against a same-source dictionary.

## 13. Addendum of 2026-11-14 — PPMd was missing from the classical set (decides nothing above)

The classical codecs of this file (`baselines.py`) are xz, zstd, brotli, bz2 and gzip: the LZ and BWT families. The
width-0 tier is a context mixer, and no context-model codec was among its competitors. Measured on 2026-11-14 with
`python -I tools/strata/viability/ppmd_baseline.py` (7-Zip PPMd var.H, orders 4-32, packed stream without the
container header), on exactly the same bytes:

| Source | Best classical above | Best PPMd | Strata `cm` / width 88 / width 208 (mean of two seeds) |
|---|---|---|---|
| sel-webster | 1.8547 | **1.6681** | 1.7388 / 1.7293 / 1.7213 |
| sel-xml | 0.3219 | **0.2778** | 0.3872 / 0.4047 / 0.4057 |
| sel-zookeeper | 0.2043 | **0.1856** | 0.1490 / 0.1382 / 0.1356 |
| sel-proxifier | 0.2416 | **0.2201** | 0.2023 / 0.1990 / 0.1971 |
| test-dickens | 2.1960 | **1.9681** | 2.1269 / 2.0653 / 2.0408 |
| test-apache | 0.3132 | **0.2576** | 0.2054 / 0.1948 / 0.1861 |
| test-linux | **0.3697** | 0.4039 | 0.3495 / 0.3203 / 0.3136 |
| test-openssh | 0.3460 | **0.3222** | 0.2506 / 0.2408 / 0.2381 |
| test-facturae | 0.3953 | **0.3845** | 0.4785 / 0.4718 / 0.4876 |
| test-ubl | **0.2470** | 0.2490 | 0.3150 / 0.3184 / 0.3278 |

Statements of this file that hold only against the LZ and BWT families, and their value with PPMd included:
- Section 12, G0d: "`cm` reaches 0.6193, 5.5 % below the best classical codec per source". With PPMd the classical
  macro is 0.5879 and `cm` is **5.3 % above** it; it wins on the two log sources (-20 %, -8 %) and loses on
  `webster` (+4.2 %) and `xml` (+39 %).
- Section 12.2: "`cm` beats the best classical codec on all four [real held-out sources] and by 6.0 % in total
  bytes; tier A by 9.1 %". With PPMd: `cm` **+4.2 %**, tier A **+0.75 %**, tier B -0.6 %; all tiers lose on
  `dickens` (+3.7 % to +8.1 %) and win on the three logs (-15 % to -22 % in total).
- Section 0, weak point 1 becomes stronger, not weaker: after G0 no tier beats the best classical codec on text or
  markup.

The G0 decision is unaffected (its rule never used a classical codec). The classical set of the specification's
gates now includes PPMd (`05-gate-amendments.md`, amendment 4), and the consequences for storage viability are in
`08-capacity-ladder.md`.
