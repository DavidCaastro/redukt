# Strata backbone — 09c. Arm P and arm P-s on the decision files (gate V7)

> 2026-11-14. Run after the registration of amendment 5 (commit e340d21), exactly as registered in
> `08-capacity-ladder.md` 6.3: no other variant, composition or table size was run on a decision file.
> G0 weights (`strata_g_bc`, seeds 1204 and 1205), rungs R0 (`cm`, 208 MAC/byte) and R1 (width 88, 27,752 MAC/byte).
> Strata bytes are ideal code lengths of the integer simulation / 8 (no range coder, no header, no `PRIMER_ID`).
> Labels: MEASURED (command given), DERIVED, INFERENCE. Parity against Mamba-130M is left blank: Mamba is run
> afterwards by the orchestrator, so no outcome O1 / O2 can be closed here.

## 0. Result first — weak points before anything else

1. **Text: nothing passes, on either set (MEASURED).** At the primary points both primed arms still lose to the
   best classical codec on the extended text class (arm P -4.2 % R0 / -2.9 % R1; arm P-s -2.6 % / -1.2 %) and on
   `dickens` of the frozen test set (arm P -3.6 % / -1.7 %; arm P-s -1.7 % R0). The single positive cell is arm P-s,
   R1, frozen `dickens`: +0.3 % (122,603 against 123,006 bytes), far from the 10 % materiality and not
   reproduced on the extended set. Text is O3 for both arms without needing Mamba. The expectation registered
   in amendment 5 holds.
2. **Logs: the specified primer (arm P) makes the result worse than no primer on both sets.** Frozen test, R0:
   saving 15.0 % unprimed -> 12.9 % at the primary point; R1: 20.2 % -> 15.2 %, and **R1 then fails the cost
   inequality** (delta 0.0486 bpb against 0.0608 needed at 10 G MAC/s; unprimed R1 passes with 0.0649). Extended:
   23.4 % -> 22.5 % (R0), 24.8 % -> 23.0 % (R1).
3. **A primed R0 passing cost and materiality on logs is not evidence for the primer.** Arm P R0 passes both on
   logs on both sets, but so does the unprimed R0 ("CM only"), with a larger saving. Read literally, the O1 / O2
   wording of amendment 4 ("any rung of arm P with a primer") would credit the primer with a result the primer
   lowers. This has to be said wherever the verdict is written.
4. **Arm P-s (not specification 3.4) is the only primed configuration that improves on the unprimed rung on
   logs**, by a small margin: frozen R0 15.0 % -> 18.2 %, R1 20.2 % -> 22.1 %; extended R0 23.4 % -> 24.6 %, R1
   24.8 % -> 25.7 %. Part of that is the larger tables alone (unprimed `L`: 15.2 % / 20.4 % and 23.9 % / 25.4 %).
   Cost and materiality pass for both rungs on both sets; the verdict depends on parity. It costs 235.4 MB of
   state per stream (38.8 MB by default), whose speed is unmeasured: the 1 us per byte of table work in the cost
   model is a hypothesis made for the default tables.
5. **Markup and structured data: every configuration loses** (synthetic XML -24 % to -38 %; JSON events + `nci`
   -12 % to -15 %). Arm P makes the synthetic XML worse (-26 % -> -33 % / -36 %); arm P-s moves it by about 2 points.
6. The same-prior classical side never mattered: zstd -19 / -22 with the primer as dictionary is larger than the
   best classical codec without prior on all 13 files at all three sizes, so B_cl is the no-prior figure
   everywhere. The conditional PPMd reference gets worse as the primer grows on every class (e.g. extended logs
   132,525 -> 140,179 -> 156,305 bytes against 126,891 without prior): a general prior does not help PPMd either.
7. Limits: ideal code lengths, not archives; two seeds that share every online table (seed spread of a class
   total: at most 0.11 % on R0, 1.04 % on R1); the exact order-2 bit-cell index of tables `L` (24 bits) still has **no
   independent reference** (`ref_int.py` only has the hashed form; the reference check covers 23 bits);
   the extended log slices come from the same three systems as the frozen samples (disjoint lines, same sources),
   so the two sets are not independent evidence for logs.
8. Correction to `09a-arm-p-selection.md` section 1: it reported 1 of 2,000 shared lines between the head of the
   full Linux log and the frozen sample. That comparison did not ignore the CR line endings of the frozen file;
   ignoring them, 2,000 of 2,000 are shared, as amendment 5 states. The registered tail slices share none
   (MEASURED again, CR ignored: 0 of 2,000 for Linux, OpenSSH and Apache).
9. Process: two evaluation rounds were moved to the background by the harness at its 180 s limit (large-table
   checkpoints of seven streams were slow to write); both finished normally. The work was then split into 112
   units of one input group each, all under the limit. One pair of rounds may have overlapped; the per-unit
   files were checked afterwards: 364 rows, 364 unique keys, no duplicate, no conflict.

## 1. What was run (MEASURED)

Configurations, each on the six frozen-test sources and the seven extended slices, R0 and R1, both seeds:
size 0 with tables `D` and `L`; arm P (specification 3.4, primer `mixI`, tables `L`) at 65,536 / 1,048,576 (primary) /
16,777,216; arm P-s (successor tables only, tables `L`) at 1,048,576 / 16,777,216 (primary). Snapshots are the ones
built for `09a` (no re-priming).

```
python -I tools/strata/viability/baselines_ext.py              # registered extended slices -> baselines-ext.csv
python -I tools/strata/viability/baselines_ext.py --frozen     # PPMd stdin 1 GB on the frozen set
# for run in strata_g_bc strata_g_bc_s1205; for each configuration; for each input group G (the six test-* sources
# together, each ext-* slice alone); repeated until it prints DONE:
python -I tools/strata/viability/primer_long.py code --run $run --comp mixI --tables D|L --size S [--keep succ] \
       --inputs G --allow-decision --minutes 1.8 --out arm-p-decision-<id>-<G>.part.csv
python -I tools/strata/viability/primer_baselines.py classical --inputs <the 13 names>   # zstd dict, conditional
python -I tools/strata/viability/arm_p_decision.py              # arm-p-decision.csv and the tables below
```

Checks: the unprimed `D` rows of the frozen test set equal the bake-off's `eval-test-on.csv` bit totals to the digit
(24 of 24); `primer_long.py verify --run strata_g_bc --tables D` passes after the harness changes made for this run
(inputs grouped by length, pause and resume inside an input, coding chunks of 16,384 bytes): 0 mismatches against
`ref_int.py`; a paused and resumed run reproduces the selection bit totals exactly (8 of 8).

Classical side (registered set): xz -9e, zstd -19 / -22, brotli -11, bz2 -9, PPMd file mode and stdin 1 GB, smallest
per file; for primed rungs the smaller of that and zstd + dictionary. Extended slices after amendment 5 (bpb):
`dickens` 1.9981, `bible` 1.3393, Apache 0.3318, Linux tail 0.3922 (xz), OpenSSH tail 0.2441, GitHub events 0.6935,
`nci` 0.4551. Frozen set: unchanged except `facturae` 0.3794 (PPMd through stdin).

Output: `data/strata/runs/arm-p-decision.csv`, 364 rows = 13 sources x 2 rungs x 2 seeds x 7 configurations, with
bytes, ideal bits, bpb, Strata bytes, best classical bytes and codec, zstd + dictionary bytes, classical side, and
the conditional xz / PPMd bytes.

## 2. Primary points, per class (DERIVED from section 3; saving against B_cl, mean of the seeds)

| Set | Class | Rung | Unprimed `D` | Unprimed `L` | Arm P, 1 MiB | Arm P-s, 16 MiB | Cost <= value at 10 G (unprimed / P / P-s) | Material (unprimed / P / P-s) |
|---|---|---|---|---|---|---|---|---|
| frozen test | text | R0 | -8.1 % | -6.7 % | -3.6 % | -1.7 % | no / no / no | no / no / no |
| frozen test | text | R1 | -4.9 % | -3.7 % | -1.7 % | +0.3 % | no / no / no | no / no / no |
| extended | text | R0 | -7.3 % | -5.3 % | -4.2 % | -2.6 % | no / no / no | no / no / no |
| extended | text | R1 | -5.0 % | -3.3 % | -2.9 % | -1.2 % | no / no / no | no / no / no |
| frozen test | logs | R0 | +15.0 % | +15.2 % | +12.9 % | +18.2 % | yes / yes / yes | yes / yes / yes |
| frozen test | logs | R1 | +20.2 % | +20.4 % | +15.2 % | +22.1 % | yes / **no** / yes | yes / yes / yes |
| extended | logs | R0 | +23.4 % | +23.9 % | +22.5 % | +24.6 % | yes / yes / yes | yes / yes / yes |
| extended | logs | R1 | +24.8 % | +25.4 % | +23.0 % | +25.7 % | yes / yes / yes | yes / yes / yes |
| frozen test | xml-synthetic (no vote) | R0 | -26.9 % | -26.2 % | -33.0 % | -23.9 % | no / no / no | no / no / no |
| frozen test | xml-synthetic (no vote) | R1 | -27.0 % | -26.5 % | -35.7 % | -26.5 % | no / no / no | no / no / no |
| extended | structured | R0 | -14.0 % | -12.8 % | -13.0 % | -11.9 % | no / no / no | no / no / no |
| extended | structured | R1 | -14.4 % | -13.1 % | -13.8 % | -12.7 % | no / no / no | no / no / no |

Positive = fewer bytes than the best classical codec. V7 takes the worse of the two sets per class:
- `text`: O3 for arm P and arm P-s (loses or is not material on both sets).
- `logs`, worse set = frozen test: arm P R0 passes cost and materiality with 12.9 % (below the unprimed 15.0 %);
  arm P R1 fails cost; arm P-s passes both on R0 (18.2 %) and R1 (22.1 %). Whether any of these is O1 or O2
  depends on parity with Mamba-130M on the identical bytes (blank). At registration the unprimed parity on the
  frozen logs was 1.29 (R0) and 1.21 (R1); the P-s gain is 3.8 % (R0) and 2.3 % (R1) of the coded bytes, so reaching
  1.10 would need Mamba to be much worse on these bytes than its repo figures (INFERENCE).
- `structured` (extended only): loses everywhere.

## 3. Per class, every configuration (MEASURED; `arm_p_decision.py`)

Columns: coded bytes per seed; B_r their mean; B_cl the classical side; saving = 1 - B_r / B_cl; delta = 8 (B_cl -
B_r) / N; value and cost in $ per raw GB from `viability.py` (10 G MAC/s, $0.014 per vCPU-hour; needed delta
0.0164 bpb for R0 and 0.0608 for R1); conditional PPMd / xz = C(primer + input) - C(primer) summed over the class
(stricter, non-deployable reference). The cost column assumes the 1 us per byte of table work of specification 7.2
for both table sizes (HYPOTHESIS; not measured for the 235 MB state).

### frozen test, class `text`: test-dickens; 500,000 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 132,924 / 132,936 | 132,930 | 123,006 | -8.1% | -0.1588 | -0.0383 | 0.0040 | no | no | - | |
| none | D | 0 | R1 | 129,136 / 129,027 | 129,081 | 123,006 | -4.9% | -0.0972 | -0.0235 | 0.0147 | no | no | - | |
| none | L | 0 | R0 | 131,215 / 131,226 | 131,221 | 123,006 | -6.7% | -0.1314 | -0.0317 | 0.0040 | no | no | - | |
| none | L | 0 | R1 | 127,644 / 127,556 | 127,600 | 123,006 | -3.7% | -0.0735 | -0.0178 | 0.0147 | no | no | - | |
| P | L | 65,536 | R0 | 130,120 / 130,129 | 130,125 | 123,006 | -5.8% | -0.1139 | -0.0275 | 0.0040 | no | no | 122,995 / 152,160 | |
| P | L | 65,536 | R1 | 126,990 / 126,949 | 126,970 | 123,006 | -3.2% | -0.0634 | -0.0153 | 0.0147 | no | no | 122,995 / 152,160 | |
| P **(primary)** | L | 1,048,576 | R0 | 127,380 / 127,389 | 127,385 | 123,006 | -3.6% | -0.0701 | -0.0169 | 0.0040 | no | no | 123,636 / 148,396 | |
| P **(primary)** | L | 1,048,576 | R1 | 125,132 / 125,092 | 125,112 | 123,006 | -1.7% | -0.0337 | -0.0081 | 0.0147 | no | no | 123,636 / 148,396 | |
| P | L | 16,777,216 | R0 | 125,312 / 125,321 | 125,317 | 123,006 | -1.9% | -0.0370 | -0.0089 | 0.0040 | no | no | 124,913 / 144,752 | |
| P | L | 16,777,216 | R1 | 123,862 / 123,884 | 123,873 | 123,006 | -0.7% | -0.0139 | -0.0034 | 0.0147 | no | no | 124,913 / 144,752 | |
| P-s | L | 1,048,576 | R0 | 126,779 / 126,791 | 126,785 | 123,006 | -3.1% | -0.0605 | -0.0146 | 0.0040 | no | no | 123,636 / 148,396 | |
| P-s | L | 1,048,576 | R1 | 123,869 / 123,813 | 123,841 | 123,006 | -0.7% | -0.0134 | -0.0032 | 0.0147 | no | no | 123,636 / 148,396 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 125,113 / 125,124 | 125,119 | 123,006 | -1.7% | -0.0338 | -0.0082 | 0.0040 | no | no | 124,913 / 144,752 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 122,599 / 122,607 | 122,603 | 123,006 | +0.3% | +0.0064 | +0.0016 | 0.0147 | no | no | 124,913 / 144,752 | |

### frozen test, class `logs`: test-apache, test-linux, test-openssh; 612,940 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 20,920 / 20,897 | 20,908 | 24,587 | +15.0% | +0.0480 | +0.0116 | 0.0040 | yes | yes | - | |
| none | D | 0 | R1 | 19,633 / 19,596 | 19,614 | 24,587 | +20.2% | +0.0649 | +0.0157 | 0.0147 | yes | yes | - | |
| none | L | 0 | R0 | 20,873 / 20,850 | 20,861 | 24,587 | +15.2% | +0.0486 | +0.0117 | 0.0040 | yes | yes | - | |
| none | L | 0 | R1 | 19,596 / 19,547 | 19,572 | 24,587 | +20.4% | +0.0655 | +0.0158 | 0.0147 | yes | yes | - | |
| P | L | 65,536 | R0 | 21,070 / 21,052 | 21,061 | 24,587 | +14.3% | +0.0460 | +0.0111 | 0.0040 | yes | yes | 26,126 / 25,700 | |
| P | L | 65,536 | R1 | 20,154 / 20,112 | 20,133 | 24,587 | +18.1% | +0.0581 | +0.0140 | 0.0147 | no | yes | 26,126 / 25,700 | |
| P **(primary)** | L | 1,048,576 | R0 | 21,414 / 21,401 | 21,408 | 24,587 | +12.9% | +0.0415 | +0.0100 | 0.0040 | yes | yes | 28,098 / 25,348 | |
| P **(primary)** | L | 1,048,576 | R1 | 20,905 / 20,816 | 20,861 | 24,587 | +15.2% | +0.0486 | +0.0117 | 0.0147 | no | yes | 28,098 / 25,348 | |
| P | L | 16,777,216 | R0 | 21,084 / 21,077 | 21,081 | 24,587 | +14.3% | +0.0458 | +0.0111 | 0.0040 | yes | yes | 30,638 / 24,816 | |
| P | L | 16,777,216 | R1 | 20,821 / 20,647 | 20,734 | 24,587 | +15.7% | +0.0503 | +0.0121 | 0.0147 | no | yes | 30,638 / 24,816 | |
| P-s | L | 1,048,576 | R0 | 20,573 / 20,549 | 20,561 | 24,587 | +16.4% | +0.0526 | +0.0127 | 0.0040 | yes | yes | 28,098 / 25,348 | |
| P-s | L | 1,048,576 | R1 | 19,517 / 19,481 | 19,499 | 24,587 | +20.7% | +0.0664 | +0.0160 | 0.0147 | yes | yes | 28,098 / 25,348 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 20,124 / 20,102 | 20,113 | 24,587 | +18.2% | +0.0584 | +0.0141 | 0.0040 | yes | yes | 30,638 / 24,816 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 19,187 / 19,121 | 19,154 | 24,587 | +22.1% | +0.0709 | +0.0171 | 0.0147 | yes | yes | 30,638 / 24,816 | |

### frozen test, class `xml-synthetic`: test-facturae, test-ubl; 122,644 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 5,634 / 5,633 | 5,633 | 4,439 | -26.9% | -0.0779 | -0.0188 | 0.0040 | no | no | - | |
| none | D | 0 | R1 | 5,663 / 5,610 | 5,637 | 4,439 | -27.0% | -0.0781 | -0.0189 | 0.0147 | no | no | - | |
| none | L | 0 | R0 | 5,603 / 5,602 | 5,602 | 4,439 | -26.2% | -0.0759 | -0.0183 | 0.0040 | no | no | - | |
| none | L | 0 | R1 | 5,645 / 5,589 | 5,617 | 4,439 | -26.5% | -0.0768 | -0.0186 | 0.0147 | no | no | - | |
| P | L | 65,536 | R0 | 5,690 / 5,689 | 5,689 | 4,439 | -28.2% | -0.0816 | -0.0197 | 0.0040 | no | no | 4,425 / 4,584 | |
| P | L | 65,536 | R1 | 5,852 / 5,802 | 5,827 | 4,439 | -31.3% | -0.0905 | -0.0219 | 0.0147 | no | no | 4,425 / 4,584 | |
| P **(primary)** | L | 1,048,576 | R0 | 5,904 / 5,904 | 5,904 | 4,439 | -33.0% | -0.0955 | -0.0231 | 0.0040 | no | no | 4,648 / 4,444 | |
| P **(primary)** | L | 1,048,576 | R1 | 6,055 / 5,992 | 6,023 | 4,439 | -35.7% | -0.1033 | -0.0250 | 0.0147 | no | no | 4,648 / 4,444 | |
| P | L | 16,777,216 | R0 | 6,034 / 6,033 | 6,034 | 4,439 | -35.9% | -0.1040 | -0.0251 | 0.0040 | no | no | 5,122 / 4,520 | |
| P | L | 16,777,216 | R1 | 6,167 / 6,117 | 6,142 | 4,439 | -38.4% | -0.1111 | -0.0268 | 0.0147 | no | no | 5,122 / 4,520 | |
| P-s | L | 1,048,576 | R0 | 5,518 / 5,515 | 5,517 | 4,439 | -24.3% | -0.0703 | -0.0170 | 0.0040 | no | no | 4,648 / 4,444 | |
| P-s | L | 1,048,576 | R1 | 5,632 / 5,595 | 5,614 | 4,439 | -26.5% | -0.0766 | -0.0185 | 0.0147 | no | no | 4,648 / 4,444 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 5,502 / 5,500 | 5,501 | 4,439 | -23.9% | -0.0693 | -0.0167 | 0.0040 | no | no | 5,122 / 4,520 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 5,628 / 5,600 | 5,614 | 4,439 | -26.5% | -0.0766 | -0.0185 | 0.0147 | no | no | 5,122 / 4,520 | |

### extended, class `text`: ext-dickens, ext-bible; 2,097,152 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 469,145 / 469,178 | 469,161 | 437,436 | -7.3% | -0.1210 | -0.0292 | 0.0040 | no | no | - | |
| none | D | 0 | R1 | 459,423 / 459,465 | 459,444 | 437,436 | -5.0% | -0.0840 | -0.0203 | 0.0147 | no | no | - | |
| none | L | 0 | R0 | 460,670 / 460,703 | 460,687 | 437,436 | -5.3% | -0.0887 | -0.0214 | 0.0040 | no | no | - | |
| none | L | 0 | R1 | 451,968 / 452,082 | 452,025 | 437,436 | -3.3% | -0.0557 | -0.0134 | 0.0147 | no | no | - | |
| P | L | 65,536 | R0 | 459,625 / 459,656 | 459,640 | 437,436 | -5.1% | -0.0847 | -0.0205 | 0.0040 | no | no | 439,583 / 541,676 | |
| P | L | 65,536 | R1 | 451,897 / 452,058 | 451,978 | 437,436 | -3.3% | -0.0555 | -0.0134 | 0.0147 | no | no | 439,583 / 541,676 | |
| P **(primary)** | L | 1,048,576 | R0 | 455,755 / 455,782 | 455,769 | 437,436 | -4.2% | -0.0699 | -0.0169 | 0.0040 | no | no | 447,106 / 534,312 | |
| P **(primary)** | L | 1,048,576 | R1 | 450,051 / 450,185 | 450,118 | 437,436 | -2.9% | -0.0484 | -0.0117 | 0.0147 | no | no | 447,106 / 534,312 | |
| P | L | 16,777,216 | R0 | 450,785 / 450,806 | 450,796 | 437,436 | -3.1% | -0.0510 | -0.0123 | 0.0040 | no | no | 463,400 / 526,796 | |
| P | L | 16,777,216 | R1 | 447,310 / 447,550 | 447,430 | 437,436 | -2.3% | -0.0381 | -0.0092 | 0.0147 | no | no | 463,400 / 526,796 | |
| P-s | L | 1,048,576 | R0 | 453,659 / 453,693 | 453,676 | 437,436 | -3.7% | -0.0620 | -0.0150 | 0.0040 | no | no | 447,106 / 534,312 | |
| P-s | L | 1,048,576 | R1 | 446,260 / 446,424 | 446,342 | 437,436 | -2.0% | -0.0340 | -0.0082 | 0.0147 | no | no | 447,106 / 534,312 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 448,737 / 448,770 | 448,753 | 437,436 | -2.6% | -0.0432 | -0.0104 | 0.0040 | no | no | 463,400 / 526,796 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 442,545 / 442,808 | 442,677 | 437,436 | -1.2% | -0.0200 | -0.0048 | 0.0147 | no | no | 463,400 / 526,796 | |

### extended, class `logs`: ext-apache, ext-linux, ext-openssh; 3,145,728 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 97,215 / 97,144 | 97,180 | 126,891 | +23.4% | +0.0756 | +0.0182 | 0.0040 | yes | yes | - | |
| none | D | 0 | R1 | 95,288 / 95,522 | 95,405 | 126,891 | +24.8% | +0.0801 | +0.0193 | 0.0147 | yes | yes | - | |
| none | L | 0 | R0 | 96,555 / 96,482 | 96,518 | 126,891 | +23.9% | +0.0772 | +0.0187 | 0.0040 | yes | yes | - | |
| none | L | 0 | R1 | 94,461 / 94,759 | 94,610 | 126,891 | +25.4% | +0.0821 | +0.0198 | 0.0147 | yes | yes | - | |
| P | L | 65,536 | R0 | 97,102 / 97,038 | 97,070 | 126,891 | +23.5% | +0.0758 | +0.0183 | 0.0040 | yes | yes | 132,525 / 133,672 | |
| P | L | 65,536 | R1 | 95,586 / 95,828 | 95,707 | 126,891 | +24.6% | +0.0793 | +0.0192 | 0.0147 | yes | yes | 132,525 / 133,672 | |
| P **(primary)** | L | 1,048,576 | R0 | 98,365 / 98,311 | 98,338 | 126,891 | +22.5% | +0.0726 | +0.0175 | 0.0040 | yes | yes | 140,179 / 132,608 | |
| P **(primary)** | L | 1,048,576 | R1 | 97,612 / 97,707 | 97,660 | 126,891 | +23.0% | +0.0743 | +0.0180 | 0.0147 | yes | yes | 140,179 / 132,608 | |
| P | L | 16,777,216 | R0 | 98,276 / 98,230 | 98,253 | 126,891 | +22.6% | +0.0728 | +0.0176 | 0.0040 | yes | yes | 156,305 / 131,856 | |
| P | L | 16,777,216 | R1 | 98,150 / 98,072 | 98,111 | 126,891 | +22.7% | +0.0732 | +0.0177 | 0.0147 | yes | yes | 156,305 / 131,856 | |
| P-s | L | 1,048,576 | R0 | 96,277 / 96,205 | 96,241 | 126,891 | +24.2% | +0.0779 | +0.0188 | 0.0040 | yes | yes | 140,179 / 132,608 | |
| P-s | L | 1,048,576 | R1 | 94,541 / 94,742 | 94,642 | 126,891 | +25.4% | +0.0820 | +0.0198 | 0.0147 | yes | yes | 140,179 / 132,608 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 95,663 / 95,591 | 95,627 | 126,891 | +24.6% | +0.0795 | +0.0192 | 0.0040 | yes | yes | 156,305 / 131,856 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 94,213 / 94,318 | 94,266 | 126,891 | +25.7% | +0.0830 | +0.0200 | 0.0147 | yes | yes | 156,305 / 131,856 | |

### extended, class `structured`: ext-gharchive, ext-nci; 2,097,152 raw bytes
| Arm | Tables | Primer | Rung | Coded bytes (seed 1204 / 1205) | B_r mean | B_cl | Saving | delta bpb | Value $/GB | Cost $/GB | Cost <= value | Material | Conditional PPMd / xz | Parity vs Mamba |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | D | 0 | R0 | 171,694 / 171,663 | 171,678 | 150,551 | -14.0% | -0.0806 | -0.0195 | 0.0040 | no | no | - | |
| none | D | 0 | R1 | 172,956 / 171,385 | 172,170 | 150,551 | -14.4% | -0.0825 | -0.0199 | 0.0147 | no | no | - | |
| none | L | 0 | R0 | 169,880 / 169,848 | 169,864 | 150,551 | -12.8% | -0.0737 | -0.0178 | 0.0040 | no | no | - | |
| none | L | 0 | R1 | 171,128 / 169,521 | 170,324 | 150,551 | -13.1% | -0.0754 | -0.0182 | 0.0147 | no | no | - | |
| P | L | 65,536 | R0 | 169,586 / 169,558 | 169,572 | 150,551 | -12.6% | -0.0726 | -0.0175 | 0.0040 | no | no | 151,619 / 170,364 | |
| P | L | 65,536 | R1 | 171,134 / 169,683 | 170,408 | 150,551 | -13.2% | -0.0757 | -0.0183 | 0.0147 | no | no | 151,619 / 170,364 | |
| P **(primary)** | L | 1,048,576 | R0 | 170,166 / 170,137 | 170,151 | 150,551 | -13.0% | -0.0748 | -0.0181 | 0.0040 | no | no | 156,041 / 168,428 | |
| P **(primary)** | L | 1,048,576 | R1 | 172,112 / 170,672 | 171,392 | 150,551 | -13.8% | -0.0795 | -0.0192 | 0.0147 | no | no | 156,041 / 168,428 | |
| P | L | 16,777,216 | R0 | 170,992 / 170,961 | 170,977 | 150,551 | -13.6% | -0.0779 | -0.0188 | 0.0040 | no | no | 163,792 / 167,360 | |
| P | L | 16,777,216 | R1 | 173,383 / 171,981 | 172,682 | 150,551 | -14.7% | -0.0844 | -0.0204 | 0.0147 | no | no | 163,792 / 167,360 | |
| P-s | L | 1,048,576 | R0 | 168,725 / 168,693 | 168,709 | 150,551 | -12.1% | -0.0693 | -0.0167 | 0.0040 | no | no | 156,041 / 168,428 | |
| P-s | L | 1,048,576 | R1 | 170,418 / 168,945 | 169,682 | 150,551 | -12.7% | -0.0730 | -0.0176 | 0.0147 | no | no | 156,041 / 168,428 | |
| P-s **(primary)** | L | 16,777,216 | R0 | 168,447 / 168,416 | 168,431 | 150,551 | -11.9% | -0.0682 | -0.0165 | 0.0040 | no | no | 163,792 / 167,360 | |
| P-s **(primary)** | L | 16,777,216 | R1 | 170,365 / 168,942 | 169,654 | 150,551 | -12.7% | -0.0729 | -0.0176 | 0.0147 | no | no | 163,792 / 167,360 | |

## 4. Per source (MEASURED)

per source, mean of the seeds, bpb (classical = registered set without prior)
| Rung | Arm | Tables | Primer | test-dickens | test-apache | test-linux | test-openssh | test-facturae | test-ubl | ext-dickens | ext-bible | ext-apache | ext-linux | ext-openssh | ext-gharchive | ext-nci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | best classical | - | - | 1.9681 | 0.2576 | 0.3697 | 0.3222 | 0.3794 | 0.2470 | 1.9981 | 1.3393 | 0.3318 | 0.3922 | 0.2441 | 0.6935 | 0.4551 |
| R0 | none | D | 0 | 2.1269 | 0.2054 | 0.3495 | 0.2506 | 0.4785 | 0.3150 | 2.1514 | 1.4280 | 0.2767 | 0.3152 | 0.1495 | 0.7795 | 0.5303 |
| R0 | none | L | 0 | 2.0995 | 0.2052 | 0.3487 | 0.2499 | 0.4761 | 0.3131 | 2.1084 | 1.4063 | 0.2747 | 0.3131 | 0.1485 | 0.7693 | 0.5266 |
| R0 | P | L | 65,536 | 2.0820 | 0.2132 | 0.3440 | 0.2554 | 0.4803 | 0.3195 | 2.1003 | 1.4065 | 0.2768 | 0.3126 | 0.1512 | 0.7639 | 0.5298 |
| R0 | P | L | 1,048,576 | 2.0382 | 0.2183 | 0.3435 | 0.2643 | 0.4984 | 0.3315 | 2.0708 | 1.4065 | 0.2819 | 0.3138 | 0.1545 | 0.7627 | 0.5355 |
| R0 | P | L | 16,777,216 | 2.0051 | 0.2188 | 0.3279 | 0.2673 | 0.5101 | 0.3385 | 2.0391 | 1.4002 | 0.2854 | 0.3094 | 0.1548 | 0.7625 | 0.5420 |
| R0 | P-s | L | 1,048,576 | 2.0286 | 0.2051 | 0.3400 | 0.2476 | 0.4671 | 0.3092 | 2.0653 | 1.3959 | 0.2743 | 0.3111 | 0.1488 | 0.7584 | 0.5288 |
| R0 | P-s | L | 16,777,216 | 2.0019 | 0.2049 | 0.3244 | 0.2468 | 0.4658 | 0.3082 | 2.0367 | 1.3870 | 0.2747 | 0.3059 | 0.1490 | 0.7540 | 0.5311 |
| R1 | none | D | 0 | 2.0653 | 0.1948 | 0.3203 | 0.2408 | 0.4718 | 0.3184 | 2.1005 | 1.4048 | 0.2760 | 0.3054 | 0.1465 | 0.7711 | 0.5424 |
| R1 | none | L | 0 | 2.0416 | 0.1947 | 0.3195 | 0.2400 | 0.4696 | 0.3176 | 2.0627 | 1.3860 | 0.2738 | 0.3031 | 0.1449 | 0.7612 | 0.5383 |
| R1 | P | L | 65,536 | 2.0315 | 0.2057 | 0.3223 | 0.2489 | 0.4894 | 0.3284 | 2.0587 | 1.3896 | 0.2769 | 0.3054 | 0.1479 | 0.7585 | 0.5417 |
| R1 | P | L | 1,048,576 | 2.0018 | 0.2150 | 0.3274 | 0.2628 | 0.5105 | 0.3373 | 2.0388 | 1.3953 | 0.2830 | 0.3094 | 0.1526 | 0.7593 | 0.5483 |
| R1 | P | L | 16,777,216 | 1.9820 | 0.2177 | 0.3152 | 0.2680 | 0.5232 | 0.3427 | 2.0183 | 1.3954 | 0.2873 | 0.3071 | 0.1542 | 0.7626 | 0.5548 |
| R1 | P-s | L | 1,048,576 | 1.9815 | 0.1968 | 0.3148 | 0.2404 | 0.4696 | 0.3172 | 2.0253 | 1.3800 | 0.2741 | 0.3023 | 0.1456 | 0.7535 | 0.5411 |
| R1 | P-s | L | 16,777,216 | 1.9616 | 0.1970 | 0.3014 | 0.2409 | 0.4708 | 0.3167 | 2.0029 | 1.3744 | 0.2747 | 0.2981 | 0.1464 | 0.7507 | 0.5436 |
