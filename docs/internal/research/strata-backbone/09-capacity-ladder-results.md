# Strata backbone — 09. Capacity ladder results and verdict of gate V7

> 2026-11-15. Verdict of gate V7 by the rule registered in `05-gate-amendments.md` (amendments 4 and 5) and the
> configurations registered in `08-capacity-ladder.md` 6.2 and 6.3. Nothing was run on a decision file outside
> those registrations. Arm P detail: `09c-arm-p-decision.md`; trainer: `09b-trainer-port.md`.
> Strata bytes are ideal code lengths of the integer model / 8 (no range coder, no header); Mamba-130M bytes are
> real `.azth` files. Labels: MEASURED (command given), DERIVED, INFERENCE.

## 0. Result first — weak points before anything else

1. **Text: O3. Nothing passes on either set, with any arm (MEASURED).** Every retrained rung still loses to PPMd:
   frozen `dickens` -5.6 % / -4.5 % / -2.2 % (R1 / R2 / R3), extended text -5.6 % / -5.1 % / -3.8 %. Mamba-130M is
   1.76-1.95x smaller on the same bytes. The owner's objective is not met on text at any tested capacity.
2. **Structured and markup: O3.** JSON events + `nci`: -16.6 % to -17.3 %; synthetic XML (no vote): -42 % to -46 %.
3. **Logs: O2 by the letter of the rule, and the letter flatters the trunk.** The retrained R1 (width 88) passes cost
   and materiality on both sets (saving 22.4 % frozen, 25.6 % extended) and passes parity only on the extended set
   (1.05; frozen 1.17), so the class verdict is O2, "cheaper and weaker". Stop rule 1 is therefore **not**
   triggered. But (DERIVED, section 2):
   - the rung without a trunk (R0, 208 MAC/byte) already saves 15.0 % / 23.4 % and has the **larger net value**:
     $0.0076 / $0.0142 per raw GB over 7 years against $0.0027 / $0.0052 for R1. The trunk adds $0.002-0.006 of
     value for $0.011 of projected cost. On the margin the trunk loses money on logs too;
   - R1 passes cost only under the 10 G MAC/s hypothesis; at the effective 2.5 G of today it fails (needs
     0.195 bpb, has 0.072-0.083). R2 and R3 fail cost at 10 G on both sets;
   - the whole net value is about half a cent to a cent and a half per GB over seven years ($5-15 per TB). Whether
     that is a product is the owner's decision (amendment 4); the recommendation here is that it is not.
4. **43x more training bytes and 10.7x more updates did not move the ratio** (retrained R1 against the G0 R1, same
   width, MEASURED): logs -2.8 % (frozen) / -1.0 % (extended); text **+0.7 % / +0.6 % (worse)**; structured
   +2.6 %; synthetic XML +14.7 %. The G0 models were not under-trained in any way that matters for unseen data.
5. **18x more compute bought 1-3 %** (R1 -> R3, 27,752 -> 499,552 MAC/byte): text -3.2 % (frozen) / -1.7 %
   (extended); logs -2.0 % / -0.7 %; structured -0.6 %. Against a cost that rises 13.5x.
6. **Stop rule 2: the condition for R4 is met by one cell and R4 was not trained.** R2 -> R3 lowers the frozen
   `dickens` total by 2.17 % (both seeds above 2 %); the extended text class gives 1.20 % and no other class
   reaches 2 %. The rule permits R4, it does not require it, and R4 cannot change any verdict: at 1,961,768
   MAC/byte it needs a delta of 3.18 bpb at 10 G MAC/s (1.60 at 20 G) and the best classical codec on these text
   files is at 1.67-1.97 bpb, so cost fails even with an output of zero bytes at the decision point. It would cost
   about 4-5 of the 8.2 remaining budgeted GPU hours to read one more point of a slope that is already 1-3 % per 4x.
   Left to the owner; not recommended. **A statement made to the owner on 2026-11-15 before this computation ("the
   2 % is not reached, R4 is not run") was wrong for the frozen text class** and is corrected here.
7. **Position against Mamba-130M on identical bytes (MEASURED, new).** Logs: Strata is at 1.04-1.08x Mamba on the
   extended set (R0 alone 1.08; on `ext-openssh` every Strata rung is smaller than Mamba, 17.7-18.5 KB against
   21.1 KB) and 1.15-1.28x on the frozen set. Text: 1.72-2.00x. Mamba ran at 648 raw bytes per second on the
   12-thread laptop (8,575,616 bytes in 13,236 s); R0 is about 170,000x less arithmetic per byte.
8. **Arm P with parity filled in** (`09c` left it blank): arm P R0 is O2 on logs by the letter, with a saving
   *below* the unprimed R0 (12.9 % against 15.0 % frozen), so it is no evidence for the primer; arm P R1 is O3
   (fails cost on the frozen set); arm P-s (not specification 3.4) is O2 on both rungs. No primed rung reaches
   parity on the frozen set (1.18-1.32).
9. Limits. (a) Ideal code lengths against real archives: the comparison favours Strata by the coder and header
   overhead it does not pay yet. (b) Cost is the hypothesis `1 us + MAC / R_mac`; gate V4 has never been measured
   because nothing exists in Rust. (c) The 13 Mamba archives were not decoded back (about 3.7 h on this machine);
   the same binary passed its roundtrip gates on 2026-11-14 (`07-legacy-fixes.md` 11). (d) The two seeds share
   every online table; seed spread of a class total is up to 2.6 % on logs for the retrained rungs. (e) The
   extended log slices come from the same three systems as the frozen samples (disjoint lines). (f) The retrained
   ladder was evaluated with the default tables only; no primer was combined with it (not registered).

**What V7 answers.** Is there a Strata size whose compute cost per raw GB is below the storage value it saves, close
to the ratio of Mamba-130M? On text and structured data: no, at any tested size, with any amount of training tried,
with or without a primer. On logs: yes for the part that is not a neural network (the context mixer of tier `cm`),
by a margin worth about a cent per GB over seven years; the trunk does not pay for itself. Capacity in the trunk is
not the lever: neither more compute, nor more data, nor a general prior moved unseen sources by more than a few
percent. The Fase 20 storage verdict stands.

## 1. What was run (MEASURED)

| Item | Value |
|---|---|
| Training | Kaggle kernel `strata-ladder` version 3 (version 2 was the same script and never left the queue), T4 x2, one seed per GPU, 210.3 minutes = 3.5 GPU h; 22,016 updates per seed, 1.44 GB seen; final training cost of the widest tier 0.98-1.00 bits/byte; export check 0 mismatches against the int64 reference |
| GPU total | 0.30 (smoke) + 3.5 = 3.8 of the 12 budgeted hours |
| Arm W evaluation | `python -I tools/strata/train/run_eval.py --allow-decision` on CPU, integer model, one stream per file, no reset, default tables; 13 decision files x 3 widths x 2 seeds = 78 rows (`data/strata/runs-ladder/kaggle-long-v3/run-ladder3-s120{4,5}/eval-decision.csv`) |
| R0 and arm P / P-s | from `data/strata/runs/arm-p-decision.csv` (`09c`); R0 is the G0 `cm` tier, not retrained, as registered |
| Mamba-130M | `redukt compress --quant q8` (default reset 4096), batch of the 13 files, real `.azth` (`data/strata/runs/mamba-h2h/out`); 13/13 succeeded, 3,215,683 tokens in 13,236 s |
| Classical side | best of xz -9e, zstd -19 / -22, brotli -11, bz2 -9, PPMd (file mode and stdin 1 GB) per file, as in `09c` |
| Verdict tool | `python -I tools/strata/viability/ladder_verdict.py` (prints sections 3 to 5 verbatim) |

Selection set, retrained ladder (MEASURED before the decision files, mean of the seeds, bpb, R1 / R2 / R3):
`webster` 1.765 / 1.746 / 1.724 (G0 R1 / R2 1.729 / 1.721; PPMd 1.668); `xml` 0.426 / 0.422 / 0.422 (classical
0.278); Zookeeper 0.137 / 0.136 / 0.138; Proxifier 0.203 / 0.203 / 0.203. No gain over G0 on unseen sources there
either, so the decision files confirm the selection set rather than contradict it.

## 2. Logs, value against cost (DERIVED from section 3; $ per raw GB over 7 years, 10 G MAC/s, $0.014 per vCPU-hour)

| Set | Configuration | Value | Cost | Net | Parity |
|---|---|---|---|---|---|
| frozen test | R0 `cm`, no primer | 0.0116 | 0.0040 | +0.0076 | 1.28 |
| frozen test | W R1 retrained | 0.0174 | 0.0147 | +0.0027 | 1.17 |
| frozen test | W R2 retrained | 0.0179 | 0.0482 | -0.0303 | 1.16 |
| frozen test | W R3 retrained | 0.0185 | 0.1982 | -0.1797 | 1.15 |
| frozen test | P-s R0, 16 MiB (235 MB of state, speed unmeasured) | 0.0141 | 0.0040 | +0.0101 | 1.24 |
| extended | R0 `cm`, no primer | 0.0182 | 0.0040 | +0.0142 | 1.08 |
| extended | W R1 retrained | 0.0199 | 0.0147 | +0.0052 | 1.05 |
| extended | W R2 retrained | 0.0204 | 0.0482 | -0.0278 | 1.04 |
| extended | W R3 retrained | 0.0203 | 0.1982 | -0.1779 | 1.04 |
| extended | P-s R0, 16 MiB | 0.0192 | 0.0040 | +0.0152 | 1.06 |

Step R0 -> R1: +$0.0058 (frozen) / +$0.0017 (extended) of value for +$0.0107 of cost. For scale: the Fase 20
threshold for logs was a compute cost of at most $0.06 per GB, and the saving of Mamba-130M itself on these logs is
worth $0.023-0.026 per GB at a measured cost of $33-60.

## 3. Per class, every judged configuration (MEASURED; output of `ladder_verdict.py`)

Columns: coded bytes per seed; B_r their mean; saving = 1 - B_r / B_cl; delta = 8 (B_cl - B_r) / N; needed = the
delta at which cost equals value at 10 G MAC/s; parity = B_r / B_mamba. "G0 R1" is the width-88 tier trained on
33.55 MB in the bake-off: a reference row, not a rung of arm W.

### frozen test, class `text`: test-dickens; 500,000 raw bytes; B_cl 123,006; Mamba-130M 66,568 (-45.9% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 132,924 / 132,936 | 132,930 | -8.1% | -0.1588 | 0.0164 | -0.0383 | 0.0040 | no | no | 2.00 | O3 |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 129,136 / 129,027 | 129,081 | -4.9% | -0.0972 | 0.0608 | -0.0235 | 0.0147 | no | no | 1.94 | O3 |
| W R1 width 88, retrained | 27,752 | 130,066 / 129,830 | 129,948 | -5.6% | -0.1111 | 0.0608 | -0.0268 | 0.0147 | no | no | 1.95 | O3 |
| W R2 width 208, retrained | 113,912 | 128,714 / 128,389 | 128,552 | -4.5% | -0.0887 | 0.1995 | -0.0214 | 0.0482 | no | no | 1.93 | O3 |
| W R3 width 456, retrained | 499,552 | 125,762 / 125,752 | 125,757 | -2.2% | -0.0440 | 0.8205 | -0.0106 | 0.1982 | no | no | 1.89 | O3 |
| P R0, primer 1,048,576 | 208 | 127,380 / 127,389 | 127,384 | -3.6% | -0.0701 | 0.0164 | -0.0169 | 0.0040 | no | no | 1.91 | O3 |
| P-s R0, primer 16,777,216 | 208 | 125,113 / 125,124 | 125,119 | -1.7% | -0.0338 | 0.0164 | -0.0082 | 0.0040 | no | no | 1.88 | O3 |
| P R1, primer 1,048,576 | 27,752 | 125,132 / 125,092 | 125,112 | -1.7% | -0.0337 | 0.0608 | -0.0081 | 0.0147 | no | no | 1.88 | O3 |
| P-s R1, primer 16,777,216 | 27,752 | 122,599 / 122,607 | 122,603 | +0.3% | +0.0064 | 0.0608 | +0.0016 | 0.0147 | no | no | 1.84 | O3 |

### frozen test, class `logs`: test-apache, test-linux, test-openssh; 612,940 raw bytes; B_cl 24,587; Mamba-130M 16,274 (-33.8% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 20,920 / 20,897 | 20,908 | +15.0% | +0.0480 | 0.0164 | +0.0116 | 0.0040 | yes | yes | 1.28 | CM only |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 19,633 / 19,596 | 19,614 | +20.2% | +0.0649 | 0.0608 | +0.0157 | 0.0147 | yes | yes | 1.21 | O2 |
| W R1 width 88, retrained | 27,752 | 18,936 / 19,215 | 19,076 | +22.4% | +0.0719 | 0.0608 | +0.0174 | 0.0147 | yes | yes | 1.17 | O2 |
| W R2 width 208, retrained | 113,912 | 18,688 / 19,152 | 18,920 | +23.0% | +0.0740 | 0.1995 | +0.0179 | 0.0482 | no | yes | 1.16 | O3 |
| W R3 width 456, retrained | 499,552 | 18,458 / 18,946 | 18,702 | +23.9% | +0.0768 | 0.8205 | +0.0185 | 0.1982 | no | yes | 1.15 | O3 |
| P R0, primer 1,048,576 | 208 | 21,414 / 21,402 | 21,408 | +12.9% | +0.0415 | 0.0164 | +0.0100 | 0.0040 | yes | yes | 1.32 | O2 |
| P-s R0, primer 16,777,216 | 208 | 20,124 / 20,102 | 20,113 | +18.2% | +0.0584 | 0.0164 | +0.0141 | 0.0040 | yes | yes | 1.24 | O2 |
| P R1, primer 1,048,576 | 27,752 | 20,905 / 20,816 | 20,861 | +15.2% | +0.0486 | 0.0608 | +0.0117 | 0.0147 | no | yes | 1.28 | O3 |
| P-s R1, primer 16,777,216 | 27,752 | 19,187 / 19,121 | 19,154 | +22.1% | +0.0709 | 0.0608 | +0.0171 | 0.0147 | yes | yes | 1.18 | O2 |

### frozen test, class `xml-synthetic`: test-facturae, test-ubl; 122,644 raw bytes; B_cl 4,439; Mamba-130M 3,669 (-17.3% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 5,634 / 5,633 | 5,634 | -26.9% | -0.0779 | 0.0164 | -0.0188 | 0.0040 | no | no | 1.54 | O3 |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 5,663 / 5,610 | 5,637 | -27.0% | -0.0781 | 0.0608 | -0.0189 | 0.0147 | no | no | 1.54 | O3 |
| W R1 width 88, retrained | 27,752 | 6,568 / 6,368 | 6,468 | -45.7% | -0.1323 | 0.0608 | -0.0320 | 0.0147 | no | no | 1.76 | O3 |
| W R2 width 208, retrained | 113,912 | 6,447 / 6,137 | 6,292 | -41.7% | -0.1209 | 0.1995 | -0.0292 | 0.0482 | no | no | 1.71 | O3 |
| W R3 width 456, retrained | 499,552 | 6,440 / 6,132 | 6,286 | -41.6% | -0.1205 | 0.8205 | -0.0291 | 0.1982 | no | no | 1.71 | O3 |
| P R0, primer 1,048,576 | 208 | 5,904 / 5,904 | 5,904 | -33.0% | -0.0955 | 0.0164 | -0.0231 | 0.0040 | no | no | 1.61 | O3 |
| P-s R0, primer 16,777,216 | 208 | 5,502 / 5,500 | 5,501 | -23.9% | -0.0693 | 0.0164 | -0.0167 | 0.0040 | no | no | 1.50 | O3 |
| P R1, primer 1,048,576 | 27,752 | 6,055 / 5,992 | 6,023 | -35.7% | -0.1033 | 0.0608 | -0.0250 | 0.0147 | no | no | 1.64 | O3 |
| P-s R1, primer 16,777,216 | 27,752 | 5,628 / 5,600 | 5,614 | -26.5% | -0.0766 | 0.0608 | -0.0185 | 0.0147 | no | no | 1.53 | O3 |

### extended, class `text`: ext-dickens, ext-bible; 2,097,152 raw bytes; B_cl 437,436; Mamba-130M 257,513 (-41.1% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 469,145 / 469,178 | 469,161 | -7.3% | -0.1210 | 0.0164 | -0.0292 | 0.0040 | no | no | 1.82 | O3 |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 459,423 / 459,465 | 459,444 | -5.0% | -0.0840 | 0.0608 | -0.0203 | 0.0147 | no | no | 1.78 | O3 |
| W R1 width 88, retrained | 27,752 | 462,152 / 462,066 | 462,109 | -5.6% | -0.0941 | 0.0608 | -0.0227 | 0.0147 | no | no | 1.79 | O3 |
| W R2 width 208, retrained | 113,912 | 459,619 / 459,478 | 459,549 | -5.1% | -0.0844 | 0.1995 | -0.0204 | 0.0482 | no | no | 1.78 | O3 |
| W R3 width 456, retrained | 499,552 | 453,701 / 454,354 | 454,028 | -3.8% | -0.0633 | 0.8205 | -0.0153 | 0.1982 | no | no | 1.76 | O3 |
| P R0, primer 1,048,576 | 208 | 455,755 / 455,782 | 455,769 | -4.2% | -0.0699 | 0.0164 | -0.0169 | 0.0040 | no | no | 1.77 | O3 |
| P-s R0, primer 16,777,216 | 208 | 448,737 / 448,770 | 448,753 | -2.6% | -0.0432 | 0.0164 | -0.0104 | 0.0040 | no | no | 1.74 | O3 |
| P R1, primer 1,048,576 | 27,752 | 450,051 / 450,185 | 450,118 | -2.9% | -0.0484 | 0.0608 | -0.0117 | 0.0147 | no | no | 1.75 | O3 |
| P-s R1, primer 16,777,216 | 27,752 | 442,545 / 442,808 | 442,677 | -1.2% | -0.0200 | 0.0608 | -0.0048 | 0.0147 | no | no | 1.72 | O3 |

### extended, class `logs`: ext-apache, ext-linux, ext-openssh; 3,145,728 raw bytes; B_cl 126,891; Mamba-130M 90,060 (-29.0% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 97,215 / 97,144 | 97,180 | +23.4% | +0.0756 | 0.0164 | +0.0182 | 0.0040 | yes | yes | 1.08 | CM only |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 95,288 / 95,522 | 95,405 | +24.8% | +0.0801 | 0.0608 | +0.0193 | 0.0147 | yes | yes | 1.06 | O1 |
| W R1 width 88, retrained | 27,752 | 94,238 / 94,694 | 94,466 | +25.6% | +0.0825 | 0.0608 | +0.0199 | 0.0147 | yes | yes | 1.05 | O1 |
| W R2 width 208, retrained | 113,912 | 93,093 / 94,272 | 93,682 | +26.2% | +0.0845 | 0.1995 | +0.0204 | 0.0482 | no | yes | 1.04 | O3 |
| W R3 width 456, retrained | 499,552 | 93,190 / 94,481 | 93,836 | +26.1% | +0.0841 | 0.8205 | +0.0203 | 0.1982 | no | yes | 1.04 | O3 |
| P R0, primer 1,048,576 | 208 | 98,365 / 98,311 | 98,338 | +22.5% | +0.0726 | 0.0164 | +0.0175 | 0.0040 | yes | yes | 1.09 | O1 |
| P-s R0, primer 16,777,216 | 208 | 95,663 / 95,591 | 95,627 | +24.6% | +0.0795 | 0.0164 | +0.0192 | 0.0040 | yes | yes | 1.06 | O1 |
| P R1, primer 1,048,576 | 27,752 | 97,612 / 97,707 | 97,660 | +23.0% | +0.0743 | 0.0608 | +0.0180 | 0.0147 | yes | yes | 1.08 | O1 |
| P-s R1, primer 16,777,216 | 27,752 | 94,213 / 94,318 | 94,265 | +25.7% | +0.0830 | 0.0608 | +0.0200 | 0.0147 | yes | yes | 1.05 | O1 |

### extended, class `structured`: ext-gharchive, ext-nci; 2,097,152 raw bytes; B_cl 150,551; Mamba-130M 124,920 (-17.0% vs classical)
| Configuration | MAC/byte | Coded bytes (1204 / 1205) | B_r | Saving | delta bpb | Needed | Value $/GB | Cost $/GB | Cost | Material | Parity | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 cm, no primer | 208 | 171,694 / 171,662 | 171,678 | -14.0% | -0.0806 | 0.0164 | -0.0195 | 0.0040 | no | no | 1.37 | O3 |
| G0 R1 width 88 (33.55 MB training) | 27,752 | 172,956 / 171,385 | 172,170 | -14.4% | -0.0825 | 0.0608 | -0.0199 | 0.0147 | no | no | 1.38 | O3 |
| W R1 width 88, retrained | 27,752 | 176,248 / 176,937 | 176,592 | -17.3% | -0.0993 | 0.0608 | -0.0240 | 0.0147 | no | no | 1.41 | O3 |
| W R2 width 208, retrained | 113,912 | 175,951 / 175,446 | 175,698 | -16.7% | -0.0959 | 0.1995 | -0.0232 | 0.0482 | no | no | 1.41 | O3 |
| W R3 width 456, retrained | 499,552 | 176,540 / 174,400 | 175,470 | -16.6% | -0.0951 | 0.8205 | -0.0230 | 0.1982 | no | no | 1.40 | O3 |
| P R0, primer 1,048,576 | 208 | 170,166 / 170,137 | 170,151 | -13.0% | -0.0748 | 0.0164 | -0.0181 | 0.0040 | no | no | 1.36 | O3 |
| P-s R0, primer 16,777,216 | 208 | 168,447 / 168,416 | 168,431 | -11.9% | -0.0682 | 0.0164 | -0.0165 | 0.0040 | no | no | 1.35 | O3 |
| P R1, primer 1,048,576 | 27,752 | 172,112 / 170,672 | 171,392 | -13.8% | -0.0795 | 0.0608 | -0.0192 | 0.0147 | no | no | 1.37 | O3 |
| P-s R1, primer 16,777,216 | 27,752 | 170,365 / 168,942 | 169,653 | -12.7% | -0.0729 | 0.0608 | -0.0176 | 0.0147 | no | no | 1.36 | O3 |

## 4. Class verdicts (worse of the sets a class appears in)

| Class | Configuration | Verdict | Per set |
|---|---|---|---|
| logs | R0 cm, no primer | CM only | frozen test: CM only; extended: CM only |
| logs | W R1 width 88, retrained | O2 | frozen test: O2; extended: O1 |
| logs | W R2 width 208, retrained | O3 | frozen test: O3; extended: O3 |
| logs | W R3 width 456, retrained | O3 | frozen test: O3; extended: O3 |
| logs | P R0, primer 1,048,576 | O2 | frozen test: O2; extended: O1 |
| logs | P-s R0, primer 16,777,216 | O2 | frozen test: O2; extended: O1 |
| logs | P R1, primer 1,048,576 | O3 | frozen test: O3; extended: O1 |
| logs | P-s R1, primer 16,777,216 | O2 | frozen test: O2; extended: O1 |
| structured | R0 cm, no primer | O3 | extended: O3 |
| structured | W R1 width 88, retrained | O3 | extended: O3 |
| structured | W R2 width 208, retrained | O3 | extended: O3 |
| structured | W R3 width 456, retrained | O3 | extended: O3 |
| structured | P R0, primer 1,048,576 | O3 | extended: O3 |
| structured | P-s R0, primer 16,777,216 | O3 | extended: O3 |
| structured | P R1, primer 1,048,576 | O3 | extended: O3 |
| structured | P-s R1, primer 16,777,216 | O3 | extended: O3 |
| text | R0 cm, no primer | O3 | frozen test: O3; extended: O3 |
| text | W R1 width 88, retrained | O3 | frozen test: O3; extended: O3 |
| text | W R2 width 208, retrained | O3 | frozen test: O3; extended: O3 |
| text | W R3 width 456, retrained | O3 | frozen test: O3; extended: O3 |
| text | P R0, primer 1,048,576 | O3 | frozen test: O3; extended: O3 |
| text | P-s R0, primer 16,777,216 | O3 | frozen test: O3; extended: O3 |
| text | P R1, primer 1,048,576 | O3 | frozen test: O3; extended: O3 |
| text | P-s R1, primer 16,777,216 | O3 | frozen test: O3; extended: O3 |
| xml-synthetic | R0 cm, no primer | O3 | frozen test: O3 |
| xml-synthetic | W R1 width 88, retrained | O3 | frozen test: O3 |
| xml-synthetic | W R2 width 208, retrained | O3 | frozen test: O3 |
| xml-synthetic | W R3 width 456, retrained | O3 | frozen test: O3 |
| xml-synthetic | P R0, primer 1,048,576 | O3 | frozen test: O3 |
| xml-synthetic | P-s R0, primer 16,777,216 | O3 | frozen test: O3 |
| xml-synthetic | P R1, primer 1,048,576 | O3 | frozen test: O3 |
| xml-synthetic | P-s R1, primer 16,777,216 | O3 | frozen test: O3 |

Class verdict under V7: `text` O3; `structured` O3; `logs` O2 (retrained R1; arm P-s R0 and R1; arm P R0 with the
caveat of section 0.8), and "CM only" passes. `xml-synthetic` does not vote.

## 5. Slope and stop rule 2 (R4 is trained only if R2 -> R3 lowers one class total by 2 % or more)

| Set | Class | R1 -> R2 | R2 -> R3 | G0 R1 -> retrained R1 (43x the training bytes) |
|---|---|---|---|---|
| frozen test | text | -1.07% | -2.17% | +0.67% |
| frozen test | logs | -0.82% | -1.15% | -2.75% |
| frozen test | xml-synthetic | -2.72% | -0.10% | +14.74% |
| extended | text | -0.55% | -1.20% | +0.58% |
| extended | logs | -0.83% | +0.16% | -0.98% |
| extended | structured | -0.51% | -0.13% | +2.57% |

Largest R2 -> R3 gain among voting classes: 2.17% -> the condition for R4 is MET
R4 (`r4`, 1,961,768 MAC/byte) would need delta >= 3.175 bpb at 10 G MAC/s and 1.596 at 20 G

Negative = fewer bytes. The last column is the effect of the larger training set and longer schedule at equal width.

## 6. Consequences

- Stop rule 1 (close the storage question for Strata) is not triggered by the letter, because of the logs O2. The
  facts behind that O2 are in section 0.3; the decision "is a cheaper and weaker log compressor a product" belongs
  to the owner. Recommendation: no further capacity work (no R4, no larger trunk, no more training data) — the
  three levers V7 tested are measured flat.
- What survives as an asset: tier `cm` — 208 MAC/byte, about 34 KB of weights, integer, bit-identical by
  construction — compresses unseen logs 15-23 % better than the best of PPMd / xz and within 8-28 % of Mamba-130M.
  It is a classical-class context mixer, not a neural backbone, and it loses to PPMd on text and to PPMd / brotli
  on structured data.
- Path 3 (v1 scope, messaging files) is not judged by V7 and is unaffected; its open gates are V1 (projected FAIL
  with PPMd in the set) and V3 (zstd + same-source dictionary), plus the unverified market (R-116).
- Stage 4a (Rust) was not needed for this verdict and should not start on the strength of it.
