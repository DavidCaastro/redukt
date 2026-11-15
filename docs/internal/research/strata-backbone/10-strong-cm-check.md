# Strata backbone — 10. Speed of tier `cm` and check E2: public context-mixing compressors

> 2026-11-15. Two measurements made after the V7 verdict (`09-capacity-ladder-results.md`), on the owner's request
> to explore the throughput problem. Check E2 was registered in `05-gate-amendments.md` (amendment 6, commit
> 136f639) before any public CM compressor was run on a selection or decision file.
> Labels: MEASURED (command given), DERIVED, INFERENCE.

## 0. Result first — weak points before anything else

1. **Check E2: NO ASSET, on both sets (MEASURED).** On unseen logs tier `cm` is 8.7 % (frozen test) and 20.2 %
   (extended) larger than the best fast public compressor per file. lpaq1 (2007, 1,220 KB/s here) is 7.3-7.9 %
   smaller than `cm` on logs, 11 % on text, 17 % on structured data. zpaq `-m5` (601 KB/s) is 17 % smaller on the
   extended logs. The statement of `09` section 6, "what survives as an asset: tier `cm`", is **withdrawn**: `cm`
   beat PPMd and xz only because no compressor of its own family was in the classical set.
2. **The expectation registered in amendment 6 was too kind to `cm`** ("lpaq1 close to `cm` on logs, likely TIE").
3. **Mamba-130M loses to free CPU compressors on everything except prose (MEASURED, identical bytes).**
   paq8px v217 `-8` (11 KB/s on one thread) against Mamba-130M Q8 (648 B/s on twelve threads): logs -33.5 % /
   -41.2 %, structured -26.7 %, synthetic invoices -39.0 %; text +58 % / +43 % (Mamba wins). paq8l (2007, 30 KB/s)
   also beats Mamba on logs (-22 % / -31 %), structured (-14 %) and invoices (-22 %). zpaq `-m5` at 601 KB/s is
   10 % smaller than Mamba on the extended logs; lpaq1 ties it there (+0.1 %). These are 171 KB to 1 MiB inputs
   with the default 4096-token reset; the 10 KB windows of the headline table of `CLAUDE.md` were not measured
   here, and a pretrained model has its largest advantage on short inputs.
4. **Speed of tier `cm` (MEASURED on a proxy, not on the specified arithmetic):** 0.53-0.63 us per byte on one
   thread (1.6-1.9 MB/s), 1.45 us per byte and thread with twelve threads loaded (0.69 MB/s each). The 1 us per
   byte of specification 7.2 holds to within a factor 2 either way: $0.002-0.006 per raw GB at $0.014 per
   vCPU-hour. So the cost side of V7's "CM only" result was right; its classical side was incomplete.
5. **The proxy has no trained weights and is within 2-5 % of `cm`** (extended logs 102,186 against 97,180 bytes;
   frozen logs 21,733 against 20,908; extended text 479,139 against 469,161). The 1,199 trained parameters of the
   width-0 tier buy a few percent; the ratio of `cm` is its online tables.
6. **Where value minus cost is largest (DERIVED, section 2):** zpaq `-m5` and lpaq1, at +$0.007 to +$0.022 per raw
   GB over seven years on logs and text. paq8l and paq8px save far more bytes and lose money for storage (cost
   $0.13 and $0.34 per GB against a value of $0.04-0.07). No compressor measured here, at any speed, earns more
   than about two cents per GB over seven years. That is the size of the storage prize, and free tools already
   take it.
7. **Ceiling of the prize (DERIVED).** With free compute and the bytes of Mamba-130M the saving against the old
   best classical codec is worth $0.17-0.22 per raw GB over seven years on text and $0.023-0.026 on logs. Against
   the best fast compressor of this check the log figure is negative on the extended set (zpaq is smaller than
   Mamba). Throughput work cannot raise these numbers.
8. Limits. (a) Speeds are wall times of whole processes on a thermally limited laptop, start-up and allocation of
   1.5-2.4 GB included; they are within a small factor, not benchmarks. (b) zpaq writes an archive: about 1 KB of
   overhead that dominates the two invoice files (it loses there for that reason) and is negligible at 1 MiB.
   (c) cmix was not run. (d) One setting per compressor, as registered. (e) `cm` is counted in ideal code lengths
   (favours `cm`). (f) The proxy and the public compressors were run on the decision files of V7 after its
   verdict; those files are no longer unseen for any later design choice. (g) The public binaries are GPL
   programs used as measuring instruments only; nothing of them is in the repository.

**What this changes.** Strata has no measured advantage left for storage or for long inputs: on text it loses to
PPMd, on logs and structured data it loses to its own family, and its trunk does not pay. The legacy Mamba path
keeps one measured advantage, prose (the best CM compressor here is 43-58 % larger), at 200x the cost per thread of
paq8px. The claims of `CLAUDE.md` against brotli and gzip on logs and XML must not be read as "better than the
best CPU compressor". What is untouched by this check: short messages (path 3), where a pretrained prior matters
most and no online compressor has time to learn — and where the comparison against this family has not been made
either (see section 4).

## 1. What was run (MEASURED)

| Item | Value |
|---|---|
| Speed proxy | `tools/strata/viability/cm_speed_proxy.rs`, `rustc -C opt-level=3 -C target-cpu=x86-64-v3`; `cm_speed_proxy <file> <threads> <repeats>`, best of the repeats. Same per-byte work as specification 2.5 / 3.1 / 3.3 with the v1 default tables (8 successor orders, 4 bit-cell orders, 38.8 MB of state, 13 mixer inputs, calibration cells, a binary range coder); stand-in tables, boot weights and order lengths, so it is not bit-compatible |
| Proxy, one thread, 1 MiB | `ext-apache` 0.528 us/byte, `ext-dickens` 0.628, `ext-gharchive` 0.576 |
| Proxy, `ext-apache`, 2 / 4 / 12 threads (one stream each) | 0.604 / 0.888 / 1.450 us per byte and stream; 3.3 / 4.5 / 8.3 MB/s in total (i5-1235U, 2 P + 8 E cores) |
| Public compressors | `python -I tools/strata/viability/strong_cm_baseline.py` -> `data/strata/runs/strong-cm.csv` (65 rows), one process at a time; settings of amendment 6 |
| Report | `python -I tools/strata/viability/strong_cm_report.py` (prints sections 2 and 3 verbatim) |
| Integrity | every output of lpaq1, zpaq and bsc decoded back and compared: 39 of 39 identical; paq8px decoded on three files: identical; paq8l not decoded |
| Binaries (sha256 of the downloaded archive) | zpaq 7.15, `mattmahoney.net/dc/zpaq715.zip`, `e85ec252…d418`; lpaq1, `…/dc/lpaq1.zip`, `e4b54c3e…3b47`; paq8l, `…/dc/paq8l.zip`, `fb72db12…74cf`; paq8px v217, GitHub `hxim/paq8px` release, `f5be98bf…2f1c`; bsc 3.4.0, GitHub `IlyaGrebnov/libbsc` release, `2b535dff…7a3a`. Kept under `data/strata/tools-ext/` (not in git) |

## 2. Speed, and per class every compressor (MEASURED; output of `strong_cm_report.py`)

Value = saving against the old best classical codec of amendment 5, in $ per raw GB over seven years; cost = $0.014
per vCPU-hour x measured seconds per byte on one thread; for `cm` the loaded-machine figure (1.45 us) is used.

### speed (seven 1 MiB files, one process at a time, wall time including start-up)
| Compressor | us per byte | KB/s | Cost $ per raw GB | Fast class |
|---|---|---|---|---|
| bsc-e2 | 0.25 | 3,983 | 0.0010 | yes |
| lpaq1-9 | 0.82 | 1,220 | 0.0032 | yes |
| paq8l-8 | 33.73 | 30 | 0.1312 | no |
| paq8px217-8 | 87.40 | 11 | 0.3399 | no |
| zpaq-m5 | 1.66 | 601 | 0.0065 | yes |
| tier `cm` (speed proxy) | 0.53 - 1.45 | 690 - 1,887 | 0.0021 - 0.0056 | - |

### frozen test, class `text`: test-dickens; 500,000 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 123,006 | +0.0% | -7.5% | +84.8% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 132,930 | +8.1% | +0.0% | +99.7% | -0.0383 | 0.0056 | -0.0440 |
| bsc-e2 | 126,260 | +2.6% | -5.0% | +89.7% | -0.0126 | 0.0010 | -0.0135 |
| lpaq1-9 | 118,537 | -3.6% | -10.8% | +78.1% | +0.0173 | 0.0032 | +0.0141 |
| paq8l-8 | 111,341 | -9.5% | -16.2% | +67.3% | +0.0451 | 0.1312 | -0.0861 |
| paq8px217-8 | 105,403 | -14.3% | -20.7% | +58.3% | +0.0680 | 0.3399 | -0.2719 |
| zpaq-m5 | 117,176 | -4.7% | -11.9% | +76.0% | +0.0225 | 0.0065 | +0.0161 |
| best fast compressor per file (F) | 117,176 | -4.7% | -11.9% | +76.0% | +0.0225 | - | - |
| Mamba-130M Q8 (real `.azth`) | 66,568 | -45.9% | -49.9% | +0.0% | +0.2181 | - | - |

B_cm / F = 1.134 -> NO ASSET (reported, no vote)

### frozen test, class `logs`: test-apache, test-linux, test-openssh; 612,940 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 24,587 | +0.0% | +17.6% | +51.1% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 20,908 | -15.0% | +0.0% | +28.5% | +0.0116 | 0.0056 | +0.0060 |
| bsc-e2 | 30,132 | +22.6% | +44.1% | +85.2% | -0.0175 | 0.0010 | -0.0185 |
| lpaq1-9 | 19,260 | -21.7% | -7.9% | +18.3% | +0.0168 | 0.0032 | +0.0136 |
| paq8l-8 | 12,711 | -48.3% | -39.2% | -21.9% | +0.0374 | 0.1312 | -0.0937 |
| paq8px217-8 | 10,822 | -56.0% | -48.2% | -33.5% | +0.0434 | 0.3399 | -0.2965 |
| zpaq-m5 | 20,195 | -17.9% | -3.4% | +24.1% | +0.0138 | 0.0065 | +0.0074 |
| best fast compressor per file (F) | 19,233 | -21.8% | -8.0% | +18.2% | +0.0169 | - | - |
| Mamba-130M Q8 (real `.azth`) | 16,274 | -33.8% | -22.2% | +0.0% | +0.0262 | - | - |

B_cm / F = 1.087 -> NO ASSET

### frozen test, class `xml-synthetic`: test-facturae, test-ubl; 122,644 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 4,439 | +0.0% | -21.2% | +21.0% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 5,634 | +26.9% | +0.0% | +53.5% | -0.0188 | 0.0056 | -0.0245 |
| bsc-e2 | 5,442 | +22.6% | -3.4% | +48.3% | -0.0158 | 0.0010 | -0.0168 |
| lpaq1-9 | 4,264 | -3.9% | -24.3% | +16.2% | +0.0028 | 0.0032 | -0.0004 |
| paq8l-8 | 2,842 | -36.0% | -49.6% | -22.5% | +0.0252 | 0.1312 | -0.1060 |
| paq8px217-8 | 2,238 | -49.6% | -60.3% | -39.0% | +0.0347 | 0.3399 | -0.3052 |
| zpaq-m5 | 7,337 | +65.3% | +30.2% | +100.0% | -0.0457 | 0.0065 | -0.0521 |
| best fast compressor per file (F) | 4,264 | -3.9% | -24.3% | +16.2% | +0.0028 | - | - |
| Mamba-130M Q8 (real `.azth`) | 3,669 | -17.3% | -34.9% | +0.0% | +0.0121 | - | - |

B_cm / F = 1.321 -> NO ASSET (reported, no vote)

### extended, class `text`: ext-dickens, ext-bible; 2,097,152 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 437,436 | +0.0% | -6.8% | +69.9% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 469,161 | +7.3% | +0.0% | +82.2% | -0.0292 | 0.0056 | -0.0349 |
| bsc-e2 | 450,800 | +3.1% | -3.9% | +75.1% | -0.0123 | 0.0010 | -0.0133 |
| lpaq1-9 | 416,981 | -4.7% | -11.1% | +61.9% | +0.0188 | 0.0032 | +0.0157 |
| paq8l-8 | 390,702 | -10.7% | -16.7% | +51.7% | +0.0431 | 0.1312 | -0.0881 |
| paq8px217-8 | 369,132 | -15.6% | -21.3% | +43.3% | +0.0629 | 0.3399 | -0.2770 |
| zpaq-m5 | 412,569 | -5.7% | -12.1% | +60.2% | +0.0229 | 0.0065 | +0.0164 |
| best fast compressor per file (F) | 412,569 | -5.7% | -12.1% | +60.2% | +0.0229 | - | - |
| Mamba-130M Q8 (real `.azth`) | 257,513 | -41.1% | -45.1% | +0.0% | +0.1658 | - | - |

B_cm / F = 1.137 -> NO ASSET (reported, no vote)

### extended, class `logs`: ext-apache, ext-linux, ext-openssh; 3,145,728 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 126,891 | +0.0% | +30.6% | +40.9% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 97,180 | -23.4% | +0.0% | +7.9% | +0.0182 | 0.0056 | +0.0126 |
| bsc-e2 | 153,386 | +20.9% | +57.8% | +70.3% | -0.0163 | 0.0010 | -0.0172 |
| lpaq1-9 | 90,116 | -29.0% | -7.3% | +0.1% | +0.0226 | 0.0032 | +0.0194 |
| paq8l-8 | 61,762 | -51.3% | -36.4% | -31.4% | +0.0400 | 0.1312 | -0.0912 |
| paq8px217-8 | 52,999 | -58.2% | -45.5% | -41.2% | +0.0454 | 0.3399 | -0.2945 |
| zpaq-m5 | 80,834 | -36.3% | -16.8% | -10.2% | +0.0283 | 0.0065 | +0.0218 |
| best fast compressor per file (F) | 80,834 | -36.3% | -16.8% | -10.2% | +0.0283 | - | - |
| Mamba-130M Q8 (real `.azth`) | 90,060 | -29.0% | -7.3% | +0.0% | +0.0226 | - | - |

B_cm / F = 1.202 -> NO ASSET

### extended, class `structured`: ext-gharchive, ext-nci; 2,097,152 raw bytes
| Compressor | Bytes | vs old best classical | vs tier `cm` | vs Mamba-130M | Value $/GB | Cost $/GB | Value - cost |
|---|---|---|---|---|---|---|---|
| old best classical (per file) | 150,551 | +0.0% | -12.3% | +20.5% | +0.0000 | - | - |
| tier `cm` (R0, ideal code length) | 171,678 | +14.0% | +0.0% | +37.4% | -0.0195 | 0.0056 | -0.0251 |
| bsc-e2 | 157,426 | +4.6% | -8.3% | +26.0% | -0.0063 | 0.0010 | -0.0073 |
| lpaq1-9 | 143,251 | -4.8% | -16.6% | +14.7% | +0.0067 | 0.0032 | +0.0035 |
| paq8l-8 | 107,619 | -28.5% | -37.3% | -13.8% | +0.0396 | 0.1312 | -0.0916 |
| paq8px217-8 | 91,619 | -39.1% | -46.6% | -26.7% | +0.0543 | 0.3399 | -0.2856 |
| zpaq-m5 | 132,352 | -12.1% | -22.9% | +5.9% | +0.0168 | 0.0065 | +0.0103 |
| best fast compressor per file (F) | 132,352 | -12.1% | -22.9% | +5.9% | +0.0168 | - | - |
| Mamba-130M Q8 (real `.azth`) | 124,920 | -17.0% | -27.2% | +0.0% | +0.0236 | - | - |

B_cm / F = 1.297 -> NO ASSET (reported, no vote)

### check E2, class `logs`: NO ASSET / NO ASSET -> NO ASSET

### per file, bytes
| Compressor | test-dickens | test-apache | test-linux | test-openssh | test-facturae | test-ubl | ext-dickens | ext-bible | ext-apache | ext-linux | ext-openssh | ext-gharchive | ext-nci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old best classical | 123,006 | 5,513 | 10,004 | 9,070 | 1,868 | 2,571 | 261,889 | 175,547 | 43,485 | 51,408 | 31,998 | 90,902 | 59,649 |
| tier `cm` | 132,930 | 4,396 | 9,458 | 7,054 | 2,356 | 3,278 | 281,984 | 187,177 | 36,271 | 41,315 | 19,593 | 102,172 | 69,507 |
| bsc-e2 | 126,260 | 6,886 | 12,228 | 11,018 | 2,300 | 3,142 | 267,182 | 183,618 | 45,280 | 69,876 | 38,230 | 101,204 | 56,222 |
| lpaq1-9 | 118,537 | 4,099 | 8,463 | 6,698 | 1,849 | 2,415 | 249,267 | 167,714 | 32,448 | 37,559 | 20,109 | 86,393 | 56,858 |
| paq8l-8 | 111,341 | 2,853 | 6,051 | 3,807 | 1,240 | 1,602 | 234,298 | 156,404 | 25,227 | 25,944 | 10,591 | 67,742 | 39,877 |
| paq8px217-8 | 105,403 | 2,488 | 5,234 | 3,100 | 984 | 1,254 | 222,336 | 146,796 | 22,721 | 22,415 | 7,863 | 56,774 | 34,845 |
| zpaq-m5 | 117,176 | 4,825 | 8,699 | 6,671 | 3,207 | 4,130 | 244,995 | 167,574 | 30,166 | 34,253 | 16,415 | 83,795 | 48,557 |
| Mamba-130M | 66,568 | 3,960 | 6,639 | 5,675 | 1,224 | 2,445 | 145,471 | 112,042 | 35,665 | 33,269 | 21,126 | 60,735 | 64,185 |

## 3. The throughput question, answered with these numbers (DERIVED)

| Lever | Status | Evidence |
|---|---|---|
| Smaller pretrained backbone | closed | R-105 (Pythia, L3TC), R-112 (own specialists), V7 (Strata ladder: 18x compute for 1-3 %, 43x data for 0-3 %) |
| Faster hardware for Mamba-130M | at best break-even on text | T4 state-traffic ceiling 55K tokens/s (DERIVED in `CLAUDE.md`); through the product $33-38 per GB |
| Skipping work (entropy gate, CM skip, prediction cache) | closed | Fases 7, 9, 14: the recurrent state needs every token |
| A cheap online compressor instead of a backbone | works, and already exists for free | this document: lpaq1 / zpaq at 0.6-1.2 MB/s take the whole positive margin |
| More compute in an online compressor (paq8 class) | buys ratio, not money | paq8px: -56 % on logs against the old classical set, at $0.34 per GB against $0.045 of value |

Storage value per raw GB is bounded by the bytes saved times $1.93 (seven years of hot storage). The best ratio
measured on each class, at zero cost, gives $0.22 (text, Mamba), $0.045 (logs, paq8px), $0.054 (structured,
paq8px). No throughput improvement moves that bound; only a more expensive byte does.

## 4. What is still open, in order of cost

1. **Short messages against this family (path 3).** R-113 compared `.azth` with brotli and zstd + dictionary on
   128-2048 byte messages. A CM compressor with a primed state (zpaq or lpaq1 fed a same-domain dictionary) was
   not in that comparison. It is the same omission that this document found for long inputs, so it has to be
   closed before any path-3 claim is repeated. Cost: CPU only, existing R-113 messages.
2. **Market of path 3 (R-116)**, unverified; no code.
3. Anything for storage: nothing measured supports starting it.
