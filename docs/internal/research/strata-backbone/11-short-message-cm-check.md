# Strata backbone — 11. Check E3: primed public context-mixing compressors against `.azth` on short messages

> 2026-11-15. The comparison that check E2 (`10-strong-cm-check.md`, section 4) left open, run on the owner's
> request. Registered in `05-gate-amendments.md` (amendment 7, commit ae8e6f7) before any public CM compressor was
> run on an R-113 message. Labels: MEASURED (command given), DERIVED, INFERENCE.

## 0. Result first — weak points before anything else

1. **Check E3 (MEASURED): `text` KEEPS, `logs` DEPENDS ON THE PRIMER, `json` LOST.** Mamba-130M's payload against
   the best primed CM compressor of the class (ratio below 1 = Mamba smaller):

   | Class | Same-source primer | Foreign primer | Verdict |
   |---|---|---|---|
   | `text` (Wikipedia, prose) | 0.750 | 0.521 | KEEPS |
   | `logs` (Apache, Linux, OpenSSH) | 2.626 | 0.628 | DEPENDS ON THE PRIMER |
   | `json` (GitHub events) | 1.462 | 1.319 | LOST |

   The outcome is the one written as expectation in amendment 7.
2. **What decides logs is whether the terminal holds a sample of its own traffic.** With 86-110 KB of earlier
   lines of the same log, paq8px codes the messages in 3.6 % of their size, lpaq1 in 5.4 %, zstd + dictionary in
   10.9 %, Mamba-130M in 9.4 %. With 110 KB of a different system's log (Zookeeper) every CM compressor is 59-107 %
   larger than Mamba. A fleet that sends the same kind of message over and over has such a sample; that was the
   premise of the zstd dictionary in R-113 as well.
3. **The JSON "foreign" primer is another hour of the same public event archive** (labelled same source in G0
   amendment 1). The JSON LOST therefore says "with a sample of the same feed"; a JSON feed of a different schema
   was not measured.
4. **lpaq1 with 9 MB of memory equals lpaq1 with 1.5 GB on these messages** (55,311 against 54,794 bytes in total,
   same-source; 95,577 against 94,986 foreign). The primed-CM result does not need a large machine: a 2007
   compressor, 9 MB of state and a 110 KB sample, against a 130 MB model.
5. **The real R-113 archive is worse than its payload by 20 bytes per message.** Counted as it ships today, and
   giving the CM side 4 bytes of framing: text 0.834 / 0.583, logs 2.919 / 0.770, JSON 1.636 / 1.483.
6. **The CM figures are differences of two real archives, not a shipped codec.** No public CM compressor has a
   dictionary or snapshot option; the payload of message m after primer D is size(C(D + m)) - size(C(D)). An empty
   message costs 0 bytes by this count (checked on a neutral file), and lpaq1 decodes D + m back on six sampled
   cases. A product would have to store the primed state (or re-read the primer per message: 0.1 s for lpaq1,
   about 10 s for paq8px on this laptop).
7. Limits. (a) Mamba is not primed; a primed Mamba was never built and would probably improve it on logs and JSON
   (INFERENCE). (b) The best member of S per class is chosen after the fact (favours the CM side). (c) paq8px shows
   a block-detection artefact on Wikipedia after a `webster` primer (126.6 bytes for a 128-byte message, worse than
   lpaq1's 70.8); the rule picked paq8l there. (d) Wikipedia and English prose are almost certainly in Mamba's
   pre-training data. (e) One primer size (110,000 bytes) votes; the whole first half (2 MB) with lpaq1 narrows the
   text gap from +50 % to +27 % and is report-only. (f) 16 messages per cell, one cut. (g) The R-113 messages have
   now been read by this family; they are no longer unseen for a later design choice, gate V3 included.

**What this changes.** The path-3 statement of `CLAUDE.md` ("`.azth` is 41-53 % smaller than brotli; wins 16 of 30
against zstd + dictionary") stays true and is no longer the relevant comparison. Against primed context mixing,
Mamba-130M keeps natural-language text in every pairing (25-48 % smaller), keeps logs only when the terminal has no
sample of its own log, and loses structured JSON. The surviving claim is: **short natural-language messages, and
any message type for which no same-source sample exists on the terminal.** For repetitive machine traffic with a
sample — the usual telemetry case — a primed CM coder with megabytes of state is 2.4-3x smaller than Mamba.

**For gate V3 (owner's call, nothing amended here).** V3 compares Strata `msg88` with zstd + same-source
dictionary. On the same messages (primer sizes differ: Strata 65,536 bytes, lpaq1 110,000, so this is indicative),
`msg88` primed gives 11,143 bytes on logs against 10,179 for lpaq1 level 0 and 6,818 for paq8px; on text 37,937
against 36,447 and 31,736; on JSON 10,526 against 8,685 and 6,199. With a foreign primer `msg88` is ahead of lpaq1
on text and logs (47,128 against 48,783; 34,115 against 37,037) and behind paq8l / paq8px. A V3 pass against zstd
would not show an advantage over this family.

## 1. What was run (MEASURED)

| Item | Value |
|---|---|
| Messages | the 480 messages of R-113, cut locally by `tools/strata/bakeoff/shortmsg.py`; alignment with the Kaggle file: 1,919 of 1,920 classical sizes identical (the one difference is a brotli size above the message length, which the notebook capped at length + 1) |
| Run | `python -I tools/strata/viability/shortmsg_cm.py --workers 6` -> `data/strata/runs/shortmsg-cm.csv` (5,280 rows, 11 arms x 480 messages); about 2 h 40 min on the laptop, paq8px taking most of it |
| Report | `python -I tools/strata/viability/shortmsg_cm_report.py` (section 2 is its output, verbatim) |
| Compressors | lpaq1 `9` and `0`, paq8l `-8`, paq8px v217 `-8`; binaries and sha256 as in `10-strong-cm-check.md` section 1 |
| Primers | `same110k`: last 110,000 bytes of the first half of the source (85,619 for Apache, 108,242 for Linux); `foreign110k`: `webster` for Wikipedia and prose, Zookeeper for the logs, the next day's hour of the event archive for JSON |
| Mamba-130M | R-113 archives (`data/benchmark/results/short-message-bench/run-1/`, Q8, CPU, roundtrip 24 of 24 there); payload = archive - 20 bytes |
| Strata rows | `data/strata/runs/strata_g_bc/shortmsg-r113-t{0,1}-*.csv` of the bake-off (`t0` = `cm`, `t1` = `msg88`; 65,536-byte primer), shown for context |
| Integrity | lpaq1 `9`: primer + message decoded back, 6 of 6 identical (one per source). paq8l and paq8px outputs were not decoded in this check |

## 2. Output of the report (MEASURED)

alignment with the R-113 file (gzip, xz, brotli, zstd without dictionary, local against Kaggle): 1919 sizes identical, 1 different

### class `text`: wikipedia, prose; 160 messages, 126,976 raw bytes
| Arm | Total bytes | % of raw | vs Mamba payload |
|---|---|---|---|
| Mamba-130M payload | 23,817 | 18.8% | +0.0% |
| Mamba-130M archive | 27,017 | 21.3% | +13.4% |
| brotli-11 | 57,997 | 45.7% | +143.5% |
| zstd-19-dict | 49,894 | 39.3% | +109.5% |
| zstd-19-fdict | 60,404 | 47.6% | +153.6% |
| strata-t0-payload-fprimer65536 | 52,445 | 41.3% | +120.2% |
| strata-t0-payload-primer65536 | 40,324 | 31.8% | +69.3% |
| strata-t1-payload | 47,546 | 37.4% | +99.6% |
| strata-t1-payload-fprimer65536 | 47,128 | 37.1% | +97.9% |
| strata-t1-payload-primer65536 | 37,937 | 29.9% | +59.3% |
| lpaq1-0+foreign110k | 48,783 | 38.4% | +104.8% |
| lpaq1-0+same110k | 36,447 | 28.7% | +53.0% |
| lpaq1-9+foreign110k | 48,458 | 38.2% | +103.5% |
| lpaq1-9+none | 64,488 | 50.8% | +170.8% |
| lpaq1-9+same110k | 35,774 | 28.2% | +50.2% |
| lpaq1-9+samehalf | 30,321 | 23.9% | +27.3% |
| lpaq1-9+zdict | 37,066 | 29.2% | +55.6% |
| paq8l-8+foreign110k | 45,685 | 36.0% | +91.8% |
| paq8l-8+same110k | 33,749 | 26.6% | +41.7% |
| paq8px217-8+foreign110k | 49,220 | 38.8% | +106.7% |
| paq8px217-8+same110k | 31,736 | 25.0% | +33.2% |

same110k: B_cm = 31,736 (paq8px217-8); Mamba payload / B_cm = 0.750 -> KEEPS
  ratio by message size (4 of 5 below 1): 128: 1.03, 256: 0.80, 512: 0.74, 1024: 0.70, 2048: 0.75
  real archive against CM payload + 4 B of framing: 0.834

foreign110k: B_cm = 45,685 (paq8l-8); Mamba payload / B_cm = 0.521 -> KEEPS
  ratio by message size (5 of 5 below 1): 128: 0.67, 256: 0.53, 512: 0.50, 1024: 0.48, 2048: 0.53
  real archive against CM payload + 4 B of framing: 0.583

### class `logs`: log-apache, log-linux, log-openssh; 240 messages, 190,464 raw bytes
| Arm | Total bytes | % of raw | vs Mamba payload |
|---|---|---|---|
| Mamba-130M payload | 17,904 | 9.4% | +0.0% |
| Mamba-130M archive | 22,704 | 11.9% | +26.8% |
| brotli-11 | 41,069 | 21.6% | +129.4% |
| zstd-19-dict | 20,744 | 10.9% | +15.9% |
| zstd-19-fdict | 42,262 | 22.2% | +136.0% |
| strata-t0-payload-fprimer65536 | 43,375 | 22.8% | +142.3% |
| strata-t0-payload-primer65536 | 12,066 | 6.3% | -32.6% |
| strata-t1-payload | 29,814 | 15.7% | +66.5% |
| strata-t1-payload-fprimer65536 | 34,115 | 17.9% | +90.5% |
| strata-t1-payload-primer65536 | 11,143 | 5.9% | -37.8% |
| lpaq1-0+foreign110k | 37,037 | 19.4% | +106.9% |
| lpaq1-0+same110k | 10,179 | 5.3% | -43.1% |
| lpaq1-9+foreign110k | 36,855 | 19.4% | +105.8% |
| lpaq1-9+none | 46,554 | 24.4% | +160.0% |
| lpaq1-9+same110k | 10,336 | 5.4% | -42.3% |
| lpaq1-9+samehalf | 10,348 | 5.4% | -42.2% |
| lpaq1-9+zdict | 12,864 | 6.8% | -28.2% |
| paq8l-8+foreign110k | 30,892 | 16.2% | +72.5% |
| paq8l-8+same110k | 7,968 | 4.2% | -55.5% |
| paq8px217-8+foreign110k | 28,512 | 15.0% | +59.2% |
| paq8px217-8+same110k | 6,818 | 3.6% | -61.9% |

same110k: B_cm = 6,818 (paq8px217-8); Mamba payload / B_cm = 2.626 -> LOST
  ratio by message size (0 of 5 below 1): 128: 2.45, 256: 2.94, 512: 3.04, 1024: 2.56, 2048: 2.41
  real archive against CM payload + 4 B of framing: 2.919

foreign110k: B_cm = 28,512 (paq8px217-8); Mamba payload / B_cm = 0.628 -> KEEPS
  ratio by message size (5 of 5 below 1): 128: 0.58, 256: 0.57, 512: 0.60, 1024: 0.64, 2048: 0.68
  real archive against CM payload + 4 B of framing: 0.770

### class `json`: json-events; 80 messages, 63,488 raw bytes
| Arm | Total bytes | % of raw | vs Mamba payload |
|---|---|---|---|
| Mamba-130M payload | 9,062 | 14.3% | +0.0% |
| Mamba-130M archive | 10,662 | 16.8% | +17.7% |
| brotli-11 | 24,152 | 38.0% | +166.5% |
| zstd-19-dict | 12,252 | 19.3% | +35.2% |
| zstd-19-fdict | 12,783 | 20.1% | +41.1% |
| strata-t0-payload-fprimer65536 | 11,564 | 18.2% | +27.6% |
| strata-t0-payload-primer65536 | 10,592 | 16.7% | +16.9% |
| strata-t1-payload | 23,578 | 37.1% | +160.2% |
| strata-t1-payload-fprimer65536 | 11,524 | 18.2% | +27.2% |
| strata-t1-payload-primer65536 | 10,526 | 16.6% | +16.2% |
| lpaq1-0+foreign110k | 9,757 | 15.4% | +7.7% |
| lpaq1-0+same110k | 8,685 | 13.7% | -4.2% |
| lpaq1-9+foreign110k | 9,673 | 15.2% | +6.7% |
| lpaq1-9+none | 26,242 | 41.3% | +189.6% |
| lpaq1-9+same110k | 8,684 | 13.7% | -4.2% |
| lpaq1-9+samehalf | 7,366 | 11.6% | -18.7% |
| lpaq1-9+zdict | 8,807 | 13.9% | -2.8% |
| paq8l-8+foreign110k | 7,834 | 12.3% | -13.6% |
| paq8l-8+same110k | 7,039 | 11.1% | -22.3% |
| paq8px217-8+foreign110k | 6,871 | 10.8% | -24.2% |
| paq8px217-8+same110k | 6,199 | 9.8% | -31.6% |

same110k: B_cm = 6,199 (paq8px217-8); Mamba payload / B_cm = 1.462 -> LOST
  ratio by message size (0 of 5 below 1): 128: 2.03, 256: 1.72, 512: 1.65, 1024: 1.52, 2048: 1.29
  real archive against CM payload + 4 B of framing: 1.636

foreign110k: B_cm = 6,871 (paq8px217-8); Mamba payload / B_cm = 1.319 -> LOST
  ratio by message size (0 of 5 below 1): 128: 1.75, 256: 1.50, 512: 1.48, 1024: 1.36, 2048: 1.18
  real archive against CM payload + 4 B of framing: 1.483

### check E3
- `text`: KEEPS
- `logs`: DEPENDS ON THE PRIMER (LOST same / KEEPS foreign)
- `json`: LOST

### mean bytes per message, same110k (cells = source x size)
| Source | Size | Mamba-130M payload | Mamba-130M archive | zstd-19-dict | lpaq1-9+same110k | lpaq1-0+same110k | paq8l-8+same110k | paq8px217-8+same110k |
|---|---|---|---|---|---|---|---|---|
| wikipedia | 128 | 44.1 | 64.1 | 73.8 | 47.0 | 47.8 | 44.4 | 40.2 |
| wikipedia | 256 | 55.8 | 75.8 | 127.1 | 87.9 | 89.9 | 83.8 | 75.9 |
| wikipedia | 512 | 91.0 | 111.0 | 210.2 | 155.4 | 159.1 | 143.1 | 133.3 |
| wikipedia | 1024 | 168.3 | 188.3 | 391.9 | 298.8 | 305.2 | 276.7 | 253.3 |
| wikipedia | 2048 | 392.8 | 412.8 | 747.4 | 580.0 | 592.8 | 538.9 | 496.2 |
| prose | 128 | 35.6 | 55.6 | 68.6 | 37.6 | 37.9 | 37.4 | 37.0 |
| prose | 256 | 59.6 | 79.6 | 120.7 | 72.1 | 73.1 | 69.4 | 68.7 |
| prose | 512 | 103.5 | 123.5 | 211.1 | 140.1 | 142.3 | 135.4 | 130.9 |
| prose | 1024 | 188.9 | 208.9 | 398.2 | 274.8 | 279.4 | 263.9 | 253.4 |
| prose | 2048 | 349.1 | 369.1 | 769.4 | 542.2 | 550.6 | 516.2 | 494.6 |
| log-apache | 128 | 34.8 | 54.8 | 34.8 | 15.3 | 14.2 | 17.4 | 13.6 |
| log-apache | 256 | 49.9 | 69.9 | 44.1 | 18.8 | 17.9 | 17.4 | 16.2 |
| log-apache | 512 | 70.0 | 90.0 | 58.7 | 25.5 | 24.6 | 21.6 | 20.9 |
| log-apache | 1024 | 87.1 | 107.1 | 87.2 | 39.1 | 37.8 | 30.6 | 28.7 |
| log-apache | 2048 | 114.1 | 134.1 | 135.2 | 62.2 | 60.8 | 46.8 | 41.4 |
| log-linux | 128 | 39.3 | 59.3 | 57.2 | 27.0 | 26.9 | 22.4 | 19.6 |
| log-linux | 256 | 48.5 | 68.5 | 62.9 | 30.5 | 30.2 | 24.4 | 20.9 |
| log-linux | 512 | 56.1 | 76.1 | 74.9 | 39.5 | 39.2 | 32.0 | 27.4 |
| log-linux | 1024 | 79.8 | 99.8 | 117.2 | 69.1 | 69.1 | 54.8 | 48.6 |
| log-linux | 2048 | 114.3 | 134.3 | 163.1 | 101.7 | 101.5 | 78.5 | 67.5 |
| log-openssh | 128 | 39.0 | 59.0 | 41.6 | 17.4 | 16.9 | 16.6 | 13.1 |
| log-openssh | 256 | 62.1 | 82.1 | 60.5 | 26.2 | 25.4 | 20.7 | 17.4 |
| log-openssh | 512 | 87.6 | 107.6 | 81.8 | 36.7 | 36.2 | 27.4 | 22.0 |
| log-openssh | 1024 | 107.4 | 127.4 | 111.7 | 54.8 | 53.9 | 37.2 | 29.7 |
| log-openssh | 2048 | 129.1 | 149.1 | 165.5 | 82.4 | 81.4 | 50.2 | 39.2 |
| json-events | 128 | 33.9 | 53.9 | 38.3 | 18.8 | 18.0 | 18.4 | 16.7 |
| json-events | 256 | 48.4 | 68.4 | 65.8 | 38.0 | 37.6 | 30.6 | 28.2 |
| json-events | 512 | 92.2 | 112.2 | 109.3 | 76.1 | 75.8 | 60.3 | 55.8 |
| json-events | 1024 | 149.2 | 169.2 | 194.9 | 140.5 | 140.8 | 113.1 | 98.1 |
| json-events | 2048 | 242.7 | 262.7 | 357.5 | 269.3 | 270.7 | 217.6 | 188.8 |

Mamba's payload is smaller than every member of S in 9 of 30 cells

### mean bytes per message, foreign110k (cells = source x size)
| Source | Size | Mamba-130M payload | Mamba-130M archive | zstd-19-fdict | lpaq1-9+foreign110k | lpaq1-0+foreign110k | paq8l-8+foreign110k | paq8px217-8+foreign110k |
|---|---|---|---|---|---|---|---|---|
| wikipedia | 128 | 44.1 | 64.1 | 92.9 | 70.8 | 70.9 | 66.1 | 126.6 |
| wikipedia | 256 | 55.8 | 75.8 | 154.9 | 126.2 | 126.8 | 116.1 | 179.3 |
| wikipedia | 512 | 91.0 | 111.0 | 260.9 | 214.2 | 215.5 | 197.9 | 265.8 |
| wikipedia | 1024 | 168.3 | 188.3 | 494.2 | 409.4 | 412.1 | 376.8 | 433.2 |
| wikipedia | 2048 | 392.8 | 412.8 | 922.0 | 764.2 | 769.4 | 704.8 | 738.4 |
| prose | 128 | 35.6 | 55.6 | 79.6 | 52.8 | 52.9 | 52.1 | 49.6 |
| prose | 256 | 59.6 | 79.6 | 137.9 | 103.0 | 103.7 | 100.4 | 96.8 |
| prose | 512 | 103.5 | 123.5 | 254.4 | 197.8 | 199.6 | 191.5 | 184.0 |
| prose | 1024 | 188.9 | 208.9 | 478.4 | 377.4 | 380.8 | 365.3 | 348.6 |
| prose | 2048 | 349.1 | 369.1 | 899.9 | 712.8 | 717.4 | 684.4 | 653.9 |
| log-apache | 128 | 34.8 | 54.8 | 93.1 | 73.6 | 74.1 | 65.6 | 61.1 |
| log-apache | 256 | 49.9 | 69.9 | 126.7 | 107.0 | 107.8 | 96.5 | 88.7 |
| log-apache | 512 | 70.0 | 90.0 | 164.0 | 143.4 | 144.2 | 126.0 | 116.1 |
| log-apache | 1024 | 87.1 | 107.1 | 200.4 | 174.5 | 175.9 | 144.1 | 131.8 |
| log-apache | 2048 | 114.1 | 134.1 | 263.1 | 218.2 | 220.4 | 173.1 | 156.8 |
| log-linux | 128 | 39.3 | 59.3 | 88.3 | 73.6 | 73.0 | 64.8 | 61.6 |
| log-linux | 256 | 48.5 | 68.5 | 102.3 | 89.1 | 88.4 | 77.9 | 72.9 |
| log-linux | 512 | 56.1 | 76.1 | 118.9 | 106.4 | 105.9 | 90.0 | 83.6 |
| log-linux | 1024 | 79.8 | 99.8 | 170.3 | 155.9 | 155.0 | 124.9 | 116.6 |
| log-linux | 2048 | 114.3 | 134.3 | 249.8 | 228.1 | 227.6 | 179.2 | 164.0 |
| log-openssh | 128 | 39.0 | 59.0 | 101.8 | 85.1 | 85.9 | 74.9 | 70.9 |
| log-openssh | 256 | 62.1 | 82.1 | 160.8 | 141.1 | 142.6 | 126.1 | 117.6 |
| log-openssh | 512 | 87.6 | 107.6 | 215.0 | 193.4 | 195.0 | 169.5 | 157.1 |
| log-openssh | 1024 | 107.4 | 127.4 | 261.6 | 231.4 | 233.8 | 194.1 | 179.6 |
| log-openssh | 2048 | 129.1 | 149.1 | 325.6 | 282.6 | 285.2 | 223.9 | 203.6 |
| json-events | 128 | 33.9 | 53.9 | 43.1 | 22.9 | 23.4 | 20.9 | 19.4 |
| json-events | 256 | 48.4 | 68.4 | 71.4 | 43.9 | 44.6 | 36.4 | 32.3 |
| json-events | 512 | 92.2 | 112.2 | 115.4 | 85.6 | 86.4 | 69.4 | 62.4 |
| json-events | 1024 | 149.2 | 169.2 | 202.3 | 158.1 | 159.2 | 126.7 | 110.0 |
| json-events | 2048 | 242.7 | 262.7 | 366.8 | 294.1 | 296.2 | 236.2 | 205.4 |

Mamba's payload is smaller than every member of S in 25 of 30 cells

### lpaq1: memory and primer (total bytes over the 480 messages)
- lpaq1-0+foreign110k: 95,577
- lpaq1-0+same110k: 55,311
- lpaq1-9+foreign110k: 94,986
- lpaq1-9+none: 137,284
- lpaq1-9+same110k: 54,794
- lpaq1-9+samehalf: 48,035
- lpaq1-9+zdict: 58,737
- Mamba-130M payload: 50,783

## 3. What is still open

1. **A primed Mamba** — measured the same day, see section 4 (E3b: still LOST, at 1.12-1.24 instead of 1.32-2.63).
   Original text: a primed Mamba (state snapshot after a same-source sample) was never measured; it is the symmetric arm and
   the only thing that could change `logs` same-source and `json`. Cost: CPU, a change to the encoder start state.
2. **Which case is the market's**: does a terminal's traffic repeat (telemetry: primed CM wins) or is it free text
   (Mamba wins)? This is R-116, still unverified; it now decides the technical choice as well as the size.
3. **Hardware (owner, 2026-11-15: leaving CPU / edge is acceptable if viability needs it).** On a satellite link
   the model has to sit on the terminal either way; an accelerator changes the cost of the text case, not the
   ratio table above.
4. Whether V3 should name this family as a comparator (owner's call; see section 0).

## 4. Addendum of the same day — checks E3b and E4 (amendment 8, registered in commit 8ce1f31 before measuring)

### 4.0 Result first

1. **E3b (MEASURED): priming halves the bytes of Mamba-130M on logs and still leaves it behind the best primed CM
   coder: LOST on the three arms**, by a much smaller margin than unprimed.

   | Arm | Mamba unprimed / B_cm | Mamba primed (8 KB) / B_cm | Verdict |
   |---|---|---|---|
   | `logs`, same-source | 2.626 | 1.237 | LOST |
   | `json`, same-source | 1.462 | 1.217 | LOST |
   | `json`, foreign (same feed, other hour) | 1.319 | 1.120 | LOST |

   B_cm is paq8px with 110,000 bytes of sample, its best case, as registered. The registered expectation (logs
   LOST at 1.3-2.0, JSON LOST or MARGINAL, Mamba ahead at equal primers) held; logs came out slightly better for
   Mamba than the expected range.
2. **At equal 8 KB primers Mamba is ahead** of paq8px (logs 8,437 against 8,879 bytes, -5 %; JSON 7,544 against
   10,235, -26 %) and of lpaq1 (-34 %, -42 %). **Primed Mamba with 8 KB also beats lpaq1 level 0 with 110 KB**
   (logs 8,437 against 10,179, -17 %; JSON 7,544 against 8,685, -13 %). It loses only to the paq8 class with the
   large sample (paq8l 7,968, paq8px 6,818 on logs).
3. **So the short-message picture for machine traffic with a sample is a near tie between very different
   machines**: paq8px + 110 KB sample (gigabytes of working memory as run here, seconds per message) is 11-19 %
   smaller than Mamba-130M + 8 KB sample (130 MB of weights), which is 13-17 % smaller than lpaq1 + 110 KB sample
   (9 MB, milliseconds). None of these gaps is the 41-53 % of R-113. A pretrained model is not what makes short
   log or JSON messages small; the sample is.
4. **E4 (MEASURED): on the 10 KB headline windows Redukt is smaller than the best public CM compressor on 6 of 12
   files** (against brotli the table of `CLAUDE.md` says 8 of 12). Margins against paq8px: `dickens` -75 %,
   `enwik8` -56 %, `samba` -87 %, the two invoice XML files -30 % / -32 %, Apache log -11 % (it was -54 % against
   brotli); `nci` +44 %, `mozilla` +33 %, `mr` +51 %, `x-ray` +64 %, `sao` +87 %, `ooffice` +117 %. The registered
   expectation held. On a cold 10 KB window the pretrained prior is worth a lot on natural language and markup and
   little on logs; E2 showed the log and markup margins reverse once the input is 171 KB or more.
5. Limits. (a) Primed payloads are differences of two real archives on both sides. (b) The Mamba primer is 8,192
   bytes because of the context the model can use; a longer primer (up to the ~16K-token drift limit) was not
   tried. (c) `text` and `logs` foreign were not re-run primed (they already KEEP). (d) paq8px and paq8l memory was
   not reduced; whether a small-memory paq8 keeps its ratio on 110 KB of sample is not measured (lpaq1 does).
   (e) E4 uses the current binary with defaults; some sizes differ from the 2026-10-31 table because routing and
   format changed since (e.g. `mr` now 3,191 B). (f) One cut, 16 messages per cell.

**Reading for path 3 (DERIVED).** The claim that survives both checks: Mamba-130M is 25-48 % smaller than any
primed public compressor on short natural-language messages, and 37 % or more smaller on logs when no same-source
sample exists. For traffic that repeats and can be sampled, the advantage over a 9 MB public coder is 13-17 % with
priming (a feature that does not exist in the product) and negative against paq8px. Whether a 130 MB model on the
terminal is worth 13-17 % is a market question (R-116), not a compression one.

### 4.1 What was run (MEASURED)

| Item | Value |
|---|---|
| Primed Mamba | `python -I tools/strata/viability/shortmsg_mamba_primed.py`: `target/maxperf/redukt.exe compress --input-dir ... --jobs 32 --backbone mamba --quant q8 --reset-every 0`, 81 archives per arm (primer + 80 primed messages), 825-923 s per arm on the laptop; rows `mamba130m+same8k` / `+foreign8k` in `data/strata/runs/shortmsg-cm.csv`. Every difference is positive (400 of 400). The archives were not decoded in this check |
| CM, 8 KB primer | `shortmsg_cm.py` (arms `ARMS_8K`): lpaq1 `0` and paq8px with the same primer bytes |
| E4 | `python -I tools/strata/viability/window10k_cm.py` -> `data/strata/runs/window10k-cm.csv` |

### 4.2 E3b, output of `shortmsg_cm_report.py` (MEASURED)

| Class, pairing | Mamba unprimed | Mamba primed | B_cm of E3 (110 KB) | primed / B_cm | Verdict | lpaq1-0, 8 KB primer | paq8px, 8 KB primer |
|---|---|---|---|---|---|---|---|
| logs, same | 17,904 | 8,437 | 6,818 | 1.237 | LOST | 12,713 | 8,879 |
  - log-apache, mean bytes unprimed -> primed (paq8px with 110 KB): 128: 34.8 -> 12.8 (13.6), 256: 49.9 -> 15.4 (16.2), 512: 70.0 -> 21.8 (20.9), 1024: 87.1 -> 32.6 (28.7), 2048: 114.1 -> 55.9 (41.4)
  - log-linux, mean bytes unprimed -> primed (paq8px with 110 KB): 128: 39.3 -> 22.1 (19.6), 256: 48.5 -> 26.9 (20.9), 512: 56.1 -> 31.9 (27.4), 1024: 79.8 -> 51.1 (48.6), 2048: 114.3 -> 77.4 (67.5)
  - log-openssh, mean bytes unprimed -> primed (paq8px with 110 KB): 128: 39.0 -> 15.9 (13.1), 256: 62.1 -> 23.4 (17.4), 512: 87.6 -> 32.5 (22.0), 1024: 107.4 -> 44.9 (29.7), 2048: 129.1 -> 62.7 (39.2)
| json, same | 9,062 | 7,544 | 6,199 | 1.217 | LOST | 13,010 | 10,235 |
  - json-events, mean bytes unprimed -> primed (paq8px with 110 KB): 128: 33.9 -> 21.2 (16.7), 256: 48.4 -> 32.2 (28.2), 512: 92.2 -> 72.9 (55.8), 1024: 149.2 -> 125.1 (98.1), 2048: 242.7 -> 220.2 (188.8)
| json, foreign | 9,062 | 7,696 | 6,871 | 1.120 | LOST | 12,402 | 9,469 |
  - json-events, mean bytes unprimed -> primed (paq8px with 110 KB): 128: 33.9 -> 24.7 (19.4), 256: 48.4 -> 35.2 (32.3), 512: 92.2 -> 74.4 (62.4), 1024: 149.2 -> 126.0 (110.0), 2048: 242.7 -> 220.6 (205.4)

Totals per source, bytes over 80 messages (Apache / Linux / OpenSSH / JSON): Mamba primed 8 KB 2,216 / 3,351 /
2,870 / 7,544; paq8px 8 KB 2,503 / 3,876 / 2,500 / 10,235; lpaq1 `0` 8 KB 3,304 / 5,281 / 4,128 / 13,010; lpaq1 `0`
110 KB 2,486 / 4,270 / 3,423 / 8,685; paq8l 110 KB 2,142 / 3,393 / 2,433 / 7,039; paq8px 110 KB 1,932 / 2,943 /
1,943 / 6,199.

### 4.3 E4, output of `window10k_cm.py` (MEASURED; bytes, first 10,240 bytes of each file)

empty-input archive bytes: {'lpaq1-9': 9, 'paq8l-8': 22, 'paq8px217-8': 10}
| File | redukt-azth | brotli-11 | xz-9e | gzip-9 | lpaq1-9 | paq8l-8 | paq8px217-8 | best CM | Redukt vs best CM | Redukt payload vs CM payload |
|---|---|---|---|---|---|---|---|---|---|---|
| apache-access.log | 281 | 595 | 692 | 698 | 506 | 372 | 315 | paq8px217-8 | -10.8% | -14.4% |
| facturae-batch.xml | 521 | 1,111 | 1,288 | 1,356 | 1,132 | 903 | 765 | paq8px217-8 | -31.9% | -33.6% |
| ubl-invoice-batch.xml | 592 | 1,116 | 1,324 | 1,379 | 1,139 | 967 | 842 | paq8px217-8 | -29.7% | -31.2% |
| nci | 806 | 925 | 1,012 | 1,184 | 891 | 650 | 561 | paq8px217-8 | +43.7% | +42.6% |
| dickens | 840 | 3,551 | 4,544 | 4,622 | 3,761 | 3,543 | 3,321 | paq8px217-8 | -74.7% | -75.2% |
| enwik8 | 1,241 | 3,024 | 3,752 | 3,814 | 3,232 | 3,017 | 2,793 | paq8px217-8 | -55.6% | -56.1% |
| samba | 350 | 2,838 | 3,632 | 3,687 | 2,957 | 2,767 | 2,623 | paq8px217-8 | -86.7% | -87.4% |
| mozilla | 2,518 | 2,522 | 2,600 | 2,992 | 2,695 | 2,139 | 1,889 | paq8px217-8 | +33.3% | +32.9% |
| mr | 3,191 | 2,942 | 2,864 | 3,477 | 2,767 | 2,235 | 2,117 | paq8px217-8 | +50.7% | +50.5% |
| ooffice | 5,284 | 3,574 | 3,588 | 3,793 | 3,587 | 2,740 | 2,432 | paq8px217-8 | +117.3% | +117.3% |
| x-ray | 7,575 | 5,837 | 5,944 | 7,534 | 5,659 | 4,861 | 4,613 | paq8px217-8 | +64.2% | +64.1% |
| sao | 11,718 | 7,651 | 7,624 | 8,176 | 7,836 | 6,546 | 6,282 | paq8px217-8 | +86.5% | +86.5% |

Redukt archive smaller than the best public CM archive: 6 of 12; payload against payload: 6 of 12
