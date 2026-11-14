# Strata backbone — 09a. Arm P on the selection set, extended decision set, classical baselines

> 2026-11-14. CPU-only part of gate V7 (`05-gate-amendments.md` amendment 4; plan `08-capacity-ladder.md` 6.3 and
> steps 1-2 of 6.4, without the Mamba-130M run). Existing G0 weights, no training, no GPU hour.
> Labels: MEASURED (command given), DERIVED (arithmetic on measured values), INFERENCE.
> **No Strata model was evaluated on a decision file.** Every Strata number below is on the four selection sources
> (300,000 bytes each); the decision files received classical codecs only. `08-capacity-ladder.md` was not edited
> (sha256 `1101d57e…` when this was written): section 6 below is the text proposed for its 6.3.

## 0. Result first — weak points before anything else

1. **A general primer as specification 3.4 defines it does not do what arm P hoped (MEASURED, selection).** Best
   configuration (largest legal tables, 1 MiB): selection macro 0.6150 -> 0.5995 bpb for R0 (-2.5 %) and
   0.6135 -> 0.6048 for R1 (-1.4 %). It is still 2.0 % above the best classical codec that has no prior at all
   (0.5879). By source: text -5.2 %, markup **+4.0 %**, Zookeeper -0.3 %, Proxifier **+6.4 %** (R0). On logs, where
   the unprimed `cm` tier wins by 13.4 %, the primer lowers the saving to 10.7 %.
2. **16 MiB buys nothing over 1 MiB.** R0, large tables: 0.5995 (1 MiB) -> 0.5999 (16 MiB). With the default
   tables a 16 MiB primer is worse than no primer (0.6193 -> 0.6214): the successor tables beyond order 3 are 100 %
   full and hold only the last few hundred KB.
3. **The trunk does not help a primed stream.** With large tables R1 is worse than R0 on the macro at every
   primer size (0.6048 against 0.5995 at 1 MiB), while unprimed it is slightly better (0.6135 against 0.6150).
   INFERENCE on the cause: the mixer of the width-88 tier adapts to a trunk that has read megabytes of the primer,
   and the input then starts with the trunk at rest.
4. **Surprise 1 — the harm comes from the primed bit cells (MEASURED, diagnostic, not a specified mode).** A snapshot
   that keeps only the successor tables and starts the calibration cells, the bit cells and the mixer from boot is
   better than the specified snapshot on all four sources: R0 macro 0.5912 (1 MiB) and 0.5891 (16 MiB), with
   markup and Proxifier back at their unprimed cost. This variant is not specification 3.4; registering it needs an
   amendment, and it was found after looking at six variants on four sources (section 4.3).
5. **Surprise 2 — the registered PPMd baseline is under-served on files of 1 MiB and more (MEASURED).** Given a
   file, 7-Zip lowers the PPMd memory to the power of two that holds 16x the file size, whatever `mem=` says (16 MB
   for a 1 MiB file). Through stdin with 1 GB the best classical codec gets smaller on four of the seven extended
   decision files (`nci` -4.3 %, GitHub events -1.1 %, `bible` -0.5 %). The 300-500 KB files of the bake-off are
   almost unaffected. Section 2.2.
6. **Surprise 3 — the first 225,216 bytes of the full OpenSSH log are the frozen-test file `test-openssh`, byte for
   byte.** The slice asked for (`ext-openssh`, first 1 MiB) therefore repeats a frozen-test file; a disjoint
   replacement is proposed and not adopted (section 1).
7. A foreign prior does not help the classical codecs either: zstd -19 with the primer as dictionary stays far
   behind (macro 0.706-0.741), and the conditional PPMd size gets worse as the primer grows (0.5872, 0.5914, 0.6166
   at 64 KiB, 1 MiB, 16 MiB against 0.5879 without prior).
8. Limits of this evidence: four sources of 300 KB, one of them text; the two seeds share every online table (they
   differ in trained weights only), so they are not independent evidence for a table effect; compositions were
   compared at 1 MiB, R0, one seed; the gain on text is concentrated at the start of the input and the decision
   inputs are 3.5x longer (section 4.4).

INFERENCE: on this evidence arm P, as specified, is expected to end O3 on every class. Knowledge held in the tables
of a general corpus is worth about 5 % on text of the same kind and nothing on other systems' logs or on markup.

## 1. Extended decision set (deliverable 1)

`tools/strata/viability/sources.py` (`EXTENDED`). Seven slices of 1,048,576 bytes.

| Name | Class | File | Byte range | sha256[:16] of the slice |
|---|---|---|---|---|
| `ext-dickens` | text | `data/benchmark/silesia/dickens` | [500,000, 1,548,576) | `f719c3e089a03786` |
| `ext-bible` | text | `data/strata/raw/canterbury/bible.txt` | [0, 1,048,576) | `a096ed965b4f9b4d` |
| `ext-apache` | logs | `data/strata/raw/loghub/Apache.log` | [0, 1,048,576) | `fbbc751c6e36a5fb` |
| `ext-linux` | logs | `data/strata/raw/loghub/Linux.log` | [0, 1,048,576) | `4703129ce7be0f88` |
| `ext-openssh` | logs | `data/strata/raw/loghub/OpenSSH.log` | [0, 1,048,576) | `c37e034391f7130d` |
| `ext-gharchive` | structured | `data/strata/raw/gharchive/2015-01-01-15.json` | [0, 1,048,576) | `45a14ab5b0814956` |
| `ext-nci` | structured | `data/benchmark/silesia/nci` | [0, 1,048,576) | `3d234780bf0ef1a8` |
| `alt-openssh-tail` (proposed, not adopted) | logs | `data/strata/raw/loghub/OpenSSH.log` | [951,424, 2,000,000) | `1c7c19c3990eca10` |

MEASURED, `python -I tools/strata/viability/sources.py --fetch` (4.7 s; plain GET, agent `Wget/1.21`, nothing else
sent): `bible.txt` (4,047,392 B) from the Canterbury large corpus archive; the first 2,000,000 bytes of the full
Apache, Linux and OpenSSH logs from the public Loghub record that `bakeoff/fetch_data.py` uses (the OpenSSH log is
the member `SSH` of that record). All downloads succeeded; nothing was substituted.

Role check (read in `bakeoff/common.py`, and enforced by `sources.check_roles()` on file paths):
- training sources are the Wikipedia slice, six Canterbury files (`world192.txt`, `lcet10.txt`, `cp.html`,
  `fields.c`, `grammar.lsp`, `xargs.1`) and eleven Loghub systems (BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android,
  Windows, Thunderbird, OpenStack, HealthApp); selection sources are `webster`, `xml`, Zookeeper, Proxifier;
- Apache, Linux, OpenSSH, `dickens`, `bible.txt`, the GitHub archive and `nci` are none of them. `bible.txt` comes in
  the same archive as the training file `world192.txt`; it is a different document.

Relation to the frozen test set (same role, so no leak, but the two sets are not independent; MEASURED by the same
command, lines of the frozen file that occur in the slice):
- `ext-openssh`: 1,999 of 2,000 lines, and the first 225,216 bytes of the slice equal the whole frozen file
  (compared byte by byte): **the Loghub sample is the head of the full log**. V7 takes the worse of the frozen
  and the extended set, so this slice adds little.
  Proposal: replace it by `alt-openssh-tail` (disjoint; classical baselines already measured below), or fetch more
  of the log and take [1 MiB, 2 MiB). Not done here: the orchestrator decides before registration.
- `ext-apache`: 0 of 2,000 (the sample is from December 2005, the slice starts in June 2005); `ext-linux`: 1 of
  2,000; `ext-dickens`: byte ranges disjoint by construction (1,004 shared lines are blank or two-word lines).
- `ext-gharchive` is the first MiB of the hour whose second half feeds the R-113 messages of `shortmsg.py`.

## 2. Classical baselines on the extended set (deliverable 1)

### 2.1 As registered in amendment 4

MEASURED, `python -I tools/strata/viability/baselines_ext.py` (7-Zip 26.03; 39 s) ->
`data/strata/runs/baselines-ext.csv`. bpb; PPMd = best of orders 4-32, file mode, 256 MB, as `ppmd_baseline.py`.

| Slice | xz -9e | zstd -19 / -22 | brotli -11 | bz2 -9 | PPMd (order) | Best as registered | bytes |
|---|---|---|---|---|---|---|---|
| `ext-dickens` | 2.4781 | 2.5046 / 2.5046 | 2.4320 | 2.2913 | 1.9984 (8) | **1.9984** | 261,928 |
| `ext-bible` | 1.6782 | 1.6913 / 1.6913 | 1.6741 | 1.5685 | 1.3455 (12) | **1.3455** | 176,360 |
| `ext-apache` | 0.3838 | 0.4279 / 0.4269 | 0.3923 | 0.3897 | 0.3318 (8) | **0.3318** | 43,485 |
| `ext-linux` | 0.3471 | 0.3993 / 0.3992 | 0.3668 | 0.5034 | 0.4069 (16) | **0.3471** | 45,492 |
| `ext-openssh` | 0.2304 | 0.3041 / 0.3039 | 0.2763 | 0.3492 | 0.2553 (8) | **0.2304** | 30,204 |
| `ext-gharchive` | 0.8196 | 0.8638 / 0.8644 | 0.7603 | 0.8947 | 0.7012 (16) | **0.7012** | 91,902 |
| `ext-nci` | 0.4927 | 0.5380 / 0.5374 | 0.4994 | 0.4946 | 0.4757 (24) | **0.4757** | 62,352 |
| `alt-openssh-tail` | 0.2464 | 0.3109 / 0.3109 | 0.2728 | 0.3395 | 0.2441 (8) | 0.2441 | 31,998 |

### 2.2 PPMd with the memory it was asked for

Found while measuring the conditional sizes of section 4: C(primer + input) came out smaller than C(primer). Cause
(MEASURED: a 1,048,576-byte file gives 207,300 B at order 16, the same bytes plus ten give 196,412 B; `mem=256m`,
`1g` and `2g` give identical sizes): with a file argument 7-Zip reduces the PPMd model memory to the power of two
that holds 16 times the file size, the model restarts at high orders, and the result is worse. Through stdin
(`-si`) the size is unknown and the memory is not reduced. Rows `ppmd-oN-1g` of `baselines-ext.csv`, and
`python -I tools/strata/viability/baselines_ext.py --frozen` -> `data/strata/runs/ppmd-baseline-1g.csv` for the
bake-off's sets.

| File | Best classical as registered | PPMd, stdin, 1 GB (order) | Best classical with it | Change |
|---|---|---|---|---|
| `ext-dickens` | 1.9984 (261,928 B) | 1.9981 (32) | **1.9981** (261,889 B) | -0.01 % |
| `ext-bible` | 1.3455 (176,360 B) | 1.3393 (24) | **1.3393** (175,547 B) | -0.5 % |
| `ext-apache` | 0.3318 (43,485 B) | 0.3318 (8) | 0.3318 | 0 |
| `ext-linux` | 0.3471 (xz) | 0.4069 (16) | 0.3471 | 0 |
| `ext-openssh` | 0.2304 (xz) | 0.2553 (8) | 0.2304 | 0 |
| `ext-gharchive` | 0.7012 (91,902 B) | 0.6935 (32) | **0.6935** (90,902 B) | -1.1 % |
| `ext-nci` | 0.4757 (62,352 B) | 0.4551 (32) | **0.4551** (59,649 B) | -4.3 % |
| selection (4), `test-dickens`, `test-apache`, `test-openssh`, `test-ubl` | | | unchanged | 0 |
| `test-linux` | 0.3697 (xz) | 0.4011 (was 0.4039) | 0.3697 | 0 |
| `test-facturae` (no vote) | 0.3845 | 0.3794 (32) | 0.3794 | -1.3 % |
| `seen-enwik8` (seen) | 2.0326 | 2.0180 (16) | 2.0180 | -0.7 % |

Proposal (stricter, so allowed by T4, but a change of the registered classical set: it needs its line in the
amendment log before the decision): PPMd is run through stdin with 1 GB. The position at registration of
`08-capacity-ladder.md` 3 does not move (its sources are unchanged), the extended set does.

## 3. Arm P harness (deliverable 2)

Code, all new, nothing under `tools/strata/bakeoff/` touched:
- `tools/strata/viability/primer_long.py` — `prime`, `code`, `verify`, `occupancy`;
- `tools/strata/viability/primer_baselines.py` — `classical` (same-prior baselines), `summary` (tables below).

### 3.1 What is computed

- **Snapshot** exactly as specification 3.4: the stream machine reads the primer from boot (successor tables,
  time runs, calibration cells, bit cells, online mixer, trunk); the snapshot keeps the successor tables, the
  calibration cells, the bit cells and the mixer weights; the input then starts with the trunk, the delay taps, the
  history, the byte count and the time runs at boot.
- **Inputs are coded by the frozen simulation** (`bakeoff/models.py` trunk and `Head.tier`, the code `evaluate.py`
  runs); reported bits are ideal code lengths of its integer probabilities. The online features are recomputed by
  `primer_long.Online` because `bakeoff/common.py` builds them from boot only, in one piece, at fixed table sizes;
  `Online` carries the state across chunks and takes the table bits as configuration.
- While the primer is read, the frozen trunk runs unchanged and the mixers of both seeds and both tiers advance in
  one batched copy of the integer branch of `Head.mix_loop` (`prime_chunk`). The first 2.4 MiB of the long run were
  produced by the unbatched frozen path; the two paths gave identical snapshots at 64 KiB for both seeds (all
  twelve arrays equal, one-off comparison, not kept as a command).
- Decision files are refused: `code --inputs test-dickens` and `--inputs ext-bible` both stop with "refusing to
  code …: not a selection source" unless `--allow-decision` is given (MEASURED; the flag was never passed).

### 3.2 Checks (MEASURED)

| Check | Command | Result |
|---|---|---|
| primed stream vs sequential integer reference `ref_int.py`, default tables | `primer_long.py verify --run strata_g_bc --tables D` | 6,000 B primer + 3,000 B of `sel-webster`, chunks of 1,000: 24,000 probabilities per tier, **0 mismatches** (R0 and R1) |
| same, longer, chunk = 65,536 | `… verify --run strata_g_bc --tables D --primer-bytes 200000 --bytes 20000 --chunk 65536 --source sel-xml` | 160,000 probabilities per tier, **0 mismatches** |
| same, large successor tables, second seed | `… verify --run strata_g_bc_s1205 --tables V --primer-bytes 40000 --bytes 6000 --chunk 777 --source sel-zookeeper` | 48,000 probabilities per tier, **0 mismatches** |
| no primer reproduces the bake-off | `… code --run <run> --tables D --size 0` vs `eval-select-on.csv` | the 16 bit totals (2 seeds x 2 tiers x 4 sources) are identical to the digit |

Not covered: tables `L` use the exact order-2 bit-cell index that the specification prescribes at 24 bits
(`idx = (c2 << 16) | (c1 << 8) | node`); `ref_int.py` only has the hashed form, so the reference check uses `V`
(same successor tables, 23 bits for that order). The exact-index line is unverified by an independent implementation.

### 3.3 Primer composition

`build_primer(comp, size)`; training sources only (any other role is refused in code). Blocks of whole lines of at
most 4,096 bytes; each source is read forward from its first byte.

| Name | Content |
|---|---|
| `mixI` | pattern `TLTLTLTLMTLTLTLTLTLM` repeated: 9 Wikipedia blocks, 9 log blocks (the 11 training systems in turn), 2 blocks of the six Canterbury files concatenated — the 45 / 45 / 10 mix of training. Every prefix is a primer of the same composition, so one pass gives all sizes |
| `catTL`, `catLT` | the same bytes as `mixI`, grouped: text, misc, logs — or logs, misc, text |
| `text`, `logs` | Wikipedia only; the 11 log systems only |

### 3.4 Table bits and state

MEASURED, `primer_long.py occupancy --run strata_g_bc --tables D|L --size 65536,1048576,16777216`.

| Tables | Successor bits (orders 1,2,3,4,6,8,12,16) | Bit-cell bits (orders 0-3) | State per stream | Successor slots valid after 64 KiB / 1 MiB / 16 MiB (orders 4-16) |
|---|---|---|---|---|
| `D` (v1 default) | 8/16/16/16/18/18/18/18 | 8/16/22/22 | 38.8 MB | 11-25 % / 64-91 % / **100 %** |
| `L` (largest legal, 2.1) | 8/16/22/22/22/22/22/22 | 8/16/24/24 | **235.4 MB** | 0.4-1.1 % / 2-14 % / 9-86 % |

- `D` saturates between 1 and 16 MiB (always-replace: the tables then hold the most recent contexts only). `L` is
  the largest configuration the specification allows (`tb <= 22`, `btb <= 24`) and is 79-86 % full at orders 12-16
  after 16 MiB. 16 MiB is also the largest primer the specification allows (`len(P) <= 2^24`), so "the largest the
  table state allows" and "16 MB" are the same point here.
- Bit cells: after 16 MiB 79 % of the order-0 cells and 8 % of the order-1 cells have a saturated count (255), i.e.
  they adapt at rate 1/128 from the first byte of the input.
- Larger tables alone, no primer: macro 0.6193 -> 0.6150 (R0), 0.6178 -> 0.6135 (R1), -0.7 %.
- The cost of a 235 MB state (cache misses per byte, snapshot copy per stream, lockstep width) is NOT measured.

### 3.5 Timings (MEASURED on the i5-1235U; Python, one thread per process)

| Step | Time |
|---|---|
| prime 64 KiB, both seeds and both tiers in one process, machine otherwise idle | 8.7 s (133 us per primer byte) |
| prime 1 MiB, one seed, two tiers, unbatched path, four such jobs in parallel | 4.2-4.3 min |
| prime 1 MiB, batched path | 2.3 min (DERIVED from the 64 KiB rate) |
| prime 16 MiB, both seeds and tiers, per table size (two such processes plus one or two other jobs in parallel) | 50.4 min (`D`), 51.0 min (`L`) of process time, in about 20 invocations |
| prime 16 MiB, batched path, idle machine | about 37 min (DERIVED from the 64 KiB rate; not measured) |
| code 4 x 300,000 bytes from a snapshot, two tiers | 70-120 s |
| same-prior classical baselines, all sizes | about 7 min |

Snapshots are cached in `data/strata/cache/primer/` (compressed: 1.2-9.9 MB up to 1 MiB; 6.5 MB for `D` and
38.3 MB for `L` at 16 MiB; 181 MB in total with the exploration snapshots).

## 4. Results on the selection set

Commands (every one resumable; `prime` and `classical` print DONE when finished):

```
python -I tools/strata/viability/primer_long.py prime --run strata_g_bc,strata_g_bc_s1205 --comp mixI --tables D
python -I tools/strata/viability/primer_long.py prime --run strata_g_bc,strata_g_bc_s1205 --comp mixI --tables L
for run in strata_g_bc strata_g_bc_s1205; for tables in D L:
  python -I tools/strata/viability/primer_long.py code --run $run --comp mixI --tables $tables \
         --size 0,65536,1048576,16777216
python -I tools/strata/viability/primer_baselines.py classical
python -I tools/strata/viability/primer_baselines.py summary
```

Outputs: `data/strata/runs/arm-p-selection.csv` (128 rows), `arm-p-classical.csv`, `arm-p-explore.csv`,
`arm-p-ablate.csv` (and the per-run files they are merged from).

### 4.1 Main table (MEASURED; primer `mixI`; Strata = mean of seeds 1204 and 1205, ideal bits, bpb)

"Best classical, no prior" = smallest of xz, zstd -19 / -22, brotli, bz2, PPMd on the input alone
(`baselines.csv`, `ppmd-baseline.csv`). zstd + dictionary = zstd -19, primer as raw-content dictionary, window
covering primer + input (deployable). The two conditional columns are C(primer + input) - C(primer), a stricter
reference that is **not a deployable mode** (PPMd through stdin, 1 GB, best order; xz -9e).

| Source | Primer bytes | R0 `D` | R0 `L` | R1 `D` | R1 `L` | Best classical, no prior | zstd -19 + dict | xz conditional | PPMd conditional |
|---|---|---|---|---|---|---|---|---|---|
| `webster` | 0 | 1.7388 | 1.7240 | 1.7294 | 1.7151 | 1.6681 | - | - | - |
| `webster` | 65,536 | 1.7074 | 1.6881 | 1.7102 | 1.6916 | 1.6681 | 2.0758 | 2.0300 | 1.6417 |
| `webster` | 1,048,576 | 1.6926 | **1.6338** | 1.7057 | 1.6501 | 1.6681 | 1.9742 | 1.9355 | 1.6136 |
| `webster` | 16,777,216 | 1.7109 | 1.6368 | 1.7312 | 1.6639 | 1.6681 | 1.9387 | 1.8900 | 1.6534 |
| `xml` | 0 | 0.3872 | 0.3863 | 0.4047 | 0.4036 | 0.2778 | - | - | - |
| `xml` | 65,536 | 0.3956 | 0.3938 | 0.4116 | 0.4096 | 0.2778 | 0.3589 | 0.3311 | 0.2876 |
| `xml` | 1,048,576 | 0.4052 | 0.4018 | 0.4184 | 0.4146 | 0.2778 | 0.3586 | 0.3299 | 0.3011 |
| `xml` | 16,777,216 | 0.4099 | 0.4045 | 0.4233 | 0.4179 | 0.2778 | 0.3583 | 0.3319 | 0.3182 |
| Zookeeper | 0 | 0.1490 | 0.1484 | 0.1381 | 0.1377 | 0.1856 | - | - | - |
| Zookeeper | 65,536 | 0.1483 | 0.1476 | 0.1404 | 0.1398 | 0.1856 | 0.2347 | 0.1956 | 0.1888 |
| Zookeeper | 1,048,576 | 0.1495 | 0.1480 | 0.1436 | 0.1420 | 0.1856 | 0.2352 | 0.1957 | 0.2023 |
| Zookeeper | 16,777,216 | 0.1476 | 0.1424 | 0.1440 | 0.1405 | 0.1856 | 0.2342 | 0.1977 | 0.2189 |
| Proxifier | 0 | 0.2023 | 0.2015 | 0.1990 | 0.1979 | 0.2201 | - | - | - |
| Proxifier | 65,536 | 0.2077 | 0.2068 | 0.2055 | 0.2042 | 0.2201 | 0.2937 | 0.2822 | 0.2305 |
| Proxifier | 1,048,576 | 0.2153 | 0.2144 | 0.2138 | 0.2125 | 0.2201 | 0.2929 | 0.2791 | 0.2484 |
| Proxifier | 16,777,216 | 0.2170 | 0.2157 | 0.2163 | 0.2148 | 0.2201 | 0.2923 | 0.2802 | 0.2758 |
| **macro** | 0 | 0.6193 | 0.6150 | 0.6178 | 0.6135 | 0.5879 | - | - | - |
| **macro** | 65,536 | 0.6147 | 0.6091 | 0.6169 | 0.6113 | 0.5879 | 0.7408 | 0.7097 | 0.5872 |
| **macro** | 1,048,576 | 0.6157 | **0.5995** | 0.6203 | 0.6048 | 0.5879 | 0.7152 | 0.6851 | 0.5914 |
| **macro** | 16,777,216 | 0.6214 | 0.5999 | 0.6287 | 0.6093 | 0.5879 | 0.7059 | 0.6750 | 0.6166 |

The two seeds agree to 0.0002 bpb on R0 and to 0.0084 on R1 in every cell (per-run rows in
`arm-p-selection.csv`).

Reading (DERIVED from the table):
- Text (`webster`), best cell R0 `L` 1 MiB: -5.2 % against the unprimed stream, **-2.1 % against the best classical
  codec without prior**, and +1.3 % above the conditional PPMd reference. The materiality threshold of V7 is 10 %.
- Logs (Zookeeper + Proxifier, mean): unprimed R0 `D` 0.1757 = -13.4 % against classical; R0 `L` with 1 MiB
  0.1812 = -10.7 %; with 16 MiB 0.1791 = -11.7 %. Unprimed R1 `D` 0.1686 = -16.9 %; R1 `L` with 1 MiB 0.1773 =
  -12.6 %. The primer costs 2.7-4.3 points of saving.
- Markup: +39 % against classical without primer, +45 % to +48 % with one.
- The Python brotli binding (1.2.1) has no dictionary argument (`compress(string, mode, quality, lgwin, lgblock)`),
  so brotli has no same-prior row.

### 4.2 Composition (MEASURED; R0, seed 1204, 1 MiB; `arm-p-explore.csv`)

`primer_long.py prime --run strata_g_bc --comp C --tables T --tiers 0 --sizes 1048576`, then
`… code --run strata_g_bc --comp C --tables T --tiers 0 --size 1048576 --out arm-p-explore.csv`.

| Primer | Tables | `webster` | `xml` | Zookeeper | Proxifier | macro |
|---|---|---|---|---|---|---|
| none | `D` | 1.7387 | 0.3872 | 0.1491 | 0.2024 | 0.6194 |
| `mixI` | `D` | 1.6926 | 0.4051 | 0.1496 | 0.2154 | 0.6157 |
| `catTL` | `D` | 1.6837 | 0.4045 | 0.1488 | 0.2150 | 0.6130 |
| `catLT` | `D` | 1.6831 | 0.4043 | 0.1500 | 0.2137 | 0.6128 |
| `text` | `D` | 1.6957 | 0.4017 | 0.1525 | 0.2108 | 0.6152 |
| `logs` | `D` | 1.7908 | 0.4057 | 0.1489 | 0.2150 | 0.6401 |
| none | `L` | 1.7239 | 0.3863 | 0.1485 | 0.2016 | 0.6151 |
| `mixI` | `L` | 1.6337 | 0.4018 | 0.1480 | 0.2144 | 0.5995 |
| `catTL` | `L` | 1.6385 | 0.4012 | 0.1473 | 0.2136 | 0.6002 |
| `catLT` | `L` | 1.6291 | 0.4014 | 0.1488 | 0.2137 | 0.5982 |
| `text` | `L` | 1.6591 | 0.3980 | 0.1513 | 0.2095 | 0.6045 |
| `logs` | `L` | 1.7684 | 0.4031 | 0.1475 | 0.2135 | 0.6331 |

- The three orderings of the same bytes are within 0.5 % of one another; `catLT` is nominally best in both table
  sizes (0.2-0.5 % below `mixI`), which is inside what one seed and four sources can resolve.
- A megabyte of logs from eleven other systems does nothing for two unseen systems (Zookeeper -0.1 % to -0.7 %,
  Proxifier +6 %) and costs 2.6-3.0 % on text. Same finding as R-112 by a different route.
- `mixI` is chosen: not distinguishable from the best ordering, and the only one whose sizes are prefixes of one
  another (one pass, no dependence on which part comes last at 16 MiB, where `D` keeps only the tail).

### 4.3 Which part of the snapshot does what (MEASURED, diagnostic; `arm-p-ablate.csv`)

`… code … --keep PARTS --out arm-p-ablate.csv`: the parts not listed start from boot. Only "all four" is
specification 3.4. Tables `L` unless stated; R0; seeds as stated.

| Kept from the snapshot | Primer | Seeds | `webster` | `xml` | Zookeeper | Proxifier | macro |
|---|---|---|---|---|---|---|---|
| nothing (no primer) | 0 | 2 | 1.7240 | 0.3863 | 0.1484 | 0.2015 | 0.6150 |
| all four (specification 3.4) | 1 MiB | 2 | 1.6338 | 0.4018 | 0.1480 | 0.2144 | 0.5995 |
| all but the mixer | 1 MiB | 1 | 1.6404 | 0.4050 | 0.1507 | 0.2177 | 0.6034 |
| all but the calibration cells | 1 MiB | 1 | 1.6343 | 0.4010 | 0.1480 | 0.2140 | 0.5993 |
| all but the bit cells | 1 MiB | 2 | 1.6295 | 0.3878 | 0.1470 | 0.2045 | 0.5922 |
| all but the successor tables | 1 MiB | 1 | 1.6997 | 0.4003 | 0.1475 | 0.2104 | 0.6145 |
| successor tables only | 64 KiB | 2 | 1.6868 | 0.3857 | 0.1470 | 0.2019 | 0.6054 |
| successor tables only | 1 MiB | 2 | 1.6307 | 0.3854 | 0.1456 | 0.2032 | 0.5912 |
| successor tables only | 16 MiB | 2 | **1.6281** | **0.3850** | **0.1412** | 0.2022 | **0.5891** |
| successor tables only, tables `D` | 1 MiB | 2 | 1.6898 | 0.3867 | 0.1471 | 0.2038 | 0.6068 |
| successor tables only, tables `D` | 16 MiB | 1 | 1.7010 | 0.3867 | 0.1471 | 0.2037 | 0.6096 |

R1 with successor tables only, `L`: 0.6059 / 0.5938 / 0.5932 at 64 KiB / 1 MiB / 16 MiB (R1 unprimed 0.6135).

- The gain on text is in the successor tables (dropping them loses it: 1.6997); the loss on markup and Proxifier
  is in the bit cells (dropping them removes it). The primed mixer helps a little; the calibration cells are neutral.
- INFERENCE on the cause: a primed bit cell carries a saturated count, so it moves at rate 1/128 from the first
  byte, while an unprimed cell starts at rate 1/4 and learns the input's low-order statistics within a few
  visits. A foreign prior with full confidence is worse than no prior. Not tested: keeping the cell
  probabilities with reduced counts.
- "Successor tables only" at 16 MiB is 0.2 % above the best classical macro without prior (0.5891 against 0.5879):
  a tie, not a win; on text it is -2.4 % against PPMd; on logs -15.4 % (unprimed -13.4 %).
- Caveat: this variant was picked after six variants were looked at on these four sources, and both seeds share
  the tables, so its advantage over the specified snapshot (1.4-1.8 % macro) is selection-fitted to some degree.

### 4.4 Where in the input the gain is (DERIVED from the `bpb@` columns; R0, `L`, mean of seeds)

| `webster` | first 10,240 B | 10,240-102,400 | 102,400-300,000 | all |
|---|---|---|---|---|
| no primer | 3.0303 | 1.9697 | 1.5416 | 1.7240 |
| `mixI` 1 MiB (3.4) | 1.6174 | 1.8781 | 1.5207 | 1.6338 |
| `mixI` 16 MiB (3.4) | 2.1201 | 1.8672 | 1.5043 | 1.6368 |
| successor tables only, 16 MiB | 2.2511 | 1.8515 | 1.4916 | 1.6281 |

The primer saves 47 % on the first 10 KB, 4.7 % on the next 90 KB and 1.4 % on the last 200 KB (3.2 % for the
tables-only variant at 16 MiB). The decision inputs are 1 MiB, so the relative gain there is expected to be
smaller than on these 300 KB inputs (INFERENCE). The 16 MiB primers are worse than the 1 MiB one on the first
10 KB in both variants; not investigated.

## 5. Proposed arm-P configuration to register (selection data only; not run on any decision file)

| Item | Value |
|---|---|
| Mechanism | specification 3.4 unchanged (snapshot keeps tables, calibration cells, bit cells, mixer) |
| Primer | `mixI` of `primer_long.build_primer`: 4,096-byte line-aligned blocks, pattern `TLTLTLTLMTLTLTLTLTLM` over Wikipedia slice / 11 training log systems in turn / 6 Canterbury files; prefixes |
| Sizes | 0; 65,536; 1,048,576; 16,777,216 (= the largest primer of 3.4). **Primary point: 1,048,576** |
| Table bits, primed rungs | `L`: successor 8/16/22/22/22/22/22/22, bit cells 8/16/24/24; 235.4 MB of state per stream (reported next to every result) |
| Reference rows | size 0 with `D` (the G0 tiers as trained and evaluated so far) and size 0 with `L` (separates table size from primer) |
| Rungs, seeds | R0 (`cm`) and R1 (width 88) of `strata_g_bc`, seeds 1204 and 1205, G0 weights as they are |
| Same-prior competitors | zstd -19 and -22 with the primer as raw-content dictionary; conditional PPMd and xz reported separately; brotli has none |

Justification (all from section 4): `L` is the only table size at which a megabyte or more of primer is kept
(`D` is 64-91 % full at 1 MiB and loses to no primer at 16 MiB); 1 MiB is the smallest size that reaches the best
macro and 16 MiB does not improve on it, so the primary point is 1 MiB and the others are reported; `mixI` is not
distinguishable from the best ordering and nests. One primary point per rung keeps the verdict from being the
best of several sizes.

Expected outcome, stated before the measurement (INFERENCE from section 4): O3 for arm P on text (gain about 2 %
against PPMd, less on 1 MiB inputs), and a smaller saving than the unprimed rung on logs.

Decision for the orchestrator, before registration and before any decision file is read by a Strata model:
whether to register, next to the configuration above, the tables-only snapshot of section 4.3 as a second,
separately labelled arm (same primer, same sizes, same tables). It is better on all four selection sources, it is
not specification 3.4 (so it needs its amendment), and it was chosen post hoc on selection data. On selection it
does not reach materiality on text either (-2.4 %).

## 6. Text proposed for `08-capacity-ladder.md` section 6.3

| Item | State |
|---|---|
| General primer | `mixI` (`tools/strata/viability/primer_long.py`), training sources only; sizes 0 / 65,536 / 1,048,576 / 16,777,216; primary point 1,048,576; tables `L` (successor 8/16/22x6, bit cells 8/16/24/24; 235.4 MB per stream) for primed rungs, `D` and `L` at size 0. Selection evidence: `09a-arm-p-selection.md` |
| Classical competitors with the same prior | zstd -19 / -22 with the primer as raw-content dictionary (window >= primer + input); brotli: the binding has no dictionary; PPMd and xz: conditional size reported separately (`primer_baselines.py`) |
| Extended decision set | `tools/strata/viability/sources.py` `EXTENDED` (7 x 1 MiB); OpenSSH slice to be decided (section 1 of 09a); classical baselines `data/strata/runs/baselines-ext.csv` |
| PPMd | to be decided: stdin with 1 GB instead of the file mode (09a section 2.2) |
| Mamba-130M on identical bytes | not done in this task |

## 7. Limitations and open problems

1. `ext-openssh` contains the frozen-test OpenSSH file (section 1). Open.
2. Registered PPMd baseline under-served on files of 1 MiB and more (section 2.2). Open: amendment line.
3. Arm P as specified shows no material gain on selection; the tables-only variant needs a decision (section 5).
4. Selection set: four sources, 300 KB, one text source whose kind (dictionary prose) is close to the primer's
   Wikipedia text. No held-out prose of another kind, no JSON, no second markup source was available in that role.
5. Seeds differ only in trained weights; every table effect is measured once, not twice.
6. Compositions compared at one size, one rung, one seed; block size (4,096), the 45 / 45 / 10 mix and the
   position of each source's cursor were not swept.
7. Table sizes between `D` and `L` were not measured; `L` costs 6x the state and its speed is unknown. The
   "1 us per byte of table work" of specification 7.2 is a hypothesis for `D` and less safe for 235 MB.
8. The exact order-2 bit-cell index at 24 bits has no independent check (section 3.2).
9. Ideal code lengths, not archives: no range coder, no header, no `PRIMER_ID` (4 B) — negligible at 300 KB.
10. Conditional sizes depend on the codec's memory and order; PPMd was given 1 GB and its best order per cell.
11. Mamba-130M on the identical bytes (step 2 of 6.4) is not part of this task.
12. Process: the harness moves a shell command to the background after 180 s. The first priming round (7.5 min)
    was moved that way, finished normally (exit 0, no process left); every later command was kept under 180 s.
    The priming job is resumable in pieces of any length (`--minutes`).
