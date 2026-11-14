# Strata backbone — 04. Adversarial design review (stage 3 of `00-charter.md`)

> Date: 2026-11-13. Reviewer: adversarial reviewer, not a co-author of the design.
> Reviewed: `docs/api/BACKBONE-STRATA-SPEC.md` (draft, not frozen), `03-bakeoff.md`, `05-gate-amendments.md`,
> `tools/strata/bakeoff/CRITERION.md`; cross-checked against `00-charter.md`, `01-integration-map.md`,
> `02-evidence-and-constraints.md`, `07-legacy-fixes.md`, `tools/strata/bakeoff/*.py`, `src/domain/coder.rs`,
> `src/application/codec.rs` and the result files under `data/strata/runs/` (read only).
> The documents changed while the review ran: `05-gate-amendments.md` gained the G0 pre-registration (08:44) and the
> specification was edited at 08:59 (bit-cell defaults 8 / 16 / 22 / 22, 38.8 MB per stream). This review covers
> the files as they stood at 09:05; spec line numbers refer to that version.
> Nothing was trained, built or committed; `cargo` was not run; the only file written in the repo is this one. The
> architect's G0 run was not touched (`data/strata/runs/strata_g_bc_smoke/` was not opened).
> Labels: **VERIFIED** = recomputed by this review from files on disk (scripts in section 9); **PROBE** = observed in
> an independent implementation written from the specification text alone for this review, by an agent that did not
> read `tools/`, `src/` or any other document (scratch directory, not part of the repo); **INFERENCE** = reasoning,
> not executed.

Severity scale:
- **Blocking**: must be fixed before the stage this review gates (freeze, then three independent implementations):
  either the teams cannot produce bit-identical output from the text, or the freeze criterion is ambiguous.
- **Grave**: must be fixed before the freeze; otherwise a gate or a reported conclusion misleads, or a less common
  path diverges between implementations.
- **Minor**: clarity, robustness, reporting; fix when convenient.

---

## 0. Summary

**Counts: 4 blocking, 4 grave, 21 minor.** All four blocking items are text fixes; none needs a new measurement.

1. **The arithmetic core holds up.** An implementation written from the text alone (PROBE) matched all 51 spot values
   of Appendix A (VERIFIED by re-running it), reproduced all 30 derived cells of table 7.2, and round-tripped
   archives of both tiers with and without a primer; the stream machine of sections 2.3-2.5 and 3 needed no guess.
   The worst-case bounds of 2.7 hold with large margins (section 3); the restated range coder matches
   `src/domain/coder.rs` line for line; a hand derivation of the first coded bit needs nothing outside the text
   (section 4). The boundedness proof and the coalescence lemma of 2.6 are correct, with minor slips (m-1, m-3).
2. **Archives and weight files are not yet independently implementable (blocking B-2, B-3, B-4).** The order of
   blocks and elements in a STRATUM is unstated and a misreading passes every CRC (B-2); the decoder's initial byte
   order and the end of STREAM are unstated, and reading the CRC trailer as stream data breaks 8 of 320 archives
   (B-3, VERIFIED); the container defers to texts outside the document — preprocessing with automatic
   content detection on by default, the archive CRC, empty input — so gate C1b cannot be met by a spec-only oracle
   (B-4).
3. **The freeze criterion exists in two disagreeing texts (blocking B-1).** Spec 9.0 still lets the seen-source
   Wikipedia slice vote and leaves "data type" undefined. The pre-registration written at 08:45 in
   `05-gate-amendments.md`, applied by `g0_decide.py`, fixes that and adds a short-message metric `S` — the segment
   where the trunk's measured value is largest — but decides a whole subsystem on one seed and sums `S` in bytes.
4. **Two reported numbers are wrong on disk.** Primed short-message archives carry `PRIMER_ID`, so their framing is
   24 B (compact) or 28 B (default), not 20 B: the record against zstd-19 + dictionary is 22 / 30 and 20 / 30 cells,
   not 24 / 30, and 17 / 25 and 15 / 25 without the five seen-source Wikipedia cells (G-1). The "frozen test macro"
   averages the seen-source `enwik8` column into the held-out sources: held-out only, the candidate is 1.9 % (not
   3.5 %) behind the GRU and zeroing the trunk costs 23.6 % (not 29 %) (m-5). Five other figures check exactly.
5. **Economics.** With the prototype's measured ratios no tier reaches a storage break-even: text loses to bz2 at any
   speed; logs need 250-890 KB/s per vCPU at $0.014/vCPU-h for a value of $0.004-0.015 per raw GB. Path 3 has no
   throughput budget, but its ratio gate (V3) and footprint gate (V6) can pass on different files: the 30 / 30 result
   used 5 MB of successor tables and no bit cells; the 0.64 MB messaging file has never been measured (G-2).
6. **Trunk.** On long streams from unseen sources the trunk is worth about nothing once bit cells exist; on short
   messages with a primer it is worth 4-27 % on held-out sources. The 08:45 G0 measures both; section 7.4 lists what
   it still lacks to settle the question.

**Verdict.** Today the specification is independently implementable for probabilities, not for weight files and
archives. After B-2, B-3, B-4 and G-3 are fixed in the text (and m-2, m-19 for robustness), three teams can produce
bit-identical output from the document alone: the probe already did, after six guesses that the text must now make
explicit. Whether
the design deserves that effort is G0's question, which needs one normative text first (B-1).

---

## 1. Findings, ordered by severity

| ID | Severity | Axis | Finding (one line) | Evidence | Correction |
|---|---|---|---|---|---|
| B-1 | blocking | D, F | G0 defined twice and inconsistently (spec 9.0 vs `05` pre-registration); the governing text decides on one seed and sums `S` in bytes | spec 974-984; `05` 13-86; `g0_decide.py` | one normative text; second seed near the bar; per-cell `S` |
| B-2 | blocking | A | Weight-file layout unstated: block order per layer, element order of `[a][b]` blocks, `KF`/`KS` order; misreadings load silently | spec 571, 596-599 (PROBE A3-A5) | explicit byte sequence; row-major; synthetic file + SHA-256 |
| B-3 | blocking | A | Decoder start (byte order of `code`) and end of STREAM unstated; reading the CRC as stream data breaks 8 / 320 archives | spec 70, 479, 497, 619 (PROBE A1-A2, VERIFIED) | write the formula; STREAM = header end .. length − 4 |
| B-4 | blocking | A, C | Container defers to texts outside the spec: preprocessing (auto by default), archive CRC, primer bytes, empty input, coder precedence | spec 15-18, 459, 617-629, 718, 745, 997; `codec.rs:839-935` | v1 forbids preprocessing; define CRC, empty input, precedence |
| G-1 | grave | D | Primed short messages counted with 20 B framing: real 24 / 28 B gives 22 / 20 of 30 (17 / 15 of 25 held-out); V3 counts seen-source cells | spec 417-420, 511-516, 1017; `shortmsg.py:122` | correct text; V3 on held-out cells, real header |
| G-2 | grave | E | V3 (ratio) and V6 (footprint) can pass on different files; the messaging file was never measured | spec 394-396, 822-824, 1017, 1020 | one messaging file passes both; sweep tables |
| G-3 | grave | A, C | Default tier of a truncated weight file undefined (terminal deployment form) | spec 531-533, 547, 713 (PROBE A7) | reject-or-clamp rule; test in C9 |
| G-4 | grave | D | Long-run exclusions omit GH Archive (V3's JSON source), the Samba project, Gutenberg | spec 923-925 | list excluded sources by source |
| m-1 | minor | B | Taps agree after 4 commits, not 3; `n` is a stored counter of unspecified width; M = 2,127 not 2,126 | spec 116, 239-240, 258, 266 | saturate `n`; fix wording |
| m-2 | minor | A, B | Reader MUST-reject list incomplete (biases, `W0`, `DEFAULT_TIER`, names): i32 wrap vs int64 | spec 279, 609, 700 (PROBE A13) | one range table, tested by C9 |
| m-3 | minor | B | Chunk-length MUST (21.6-28.4 KB) violated by bake-off and G0 (8,192 B) and by the long run's short chunks | spec 895-896; `05` 25-27 | `M` from the schedule; long chunks only |
| m-4 | minor | C | PROFILE 0x10 should be written explicitly; the RWKV quant decision (07 §7) is left open, not blocked | spec 625; `07` 406-414 | per-backbone PROFILE table |
| m-5 | minor | D | Test "macro" averages seen-source `enwik8`: +1.9 % vs GRU (not 3.5 %), trunk zeroed +23.6 % (not 29 %) | `03` 22-25, 233-252; spec 314 | correct the tables |
| m-6 | minor | C | API has one `--weights` for every backbone | spec 739-744; `01` §8 | Strata-specific flag |
| m-7 | minor | C | `TIER_ID` covers the whole header, not only tier k | spec 606-608 | state it or redefine |
| m-8 | minor | D | Gutenberg etexts trained on (`world192`, `lcet10`); PG boilerplate in `webster` / `dickens` | `common.py:63-64`; `03` 116 | drop or strip PG etexts |
| m-9 | minor | D | G0b compares two implementations by the same author | spec 982; `05` 53-55 | not T2 evidence |
| m-10 | minor | D | D1 (gears) rests on `webster` and one seed; ratio means and the test set favour `strata_f` | runs CSV (VERIFIED) | re-test or keep optional |
| m-11 | minor | D | V1 counts the synthetic XML; V2 compares CRLF bytes with LF figures | spec 1015-1016; `02` E.1 | report apart; identical bytes |
| m-12 | minor | B, D | C4a loose (1.25 late-window factor vs 1.043 measured) | spec 998 | tighten |
| m-13 | minor | C, E | 38.8 MB per stream × lockstep width (9.9 GB at 256); per-message boot cost absent from the cost model | spec 7.1-7.2 | memory cap; lazy init |
| m-14 | minor | E | Crossover table: X1 not "equal bits", X4 / X5 seen-source, X7 omits −0.18 %, X8 unmeasured, X12 square sizes | spec 842-856 | relabel |
| m-15 | minor | E | Throughput budgets quoted with Mamba's margins and per-core figures | spec 831-832 | Strata's margins, per vCPU |
| m-16 | minor | A | Clarity: HEADER_LEN scope, unused `i` in `ctx_hash`, "trace updates", `conf_i` on a miss, name clashes | spec 145, 174, 178, 230, 408, 540, 618 | reword |
| m-17 | minor | C | STRATA block has no extension room besides ARITH values | spec 626 | extension bit |
| m-18 | minor | D | Bake-off sources modified after the measurement (hashes no longer match `03`); nothing committed | `03` 93-98; sha256 today | frozen copy; re-check |
| m-19 | minor | A | History keeps `max len[i]` bytes but bit cells need `c3` (bits differ when `max len < 3`) | spec 116, 372-373 (PROBE A6) | keep `max(max len, 3)` bytes |
| m-20 | minor | A | "No operation may wrap" (line 61) vs explicit mod 2^32 / mod 256 in 3.1, 3.3, 4.2 | spec 61, 331-332, 379-381, 470-482 | "except where mod is written" |
| m-21 | minor | A | Trainer text: forward pass does not clamp biases (exporter does: C10 can fail); no boot weight for bit-cell inputs; Q0 rounding unstated | spec 279, 584, 877-880 (PROBE A14) | clamp in the forward; state both |

---

## 2. Details

### B-1 (blocking) — G0 is defined twice, and the governing text has three weak points

**Evidence.** The specification's G0 (9.0, lines 974-984) keeps a trunk tier "for a data type" when it is 5 % better
"per selection source and on the seen-source slice". While this review was in progress the architect wrote a
pre-registration (`05-gate-amendments.md` lines 13-86, "written 2026-11-13 08:45, before any G0 code was written or
run") and a decision script (`tools/strata/bakeoff/g0_decide.py`, 08:58) that implement a *different* rule: two macro
metrics on selection data only, `L` (bpb of the four long streams) and `S` (total payload bytes of R-113-style
messages cut from the selection sources, with a same-source and a foreign primer); a saving-per-MAC criterion
(`s >= 5 %` and `e >= 5 %` per 100K MAC/byte, plus a marginal test of tier B over tier A); the seen-source slice
explicitly excluded; bit-cell table bits fixed at 8 / 16 / 22 / 22. Spec 9.0 was not updated (the spec was edited at
08:59 elsewhere, e.g. line 395).

**Assessment.** The 08:45 text is a real improvement; against 9.0 alone this review would have raised: the seen-source
Wikipedia slice could keep the trunk for "text" (post hoc, with cells, the trunk of `strata_g` is worth 2.9 % on
`webster` but 14.3 % on `enwik8`); "data type" was undefined (`zookeeper` 13.5 % vs `proxifier` 2.9 %); the
short-message segment — the only one Fase 20 leaves open, where the prototype's trunk is worth 4-27 % on held-out
messages — was not measured at all. Three weak points remain in the governing text:
1. *One seed against a 5 % bar.* VERIFIED from the post-hoc numbers: on the first 100 KB the `strata_g` trunk is worth
   2.7 % of the selection macro once cells are added, so `L` is likely to land near the bar, where seed noise (never
   measured; `CRITERION.md` line 73) decides.
2. *`S` is a byte total.* By analogy with the prototype's test-source messages (primed payloads per set of five sizes:
   prose 1,302 B, Apache 205 B), `S` will be dominated by `webster` text, and the 128-512 B messages, where framing and
   priors matter most for path 3, contribute roughly a quarter of it (INFERENCE). A trunk that helps only long text
   messages passes; one that helps only short log messages may not.
3. *Training chunks stay at 8,192 bytes*, below the specification's own MUST (m-3); the report-only "foreign" JSON
   primer is another hour of the same archive (GH Archive), i.e. the same source by the project's by-source rule.

**Failure scenario.** The spec is frozen with 9.0 as written while G0 was decided under `05`: the frozen document states
a rule that was not applied, the T1 trail is broken, and a later reader re-applies 9.0 to the same numbers and reaches
a different verdict.
**Correction.** Replace the G0 table of 9.0 by a pointer to the `05` pre-registration (one normative text). Before
reading G0, add in `05` (dated): if `s` on a metric lies within ±2 points of 5 %, a second seed is trained before
deciding; `S` is also reported as the mean over (source, size) cells of the log-ratio and per traffic type, next to
the byte total; the foreign JSON primer comes from another event source, or the cell is labelled same-source.

### B-2 (blocking) — the weight-file layout is not fully stated, and misreadings pass every check

**Evidence.** Spec 5.5, lines 596-599: the first row reads "for `l = 1 .. L`: `W_{l,k}`", followed by separate rows
`B_{l,k}` and `WG_{l,k}, BG_{l,k} (if GEARED)` — per-layer interleaving (`W_1 B_1 WG_1 BG_1 W_2 ...`) or grouped
(`W_1 .. W_L, B_1 .. B_L, ...`)? Line 571: `KF, KS | u8 [L][d] each` — all of `KF` then all of `KS`, or per layer?
No section says whether `[a][b]` blocks (`EMB`, `W`, `HW`, `Q0`, `W0`, ...) are row-major. PROBE: each alternative
reading was implemented on a synthetic file; all load, pass `STRATUM_LEN`, every CRC and `TIER_ID`, and give
different probabilities (with unchecked arithmetic) or a bound error later.
**Failure scenario.** The trainer's exporter writes one order, the oracle reads another: the oracle produces wrong
probabilities that surface only at C1, and the dispute can only be settled by reading another team's code, which rule
T2 forbids. A round trip inside one team cannot catch it (both sides use the same reading).
**Correction.** Replace the tables by an explicit byte sequence ("for l in 1..L: W, B, then if GEARED: WG, BG; then
HW, GW, HB, GB, W0, STRATUM_LEN, TIER_ID"; "KF[0..L-1] then KS[0..L-1]"), state "row-major, last index fastest", and
add one synthetic weight file with its SHA-256 to Appendix A.

### B-3 (blocking) — the decoder's start and the end of STREAM are not stated

**Evidence.** Line 479: "code = first 4 stream bytes (bytes past the end read as 0)". Line 70 says multi-byte fields are
little-endian, but the coder reads `code` big-endian (`coder.rs:234-237`). "The end" is not defined: the archive
continues with the 4-byte CRC (line 619), and the tight flush of 4.3 relies on zeros after the last emitted byte.
VERIFIED (re-run of the probe's script): with zeros past the end, 0 failures on 20,000 random bit streams and 320
synthetic archives; with four arbitrary bytes after the stream (random streams) or the real CRC trailer (archives) read
as stream continuation, **273 of 20,000 streams and 8 of 320 archives decode wrongly** (the rounded-up `low` plus up
to 2^24 − 1 of trailer bits can pass `low + range` when the final range is below about 2^25). PROBE: a little-endian
`code` fails the CRC.
**Failure scenario.** An oracle that decodes from the archive buffer sequentially reads the CRC as stream data and
fails on a few percent of archives: rare enough to be mistaken for a coder bug, frequent enough to fail C3.
**Correction.** Write `code = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3` as an explicit exception to line 70; define
STREAM as `archive[header_size .. len − 4)` and "positions at or after `len − 4` read as 0"; state that `lo` in the
decoder is the `cdf_low` of the decoded bit.

### B-4 (blocking) — the container defers to texts outside the specification

**Evidence.** Lines 15-18 claim that sections 2-7 alone suffice, but the text defers elsewhere at line 459 (the coder
is `RangeEncoder::encode` of `coder.rs`), 617 and 624 (SHUFFLE / DELTA bytes, E8E9 / shuffle / delta "keep their
meaning"), 629 (TOTAL_BYTES and CRC "as in v4/v5"), 718 (preprocessing "stays available"), 745 (TOTAL_BYTES enforced
"as the existing loops") and 997 (empty input "keeps today's behaviour"). The default CLI mode is automatic:
`src/application/codec.rs:839-935` runs content detection (`content_detect`, `record_detect`, `detect_e8e9`) and may
apply E8E9, a byte shuffle with a detected record stride, and delta coding. PROBE: "CRC-32 of the bytes that entered
the coder" admits original bytes, preprocessed bytes, or primer + message (primer bytes are run through the machine);
the primer could be run raw or through the archive's preprocessing; 4.3 applied literally to an empty input gives a
one-byte stream and a 25-byte archive, which may not be "today's behaviour".
**Failure scenario.** Gate C1b requires identical archives from `REF compress` and `redukt compress` with default flags
on 12 inputs that include PE executables and binary records (`mozilla`, `ooffice`, `sao`, `x-ray`). A spec-only oracle
can reproduce neither the automatic decision nor the transforms, and differs on the CRC definition: C1b fails, and
fixing it requires reading Rust code. No Strata model was ever trained or evaluated on preprocessed bytes either.
**Correction.** For v1: `FLAGS` 0x04 / 0x08 / 0x10 MUST be 0 for BACKBONE 4, the auto stage is skipped for
`--backbone strata`, and `--e8e9 / --shuffle / --delta` other than `off` are errors (as the spec already does for
`--quant` and others, lines 715-717); CRC-32 is "of the original message bytes; the primer is never included"; the
empty-input archive (or its rejection) is spelled out; and "sections 4.2-4.3 take precedence over `coder.rs`" (today
they agree: VERIFIED line by line). Preprocessing becomes a hook with its own normative text.

### G-1 (grave) — short-message results count the wrong framing and seen-source cells

**Evidence.** Spec line 511 itself says a primer adds 4 B (`PRIMER_ID`); 5.6 gives 16 B compact / 20 B default. Yet
lines 417-420 and 513-516 and `03-bakeoff.md` lines 50-54 and 342-350 count primed archives with "the full 20 B of v5
framing"; `shortmsg.py:122` adds 0 / 4 / 20 B to every model. VERIFIED recount from
`data/strata/runs/strata_g/shortmsg*.csv` (cell means, 16 messages per cell):

| Strata + primer vs zstd-19 + dictionary | all 30 cells | 25 held-out cells (no Wikipedia) |
|---|---|---|
| payload + 4 B (R-114 target) | 30 | 25 |
| + 20 B (as reported) | 24 | 19 |
| **+ 24 B (real: compact header 16 + `PRIMER_ID` 4 + CRC 4)** | **22** | **17** |
| **+ 28 B (real: default header 20 + `PRIMER_ID` 4 + CRC 4)** | **20** | **15** |

Losing cells at +24 B: all five JSON sizes, Apache 128 and 256 B, OpenSSH 128 B. Per message rather than per cell
mean, 328 of 480 messages beat zstd + dictionary at +24 B. Gate V3 (line 1017) counts ">= 24 of 30 cells" over the six
R-113 sources, one of which (Wikipedia) is a training source of every Strata model (rule T3).
**Correction.** Fix the three passages; state V3 over held-out sources only (25 cells, threshold 20 / 25, or replace
Wikipedia by an unseen text source) with the real header of the configuration under test; keep "+ 4 B" as the R-114
target, labelled as such. The fairness of the rest of the protocol is discussed in section 6.

### G-2 (grave) — the ratio gate and the footprint gate of path 3 measure different configurations

**Evidence.** The 30 / 30 short-message result came from the prototype: tier B (~114K MAC), successor tables of
2^16 / 2^18 slots (5 MB), **no bit cells** (`03-bakeoff.md` 7.6, Appendix B). The spec's messaging file is different
(successor tables of 12 bits beyond order 2, bit-cell orders 0-1, about 0.64 MB; lines 394-396, 822-824) and has never
been run. V3 (line 1017) names no weight file or tier; V6 (line 1020) is measured on `base` "with messaging tables".
**Failure scenario.** V3 passes with large tables, V6 passes with small tables, and no file that ships passes both.
INFERENCE: a 64 KB primer holds tens of thousands of distinct contexts per order; 4,096-slot tables keep at most 4,096
of them under always-replace, so the primer's benefit (the decisive lever, 10 -> 30 cells) should shrink.
**Correction.** Name one messaging weight file (tier, table bits, NBC) and require V3 and V6 on it in the same run;
sweep successor and bit-cell table bits at message scale (128-2,048 B, 64 KB primer) before choosing it.

### G-3 (grave) — default tier of a truncated weight file is undefined

**Evidence.** A file "MAY end after any complete stratum" and then provides tiers `0 .. k` (lines 531-533), but
`DEFAULT_TIER` lives in the HEADER (line 547), which truncation does not change, and `--tier` defaults to it (line 713).
**Failure scenario.** The path-3 deployment is exactly a truncated file on a terminal. With `DEFAULT_TIER = 3` and a file
cut after stratum 2, one implementation errors, another falls back to the largest tier present, a third to tier 0:
default compression differs between teams, and A4(b) ("the default tier is fixed and reproducible") is broken.
**Correction.** Specify one rule — "reject a default-tier compress when `DEFAULT_TIER` is absent" or "default =
`min(DEFAULT_TIER, last tier present)`" — and test it in C9.

### G-4 (grave) — the long run's exclusions do not name the sources of the path-3 gate

**Evidence.** Spec 8.6 (line 923) trains on "public source code, JSON and CSV corpora chosen so that none shares a source
with a test file". The frozen test list (line 925) does not contain the JSON source of V3 (GH Archive events,
`data/strata/raw/gharchive/2015-01-01-15.json`), and it excludes the Silesia `samba` file but not the Samba project's
source code (`02-evidence` E.3 item 1: "systems, not files").
**Failure scenario.** The trainer, reading only the spec (T2), includes other hours of GH Archive — the most common public
JSON corpus — and the five JSON cells of V3, today its worst cells, become seen-source unnoticed. The same holds for
`samba` with a GitHub code corpus.
**Correction.** List in 8.6, by source: GH Archive (any date), the Samba project, Wikipedia (any dump), Project
Gutenberg (m-8), anything produced by `scripts/prepare-benchmark-data.py`, the Loghub systems used for selection and
test.

### Minor findings

**m-1 — details of the length argument (axis B).** The boundedness proof and the coalescence lemma of 2.6 are correct
(re-derived: `S'` lies between `S` and `U`; the bracket lies between `floor(D/2^k)` and `ceil(D/2^k)`; since `U` is a
multiple of `2^12` and `k <= 12`, every step with differing read-outs is followed by a strict decrease of `D`).
Slips: (a) "the delay taps agree after 3 bytes" (line 258): `in[3]` agrees only after the 4th commit; (b) "No counter,
position or length is stored anywhere" (lines 239-240) is false: the byte count `n` (line 116) grows without bound and
is compared with `len[i]`; (c) the width of `n` is not given — stored in 16 bits it wraps and re-deactivates every
order for 32 bytes every 64 KiB; (d) line 266 gives 2,126 bytes for `kmax = 8`, the formula gives 2,127.3. Fix: "`n`
saturates at `max len[i]`", and correct (a), (b), (d).

**m-2 — the reader's MUST-reject list is incomplete (axes A, B).** Section 2.7 relies on biases within ±(2^24 − 1)
"clamped by the exporter", but 5.5 and 6.3 make only int8 −128 a reader MUST; nothing bounds `W0_k` (±`W_MAX`),
`DEFAULT_TIER < K`, or requires distinct tier names. PROBE: a CRC-valid file with a bias of 2^31 − 1000 loads and fails
at byte 0 in an exact reference, while a Rust release build would wrap silently; a `W0` above `W_MAX` is used raw by
one reading and clamped at load by another. Fix: one table "field, legal range, reader action", exercised by C9.

**m-3 — the training-chunk rule is violated by every measured model and contradicts the long-run plan (axis B).**
Lines 895-896: chunk length MUST be at least `(L + 1) · M`. For the v1 schedules (input shifts up to 5, layer shifts up
to 10, L = 3) that is 21.6-28.4 KB; the bake-off and G0 (`05` lines 25-27) train on 8,192-byte chunks, and the long run
makes 25 % of its chunks 128-4,096 B. The coverage half of the 2.6 argument therefore applies to no measured model
(the measured curves are still flat to 2 MiB, so this is a gap in the argument, not an observed failure). Fix: define
`M` from the schedule in the file, apply the MUST to the long chunks only, make the trainer check it.

**m-4 — PROFILE pin and the pending RWKV decision (axis C).** Line 625 pins PROFILE to 0x10, "the value every non-Mamba
backbone gets today"; today that value comes from `NumericContract::CURRENT` through the shared builders, so a future
contract bump would change Strata headers. The Strata builder should write 0x10 explicitly. The spec consumes no free
header resource (PROFILE bit 7, QUANT codes 2-15, contract code 3), so it leaves room for the RWKV quantisation
decision of `07-legacy-fixes.md` section 7; option (a) there (QUANT for RWKV) is the only fail-early route, because
RWKV archives carry no BACKBONE byte. Recommendation: resolve both in the same FORMAT-SPEC revision with one
per-backbone table of allowed PROFILE values.

**m-5 — seen-source averaged into the held-out macro (axis D).** VERIFIED: the "macro" column of `03-bakeoff.md` 7.3 is
the mean of seven columns including `seen-enwik8` (e.g. `strata_g` B: 6.1376 / 7 = 0.8768). Held-out only: `strata_g` B
0.7016, `gru` 0.6888 (+1.9 %, not +3.5 %), `strata_f` B 0.6904 (better than the chosen geared trunk on the test set);
trunk zeroed 0.8670 (+23.6 %, not +29 %). `02-evidence` E.3 item 5 forbids this averaging. Fix the table, spec line
314 and `03-bakeoff.md` lines 22-25 and 251-252.

**m-6 — API weights flag (axis C).** `redukt api` has one `--weights` value for every backbone (`01-integration-map.md`
section 8: RWKV reads it directly, Mamba falls back to it). With `--backbone strata,rwkv` one of them receives the
other's file. Fix: a Strata-specific flag (e.g. `--strata-weights`) in 6.4 and 6.5.

**m-7 — `TIER_ID` covers more than tier k (axis C).** Lines 606-608 say `TIER_ID` "identifies everything tier k depends
on and nothing else", but it is a CRC of the whole prefix, which contains the HEADER (all widths, names,
`DEFAULT_TIER`, `TRAIN_ID`) and GLOBALS sized for the largest width. Re-exporting identical tier-k weights with another
`TRAIN_ID` or top tier makes old archives fail with "different weights". Conservative and safe, but the sentence is
false: state it, or compute `TIER_ID` over the tier's own dependencies.

**m-8 — Project Gutenberg text in training (axis D).** `common.py:63-64` trains on `world192.txt` and `lcet10.txt`, both
Project Gutenberg etexts; `world192.txt` contains the PG legal "small print" (7 matches), which appears 88 times in
`webster` (selection) and 111 times in `dickens` (frozen test and the "prose" messages). `03-bakeoff.md` line 116 says
"No Project Gutenberg text ... were trained on". The byte share is small and the online tables learn repeated
boilerplate anyway, so the effect is likely small (INFERENCE), but the statement is false; the long run should drop
PG etexts or strip their headers.

**m-9 — G0b is self-agreement (axis D).** G0b compares the trainer's simulation with `ref_int.py`, both written by the
architect. A useful design check, but not T2 evidence; only C1 / C10 against the oracle are.

**m-10 — D1 (gears) rests on one source and one seed (axis D).** VERIFIED: the −3.0 % of `strata_g` over `strata_f`
comes from `webster` (ratio 0.957); `strata_f` is better on `zookeeper` and `proxifier`. With the geometric mean of
per-source ratios the difference is 0.5 % (inside the band, where the tie rule picks `strata_f`), and on the held-out
test set `strata_f` is 1.6 % better. The pre-registered arithmetic mean makes the decision procedurally valid, but a
macro of bpb weights `webster` about ten times more than a log source. Fix: re-test D1 with two seeds, report per-source
ratios next to every macro, or keep `GEARED` optional (it is already a legal file).

**m-11 — V1 and V2 inputs (axis D).** V1 counts the two synthetic XML files as two of six votes although `02-evidence`
E.1 declares them invalid as a generalisation gate; V2 compares Strata on the local CRLF copies with Mamba figures
measured on LF copies (1-4 % difference, `KF:105-106`). Fix: report the XML separately; run both on identical bytes.

**m-12 — C4a is loose (axes B, D).** The first condition (2 MiB within 3 % of 10 KB) is met trivially by any file whose
first 10 KB is expensive (`dickens`: 0.84x); the second allows a 25 % rise of the late window where both the candidate
and the LSTM measured 4.3 %. A gradual 20 % degradation of the online part would pass. Consider 1.10, and comparing with
the control's ratio on the same windows.

**m-13 — memory per stream in lockstep and per message (axes C, E).** 38.8 MB per stream at the default tables (26.2 MB
before the 08:59 edit); `--jobs` up to `MAX_SLOTS` = 256 means 9.9 GB; the dock reports state in `/v1/status` but has no
cap. The boot (or primer clone) of 38.8 MB per message is absent from the time model and dominates short messages
(section 7.2). Fix: a memory cap on lockstep width in 6.7, lazy initialisation (generation counters) or
messaging-size tables, and a boot term in 7.1.

**m-14 — crossover table (axis E).** See 7.3: X1 is not an "equal bits" statement; X4 and X5 mix seen-source evidence;
X7 and line 313 give +0.04-0.35 % and omit the measured −0.18 % (`strata_f` A); X8's cost of nesting was never
measured; X12 quotes 256 KiB / 1 MiB per layer (square sizes) where the triangular layout gives 193.5 / 769.5 KiB with
gates. Label them accordingly.

**m-15 — throughput budgets quoted against the wrong ratio and unit (axis E).** Lines 831-832 compare per-core hypotheses
with per-vCPU budgets derived from Mamba-130M's margins. With Strata's own measured margins the logs budget is
250-890 KB/s per vCPU and text has none (7.1). Decision-neutral (V4 is expected to fail), but it is the only
throughput target in the document, and `02-evidence` B.14 records the same accounting error once already.

**m-16 — clarity (axis A).** HEADER_LEN (line 540) does not say whether MAGIC is counted; `ctx_hash(..., i)` (line 174)
takes an order index the definition never uses; "trace updates" (line 230) is undefined; `run_i` and `conf_i` (lines
176-178) are undefined on a miss (never used then; PROBE A15); names clash: `S0` is both the input accumulator (line
109) and the header byte (line 618), `n` both the byte count and the calibration counts (lines 408-410), `k` both the
tier index and the shift (line 145).

**m-17 — no extension room in the STRATA block (axis C).** `S0` uses all eight bits; hooks H1-H6 can only be signalled
by ARITH values 2-8 (seven left), which mixes "arithmetic version" with "optional feature". Consider an extension bit,
or "ARITH >= 2 implies an extension byte".

**m-18 — the measured bake-off can no longer be re-run from the tree (axis D).** `03-bakeoff.md` lines 93-98 quote the
sources by hash; VERIFIED today: `CRITERION.md` is unchanged (`c29fa870…`), but `common.py`, `models.py`, `train.py` and
`evaluate.py` were modified at 08:47 for G0 (now `a396644d…`, `e7d8d532…`, `a91c34c5…`, `b7fef5bd…`), and nothing is
committed. `lstm/eval-select-on.csv` was rewritten at 08:47:50 with the same four values, which is consistent with the
modified code reproducing at least that control (INFERENCE). Fix: keep a copy of the measured sources, and record in
G0 that the cells-off path reproduces `strata_g` and `gru` exactly.

**m-19 — history length vs bit-cell context (axis A).** Line 116 keeps "the last `max len[i]` bytes"; bit cells need
`c1..c3` (lines 372-373). With context lengths {1, 2} and a bit-cell order of length 3 — a legal file — `c3` is not in
the history. PROBE: the two readings ("0" vs "the real byte") give different `p1` from bit 64 on. Fix: keep
`max(max len[i], max blen[m])` bytes, or define `c1..c3` from the delay taps `in[0..2]` with BOS mapped to 0.

**m-20 — "no wrap" vs explicit modular arithmetic (axis A).** Line 61: "No operation in sections 2 to 4 may wrap"; yet
3.1 (lines 331-332), 3.3 (lines 379-381) and 4.2 (lines 470-482) compute `mod 2^32` and `mod 256`. Fix: "no
operation may wrap except where `mod` is written explicitly".

**m-21 — trainer text (axis A, team T).** Line 279 says the exporter clamps biases to ±(2^24 − 1), but the trainer's
forward values (lines 877, 879: `round(4096 b)`, `round(2048 .)`) are not clamped, so a large trained bias makes the
exported model differ from the trained one and C10 fail; line 880 gives boot mixer weights for the trunk input and the
orders but none for the bit-cell inputs; line 584 does not say how `Q0` is rounded (PROBE A14: `(hits + 1) / (total + 2)
· 2^22` can be exactly 1024.5). Q0 is data in the file, so only trainer reproducibility is affected. Fix: clamp in the
forward pass, state the bit-cell boot weight and the rounding.

---

## 3. Worst-case arithmetic (axis A, VERIFIED by hand and by script)

| Quantity | Worst case | Width | Margin |
|---|---|---|---|
| `a[c]`, `ag[g]`, `g[i]`, `z` at `d = 1,024` (`large`) | 127 · 127 · 1,024 + (2^24 − 1) = 33,293,311 | i32 | 64x |
| same at `d = 4,096` (largest legal) | 66,064,384 + 16,777,215 = 82,841,599 | i32 | 26x |
| `S − (u << 12)` | 2 · 520,192 = 1,040,384 | i32 | 2,000x |
| `x[m] · err + 2^(MIX_SH − 1)` | 3,071 · 65,535 + 2^19 = 201,782,273 | i32 | 10x |
| `dot` | 13 · (2^19 − 1) · 3,071 = 20,931,109,901 (2^34.3) | **i64 required** | — |
| `vpmaddubsw` pair sum | 2 · 127 · 127 = 32,258 | i16 | 1.016x, **only with the sign-transfer form** |

Two conditions are implicit and should be written: (1) biases within ±(2^24 − 1) and boot mixer weights within ±`W_MAX`
hold *because the reader rejects anything else* (m-2); (2) the `u8 × i8` kernels must use the sign-transfer operand
form (|h| and sign(W, h)); the offset-binary form (h + 128) saturates (2 · 255 · 127 = 64,770).

## 4. Hand derivation of the first coded bit (axis A)

Tier width 0, NORD = 3, NBC = 2, `MIX_SH` = 14, a trainer's free choices `HB_0[1] = −1000` and boot weights
`W0_0[0] = [65536, 16384 × 5]`; first byte `0x61`:

1. `n = 0`, so every order is inactive (line 172); `top = 0`, `set = 0`.
2. `node = 1`; `z = HB_0[1] = −1000`; `x[0] = clamp(asr(−996, 3)) = −125` (floor, line 58).
3. Bit cells at boot: `qb = 2^21`, `ST[2048] = 0` (Appendix A), so `x[4] = x[5] = 0`.
4. `dot = 65536 · (−125) = −8,192,000`; `st = asr(dot, 16) = −125`; `p1 = SQ[2946] = 24,923` (table regenerated from 5.4;
   the file is the authority).
5. Bit 0 of `0x61` is 0: `split = 40,613`; `r = 0xFFFFFFFF >> 16 = 65,535`; `low = 0`;
   `range = 65,535 · 40,613 = 2,661,572,955` (no renormalisation).
6. `err = −24,923`; `w[0][0] = 65536 + asr(−125 · −24,923 + 8,192, 14) = 65,726`; the other weights are unchanged.
7. Bit cell of node 1, both orders: `qb = 2^21 + asr(0 − 2^21, QSHIFT[0] = 2) = 1,572,864`, `nb = 1`.

Everything above follows from lines 56-233, 360-389 and 439-483 without a guess, and the accumulator example of
Appendix A (shift 3: 51,200 / 96,000 / 135,200 / 53,276) re-derives exactly. The text stops being sufficient one step
further: reading a stratum (B-2), starting and ending the decoder (B-3), choosing the default tier of a truncated file
(G-3), producing an archive with default flags (B-4).

## 5. Independent spec-only implementation probe (PROBE)

A separate agent implemented the specification in plain Python from the text alone (no access to `tools/`, `src/` or
other documents): table generation, `ctx_hash`, bit cells, the stream machine with gears, primer snapshot, range coder
with tight flush, weight-file writer and reader with every CRC, and the AZTH header.

| Check | Result |
|---|---|
| Appendix A spot values (asr, accumulator, 9 SQ, 8 ST, 12 QSHIFT, 3 SHA-256, 5 `ctx_hash`, CRC-32) | 51 of 51 match (VERIFIED: re-run) |
| Table generation | exact rational arithmetic, Python `math` and numpy give identical SQ / ST; closest pre-rounding value to a .5 boundary 9.8e-5 (ST[830]) (VERIFIED) |
| Table 7.2 | all 30 derived cells reproduced from 5.2-5.5 and 7.1 |
| Round trip | synthetic file (K = 2, widths 0 / 16, L = 2, geared, NORD = 3, lengths 1 / 2 / 4, NBC = 2, `MIX_SH` = 14), 2,048-byte text decoded by a separately built decoder; default and compact headers, with and without primer, lengths 0, 1, 2, 3, 7 and 128 as well: all pass; coverage includes gain 0, mixer weights at `W_MAX`, `h` at ±127, every `top` |
| Tight flush | 0 failures on 20,000 random streams; 123 failures without the `first` condition, which supports 4.3 (VERIFIED) |
| Truncation | a file cut after stratum 0 loads with the same `TIER_ID`; truncation inside a block and flipped bytes are rejected |

What the implementer had to guess: the decoder's start and the end of STREAM (B-3), the block and element order of the
weight file (B-2), and that `lo` in the decoder is the `cdf_low` of the decoded bit. The stream machine itself
(2.3-2.5, 3) needed no guess. Ambiguities with **demonstrated** bit differences: A1 decoder byte order, A2 CRC read as
stream (both B-3), A3-A5 weight layout (B-2), A6 history length (m-19). With bit differences **by reasoning**: A7
default tier (G-3), A8-A10 and A12 container texts (B-4), A13 reader ranges (m-2), A14 Q0 rounding (m-21).
Undetermined from the text: A11 precedence between 4.2 and `coder.rs` (B-4; they agree today, VERIFIED). Harmless:
A15 (m-16). Internal contradictions it reported, each checked against the text: self-sufficiency vs deferrals (B-4),
LE convention vs big-endian `code` (B-3), history vs `c3` (m-19), "no wrap" vs explicit mod (m-20), "no counter" and
"3 bytes" (m-1), 2,126 vs 2,127 (m-1), exporter clamp vs unclamped forward and missing bit-cell boot weight (m-21),
square vs triangular layer sizes (m-14).

## 6. Experimental rigour (axis D): figures checked against the files on disk

| Claim | File | Recomputed | Verdict |
|---|---|---|---|
| D2: `strata_g` 0.7480 vs `gru` 0.7518, ratio 0.995 (TIE) | `runs/strata_g/eval-select-on.csv`, `runs/gru/eval-select-on.csv` | 0.7480 / 0.7518 = 0.995 | exact |
| Test bytes over six sources 171,490 vs best classical 168,216 (+1.9 %) | `runs/strata_g/eval-test-on.csv`, `runs/baselines.csv` | 171,492 vs 168,214 (+1.95 %) | exact (rounding) |
| Online off x6.3-6.6; trunk off +11-14 % (selection) | `eval-select-off.csv`, `eval-select-on-notrunk.csv` of four runs | x6.30-6.62; +10.6 % to +14.4 % | exact |
| Post-hoc bit cells −9.7 / −20.3 / −10.8 / −21.0 %; trunk worth −0.6 / 0.0 / −14.5 / +8.4 / −17.8 % (`lstm`) | `runs/lstm/explore-bitcells-*.txt`, `post-decision.log` | same | exact |
| Length curves, late / early 1.043 (`strata_g` B, `dickens`), 1.043 (`lstm`) | `runs/strata_g/eval-curve-on.csv`, `runs/lstm/eval-curve-on.csv` | same | exact |
| Primed short messages: 24 of 30 with 20 B | `runs/strata_g/shortmsg*.csv` | 24 at 20 B, but the real framing is 24-28 B: 22 / 20 (17 / 15 held-out) | **misreported** (G-1) |
| Frozen test macro 0.8768 vs GRU 0.8475 ("3.5 % worse"); trunk zeroed +29 % | `eval-test-on*.csv` | macro includes seen `enwik8`; held-out only 0.7016 vs 0.6888 (+1.9 %); trunk zeroed 0.8670 (+23.6 %) | **misreported** (m-5) |

- **Leakage.** enwik8 is excluded from training and labelled seen-source; the enwik9 slice is verified by hash. Two
  leaks remain: Project Gutenberg boilerplate (m-8) and the seen-source Wikipedia cells inside the short-message tally
  and V3 (G-1). Synthetic XML: the two files are distinct templates and neither was trained on, so there is no
  template leak; but each is 20 copies of one template, which is why `02-evidence` E.1 declares them invalid as a
  generalisation gate, and V1 still counts them as two of six votes (m-11). Loghub "unseen" is not "dissimilar":
  `zookeeper` shares the Java/log4j format of the Hadoop / Spark / HDFS training logs, which is why the trunk helps
  there (13-15 %) and not on `proxifier`.
- **Equal budgets.** Verified in `models.py:264-299, 413-417`: the same head, gains and mixer are counted for every
  model, trunk MACs analytically, same data order, steps and optimiser. Declared differences: controls in float (which
  favours them), one-layer controls vs a three-layer candidate, the candidate at 89 % of the MAC cap and with 117 % of
  the GRU's parameters (inside the cap). The budgets are equal in the sense pre-registered.
- **Thresholds after data.** D3 (disclosed); G0's 5 % bar was set after the post-hoc exploration, and the 9.0 version is
  not purely "stricter" as `05` says, because it can also *keep* a trunk on seen-source evidence; the 08:45
  pre-registration removes that (B-1); V3's second pairing is genuinely stricter; C4a's 1.25 is loose (m-12).
- **Short messages vs zstd + dictionary.** Data budget: the zstd dictionary is trained on the whole first half of the
  source (up to 2 MB) and may select from it, while the primer is the last 64 KB of the same half: this favours zstd.
  zstd framing is minimal (magicless, no checksum, no content size, no dictionary id: `shortmsg.py:61-63`) and
  self-terminating, while a Strata payload cannot be decoded without its length: the "+ 0 B" column flatters Strata
  and "+ 4 B" is the fair one. The same-source pairing is a best case for both (V3 adds the second pairing). Remaining
  unfairness: the framing of primed archives and the seen-source cells (G-1).
- **Do the conclusions follow?** Mostly, and `03-bakeoff.md` states its weak points first. Exceptions: framing (G-1),
  the seen-source macro (m-5), "no Project Gutenberg text" (m-8), the integer-cost range without the −0.18 % case
  (m-14), and D1, whose −3.0 % comes from one source of four (m-10).

## 7. Economics and scale (axis E) and the trunk (axis F)

### 7.1 Which tier, if any, reaches the speed it needs

Break-even throughput `R* = P · 10^9 / (3600 · V)`, `V = Δbpb / 8 · $0.023 · 84` (model of `02-evidence` C.1), with the
prototype's measured ratios on the frozen test set (VERIFIED, `economics.py`):

| Prototype tier | Text (`dickens`) | Apache | Linux | OpenSSH | Invoice XML |
|---|---|---|---|---|---|
| A (~27.7K MAC, between `nano` and `micro`) | loses to bz2: no speed suffices | 289 KB/s ($0.0135/GB) | 885 KB/s ($0.0044/GB) | 364 KB/s ($0.0107/GB) | loses |
| B (~114K MAC, between `micro` and `small`) | loses | 252 KB/s ($0.0154/GB) | 866 KB/s ($0.0045/GB) | 366 KB/s ($0.0106/GB) | loses |
| trunk zeroed (proxy for `cm`, no bit cells) | loses | 8,946 KB/s | loses | loses | loses |

KB/s per vCPU at $0.014/vCPU-h (×1.57 at $0.022). The spec's hypotheses for these widths are 60-570 KB/s **per core**
(lines 812-819), up to half that per vCPU on SMT hosts. No tier is viable for storage, as the spec already expects for
V4; the bullet at lines 831-832 ("logs need 110-200 KB/s per vCPU, reachable by `cm` .. `micro`") quotes Mamba-130M's
margins and per-core figures and understates Strata's own requirement 2-8x (m-15). Bit cells may narrow the log gaps,
but a 10-20 % gain does not change the order of magnitude, and decompression (same cost again per read) is not even in
the model. Path 3 has no throughput budget; its constraints are footprint, latency per message and determinism.

### 7.2 State per stream

- Default tables: 38.8 MB per stream since the spec's 08:59 edit (bit cells 8 / 16 / 22 / 22 = 33.8 MB, successor
  tables 4.98 MB; verified against the formula of 7.1 and table 7.2; 26.2 MB in the version first read). Lockstep width
  32 -> 1.24 GB, 64 -> 2.5 GB, `MAX_SLOTS` 256 -> 9.9 GB: the dock needs a memory cap (m-13). The lockstep benefit
  (sharing weight fetches) is irrelevant for tiers whose weights fit in L2; the tables, private per stream, are what
  thrash the cache, which the X13 row already suspects.
- Boot or primer clone of 38.8 MB per message costs ~2-4 ms (memset / memcpy at 10-20 GB/s, INFERENCE), more than
  coding a 128-2,048 B message at the hypothesised 1 µs/byte; the cost model of 7.1 does not contain it (m-13).
- Terminal: 39 MB per stream fits a Linux-class terminal but not the spec's own messaging target; the messaging
  configuration (0.64 MB) is the right target and is unmeasured for ratio (G-2).

### 7.3 Crossover table

Mostly honestly labelled. X3, X6, X7, X9 and X11 are measured; X12, X13, X15 and X16 are hypotheses, and only X12 and
X13 are calibrated at run time (the others change bits, so they can only be properties of the weight file, which the
spec should say). Not justified as written: X1 is a compute identity at equal `(d, L)` and says nothing about equal
bits, although "pays" is defined as "at equal bits" (the repo's only byte-level data point, MambaByte-353M, was 37 %
worse); X4 and X5 include seen-source numbers; X7 omits the −0.18 % case; X8's "nesting is free" was never measured
against the same tier trained alone; X12's sizes are square (m-14). Folklore is the exception, not the rule.

### 7.4 Is the trunk justified in v1?

**For.** (1) On the only open segment, short messages with a primer, zeroing the trunk costs 4-9 % on JSON and
18-27 % on logs and prose, all held-out (`03-bakeoff.md` 7.6), and drops the record against zstd + dictionary from 30
to 21 cells at + 4 B. (2) On sources that resemble the training data (`zookeeper`, Java service logs) it is worth
13-15 % even with bit cells. (3) At `nano` / `micro` it is cheap next to the tables (15-44K MAC/byte), and it is the part
that gives a prior from byte 0. (4) Without it Strata is a hand-written context mixer, a product the market already
has (lpaq / paq lineage).

**Against.** (1) On long streams from unseen sources, with bit cells, it is worth 0-3 % (`webster`, `proxifier`) and
costs 4-8 % on `xml`: the R-105 / R-112 prior holds with the online part present. (2) The ablation overstates it: the
boot weights, gains and calibration were trained with the trunk present, so zeroing it handicaps the rest; a width-0
tier trained as such will do better (G0 measures this). (3) With bit cells in the primer snapshot, part of what the
trunk adds on messages may move to the primed cells; the prototype's primer primed no cells. (4) It is the reason for
the QAT trainer (3x slower per step than `nn.LSTM`), the GPU hours, the tiered weight format, the SIMD kernels and the
execution autoscaler; the `cm` tier needs none of them. (5) On text-like messages Mamba-130M is still 1.9x smaller
(prose 2 KB: 349 vs 662 B), which is abandon criterion 5's territory.

**Experiment that settles it.** The 08:45 G0 pre-registration is most of it: the width-0 tier is trained as a tier (its
node biases, gains and boot weights get gradient only from its own loss, so the nested run is a fair stand-in for
training it alone), bit cells are in the loop, `L` and `S` use same-source and foreign primers, the seen-source slice
is excluded. What it lacks for a final answer (B-1): (a) a second seed whenever `s` lands within ±2 points of 5 %;
(b) `S` broken down by traffic type (text, markup, logs) and as a mean of per-cell log-ratios, with the path-3 traffic
types named in advance; (c) a classical context-mixing baseline primed the same way (compress primer + message,
subtract the primer's cost), so that the `cm` tier is judged against the nearest existing product; (d) the planned
report-only confirmation on the R-113 test messages with a foreign primer, counted at payload + 4 B with the
seen-source cells excluded. Rule: keep the trunk in v1 only if it beats `cm` by >= 5 % on `S` for the named path-3
traffic types, with a gap larger than the seed spread; if it passes on `S` only, v1 is a messaging product (as `05`
already says) and the long-stream claims are dropped; if it fails on `S`, the trunk leaves v1 whatever `L` says.

---

## 8. Questions the specification must answer before it is frozen

1. Which text governs G0 — 9.0 or the 08:45 pre-registration — and what happens when a saving lands near 5 % (second
   seed)? How is `S` aggregated and broken down by traffic type?
2. What is the exact byte sequence of a STRATUM (block order per layer, element order of every multi-dimensional
   block, `KF` / `KS` order)? Can Appendix A carry one synthetic weight file with its SHA-256?
3. How does the decoder form `code` from the first four STREAM bytes, and where exactly does STREAM end?
4. Is preprocessing allowed for Strata v1? If so, where are the transforms and the automatic detection defined
   normatively, is the primer preprocessed, and is the CRC computed on original or transformed bytes?
5. What archive does an empty input produce (or is it rejected), and which governs if section 4.2-4.3 and
   `coder.rs` ever differ?
6. What does a reader do when `DEFAULT_TIER` is absent from a truncated file?
7. Which weight file (tier, table bits, NBC) is the path-3 messaging file, does it pass V3 and V6 in the same run, which
   framing does V3 count for primed archives, and are seen-source cells excluded?
8. Which corpora are excluded from the long run by source (GH Archive, the Samba project, Project Gutenberg, any
   Wikipedia dump, the Loghub systems used for selection and test)?
9. What is the legal range of every field, and which values does a reader reject (biases, `W0`, `DEFAULT_TIER`, names,
   schedules)?
10. What is the width of the byte count `n`, and how many history bytes are kept when `max len[i] < 3` while bit cells
    use `c3`?
11. How is the chunk-length rule computed from a schedule, and does it apply to G0 and to the long run's short chunks?
12. Does the trainer clamp biases in its forward pass (so that C10 can hold), what boot weight do bit-cell inputs get,
    and how is `Q0` rounded?
13. Is `GEARED` kept on the evidence of one source and one seed, or re-tested?
14. Does the Strata header builder write PROFILE 0x10 explicitly, and is the RWKV quantisation level (`07` section 7)
    decided in the same format revision?
15. How does `redukt api` receive Strata's weight file when several backbones are loaded?
16. Which memory cap bounds lockstep width at ~39 MB per stream, and how is the per-message boot or clone cost handled?
17. What are Strata's own throughput budgets per vCPU, given its measured margins over classical codecs?
18. Is `TIER_ID` meant to cover the whole header and GLOBALS?
19. Will V1 report the synthetic XML separately, and will V2 compare identical bytes (CRLF vs LF)?
20. Is C4a's 1.25 factor intended?
21. How will future options be signalled in the STRATA block once ARITH values run out?
22. Where is the source code that produced the bake-off numbers preserved, now that it was modified for G0?

---

## 9. Commands used by this review (all read-only on the repo)

```
python -I <scratch>/review/shortmsg_framing.py data/strata/runs/strata_g     # G-1: recount at 0/4/20/24/28 B
python -I <scratch>/review/macros.py data/strata/runs                        # m-5, m-10: held-out macros, ratio means
python -I <scratch>/review/economics.py                                      # 7.1: break-even speed per vCPU
python -I <scratch>/review/hand_steps.py                                     # section 4: first coded bit
python -I <scratch>/t2probe/t_appendix.py                                    # section 5: Appendix A, 51 checks
python -I <scratch>/t2probe/t_flush.py                                       # B-3: tight flush, CRC read as stream
grep -c -a -i "small print" data/benchmark/silesia/{dickens,webster} data/strata/raw/canterbury/world192.txt  # m-8
sha256sum tools/strata/bakeoff/{CRITERION.md,common.py,models.py,train.py,evaluate.py}                       # m-18
```

`<scratch>` is the session scratch directory of this review (outside the repo). The probe's full implementation
(`strata_probe.py`), synthetic weight-file generator (`synth.py`), round-trip test (`t_roundtrip.py`), ambiguity
experiments (`t_ambig.py`) and table 7.2 check (`t_table72.py`) are in `<scratch>/t2probe/`.

---

## 10. Re-review of the fixes (2026-11-13, 13:15-13:30)

> Request: verify B-1 .. B-4 and G-1 .. G-4 after the author's fixes and the freeze "conditioned on the reviewer's
> check" (spec header, lines 3-10; `05-gate-amendments.md` amendments 1 and 2; `03-bakeoff.md` section 12).
> Versions checked: specification sha256 `c37cc06dda5e0a21…` (1,246 lines, edited 13:13:21, unchanged during the
> re-review), `05-gate-amendments.md` `c9efaf148bd106e3…`, `03-bakeoff.md` `dbd821dca0329c7a…`. Same rules as above:
> read-only except this file; the spec-only probe was re-run by the same independent agent on the revised text.

### 10.1 Status of the blocking and grave findings

| ID | Status | Evidence |
|---|---|---|
| B-1 | **closed** | Spec 9.0 (lines 1045-1055) now only points to `05`. The first 120 lines of the current `05` hash to `30a8cb51287c…`, the value logged in `00-charter.md` at 09:16 (VERIFIED): the registered rule is unchanged, amendment 2 was appended after it. Every G0 selection result was written after 09:16 (first: `lstm_bc` 09:17:30; candidate seeds 10:08 and 10:32), the decision at 11:05:16 after the last `S` file (11:05:09), the first test-set result at 11:19. `g0_decide.py` implements the registered rule line by line (per-cell geometric `S` over 40 cells, keep only if both seeds pass, worse-of-two G0a, seen source excluded); re-running it reproduces `g0-decision.txt` byte for byte, and an independent recomputation (duplicates removed; 16 messages per cell, 40 cells per tier) gives the same savings: tier A 9.93 / 10.20 %, tier B 12.59 / 13.05 %, marginal B over A 3.43 / 3.68 % per 100K MAC/byte. The decision (width 88 kept on `S` only, width 208 dropped, G0a TIE 1.0091 / 1.0044, G0 PASS) follows from the numbers on disk. The script was last edited at 09:20, after the registration and 2.5 minutes after the first control's selection file; since it matches the registered text, the overlap is immaterial |
| B-2 | **closed** | PROBE: the explicit sequences match the choices the probe had guessed, so its writer's output is byte-identical to before, and every derived cell of table 7.2 re-derives (MAC 208 / 27,752; parameters 1,199 / 73,392; prefixes 34,530 / 112,510 B; bytes touched 0 / 27,632; state 38,811,296 / 38,812,704 B; stand-alone general file 33,816 B). A reader that misreads the element order still loads silently and changes `p1` (shown again): only C9b's shared synthetic file catches it, which is acceptable because C9b is now a gate. Text: 5.1 (lines 550-552) row-major rule; 5.3 (588-589) explicit GLOBALS sequence, all `KF` then all `KS`; 5.5 (621-629) explicit STRATUM sequence, width-0 stratum; C9b (1076) cross-team layout gate. The prefix sizes of table 7.2 re-derive from the sequences: 34,530 B (tier 0 of a two-tier file), 33,816 B (stand-alone `K = 1`), 112,510 B (`msg88`) (VERIFIED by hand) |
| B-3 | **closed** | PROBE: round trips pass for both tiers, with and without primer, compact and default headers, lengths 1-2,048; reading the CRC as stream data still breaks 8 of 320 archives, which the text now forbids. Text: 4.2 (lines 500-505): big-endian `code` as an explicit exception, STREAM = `archive[header_size .. length − 4)`, reads past STREAM return 0, `lo` of the decoded bit |
| B-4 | **closed** | PROBE: no outside text is needed for a v1 archive; FLAGS 0x48 and 0x44, PROFILE 0x11, ARITH 2 and an absent TIER are rejected; empty input fails on both sides. Text: 5.6 (677) FLAGS exactly 0x40, no SHUFFLE / DELTA byte; 6.4 (781-782) auto detection skipped, `--e8e9 / --shuffle / --delta` other than `off` are errors; 5.6 (682) CRC-32 of the message bytes, primer never included; 5.6 (687-688) empty input rejected, `TOTAL_BYTES = 0` rejected (today's behaviour of the other backbones: `codec.rs:552-555`, VERIFIED); 5.6 (690-691) 4.2-4.3 take precedence over `coder.rs`; 6.5 (809-810) stop after `TOTAL_BYTES`, then CRC |
| G-1 | **closed** | Real framing in 4.4 (538-541), 3.4 (433-441), X10 (917) and `03-bakeoff.md` line 56; V3 (1089) votes on the 25 held-out cells, 24 B of framing, threshold 20 of 25. The new G0 counts re-derive from the R-113 files: width 88 with a same-source primer 24 / 30 (19 / 25 held-out), foreign primer 2 / 30 (0 / 25), foreign primer vs foreign dictionary 25 / 30 (20 / 25), `cm` same-source 22 / 30 (18 / 25) (VERIFIED) |
| G-2 | **partial** | Fixed: V6 is measured on the same file `M` as V3, in the same run (1092). Open: the "pre-registered sweep" that chooses `M` (line 1094) states only its grid; its objective, its constraint, its seeds and whether the tier dimension (`cm` vs `msg88`) is re-decided are not written anywhere, and nothing is registered in `05`. The grid's bit-cell option "orders 0-3" cannot meet V6's 32 MB at the default sizes (33.8 MB) and has no other sizes. Section 3.3 (line 412) still says a messaging file uses orders 0-1 only. See N-1 |
| G-3 | **closed** | PROBE: a file cut after stratum 0 with `DEFAULT_TIER = 1` compresses with tier 0 by default and round-trips. Text: 6.4 (775-776): default tier = `min(DEFAULT_TIER, highest tier present)`; C9 (1075) tests it |
| G-4 | **closed** | 8.6 (992) excludes by source GH Archive (any hour), the Samba project, Wikipedia, Project Gutenberg (with the Canterbury etexts named), `prepare-benchmark-data.py` output, Silesia, and the Loghub selection and test systems |

Minor findings: m-1, m-2, m-3, m-4, m-5, m-6, m-7, m-8, m-9, m-10, m-11, m-12, m-13, m-15, m-17, m-18, m-19, m-20,
m-21 closed (m-13: the boot term added to 7.1 is right in order of magnitude — a copy or a clear of 38.8 MB took
2.9 ms on the dev laptop, 13 GB/s, MEASURED with `boot_copy.py`, against the spec's conservative 4-8 ms; m-18 VERIFIED:
the frozen copies in `tools/strata/bakeoff/frozen-bakeoff-2026-11-12/` hash to
`29fc2638…`, `bef57807…`, `6e77a3e5…`, `5e1d9ed6…`); m-14 and m-16 partly (N-3; the `n` / `k` name clashes are kept
and declared).

Report-only numbers of `03-bakeoff.md` section 12 also re-derive from the files (VERIFIED, scripts `g0_check.py`,
`g0d_check.py`, `r113_check.py`, `economics_g0.py`): G0d cells on the selection messages, width 88 against zstd-19 +
same-source dictionary, 17 / 8 / 5-6 of 20 at + 4 / + 24 / + 28 B, foreign primer 3 / 0 / 0, foreign primer against
foreign dictionary 20 / 12 / 9-10; `cm` on long selection streams −6.3 % (`webster`), +20.3 % (`xml`), −27.0 %
(Zookeeper), −16.2 % (Proxifier) against the best classical codec, −5.5 % on the macro; on the four real held-out
test sources `cm` is 6.0 % and width 88 9.1-9.2 % smaller in total than the best classical codec per file (ideal code
lengths, no container).

### 10.2 Independent spec-only probe, re-run on the revised text

The same agent re-implemented the changed parts from the revised specification alone (it recorded the same sha256,
`c37cc06d…`, at the start and the end of its work and opened no other file). VERIFIED by re-running its scripts
(`<scratch>/t2probe/t2_appendix.py`, `t2_roundtrip.py`): Appendix A 46 of 46 spot values match; every derived cell of
the new table 7.2 matches; round trips pass for tiers 0 and 1, with and without primer, compact and default headers,
lengths 1-2,048; an empty input is rejected by `compress` and an archive with `TOTAL_BYTES = 0` by the decoder; a file
cut after stratum 0 with `DEFAULT_TIER = 1` defaults to tier 0 and round-trips; 13 corrupt weight files and 5 bad
container fields (FLAGS 0x48 / 0x44, PROFILE 0x11, ARITH 2, absent TIER) are rejected; the tight flush still has 0
failures on 20,000 random streams. The revision moved no bit for the readings the probe had used: its archives are
byte-identical to those of its first implementation. All of its earlier ambiguities A1-A15 are resolved in the text;
A4 and A5 (element order, `KF` / `KS` order) remain silent if misread, and only C9b can catch them; the per-layer
block order (A3), if misread, now also fails at load through the bias range check. No new ambiguity changes a
probability for a valid file; the gaps it found are in N-3 and N-6 below.

### 10.3 New findings introduced by the fixes or by the G0 result

| ID | Severity | Finding | Evidence | Correction |
|---|---|---|---|---|
| N-1 | grave | The trunk's only v1 role (messaging) rests on `S` measured with the default 38.8 MB tables; the shipping messaging file `M` will use other tables, chosen by a sweep with no registered selection rule, and nothing re-applies G0c's keep rule at `M`'s tables. The trunk, its trainer and the GPU plan could stay in v1 without a measurement at the configuration that ships | spec 9.2 line 1094, 8.8 lines 1027-1034, 7.2 lines 877-879; `05` has no sweep entry | Register in `05`, before the sweep: objective (per-cell `S` on selection sources, both primer cases, both seeds), constraint (V6), candidate sizes for bit-cell orders 0-3 that fit 32 MB, and "`msg88` is kept in `M` only if G0c's rule (>= 5 %, >= 5 % per 100K MAC/byte, both seeds) holds at `M`'s tables; otherwise `M` is `cm`-only" |
| N-2 | minor | Axis A0 asks to state where the design wins, ties or loses against published cells. G0a (and D2 / D3) compare on long streams, where the kept trunk is worth 0.0-0.5 %; on short messages, the only use that keeps it, no GRU / LSTM comparison exists | spec 2.9 line 328; `05` line 70-73 | Say so in 2.9 ("tie on long streams; not compared on messages"), or score the float controls on `S` with a float reference |
| N-3 | minor | Stale text after the freeze: 2.9 (331-332) "the document as a whole is not [frozen]"; 3.3 (382-384) "has to be repeated ... specified, not validated"; 3.3 (412) messaging bit cells "orders 0-1 only" vs the 9.2 grid {0-1, 0-3}; X4 (911) "G0c decides per data type"; X6 (913) "frozen test: +0.4 % macro", the old seen-source-inclusive macro (held-out only the geared trunk is 1.6 % worse); 2.5 (232) `n = n + 1` while 2.2 (127) and 2.6 (251) make `n` saturate (no bit difference while `n` cannot wrap); PROBE: 5.2 (574) still describes the old default-tier rule (6.4, 775, overrides it); Appendix B (1227) "pending gate G0"; 8.5 (985) "the ladder of 7.2" and 8.8 (1019-1023, the `large` tier, six tiers) no longer fit the two v1 tiers; 6.7 (836) 2 GiB / 38.8 MB is 55 streams, not 52; 8.3 (962) "21.6-28.4 KB": the formula as written gives 28.4 KB (kmax 10, L = 3); the 21.6 KB lower figure came from this review's m-3 (a per-layer sum) and does not follow from the text | lines cited | Edit; no bit consequence |
| N-4 | minor | 7.2 (lines 897-898) says the G0 margins "do not change the storage verdict". With G0's held-out ratios (VERIFIED, `economics_g0.py`), the break-even speeds per vCPU at $0.014/vCPU-h are, for `cm`: `dickens` 233, Apache 150, OpenSSH 169, Linux ~800 KB/s; for `msg88`: 120-155 KB/s except Linux 320-330 KB/s — inside the spec's own speed hypotheses (~1,000 and 260-420 KB/s per core). The value at stake is small ($0.005-0.026 per raw GB over 7 years, decompression not counted) and the competitor at that speed is a classical context mixer, not xz; but the sentence prejudges V4 | spec 7.2, V4 | "V4 decides"; compare `cm` with a classical CM at equal speed |
| N-5 | minor | V3's second pairing needs, for the JSON cells, a primer and dictionary source that is neither a training source nor GH Archive (now excluded by source); none exists on disk (`02-evidence` E.2), and the G0 report used another GH Archive hour, labelled same-source | spec V3 (1089); `05` line 118-120 | Name the JSON source of the second pairing before the run, or let the five JSON cells vote only in the first pairing (stated in advance) |
| N-6 | minor | Reader-validation gaps (PROBE): the table of 5.5 names "shifts" but not `TAP`, so a literal reader accepts `TAP = 7` and indexes `in[7]` at the first byte (crash, shown); the NAME rule does not settle empty names or a NUL inside a name (two readers disagree, shown); `NORD = 0` leaves "`max len[i]`" undefined (no bit effect, shown); nothing says to reject inputs above 2^32 − 1 bytes or archives shorter than header + 4 B | spec 5.5 (657-659), 2.2 (127), 5.6 (682), 4.2 (502) | Add `TAP` 0..3; "1-8 printable bytes, then zeros"; "`n` bound 0 when `NORD = 0`"; the two rejections |

Observation (VERIFIED, not a defect): the trunk's `S` saving is concentrated. Per primer case, width 88 saves
6.5 / 6.9 % with a same-source primer (logs about 12 %, `webster` and `xml` about 1 %) and 13 % with a foreign primer;
the keep rule holds in each case separately. Where primed Strata beats a same-source zstd dictionary at all (same-source
primer, real framing), the trunk adds least on text.

### 10.4 Does the freeze hold?

**Yes.** No blocking finding is reopened: B-1 to B-4 are closed, and an implementation written from the revised text
alone reproduces Appendix A, table 7.2 and its own archives without a guess. G0 was decided by the rule registered at
09:16 (hash verified), on results produced after that registration, and the decision re-derives from the files on
disk. G-1, G-3 and G-4 are closed; G-2 is partial.

Two conditions, neither of which touches the normative arithmetic or format:
1. **N-1** (grave), with the open half of G-2, must be registered in `05-gate-amendments.md` before the messaging
   sweep runs: the only v1 use of the trunk (the width-88 messaging tier) is justified at table sizes that will not
   ship, and the sweep that chooses the shipping file has no stated rule.
2. The text fixes of N-3 and N-6 (stale sentences, 55 not 52 streams, the chunk figure, `TAP` in the validation
   table, the NAME rule) before team O writes `make_synth.py` and the C9 tests.

Left open after the re-review: 1 grave (N-1), G-2 partial, 5 minor (N-2 to N-6).

Commands of the re-review (read-only on the repo; `<scratch>` as in section 9):

```
head -n 120 docs/internal/research/strata-backbone/05-gate-amendments.md | sha256sum      # 30a8cb51287c… (B-1)
python -I tools/strata/bakeoff/g0_decide.py --g0b pass                    # == data/strata/runs/g0-decision.txt
python -I <scratch>/review/g0_check.py data/strata/runs                   # S and L recomputed, duplicates removed
python -I <scratch>/review/g0_split.py data/strata/runs                   # saving per primer case and traffic type
python -I <scratch>/review/g0d_check.py data/strata/runs                  # G0d cells
python -I <scratch>/review/r113_check.py data/strata/runs/strata_g_bc     # R-113 report-only cells
python -I <scratch>/review/economics_g0.py data/strata/runs               # N-4
python -I <scratch>/review/boot_copy.py                                   # m-13 boot term
python -I <scratch>/t2probe/t2_appendix.py; python -I <scratch>/t2probe/t2_roundtrip.py   # 10.2
sha256sum tools/strata/bakeoff/frozen-bakeoff-2026-11-12/*                # m-18
```
