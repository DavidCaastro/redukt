# Strata backbone — 05. Gate amendments (rule T1 of `00-charter.md`)

Every change to a gate of `docs/api/BACKBONE-STRATA-SPEC.md` section 9 made after a related measurement was
seen, with the date and the reason. No gate measured on the Rust implementation exists yet; the entries below
were made after the design-stage bake-off (`03-bakeoff.md`), so they are recorded here to keep the trail complete.

| Date | Gate | Change | Seen before the change | Direction | Reason |
|---|---|---|---|---|---|
| 2026-11-13 | G0 (new, 9.0) | design gate added: repeat the bake-off with the bit cells of section 3.3 in the training loop and the width-0 tier trained as a tier; keep a trunk tier only where it buys >= 5 % | bake-off results and the post-hoc bit-cell exploration (`03-bakeoff.md` section 8) | stricter (adds a gate before any Rust or GPU work) | section 3.3 was added after the bake-off and has never been measured in the loop |
| 2026-11-13 | V3 | must pass in both pairings: primer + dictionary from the message source, and primer + dictionary from a different source of the same type | prototype short-message result with a same-source primer (`03-bakeoff.md` 7.6: 30 of 30 cells) | stricter | the same-source pairing is a best case for both codecs; a gate that only measures it would be too easy |
| 2026-11-13 | C1b, C2d | primer archives added to the bit-identity and batch == single sets | — | stricter | the primer (section 3.4) is a new v1 feature |

## G0 — pre-registration (written 2026-11-13 08:45, before any G0 code was written or run)

Request of the coordinator (2026-11-13): run G0 as defined in specification 9.0, on the same selection set,
against the same controls and the best classical codec; the frozen test set only after the decision; an explicit
rule, in saving per MAC, for keeping the trunk or shipping the width-0 tier (`cm`) alone; a primer from a different
source than the zstd dictionary in the short-message comparison.

### Design parameters fixed before the run (not gates)

- Bit cells of specification 3.3: orders 0, 1, 2, 3 with 8, 16, 22, 22 table bits. (The draft's 8 / 16 / 20 / 22
  put the 65,536 possible order-2 contexts into 4,096 buckets; 22 bits gives 16,384.) The specification's default
  follows what G0 measures.
- Everything else exactly as in the bake-off: data and sampler, seed 1204, 2,048 steps of 64 x 256 bytes over
  chunks of 8,192, AdamW 2e-3 with the same schedule, budgets tier A <= 32,000 MAC/byte and 96,000 parameters,
  tier B <= 128,000 and 256,000.

### Models

| Name | What |
|---|---|
| `strata_g_bc` | the geared candidate chosen by D1, with bit cells; three nested tiers trained in one run: `cm` (width 0), A (width 88), B (width 208); integer arithmetic |
| `gru_bc`, `lstm_bc` | tier-B controls (same cells and sizes as the bake-off) in the same harness, bit cells included; float |
| `gru_a_bc`, `lstm_a_bc` | tier-A controls, same; float |
| best classical | xz -9e, zstd -19 / -22, brotli -11, bz2 -9, gzip -9 on the same bytes |

### Metrics (selection data only)

- `L(model, tier)`: macro bpb over the four selection sources (first 300,000 bytes of `webster`, `xml`,
  Zookeeper, Proxifier), each one stream from the boot state, no reset, as in the bake-off.
- `S(tier)`: total payload bytes of `strata_g_bc` on a selection short-message protocol: the R-113 cutting rule
  (16 messages of 128 / 256 / 512 / 1,024 / 2,048 bytes from the second half of each source, at line starts) on
  `webster` and `xml` (first 4,000,000 bytes) and Zookeeper and Proxifier (2,000,000 bytes); each message coded by
  the integer reference from the primer snapshot of specification 3.4, real range-coder bytes with the tight flush;
  summed over two primer cases: **same-source** primer (the last 65,536 bytes of the first half of the message
  source) and **foreign** primer (the same position in a different source of the same type: `webster` <-
  `lcet10.txt`, `xml` <- the Wikipedia training slice (first 4,000,000 bytes), Zookeeper <- Proxifier, Proxifier
  <- Zookeeper).

### Decision rule

- **G0b, integer contract.** Trainer simulation == sequential integer reference with bit cells: 0 mismatching
  probabilities on 12,000 bytes of `webster` for each of the three tiers, and a 2,000-byte round trip per tier.
  Must pass; a failure is reported, its cause fixed and G0b re-run before any other decision is taken.
- **G0c, keep the trunk or ship `cm` alone, in saving per MAC.** For a trunk tier `k` and a metric `X` in
  {`L`, `S`}: `s = 1 - X(k) / X(cm)` (relative saving over the `cm` tier of the same weight file) and
  `e = s / (dMAC / 100,000)` with `dMAC = MAC/byte(k) - MAC/byte(cm)` (% saving per 100,000 MAC/byte).
  - Tier A is kept if, for `L` or for `S`, `s >= 5 %` and `e >= 5 %` per 100K MAC/byte.
  - Tier B is kept if, for the same metric, `s >= 5 %` and `e >= 5 %` against `cm`, **and** its marginal
    efficiency over tier A, `(1 - X(B) / X(A)) / ((MAC(B) - MAC(A)) / 100,000)`, is also `>= 5 %` per 100K
    MAC/byte.
  - **No tier kept: v1 ships the `cm` tier only.** Specification sections 2.3, 2.4, 8 and the GPU plan leave v1;
    the weight file holds the width-0 tier's trained node biases, gains, boot weights and tables.
  - A tier kept on `S` only: the trunk stays for messaging weight files (path 3); the default for long inputs is
    `cm`.
  - Rationale: 5 % is the repo's unit for ratio changes; the bake-off's tier-B budget (~100K MAC/byte) has to buy
    at least that unit. Tiers wider than B (`small` .. `large` of the ladder) are not justified by this rule; they
    need their own measurement.
- **G0a, trunk vs published cells**, for the widest kept tier, always on `L` (the controls have no integer
  reference for `S`): `S_ = L(strata_g_bc, that tier)`, `C` = the lower `L` of the two controls of the same budget
  (tier B: `gru_bc`, `lstm_bc`; tier A: `gru_a_bc`, `lstm_a_bc`). WIN if `S_ < 0.98 C`, TIE within +-2 %, LOSS
  above `1.02 C`. Not applicable when no tier is kept.
- **G0d**, reported only: `L` of every model against the best classical codec; the two `S` cases against zstd-19
  with a same-source dictionary and with a dictionary trained on the foreign source.
- **G0 passes** if G0b passes and either (i) a trunk tier is kept and G0a is WIN or TIE for the widest kept tier,
  or (ii) no trunk tier is kept (v1 = `cm`). It fails otherwise. The specification is frozen only if G0 passes.
- Not used for any decision: the frozen test set, the seen-source slice, D3 of the bake-off (written after the
  test set was visible), wall-clock time, float twins.

### Report-only, after the decision is written

Frozen test set with every G0 model; the R-113 short messages (test sources) with no primer, a same-source primer
and a foreign primer (Wikipedia and prose <- `lcet10.txt`; the three logs <- Zookeeper; JSON <- another hour of
the same public event archive, 2015-01-02 15:00), against zstd-19 with a same-source dictionary (the case
unfavourable to Strata) and with a dictionary trained on the primer's source; length curves if time allows.

## G0 — amendment 1 (written 2026-11-13 09:16, before any G0 selection or test result was read): one definition

Resolves finding B-1 of `04-design-review.md`. What had been seen when this was written: training-set costs of the
G0 runs in their training logs (up to step 150-600), and the 64-step smoke model's integer check on 12,000 bytes of
`webster` (3.0 bpb, an untrained model). No selection-set or test-set result of any G0 model had been read.

1. **One normative text.** G0 is defined only in this file: the 08:44 pre-registration as amended here. Section 9.0
   of the specification refers to this file and carries no rule of its own.
2. **Votes come only from sources never seen in training.** `L` uses the four selection sources; `S` uses messages
   cut from the same four sources. No primer comes from a training source: the foreign primers are now the other
   selection source of the pair, `webster` <-> `xml` and Zookeeper <-> Proxifier (08:44 used `lcet10.txt` and the
   Wikipedia slice, both training sources). The seen-source `enwik8` slice, every training source and every test
   source are excluded from every vote; they may be reported.
3. **`S` per cell, not in bytes.** For a tier `k`: `s_S(k) = 1 - exp(mean over the 40 cells (4 sources x 5 sizes x
   2 primer cases) of ln(P_k / P_cm))`, `P` = mean payload bytes of the 16 messages of the cell, so every (source,
   size, primer) cell weighs the same. The byte total and the values per traffic type (`webster` + `xml`; the two
   logs) are reported next to it and decide nothing. `e` and the marginal test of tier B over tier A use this `s`.
4. **Seed variance.** A second seed of the candidate, `strata_g_bc` with seed 1205 (initialisation and data order;
   same `Q0`, everything else identical), is trained and evaluated in every case; G0 is decided only when both seeds
   have `L` and `S`.
   - G0c: a tier is kept on a metric only if the keep conditions hold for **each** seed separately. The spread
     `|s(1204) - s(1205)|` is reported per tier and metric.
   - G0a: the verdict (WIN / TIE / LOSS, same bands) is computed for each candidate seed against the better
     control of that budget (controls: seed 1204 only); **G0a is the worse of the two verdicts** (LOSS worse than
     TIE worse than WIN).
   - Controls are not re-trained with a second seed (time); their seed noise is unknown and is stated as a
     limitation next to the result.
5. **Known weak points kept, not changed** (G0 must stay comparable with the bake-off): training chunks of 8,192 B,
   below the specification's chunk-length rule (review m-3); Project Gutenberg etexts in the training data (m-8);
   G0b is a consistency check between two implementations by the same author, not evidence under rule T2 (m-9).
6. **Report-only primers of the R-113 test protocol** (after the decision): Wikipedia and prose <- `webster`; the
   three logs <- Zookeeper; JSON <- another hour of the same public event archive, labelled **same source** (it is
   the same source under the by-source rule).

## Amendment 2 (2026-11-13 10:15): gates of section 9.1-9.2 changed after review 04

These gates measure the Rust implementation, which does not exist yet; they were changed after the reviewer and the
architect had seen the prototype's numbers, so each change is recorded here. All but V1 make a gate stricter or
remove an ambiguity.

| Gate | Change | Direction | Reason |
|---|---|---|---|
| V3 | votes on the 25 held-out cells only (Wikipedia, a training source, excluded), threshold 20 of 25, counted with the real framing of a primed archive (24 B compact), on a named messaging file `M`, in both pairings | stricter | review G-1 (framing was 20 B in the first count; seen-source cells voted) |
| V6 | measured on the same file `M` as V3, in the same run | stricter | review G-2 (ratio and footprint could pass on different files) |
| V1 | votes on the 4 real held-out sources (3 of 4 and total bytes); the two synthetic invoice files are reported apart | changed: two votes removed. With the prototype's numbers V1 goes from 3 of 6 wins to 3 of 4, and the total over the 4 is 0.2 % above the classical total (still a fail) | review m-11: `02-evidence` E.1 declares the synthetic XML invalid as a generalisation gate |
| V2 | Strata and Mamba-130M on identical bytes | stricter | review m-11 (CRLF vs LF copies) |
| C3 | empty input rejected; primer archives included | clarified | review B-4 |
| C4a | late window <= 1.10 x early window (was 1.25) | stricter | review m-12 (1.25 would hide a 20 % drift; the prototype measured 1.043) |
| C9, C9b | the reader-validation table of 5.5 and the default tier of a truncated file are tested; new C9b checks the weight-file layout across teams | new / stricter | review B-2, G-3, m-2 |

## Amendment 3 (2026-11-14): accelerator path reserved (portability principle)

Requested by the owner on 2026-11-14, before any Rust, reference or device code exists. The question was raised on
2026-11-13 while answering whether the design parallelises and adapts to the hardware: the frozen v1 text defined no
inference path on an accelerator (GPU appeared only for training, section 8.8), while the repository's portability
principle (`CLAUDE.md`) asks that every backbone be demonstrable on CPU and on any accelerator through the same format
and CLI, with the CPU as the reference.

Seen before this amendment: nothing new. No gate has been measured on a Rust implementation, no device executor
exists, and no number was produced or read for this change. The only figures used are DERIVED from section 7 of the
specification (MAC per byte, state bytes, the 10-20 G MAC/s and ~1 us per byte hypotheses of 7.2) and from the
FFI v2 logits size; they are labelled as such in section 6.9.

| Gate / section | Change | Direction | Reason |
|---|---|---|---|
| C12 (new, 9.1) | Accelerator == CPU, deferred until a device executor exists (like C11): identical sha256 of the C1b set and the C2d batches with the device executor forced and with the CPU executor forced (encoder windows included), decode in both directions across devices, and the load-time probe must reject two deliberately faulty executors (`asr` rounding toward zero; clamp to -128) with an unchanged archive after the fallback | stricter (adds a gate) | an accelerator must give the same bits as the CPU reference by construction (axiom A1); the gate makes that claim testable before any device code is written |
| 6.9 (new) | offload boundary = section 2.3 only; exact contract, the device is part of the execution plan and is never written to the header; encoder windows (the trunk depends only on the bytes); data exchanged per byte (`d + NORD` bytes to the host); device arithmetic traps; load-time probe with an exact fallback; reserved ABI (`docs/api/PLUGIN-FFI-STRATA.md`, slot model of FFI v2, the weight file itself as input); the seam stage 4a must keep; expected value | no threshold | portability principle; the R-110 lesson (a device path is capped by its per-step host copy) applied before implementation |
| 3.5, 6.2, 6.5, 6.7, X17, section 10, risk 10, Appendix D, status block | cross-references: H7 as an execution hook with no encoding; `begin_byte` as the only offloadable call; Phase B as one device call; a "trunk executor" row in the autoscaler table (v1: CPU only); crossover row X17; ownership of C12; risk row 10 | no threshold | consistency of the frozen text with 6.9 |

Effect on the freeze: none of B-1..B-4 or G-1..G-4 is reopened. The arithmetic of sections 2-4, the weight file and
the AZTH declaration of section 5, the v1 scope fixed by G0 and every existing gate threshold are unchanged, so the
freeze of 2026-11-13 stands. v1 ships no accelerator code; the only obligation on stage 4a is the seam of 6.9.6
(a plain module boundary, no dead code).

## Amendment 4 (2026-11-14): owner's objective, PPMd in the classical set, capacity-ladder gate V7

Requested by the owner on 2026-11-14, after asking what objective the effort serves. The owner's statement, in
substance: Strata is the project's intended backbone, built from scratch on the lessons learned, and it should reach
what the product reaches today with a process designed for the codec. The economic reading given to the owner the
same day: the storage saving per GB is fixed and grows linearly with the retention time, so the only variable that
can close the Fase 20 gap is the compute cost per byte, which is the size of the model. The question this amendment
turns into a gate is therefore: **on held-out data, is there a Strata size whose compute cost per raw GB is below the
storage value of the bytes it saves against the best classical codec, and how close to Mamba-130M's ratio is it?**

Seen before this amendment (disclosed in full):
- every result of `03-bakeoff.md`, including the frozen test set of its section 12.2 (report-only, read on
  2026-11-13);
- the PPMd baseline of `tools/strata/viability/ppmd_baseline.py`, run and read on 2026-11-14 before this text was
  written (`03-bakeoff.md` section 13);
- the output of `tools/strata/viability/viability.py` for the three trained rungs (`08-capacity-ladder.md`
  section 3, "position at registration").

Not seen, because it does not exist: any Strata model wider than 208, any model trained on more than the 33.55 MB of
the bake-off or for more than 2,048 steps, any measurement of arm P below, any Rust throughput.

| Gate / section | Change | Direction | Reason |
|---|---|---|---|
| 9.2 preamble; V1, V4 and every "best classical" figure from now on | the classical set is xz -9e, zstd -19 and -22, brotli -11 (lgwin 24), bz2 -9 and **PPMd** (7-Zip PPMd var.H, model orders 4, 6, 8, 12, 16, 24, 32 with 256 MB, smallest packed stream, container header excluded). zstd -22 and bz2 -9 are written down because the bake-off already used them | stricter | the width-0 tier is a context mixer; a classical set made only of the LZ and BWT families leaves out its closest classical relative. MEASURED effect on the frozen test set (ideal code lengths, mean of two seeds): the best classical total over the four real held-out sources falls from 163,699 B to 147,595 B; `cm` goes from -6.0 % to **+4.2 %**, width 88 from -9.2 % to **+0.75 %**, width 208 from -10.4 % to -0.6 %. Every tier now loses on `dickens` (+3.7 % to +8.1 %) and still wins on the three logs. Selection macro of the best classical codec: 0.6556 -> 0.5879 (the G0 tiers are at 0.614-0.619). **Projected V1: FAIL for both v1 tiers** (3 of 4 sources, but the total is above the classical total) |
| V7 (new, 9.2) | capacity ladder, defined below | new gate; no existing threshold is touched | the owner's objective needs a measurement with a rule written before it |
| `00-charter.md` section 1 | objective recorded (section 1b) | — | owner's instruction |

The G0 decision is not reopened: its rule compared tiers with each other and with equal-budget controls, never with a
classical codec, so the classical set does not enter it.

### V7 — capacity ladder (pre-registered 2026-11-14, before any ladder model exists)

**Arms.**
- Arm W (capacity in compute): one model per rung, each trained alone or nested, on a training set at least 10x the
  bake-off's bytes. Rungs by cap on MAC per byte: R0 = 208 (`cm`), R1 <= 32,000, R2 <= 128,000, R3 <= 512,000,
  R4 <= 2,000,000. R4 is a slope probe: by the ceilings of `08-capacity-ladder.md` section 4 it cannot be viable for
  storage on logs at any ratio, and on text only at 20 G MAC/s with an output close to zero bytes.
- Arm P (capacity in memory, constant compute): the primer mechanism of specification 3.4 with a general primer built
  from training sources only, sizes 0, 64 KB, 1 MB, 16 MB and the largest the table state allows, on rungs R0 and R1.
  CPU only. Its classical competitors receive the same prior where they can (zstd -19 with the primer bytes as a
  dictionary; brotli with a custom dictionary if the binding allows); PPMd has no such mode and is reported without.

The exact configuration of every model (widths, depth, table bits, steps, training sources and weights, seeds) and
the extended decision files are written in `08-capacity-ladder.md` and its sha256 is logged in the charter ledger
**before the first ladder model is evaluated on any decision file**. Hyper-parameters are chosen on the selection
set only.

**Data.** By-source split as in the bake-off: no source contributes to more than one role. The decision is read on
(a) the existing frozen test set, classes `text` (`dickens` 500 KB) and `logs` (Apache, Linux, OpenSSH), and (b) an
extended set with at least two text sources, three log systems and two real markup or structured sources, each of
1 MB or more where the source has that much. A class verdict is the **worse** of (a) and (b). The two synthetic
invoice files are reported and do not vote (as in V1).

**Quantities** (per rung `r` and class `c`; computed by `tools/strata/viability/viability.py`):
- `B_r`, `B_cl`, `B_mamba`: total coded bytes of the class for the rung, for the best classical codec of each file,
  and for Mamba-130M on the identical bytes (Mamba re-run locally, real `.azth`; the first 1 MiB of a file is enough).
  Ideal code lengths of the integer reference are accepted for `B_r` until the Rust backbone exists, and labelled.
- `delta = 8 (B_cl - B_r) / N` bpb; `saving = 1 - B_r / B_cl`; `parity = B_r / B_mamba`.
- `seconds per byte = 1 us + MAC_r / R_mac` (the hypothesis of specification 7.2); `R = 1 / seconds`.
- `viable <=> delta >= 16.10 / R[KB/s]` (gate V4's inequality: $0.014 per vCPU-hour, $0.023 per GB-month, 7 years).

**Thresholds.** Decision point `R_mac = 10 G MAC/s` (the lower hypothesis of 7.2); 2.5 G (today's effective rate,
`02-evidence` C.5), 20 G and the $0.022 price are reported next to it.
- cost: `viable` at the decision point;
- materiality: `saving >= 10 %` (a codec a few percent ahead of PPMd is not a product);
- parity: `parity <= 1.10` ("the same as Mamba-130M"); `<= 1.30` is reported (V2's threshold).

**Outcomes, per class.**
- O1, objective met: a rung of arm W with width > 0, or any rung of arm P with a primer, passes cost, materiality and
  parity. Conditional on gate V4 measured on the Rust implementation.
- O2, cheaper and weaker: cost and materiality pass, parity does not. Reported with its distance to Mamba; whether
  that is a product is the owner's decision.
- O3, not viable: otherwise.
- Separately reported: R0 without a primer (a classical-class context mixer). It passing cost and materiality is not
  the objective; it is recorded as "CM only".

**Stop rules.** (1) If no class reaches O1 or O2 with a trunk rung or a primed rung, the trunk does not pay for
storage at any tested capacity: the storage question is closed for Strata and no further capacity work starts
without new evidence; the path-3 scope of v1 is unaffected. (2) R4 is trained only if R2 -> R3 lowers the total
bytes of at least one class by 2 % or more. (3) GPU budget and order as in charter 5b (12 h ceiling; CPU end to end,
then a smoke run of at most 30 minutes, then the long runs).

**New constants and where the trained rungs stood when they were chosen** (the 10 % and 1.10 are judgements made with
the position of R0-R2 known; everything else was already registered in V4, 7.2 or `scripts/breakeven.py`): on the
frozen test set with PPMd in the classical set, `logs`: R0 saving 15.0 %, parity 1.29, cost passes; R1 20.2 %, 1.21,
cost passes at 10 and 20 G and fails at 2.5 G; R2 22.0 %, 1.18, cost fails at all three rates. `text`: every rung
loses to PPMd (+3.7 % to +8.1 %), parity 1.92-2.00. So at registration: `logs` = "CM only" passes and R1 would be O2;
`text` = O3; nothing is O1. The unknowns V7 measures are better-trained and larger rungs, and arm P.

Effect on the freeze: the arithmetic of sections 2-4, the weight file and the AZTH declaration of section 5 and the
v1 scope fixed by G0 are unchanged, so the freeze of 2026-11-13 stands. Ladder models above width 88 and general
primers are not part of v1; a rung that reaches O1 or O2 enters the product only through its own amendment.

## Amendment 5 (2026-11-14): V7 details fixed before any decision evaluation (PPMd memory, extended slices, arm P-s)

Written after the arm-P exploration on the selection set (`09a-arm-p-selection.md`) and the smoke run of the
trainer, and before any Strata model (primed, retrained or not) was evaluated on a decision file of V7.

Seen before this amendment: classical codec sizes on the seven extended slices (`baselines-ext.csv`; classical
codecs only), the arm-P results on the four selection sources, the training costs of the smoke run. Not seen: any
Strata result on the extended set; any primed or retrained Strata result on the frozen test set.

| Gate / section | Change | Direction | Reason |
|---|---|---|---|
| classical set, PPMd | PPMd is run through standard input with 1 GB of model memory (`baselines_ext.py`), not on a file: given a file, 7-Zip lowers the PPMd memory to about 16x the file size whatever `mem=` says. The best PPMd of each file is the smaller of the two modes | stricter | MEASURED on the extended slices: 4 of 7 improve (`nci` -4.3 %, the others <= 1.1 %); selection and frozen-test figures are unchanged |
| V7 extended set | fixed as: `dickens` [500,000, 1,548,576); `bible.txt` [0, 1 MiB); full Loghub `Apache.log` [0, 1 MiB); `Linux.log` and `OpenSSH.log` [951,424, 2,000,000); GitHub event archive hour 2015-01-01-15 [0, 1 MiB); Silesia `nci` [0, 1 MiB). Classes: text = `dickens` + `bible`; logs = Apache + Linux + OpenSSH; structured = the JSON hour + `nci` (reported and judged as a third class, labelled; the frozen test set has no such class, so for it the extended set alone decides) | neutral / stricter | the first MiB of the full Linux and OpenSSH logs contains every line of the frozen-test samples (MEASURED: 2,000 of 2,000 lines each), so those slices would have counted the same bytes twice; the tails share no line with them. Apache shares none |
| V7 arm P | registered configuration: specification 3.4 unchanged; primer `mixI` (training sources only); sizes 0, 65,536, 1,048,576, 16,777,216 with **1,048,576 as the primary point**; primed rungs use the large tables `L` (successor 8/16/22x6, bit cells 8/16/24/24; 235.4 MB of state per stream); rungs R0 and R1 with the G0 weights, seeds 1204 and 1205 | — | chosen on selection data only (`09a` section 5) |
| V7 arm P-s (new, labelled) | a second primed arm: the snapshot keeps only the successor tables (bit cells, calibration cells and mixer start from boot). Same primer, tables and rungs; sizes 1,048,576 and 16,777,216, **primary point 16,777,216**. It is NOT specification 3.4: an outcome obtained with it enters the product only through an arithmetic amendment | adds one more chance to pass, disclosed | on selection the primed bit cells are what hurts (`09a` 4.3); the variant was picked among six on four sources, which is selection-set fitting and is why it is a separate, labelled arm |
| V7 verdict for primed rungs | a primed rung is judged at its primary point only; the other sizes are reported. The classical side of a primed rung is the smaller of: best classical without prior, zstd -19 / -22 with the primer as raw-content dictionary. Conditional PPMd / xz (C(primer + input) - C(primer)) are reported next to it as a stricter, non-deployable reference | stricter than "any size" | one point per rung keeps the verdict from being the best of several sizes |

Expectation written before the measurement (INFERENCE from the selection set): arm P ends O3 on text (about -2 %
against PPMd at best, and the gain is front-loaded: 47 % on the first 10 KB, 1.4 % on the last 200 KB of a 300 KB
input) and lowers the unprimed saving on logs.
