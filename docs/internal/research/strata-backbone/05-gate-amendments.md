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
