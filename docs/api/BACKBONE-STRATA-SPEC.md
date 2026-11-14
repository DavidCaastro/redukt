# Strata backbone — normative specification (arithmetic version 1)

> Status: **FROZEN for v1 (arithmetic version 1), 2026-11-13.** Design gate G0 passed (rule in
> `docs/internal/research/strata-backbone/05-gate-amendments.md`, decision in `03-bakeoff.md` section 12) and the
> four blocking findings of the adversarial review (`04-design-review.md`: B-1 to B-4) and its grave findings
> (G-1 to G-4) are fixed in this text; the reviewer has not yet re-checked the fixes, and a re-check that reopens a
> blocking finding withdraws the freeze by amendment. Every later change goes through a dated amendment in
> `05-gate-amendments.md` (rule T1).
> **v1 scope, fixed by G0:** the width-0 tier `cm` (no trunk) for long inputs; one trunk tier of width 88 for
> messaging weight files (path 3). Wider tiers are not part of v1 (section 7.2).
> Date: 2026-11-13. Working name: Strata. Nothing in this document is implemented in Rust yet.
> Scope: an integer, coder-domain backbone for Redukt, its weight file, its AZTH declaration, its integration
> module ("the dock"), its cost model, its training recipe, its gates and its test plan.
> Evidence: the design choices marked **MEASURED** were measured by the bake-off of
> `docs/internal/research/strata-backbone/03-bakeoff.md` (commands there). Everything else is **DERIVED**
> (arithmetic on stated inputs) or **HYPOTHESIS** (to be calibrated; a gate says how).
> The key words MUST, MUST NOT, SHOULD and MAY are normative.

Three independent teams (Rust runtime, numpy reference, PyTorch trainer) MUST be able to produce bit-identical
probabilities and archives from sections 2 to 7 alone. If two readings of a sentence give different bits, the
sentence is a defect of this document: it is fixed here first (rule T2), never settled by reading another
team's code.

---

## 0. Summary

Strata predicts **bits, in the coder's own domain**. A byte is eight binary decisions; for each decision the
model produces a 16-bit integer probability that goes straight into the existing AZTH range coder. There is no
softmax, no logit vector, no floating point anywhere between the input byte and the coded bit.

It has three parts, all integer:

1. **Trunk** — a stack of *strata layers*. Each layer is one int8 matrix followed by saturation and by a bank
   of leaky integer accumulators with fixed power-of-two time constants. There is **no learned recurrent
   weight**: memory lives only in accumulators whose contraction is fixed by the arithmetic, which is what makes
   the state provably bounded and provably forgetful (section 2.6).
2. **Online context tables** — small per-stream hash tables that remember, for eight context lengths, which byte
   followed each context last time, with online-calibrated confidence. They are updated identically by the
   compressor and the decompressor. They are the part that adapts to a source the model has never seen. They
   come with adaptive low-order bit statistics ("bit cells", section 3.3) for the bytes that no table predicts.
3. **Head and mixer** — per bit, the trunk's opinion (one int8 dot product with a node vector of a binary tree)
   and the tables' evidence are added in the logistic domain with weights that start from learned boot values
   and are adapted online by an integer gradient step. The trunk also emits per-context gains: the state learns
   which contexts to trust.

Model sizes are **nested by construction**: channels are ordered in strata, a channel only reads channels of
its own or lower strata, so a smaller tier is an exact prefix of a larger one — in weights *and in activations*.
One weight file, truncated at a stratum boundary, is a valid smaller model; one training forward pass trains
every tier.

A stream may start from a **primer** (section 3.4): the state left by reading a dictionary-like text that both
sides hold; on short messages it is the decisive lever (with real framing it does not yet beat a same-source zstd
dictionary on the selection sources: `03-bakeoff.md` 12).

**What G0 measured (`03-bakeoff.md` 12).** With the bit cells of 3.3 in the training loop, the width-0 tier alone
(208 MAC/byte) codes the four selection sources at 0.619 bpb, 5.5 % below the best classical codec per source, and
a trunk adds 0.0-0.9 % on long streams; on short messages the width-88 trunk saves 10 % per cell (16-17 % on logs).
Hence the v1 scope above.

What is new here is the combination and its derivation from the codec; the ingredients have relatives in
prior work, listed without softening in section 12.

---

## 1. Notation and conventions

- Integers are two's complement. `asr(x, k)` is the arithmetic right shift: `asr(x, k) = floor(x / 2^k)` for
  any sign of `x`, `k >= 0`. `x << k` is multiplication by `2^k` (never applied to a value that would overflow
  the stated width). `clamp(x, lo, hi) = min(max(x, lo), hi)`.
- No operation in sections 2 to 4 may wrap, except where `mod` is written explicitly (hashes of 3.1 and 3.3, the
  coder of 4.2): every accumulator has a proven bound (section 2.7). An implementation that detects a value
  outside its bound MUST treat it as an internal error.
- "int8" values in this document are restricted to **[-127, 127]**; the value -128 never appears in weights,
  activations or embeddings (the exporter and the saturations guarantee it). This is what lets
  `u8 x i8 -> i16` SIMD multiply-add kernels be exact (section 6.6).
- Bytes of the input are `x[0] .. x[N-1]`. Bits of a byte are numbered `j = 0 .. 7` from the **most
  significant** (`bit_j(v) = (v >> (7 - j)) & 1`). `prefix_j(v) = v >> (8 - j)` is the value of the first `j`
  bits (`prefix_0 = 0`).
- `BOS = 256` is the input symbol that stands for "before the first byte".
- Multi-byte fields of files are little-endian.
- All sums of products are exact integer sums; the order of accumulation is free (integer addition is
  associative), which is the whole basis of ISA-, thread- and batch-independence.

---

## 2. Architecture

### 2.1 Configuration (read from the weight file, section 5)

| Symbol | Meaning | v1 constraints |
|---|---|---|
| `K` | number of tiers, indexed `k = 0 .. K-1` | `1 <= K <= 8` |
| `d_k` | cumulative width of tier `k` | `0 <= d_0 < d_1 < ... < d_{K-1} <= 4096`, multiples of 8 |
| `L` | number of strata layers (same for every tier) | `1 <= L <= 16` |
| `GEARED` | whether layers have gates (section 2.4) | flag |
| `tap[c]` | which delayed input feeds input channel `c` | `0..3` |
| `kin[c]` | shift of input channel `c` | `0..12` |
| `kf[l][c]`, `ks[l][c]` | shifts of channel `c` of layer `l` (fast, slow; equal when not geared) | `0..12` |
| `NORD`, `len[i]`, `tb[i]` | context orders: count, length in bytes, log2 of table slots | `NORD <= 8`, `1 <= len[i] <= 32`, strictly increasing; `8 <= tb[i] <= 22`; `tb = 8` when `len = 1`, `tb = 16` when `len = 2` |
| `MIX_SH` | mixer learning-rate shift | `12..20` |
| `NBC`, `blen[m]`, `btb[m]` | bit-cell orders (section 3.3): count, context length, log2 of cells | `NBC <= 4`; `blen` in `{0, 1, 2, 3}`, strictly increasing; `btb = 8` for length 0, `16` for length 1, `16..24` otherwise |

Constants of arithmetic version 1 (not configurable): `A = 5` (activation scale 32), `F = 12` (accumulator
fraction bits), weight shift 7, head shift 3, gain shift 6, `ST_MAX = 3071`, `Q_BITS = 22`, `NC = 32`
confidence buckets, mixer weights Q16 clamped to `W_MAX = 2^19 - 1`, `N_IN = 1 + NORD + NBC`,
`NSET = 8 * (NORD + 1)`.

The **stratum** of channel `c` is `s(c) = min{k : c < d_k}`. Channel `c` of a layer reads exactly the channels
`c' < d_{s(c)}` of the previous layer. Tier `k` consists of the channels `c < d_k`; since no channel reads a
higher stratum, the values of those channels do not depend on which tier is being run.

Tier `k` with `d_k = 0` is legal (the trunk is empty; the head sees only its biases, the tables and the mixer).

### 2.2 Per-stream state

| Item | Type and range | Initial value (boot) |
|---|---|---|
| `in[0..3]` | last four inputs, `in[0]` the most recent | `BOS, BOS, BOS, BOS` |
| `S0[c]`, `c < d` | i32, `|S0| <= 127 * 2^12` | 0 |
| `S[l][c]`, `l = 1..L` | i32, same bound | 0 |
| successor tables `T[i][slot]` | (valid, check 8 bit, succ 8 bit, run 0..3) | all invalid |
| `trun[i]` | 0..255 | 0 |
| calibration cells `q[i][conf][j]`, `n[i][conf][j]` | 22-bit, 0..255 | `Q0[i][conf][j]`, 0 |
| bit cells `qb[m][idx]`, `nb[m][idx]` (section 3.3) | 22-bit, 0..255 | `2^21`, 0 |
| mixer weights `w[set][m]` | i32, `|w| <= W_MAX` | `W0_k[set][m]` (tier `k`) |
| history of the last `max len[i]` bytes; byte count `n`, saturating at `max len[i]` (so it is bounded) | | empty, 0 |

There is no reset operation other than "back to boot" (or back to the primer snapshot of section 3.4)
between files. The boot state is entirely defined by
the weight file: the first byte of a stream is predicted by the trained response to four `BOS` inputs, not by a
uniform distribution.

### 2.3 Byte step, part 1: the trunk

Executed once per byte, before the byte is coded (encoder) or decoded (decoder), with the state left by the
previous bytes. `d` below is the width of the tier in use.

```
# input bank
for c in 0 .. d-1:
    v      = EMB[ in[tap[c]] ][c]                       # int8 table lookup, EMB is [257][d]
    S0[c]  = S0[c] - asr(S0[c] - (v << 12), kin[c])
    h[c]   = asr(S0[c], 12)

# strata layers
for l in 1 .. L:
    for c in 0 .. d-1:                                   # all rows read the h of the previous layer
        a[c] = B[l][c] + sum over c' < d_{s(c)} of W[l][c][c'] * h[c']          # i32
        u[c] = clamp(asr(a[c] + 64, 7), -127, 127)
    if GEARED:
        for g in 0 .. d/8 - 1:                           # one gate per group of 8 consecutive channels
            ag[g]   = BG[l][g] + sum over c' < d_{s(8g)} of WG[l][g][c'] * h[c']
            open[g] = (ag[g] >= 0)
    for c in 0 .. d-1:
        k      = kf[l][c] if (not GEARED or open[c / 8]) else ks[l][c]
        S[l][c] = S[l][c] - asr(S[l][c] - (u[c] << 12), k)
        h'[c]  = clamp(h[c] + asr(S[l][c], 12), -127, 127)
    h = h'

# trunk-driven gains, one per context order (tier k)
for i in 0 .. NORD-1:
    g[i]    = GB_k[i] + sum over c < d of GW[i][c] * h[c]
    gain[i] = clamp(asr(g[i] + 32, 6), 0, 127)          # Q5: 32 means 1.0
```

`W[l]` is `d x d` int8 with `W[l][c][c'] = 0` (not stored) for `c' >= d_{s(c)}`; `B`, `BG`, `GB_k` are i32;
`EMB`, `GW`, `WG` are int8. With shift 0 the accumulator update reduces to `S = v << 12`, i.e. a plain
feed-forward channel; with `tap[c] > 0` and `kin[c] = 0` the channel is an exact delay tap.

### 2.4 Gears (optional feature, flag `GEARED`)

A geared channel has two legal time constants and an input-driven gate picks one per byte. The gate is computed
from the previous layer only, never from the channel's own accumulator, so it cannot create a feedback loop.
Both shifts are finite (`<= 12`): a closed gate still leaks. Whether v1 weights use gears is decided by the
bake-off (section 2.9).

### 2.5 Byte step, part 2: lookups, eight bit steps, commit

```
# LOOKUP (once per byte)
for i in 0 .. NORD-1:
    if n < len[i]:  order i is inactive for this byte (pred_i = none)
    else:
        (slot, check) = ctx_hash(last len[i] bytes)                    # section 3.1 (len[i], tb[i])
        e = T[i][slot]
        if e.valid and e.check == check:  pred_i = e.succ; run_i = e.run
        else:                             pred_i = none; run_i = 0
        conf_i = 8 * run_i + min(bit_length(trun[i]), 7)               # 0 .. 31

# BIT STEPS, j = 0 .. 7, with p = prefix_j(byte) known to both sides
node    = (1 << j) | p                                                  # 1 .. 255
z       = HB_k[node] + sum over c < d of HW[node][c] * h[c]             # i32
x[0]    = clamp(asr(z + 4, 3), -3071, 3071)                             # the trunk's stretch (Q8 nats)
top     = 0
for i in 0 .. NORD-1:
    live_i = (pred_i != none) and (prefix_j(pred_i) == p)
    if live_i:
        eb_i     = bit_j(pred_i)
        ev       = (2 * eb_i - 1) * ST[ q[i][conf_i][j] >> 10 ]
        x[i + 1] = clamp(asr(ev * gain[i], 5), -3071, 3071)
        top      = i + 1
    else:
        x[i + 1] = 0
for m in 0 .. NBC-1:
    x[1 + NORD + m] = ST[ qb[m][idx_m] >> 10 ]                          # bit cells, section 3.3
set  = j * (NORD + 1) + top
dot  = sum over m in 0 .. N_IN-1 of w[set][m] * x[m]                    # i64
st   = clamp(asr(dot, 16), -3071, 3071)
p1   = SQ[st + 3071]                                                    # P(bit = 1) * 65536, in 1 .. 65535

code or decode the bit b with p1 (section 4)

# UPDATE (after b is known)
err = (b << 16) - p1
for m in 0 .. N_IN-1:
    w[set][m] = clamp(w[set][m] + asr(x[m] * err + (1 << (MIX_SH - 1)), MIX_SH), -W_MAX, W_MAX)
for each live i:
    hit = (eb_i == b)
    q[i][conf_i][j] = q[i][conf_i][j] + asr((hit ? 1 << 22 : 0) - q[i][conf_i][j], QSHIFT[ n[i][conf_i][j] ])
    n[i][conf_i][j] = min(n[i][conf_i][j] + 1, 255)
for m in 0 .. NBC-1:
    update the bit cell (qb[m][idx_m], nb[m][idx_m]) with b                 # section 3.3

# COMMIT (after the 8 bits, byte value v)
for i in 0 .. NORD-1:
    if order i was inactive:  trun[i] = 0
    else:
        hit = (pred_i != none) and (pred_i == v)
        T[i][slot] = (valid, check, succ = v, run = hit ? min(run_i + 1, 3) : 0)
        trun[i]    = hit ? min(trun[i] + 1, 255) : 0
in = [v, in[0], in[1], in[2]];  append v to the history;  n = n + 1
```

`HW` is `[256][d]` int8 indexed by `node` (row 0 unused), `HB_k` is `[256]` i32 per tier, `SQ`, `ST`, `QSHIFT`,
`Q0` are tables of the weight file (section 5.4). `x * err` fits in 30 bits; `dot` MUST be accumulated in
64 bits.

The order of evaluation that matters is exactly the one written: within a byte the eight bit steps are
sequential (each uses the mixer set `(j, top)`, and different `j` never share a set or a cell, so an encoder MAY
evaluate the eight steps of a known byte in any order or in parallel); successor-table, `trun` and history
updates happen at COMMIT; trunk accumulators are updated in part 1 of the next byte.
The trunk of byte `t + 1` may be computed any time after COMMIT of byte `t`.

### 2.6 Why the state cannot degenerate with length

**Boundedness.** For an accumulator `S` updated by `S' = S - asr(S - U, k)` with `U = u << 12`,
`|u| <= 127`, `0 <= k <= 12`: if `S >= U` then `0 <= asr(S - U, k) <= S - U`, so `U <= S' <= S`; if `S < U`
then `S - U <= asr(S - U, k) <= -1`, so `S < S' <= U`. Hence `S'` lies between `S` and `U`, and
`|S| <= 127 * 2^12` is invariant; the read-out `asr(S, 12)` is in `[-127, 127]` with no clamp. Every other
state item is clamped or saturating by definition (section 2.2); the byte count `n` saturates at `max len[i]`.
No position or length is stored anywhere: the machine that codes byte `2^31` is the machine that codes byte
`2^11`.

**Monotone coalescence (lemma).** Take two copies of one accumulator, `S1 >= S2`, driven by the same `U` and
the same shift `k`, and let `D = S1 - S2`. Then

```
D' = D - [ asr(S1 - U, k) - asr(S2 - U, k) ],   with   floor(D / 2^k) <= [...] <= ceil(D / 2^k)
=> 0 <= D' <= D - floor(D / 2^k)
```

so `D` never grows, shrinks geometrically (factor `1 - 2^-k`) while `D >= 2^k`, and is 0 after one step when
`k = 0`. Moreover the bracket is at least 1 whenever a multiple of `2^k` lies in `(S2, S1]`; the read-outs
`asr(S1, 12)` and `asr(S2, 12)` differ only if a multiple of `2^12` lies in `(S2, S1]`; since `k <= 12`,
**every step at which the two read-outs differ is followed by a strict decrease of `D`**. Therefore the number
of steps at which the read-outs differ is at most `D` at the time it falls below `2^k`, i.e. finite, and at most
`2^k (1 + ln(D_0 / 2^k)) + 2^k` in total.

**Finite disagreement (theorem).** Run the trunk from two different states on the same byte sequence. The
delay taps agree after 4 commits. Layer 0 then receives identical `U`, so by the lemma its outputs differ at
finitely many steps. Layer `l` receives inputs (and, when geared, gate decisions) that differ only at the
finitely many steps where layer `l - 1` differed; each such step perturbs its accumulators by at most
`254 * 2^12`, after which the lemma applies again. By induction the trunk outputs of the two runs differ at a
number of steps bounded by a constant that depends on `(L, d, 2^12, max shift)` and **not on the length of the
sequence**; outside those steps the two runs produce identical `h` and identical gains.

**Consequence for length.** Compare the real run at position `t` with a run started from boot at `t - M`.
With `M >= 2^kmax * ln(254 * 2^12 / 2^kmax)` per layer (2,127 bytes per layer for `kmax = 8`; 7,090 for
`kmax = 10`) every accumulator pair is inside its dead band, so the two runs differ by less than one read-out
unit, at sporadic steps drawn from a finite budget. Any state the trunk reaches at any length is therefore, up
to that residue, a state it reaches within the first `(L + 1) M` bytes of some stream — the regime covered by
training, which carries state across at least that many bytes (section 8.3). There is no weight whose repeated
application could drift, because no weight is applied recurrently: this is the structural difference from the
measured failure of Mamba-130M (`02-evidence-and-constraints.md` B.1). This argument covers the trunk; the
online tables and mixer keep unbounded history on purpose and are covered by boundedness plus gate C4.

### 2.7 Accumulator bounds (for implementers)

| Value | Bound | Width |
|---|---|---|
| `a[c]`, `ag[g]`, `g[i]`, `z` | `127 * 127 * d + |bias| <= 66,064,384 + 2^24` at `d = 4096` (biases are clamped to `+-(2^24 - 1)` by the exporter) | i32 |
| pairwise products in SIMD | `2 * 127 * 127 = 32,258 < 2^15` | i16 without saturation |
| `S0`, `S[l]` and `u << 12` | `127 * 2^12 = 520,192` | i32 |
| `ev * gain[i]` | `3071 * 127 = 390,017` | i32 |
| `x[m] * err` | `3071 * 65,535 < 2^28` | i32 |
| `w[set][m] * x[m]`, `dot` | `13 * (2^19) * 3071 < 2^35` | i64 |
| `q` | `0 <= q < 2^22` (from below it stops within `2^QSHIFT` of `2^22`) | u32 |

### 2.8 Channel schedules of v1 weights

Schedules are data (section 5.3). The v1 trainer writes periodic patterns of period 8, so that every stratum
contains every kind of channel:

| `c mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `tap` | 0 | 1 | 2 | 3 | 0 | 0 | 1 | 0 |
| `kin` | 0 | 0 | 0 | 0 | 1 | 2 | 3 | 5 |
| layer shift, not geared (`kf = ks`) | 0 | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
| layer shifts, geared (`kf` / `ks`) | 0 / 2 | 0 / 3 | 0 / 4 | 0 / 5 | 1 / 6 | 1 / 7 | 2 / 8 | 2 / 10 |

Half of the input channels are exact delay taps (the last four bytes, verbatim); the rest are traces of the
byte embedding at four time scales. These are the patterns measured in the bake-off; other patterns are legal
weight files.

### 2.9 Decision of the bake-off

MEASURED (`docs/internal/research/strata-backbone/03-bakeoff.md`; criterion pre-registered in
`tools/strata/bakeoff/CRITERION.md`; decisions taken on four selection sources never trained on):

| Decision | Result |
|---|---|
| D1: fixed vs geared time constants | geared wins, 0.7480 vs 0.7708 bpb (-3.0 %) at 3 % fewer MACs. **v1 weights are geared (`GEARED = 1`).** The fixed variant stays a legal weight file |
| D2: this trunk vs published cells, tier B (~120K MAC/byte) | **TIE**: 0.7480 (integer) vs GRU 0.7518 and LSTM 0.7548 (float) |
| D3: same at tier A (~30K MAC/byte), nested vs trained alone | **TIE**: 0.7511 vs LSTM 0.7484 and GRU 0.7506 |
| Integer simulation vs the same model in float | +0.04 % to +0.35 % |
| Frozen test set (report-only) | 1.9 % worse than the GRU in macro over the six held-out test sources (the seen-source `enwik8` slice excluded), all of it on text; equal or better on logs |
| G0 (bit cells in the loop, two seeds of the candidate) | PASS. Trunk on long streams: 0.0-0.9 % over the width-0 tier, so long inputs use `cm`; on short messages the width-88 tier saves 9.9 / 10.2 % per cell (36-37 % per 100K MAC/byte) and is kept for messaging files; the width-208 tier is not kept (marginal 3.4 / 3.7 % per 100K MAC/byte over width 88); G0a TIE (1.0091 / 1.0044 against the LSTM of the same budget) |
| Robustness of D1 | rests mostly on `webster` and one seed: with the geometric mean of per-source ratios the difference is 0.5 %, inside the band (review 04, m-10). `GEARED` therefore stays a file option, not a requirement |

The tie is between trunks that are a minor term on long streams from unseen sources (risk 1). By the criterion
the trunk is frozen as designed; the document as a whole is not (gate G0).

---

## 3. Online adaptation

### 3.1 Context hash

For order `i` with length `len` and `tb` table bits, over the last `len` bytes `c_1 .. c_len` (oldest first):

```
len = 1:  slot = c_1;                 check = 0
len = 2:  slot = (c_1 << 8) | c_2;    check = 0
else:     h = 0
          for m in 1 .. len:  h = ((h + c_m + 1) * 0x9E3779B1) mod 2^32
          h = h xor (h >> 15);  h = (h * 0x85EBCA6B) mod 2^32;  h = h xor (h >> 13)
          slot = h >> (32 - tb);      check = h & 0xFF
```

An implementation MAY compute it incrementally; only the values are normative.

### 3.2 What adapts, and with which rule

| Component | Updated when | Integer rule | Purpose |
|---|---|---|---|
| Successor tables `T[i]` | COMMIT of every byte, every active order | overwrite `(check, succ, run)` as in 2.5; always-replace | "what followed this context last time", exact repetition at 8 scales |
| Time runs `trun[i]` | COMMIT | saturating count of consecutive hits of order `i` | match-length-like confidence |
| Calibration cells `q, n` | every bit of every live order | `q += asr(target - q, QSHIFT[n])`, `QSHIFT[n] = clamp(bit_length(n + 2), 2, 7)` | measured reliability of (order, confidence, bit position) in *this* stream |
| Mixer weights `w[set]` | every bit, the selected set | integer SGD on coding cost, rounded, clamped | how much to trust the trunk and each order here |
| Bit cells `qb, nb` (3.3) | every bit, every cell order | same rule as the calibration cells, target = the bit | adaptive statistics for the bytes no table predicts |
| Trunk accumulators | every byte | fixed contraction (2.3) | short and medium memory, pretrained behaviour |

Nothing else changes during coding: matrices, embeddings, node vectors, gain vectors and tables of the weight
file are read-only. Every adaptive quantity starts from a value stored in the weight file (`Q0`, `W0_k`; the bit
cells start at 1/2), so a
128-byte message is coded with pretrained calibration and a 2 GB file ends with its own.

The fusion of network and context mixing is in three places: the tables' evidence enters the same logistic sum
as the trunk's node logit; the trunk scales each order's evidence per byte (`gain[i]`, a learned function of
the state: the state decides which contexts to consult); and the whole is trained with the tables and the
mixer running in the loop (section 8), so the network is trained as a component of an adaptive system, not as a
stand-alone predictor.

### 3.3 Bit cells: adaptive low-order bit statistics (added after the bake-off; measured in the loop by G0)

**Why this section exists.** The bake-off prototype had no adaptive statistics for the bytes that no successor
table predicts. MEASURED (`03-bakeoff.md` section 8): those bytes are 4-30 % of the input and carry 70-79 % of
the coded bits, at 4.6-5.8 bits per byte, because the only predictor left for them is the static trunk, which is
at 4-6 bits per byte on a source it was not trained on. A post-hoc experiment that adds the cells below to a
trained model without retraining anything lowered the selection-set cost by 9-27 % and made the trunk nearly
irrelevant on those sources. The cells are therefore part of v1, **but the bake-off has to be repeated with
them in the training loop before this document is frozen (gate G0, section 9.0)**: everything in this section is
specified, not validated.

Configuration (weight file): `NBC <= 4` cell orders with context lengths `blen[m]` in `{0, 1, 2, 3}`, strictly
increasing, and `btb[m]` table bits (`8` for length 0, `16` for length 1, `16..24` otherwise). `c1`, `c2`, `c3`
are the three most recent bytes (`c1` the latest), read from the delay taps: `c1 = in[0]`, `c2 = in[1]`,
`c3 = in[2]`, with `BOS` read as 0. They do not depend on the history length.

```
index of the cell of order m for the bit step with tree node `node` (1 .. 255):
    blen 0:  idx = node
    blen 1:  idx = (c1 << 8) | node
    blen 2:  h = (c2 * 0x9E3779B1 + c1) mod 2^32
    blen 3:  h = (((c3 * 0x9E3779B1 + c2) mod 2^32) * 0x9E3779B1 + c1) mod 2^32
    blen 2 or 3:  h = (h * 0x85EBCA6B) mod 2^32;   idx = ((h >> (40 - btb[m])) << 8) | node
    (blen 2 with btb = 24 uses the exact index  idx = (c2 << 16) | (c1 << 8) | node)
cell: (qb, nb), boot value (2^21, 0)
PROB:    x[1 + NORD + m] = ST[ qb[m][idx] >> 10 ]                       # always live, signed toward bit = 1
UPDATE:  qb[m][idx] = qb[m][idx] + asr((b << 22) - qb[m][idx], QSHIFT[ nb[m][idx] ])
         nb[m][idx] = min(nb[m][idx] + 1, 255)
```

The mixer then has `N_IN = 1 + NORD + NBC` inputs; nothing else changes (weight sets, update rule, clamps).
Within a byte the eight bit steps touch eight different cells of each order, as with the calibration cells.
The cells are pure functions of the bytes, so the trainer computes them in the data pipeline exactly like the
successor evidence (section 8.3); their boot mixer weights are trained.

Memory: `4 * sum_m 2^btb[m]` bytes per stream. The values of the experiment (orders 0-3 with 8 / 16 / 24 / 22
bits) need 84 MB per stream, which is not a shippable default; the v1 default is 8 / 16 / 22 / 22 bits (33.8 MB; fixed before gate G0, `05-gate-amendments.md`)
and a messaging file uses orders 0-1 only (0.26 MB). Table sizes were not swept: **HYPOTHESIS**. A nibble-slot
layout with check bytes (as in lpaq) is the known better structure for memory and cache locality and is hook H6.

### 3.4 Primer (optional; declared in the header)

A primer is a byte string `P` that both sides hold: the counterpart of a zstd dictionary, typically past traffic
of the same kind. With a primer every stream starts from a snapshot instead of from boot:

```
snapshot(P):
    run the stream machine of section 2 from boot over the bytes of P exactly as when compressing P
    (begin, eight bit steps with their updates, commit), discarding the coded bits;
    then set back to their boot values: the delay taps in[0..3], the accumulators S0 and S[l], the byte history,
    the byte count n and every trun[i];
    keep: the successor tables T, the calibration cells (q, n), the bit cells (qb, nb), the mixer weights w.
```

A primed message therefore still starts as a stream start (orders inactive until enough bytes have been seen,
trunk at rest) but with the tables, statistics and mixer of the primer's traffic. `len(P) <= 2^24`. The archive
declares `PRIMER_ID` = CRC-32 of `P` (section 5.6); a decoder without a primer of that id refuses to decode.

MEASURED (`03-bakeoff.md` 7.6; prototype without bit cells): on the 480 messages of R-113, a 64 KB primer from
the same half of each source that trains the zstd dictionary turns the payload's record against zstd-19 +
dictionary from 10 to 30 cells of 30 with payload + 4 B (the R-114 target); with the real framing of a primed
archive (24 B compact, 28 B default) 22 and 20 cells of 30, and 17 and 15 of the 25 cells without the
seen-source Wikipedia (review 04, G-1). G0 then measured
both pairings (`03-bakeoff.md` 12.2; width-88 tier with bit cells): with a primer from the message's own source it
wins 24 of 30 cells with real framing (19 of 25 held-out; gate V3 asks 20 of 25); with a primer from another
source it wins 2 of 30 (0 of 25) against a same-source zstd dictionary, and 25 of 30 against a dictionary trained
on the primer's source.

### 3.5 Scope of v1 and reserved hooks

v1 implements exactly sections 3.2 to 3.4. The following are specified as hooks only; their encodings are reserved
(weight-file FLAGS bits 1-15, ARITH values 2-8) and a v1 decoder MUST reject files and archives that use them:

| Hook | Idea | Where it will plug in |
|---|---|---|
| H1 learned context keys | the trunk emits `KEYB` sign bits per byte; they select a slot of an extra successor table ("order" `NORD`): contexts defined by the state instead of by literal bytes | weight-file flag bit 1, extra order with `len = 0`; needs in-loop table simulation in the trainer |
| H3 second candidate | two successors per slot | new table kind |
| H4 final probability refinement (APM/SSE) | adaptive map on `(order-1 context, st)` | after the mixer |
| H5 table size by archive | larger tables for large inputs, declared per archive | new ARITH value |
| H6 nibble-slot bit cells | hashed slots with check bytes, one slot per (context, nibble), as in lpaq | new cell-table kind |

(H2, the primer, was promoted into v1 as section 3.4 after the bake-off measured it.)

---

## 4. Output interface: binary decisions into the AZTH range coder

### 4.1 Interface

Strata does not produce logits and does not use the floating-point mixer / softmax / `Cdf::from_probs` path of
the existing backbones (`01-integration-map.md` fact 4). It produces, per bit, `p1` in `1 .. 65535` and
consumes the coded bit. The existing token-level CM ensemble is **not** run for Strata archives (FLAGS bit 0x80
is 0, section 5.6): Strata carries its own integer mixing path, which is what gives determinism end to end —
there is no float between the input byte and the range coder on either side.

### 4.2 Coding one bit

The existing range coder is used unchanged with total `2^16`:

```
split = 65536 - p1
bit 0  ->  encode(cdf_low = 0,     cdf_high = split, total = 65536)
bit 1  ->  encode(cdf_low = split, cdf_high = 65536, total = 65536)
```

where `encode` is `RangeEncoder::encode` of `src/domain/coder.rs`, restated here because it is normative for
the reference implementation:

```
state: low (u64) = 0, range (u32) = 0xFFFFFFFF, cache (u8) = 0, ff (count) = 0, first = true, out = []
encode(lo, hi, total):
    r = range / total                                   # integer division; total = 65536 -> r = range >> 16
    low = low + lo * r
    range = (hi < total) ? r * (hi - lo) : range - r * lo
    while range < 2^24:  shift_low();  range = range << 8
shift_low():
    if (low mod 2^32) < 0xFF000000 or (low >> 32) != 0:
        carry = low >> 32
        if not first: out.append((cache + carry) mod 256)
        first = false
        append ff copies of ((0xFF + carry) mod 256);  ff = 0
        cache = (low >> 24) & 0xFF
    else:
        ff = ff + 1
    low = (low & 0x00FFFFFF) << 8
decoder: low (u32) = 0, range = 0xFFFFFFFF, code = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3, the first four
         STREAM bytes, BIG-endian (the one exception to the little-endian rule of section 1); every read at or
         past the end of STREAM returns 0. STREAM = archive[header_size .. archive_length - 4): the CRC trailer
         is NOT stream data (reading it as such breaks archives: review 04, B-3)
decode bit:  r = range >> 16;  cum = min(((code - low) mod 2^32) / r, 65535);  bit = (cum >= split)
             lo = cdf_low of the decoded bit (0 or split);  low = (low + r * lo) mod 2^32;  range as in encode
             while range < 2^24:  code = ((code << 8) | next byte) mod 2^32;  low = (low << 8) mod 2^32;  range <<= 8
```

### 4.3 Stream termination (tight flush)

A Strata stream ends with a one-byte flush instead of the five bytes of `RangeEncoder::finish`:

```
finish_tight():
    low = (low + 0x00FFFFFF) & ~0x00FFFFFF        # 64-bit; the result is inside the final interval because range >= 2^24
    shift_low()
    if not first: out.append(cache)               # the initial cache is a dummy and is never written
    append ff copies of 0xFF
```

The decoder needs no change: it reads zeros past the end of STREAM (4.2). MEASURED (design stage,
`03-bakeoff.md` section 6): 120,000 random bit streams round-trip with this rule and are exactly 4 bytes shorter
than with the legacy flush; the first draft of the rule omitted the `first` condition and failed 8 of 30,000
streams, which is why the condition is spelled out.

### 4.4 Fixed cost of a short message, and what is left for the framing (R-114)

| Term | Today (Mamba Q8, `02-evidence` D.2) | Strata in the AZTH v5 container | Who owns the rest |
|---|---|---|---|
| Header | 16 B | 16 B (compact, no weights id) or 20 B (default) | R-114: MAGIC 8 + TOTAL_BYTES 4 can shrink to 2-3 B |
| CRC-32 | 4 B | 4 B | R-114 (link-layer integrity may replace it) |
| First symbol | ~2 B (uniform over 50,280 tokens) | the first byte is coded by the trained response to `BOS` (its cost is inside the measured payloads of `03-bakeoff.md`) | backbone: done |
| Coder flush | ~5 B | 1 B | backbone / dock: done |
| Reset declaration | 1 B (`RESET_LOG2`) | none (no reset exists) | backbone: done |
| Cold online models | CM ensemble worse than backbone alone on one message | mixer and cells start from trained boot values; empty tables contribute exactly 0; with a primer (3.4) they start from the primer's traffic | backbone: done (+4 B for `PRIMER_ID`) |

So the backbone removes about 6 B of the ~27 B and removes the cold-start penalty; the remaining 20 B are
container framing and belong to R-114. MEASURED (`03-bakeoff.md` 7.6): with a same-source primer the payload is
smaller than zstd-19 + dictionary in 30 of 30 cells with payload + 4 B, and with the real framing of a primed
archive (24 B compact / 28 B default) in 22 / 20 of 30 cells (17 / 15 of the 25 held-out cells); without a
primer in 10 of 30 (payload). Gate V3 reports the payload and the real `.azth` size.

---

## 5. Weight file and AZTH declaration

### 5.1 File layout

One file, `strata.bin` (the `.bin` extension is already ignored under `weights/`), default location
`weights/strata-v1/strata.bin`. All integers little-endian, no padding, no alignment requirement. Every
multi-dimensional block `T[a][b]` (and `T[a][b][c]`) is stored **row-major: the last index varies fastest**;
the blocks of a section follow each other exactly in the order of the byte sequences below, nothing else.

```
FILE     = HEADER  GLOBALS  STRATUM[0]  STRATUM[1]  ...  STRATUM[K-1]
```

A file MAY end after any complete stratum: it then provides tiers `0 .. k`. A reader MUST accept such a prefix
and MUST reject a file that ends inside a block. **A smaller tier's weight file is a byte prefix of a larger
one.**

### 5.2 HEADER

| Offset | Size | Field | Value / meaning |
|---|---|---|---|
| 0 | 8 | MAGIC | `"STRATAW1"` |
| 8 | 4 | HEADER_LEN | bytes of HEADER from offset 0 (MAGIC included) to the end of HEADER_CRC |
| 12 | 2 | ARITH | 1 |
| 14 | 2 | FLAGS | bit 0 `GEARED`; bits 1-15 reserved, MUST be 0 (bit 1 is hook H1) |
| 16 | 1 | K | number of tiers in the full file |
| 17 | 1 | L | number of layers |
| 18 | 1 | NORD | number of context orders |
| 19 | 1 | MIX_SH | |
| 20 | 1 | DEFAULT_TIER | tier used when the user does not choose one |
| 21 | 1 | NBC | number of bit-cell orders (section 3.3) |
| 22 | 2 | reserved | 0 (a reader MUST reject other values) |
| 24 | 2 x K | WIDTH[k] | `d_k` |
| ... | 8 x K | NAME[k] | ASCII, zero padded (`cm`, `nano`, `micro`, `small`, `base`, `large`) |
| ... | NORD | LEN[i] | context lengths |
| ... | NORD | TB[i] | table bits |
| ... | NBC | BLEN[m] | bit-cell context lengths |
| ... | NBC | BTB[m] | bit-cell table bits |
| ... | 16 | TRAIN_ID | free-form identifier of the training run (not interpreted) |
| ... | 4 | HEADER_CRC | CRC-32 (IEEE, the one of the AZTH container) of all preceding header bytes |

### 5.3 GLOBALS

Byte sequence, lengths implied by the header (`d = d_{K-1}`): `SQ`, `ST`, `QSHIFT`, `Q0`, `TAP`, `KIN`,
`KF[0]` .. `KF[L-1]`, then (only when `GEARED`) `KS[0]` .. `KS[L-1]`, then `GLOBALS_CRC`:

| Block | Type and shape | Notes |
|---|---|---|
| `SQ` | u16 `[6143]` | squash table, index `st + 3071`; values in `1 .. 65535`, non-decreasing |
| `ST` | i16 `[4096]` | stretch table, index `q >> 10`; values in `[-3071, 3071]`, non-decreasing |
| `QSHIFT` | u8 `[256]` | values in `2 .. 7` |
| `Q0` | u32 `[NORD][32][8]` | boot calibration, values in `[2^10, 2^22 - 2^10]` |
| `TAP` | u8 `[d]` | |
| `KIN` | u8 `[d]` | |
| `KF`, `KS` | u8 `[L][d]` each | `KS` present only when `GEARED` |
| GLOBALS_CRC | u32 | CRC-32 of the GLOBALS bytes |

The tables are **data**. No implementation regenerates them; the trainer writes them once
(section 5.4) and the reference and the runtime read them. This removes every transcendental function from the
runtime and from the oracle.

### 5.4 How the trainer generates the tables (informative for readers, normative for the trainer)

```
SQ[i]     = clamp(round_half_even(65536 / (1 + exp(-(i - 3071) / 256))), 1, 65535)       in IEEE double
ST[k]     = clamp(round_half_even(256 * ln(x / (1 - x))), -3071, 3071),  x = (k + 0.5) / 4096
QSHIFT[n] = clamp(bit_length(n + 2), 2, 7)
Q0        = clamp(round_half_even((hits + 1) / (total + 2) * 2^22), 2^10, 2^22 - 2^10), counted on training
            data (8.4) in IEEE double
```

The file is the authority: the v1 release notes MUST publish the SHA-256 of the `SQ`, `ST` and `QSHIFT` blocks
so that a second trainer can check it reproduces them.

### 5.5 STRATUM[k]

Let `lo = d_{k-1}` (0 for `k = 0`), `hi = d_k`, `m = hi - lo`. Byte sequence of a stratum:

```
EMB_k
for l = 1 .. L:   W_{l,k}   B_{l,k}   [WG_{l,k}   BG_{l,k}   only when GEARED]
HW_k   GW_k   HB_k   GB_k   W0_k   STRATUM_LEN   TIER_ID
```

A stratum with `m = 0` (the width-0 tier) has empty `EMB`, `W`, `B`, `WG`, `BG`, `HW`, `GW` blocks.

| Block | Type and shape | Meaning |
|---|---|---|
| `EMB_k` | i8 `[257][m]` | columns `lo .. hi-1` of the embedding table (row 256 = BOS) |
| for `l = 1 .. L`: `W_{l,k}` | i8 `[m][hi]` | rows `lo .. hi-1` of layer `l`, each reading channels `0 .. hi-1` |
| `B_{l,k}` | i32 `[m]` | biases of those rows |
| `WG_{l,k}`, `BG_{l,k}` (if `GEARED`) | i8 `[m/8][hi]`, i32 `[m/8]` | gates of those rows |
| `HW_k` | i8 `[256][m]` | columns `lo .. hi-1` of the node matrix (row 0 is zero) |
| `GW_k` | i8 `[NORD][m]` | columns `lo .. hi-1` of the gain matrix |
| `HB_k` | i32 `[256]` | node biases of tier `k` (complete, not incremental) |
| `GB_k` | i32 `[NORD]` | gain biases of tier `k` |
| `W0_k` | i32 `[NSET][N_IN]`, `N_IN = 1 + NORD + NBC` | boot mixer weights of tier `k` |
| STRATUM_LEN | u32 | bytes of this stratum before this field (lets a reader check it parsed the right amount) |
| TIER_ID | u32 | CRC-32 of every file byte from offset 0 up to and including STRATUM_LEN of this stratum |

`TIER_ID` of tier `k` is a CRC over the whole prefix, so it covers tier `k`'s blocks **and** the HEADER (all
widths, names, `DEFAULT_TIER`, `TRAIN_ID`) and GLOBALS (sized for the widest tier): re-exporting identical tier-k
weights under another `TRAIN_ID` or with another top tier changes it, and old archives then fail early with
"different weights". This is deliberate (conservative). It is available from a prefix file.

Reader validation (a reader MUST reject the file, at load, when any of these fails; gate C9 exercises each row):

| Field | Legal values |
|---|---|
| every i8 value | `[-127, 127]` (never -128) |
| `B`, `BG`, `HB`, `GB` | `[-(2^24 - 1), 2^24 - 1]` |
| `W0_k` | `[-W_MAX, W_MAX]` |
| `K`, `L`, widths, `NORD`, `LEN`, `TB`, `NBC`, `BLEN`, `BTB`, `MIX_SH`, shifts | the ranges of 2.1 and 3.3 |
| `DEFAULT_TIER` | `< K` |
| `NAME[k]` | printable ASCII, distinct for distinct `k` |
| `SQ`, `ST`, `QSHIFT`, `Q0` | the ranges and monotonicity of 5.3 |
| reserved fields and FLAGS bits | 0 |
| `STRATUM_LEN`, CRCs, `TIER_ID` | equal to the recomputed values; the file ends exactly after a stratum |

### 5.6 Declaration in the AZTH header

Strata takes **BACKBONE id 4** (the next free id). The container stays AZTH v5; nothing is added to FLAGS or
PROFILE. One block is appended after the BACKBONE byte, which today is the last header field:

```
MAGIC "AZTH\x05\0\0\0"   TOTAL_BYTES (4)   FLAGS = 0x40 (1)   PROFILE = 0x10 (1)   BACKBONE = 0x04 (1)
STRATA block:  SFLAGS (1)   [WEIGHTS_ID (4, LE)]   [PRIMER_ID (4, LE)]
STREAM (range coder bytes, tight flush)   CRC32 (4)
```

| Field | Rule for BACKBONE = 4 |
|---|---|
| FLAGS | exactly `0x40` in v1: the preprocessing bits 0x04 / 0x08 / 0x10 (E8E9, shuffle, delta), 0x01 (HP), 0x02 (EMB), 0x20 (SKIP) and 0x80 (ENHANCED_CM) MUST be 0, so no SHUFFLE or DELTA byte is ever present. Preprocessing is not part of Strata v1 (no Strata model was trained or measured on preprocessed bytes); it is a hook with its own future text |
| PROFILE | MUST be `0x10`, written explicitly by the Strata header builder (not derived from the process's numeric contract). Its QUANT and NUMERIC_CONTRACT fields carry no information for Strata; bit 6 (reset period) and bit 7 MUST be 0 |
| SFLAGS | bits 0-2 `TIER`; bit 3 `WID` (weights id follows); bits 4-6 `ARITH - 1` (0 for this document; any value >= 1 announces a later arithmetic version, which MAY carry extension bytes after the ids); bit 7 `PRIMER` (primer id follows) |
| WEIGHTS_ID | `TIER_ID` of the tier used, present when `WID = 1` (default). `--strata-compact` omits it (messaging) |
| PRIMER_ID | CRC-32 of the primer bytes (section 3.4), present when `PRIMER = 1` |
| TOTAL_BYTES, CRC32 | `TOTAL_BYTES` = number of message bytes (u32 LE); `CRC32` = CRC-32 (IEEE) of the message bytes exactly as given to `compress` (no preprocessing exists in v1); the primer is never included |

Header size: 16 B compact, 20 B default, +4 B with a primer; every archive also ends with the 4-byte CRC, so the
fixed framing is 20 / 24 B (compact; without / with primer) or 24 / 28 B (default).

**Empty input.** `compress` of an empty message MUST fail with an error before any archive is written (the
behaviour of the other backbones today); a decoder MUST reject `TOTAL_BYTES = 0` for BACKBONE 4.

**Precedence.** Sections 4.2 and 4.3 are the normative coder for Strata archives; they agree today with
`src/domain/coder.rs`, and if the two ever differ this document wins.

**Decoder behaviour.**

| Decoder | Archive | Result |
|---|---|---|
| any build without Strata (v5 as of 692d985) | BACKBONE = 4 | `AZTH v5 unknown backbone: 4`, raised in `parse_header` before any model load (`01-integration-map.md` 3.2). A v4-only build stops earlier at `unsupported AZTH version 5` |
| Strata build | v2-v5 with BACKBONE 0-3 | byte-for-byte the existing parse path: the STRATA block is read only after the id 4 is seen, so no existing field moves and `header_size` is unchanged for ids 0-3 |
| Strata build | BACKBONE = 4 with PROFILE != 0x10, a forbidden FLAGS bit or an unknown ARITH | error before model load, naming the field |
| Strata build | `PRIMER = 1` and no primer given, or a primer whose CRC-32 differs; or a primer given and `PRIMER = 0` | error before decoding, printing both ids |
| Strata build | TIER not provided by the loaded weight file | error before decoding, naming both |
| Strata build | WEIGHTS_ID present and different from the loaded `TIER_ID` | error before decoding, printing both ids (today a wrong weights file is only caught by the CRC after decoding everything) |
| Strata build | truncated or corrupt stream | CRC-32 mismatch after decoding, as today |

Encoders for BACKBONE 0-3 are not modified: their archives MUST be bit-identical before and after Strata is
added (gate C5).

---

## 6. The dock

The dock is everything between the Strata model and the rest of Redukt. It is a first-class deliverable with
its own tests (section 10): a correct model behind a wrong dock produces wrong archives.

### 6.1 Deviation from axiom A7, stated and justified

A7 asks a backbone to plug in through the `Backbone` trait and to touch the core only for registration and
header declaration. The trait's only output is `f32` logits over the vocabulary, consumed by a floating-point
mixer and softmax (`01-integration-map.md` facts 2 and 4). A coder-domain integer model cannot pass through that
interface without (a) evaluating all 255 nodes per byte (32 times the head cost), (b) converting integers to
logarithms in floating point and (c) inheriting the float path it exists to remove. Strata therefore does **not**
implement `Backbone`. The dock adds one trait and one engine next to the existing ones; the existing trait,
engines and loops are not edited. Core files touched, all additively: `backbone.rs` (one enum variant),
`coder.rs` (three methods), `codec.rs` (header block, flags, dispatch), `lockstep.rs` and `api.rs` (dispatch),
`model_loader.rs` (one error arm), `main.rs` (usage). Rejected alternative: an adapter that fills `out_logits`
from the tree — kept out of v1 for the three reasons above.

### 6.2 Trait

```rust
/// A backbone that predicts binary decisions in the coder's domain and owns its online adaptation.
pub trait CoderBackbone: Sync {
    type Stream: Send + Clone;                           // every per-stream item of section 2.2
    fn tiers(&self) -> &[TierInfo];                      // name, width, TIER_ID, MAC/byte, stream bytes
    fn new_stream(&self, tier: usize) -> Self::Stream;   // boot state
    fn boot(&self, s: &mut Self::Stream);                // back to boot (slot reuse in lockstep)
    /// Snapshot of section 3.4: boot, run `primer`, reset the trunk part. Primed streams are clones of it.
    fn prime(&self, tier: usize, primer: &[u8]) -> Self::Stream;
    /// Trunk + gains + lookups of the next byte for every stream (all of the same tier). Batch == single.
    fn begin_byte(&self, streams: &mut [&mut Self::Stream]);
    /// Computes p1 and leaves the mixer inputs of this bit in the stream for `bit_update`.
    fn bit_p1(&self, s: &mut Self::Stream, j: u32, prefix: u32) -> u16;
    fn bit_update(&self, s: &mut Self::Stream, bit: u32);
    fn commit_byte(&self, s: &mut Self::Stream, byte: u8);
}
```

No method returns floats, allocates per call, or can fail after construction; loading returns `AzResult`
(unlike `load_dyn_model`, a bad weight file is an error, not a panic). `begin_byte` is the only method that may
use the worker pool.

### 6.3 Registration and loading

- `BackboneChoice::Strata`: header byte 4, CLI name `strata`, `Display` "strata". `from_header_byte(4)` and
  `from_str_opt("strata")` MUST be added (both are non-exhaustive matches that compile without them); the
  list of valid names in the error text of `from_cli` (added to the working tree on 2026-11-12 by the parallel
  R-104 work) gains `strata`.
  `route_backbone` is unchanged: `auto` stays Mamba, so no default output changes.
- Weights: `--strata-weights <dir|file>` (a Strata-specific flag: `--weights` keeps its meaning for the other
  backbones, so `redukt api --backbone strata,rwkv` cannot hand one backbone the other's file), default
  `weights/strata-v1/strata.bin`; the file is read once into memory,
  every CRC of 5.2-5.5 is verified, every range constraint is checked, and a `StrataModel` holding the complete
  tensors (strata concatenated per tier boundary) is built. Load failures are messages, not panics.
- `weights/MANIFEST.md` gets an entry (provenance, training corpus, licence of the corpora, SHA-256).

### 6.4 CLI

```
redukt compress   --input F --output F.azth --backbone strata [--tier NAME|INDEX|auto] [--strata-weights W]
                  [--primer P] [--strata-compact] [--target-kbps X]
redukt compress   --input-dir D --output-dir O --backbone strata --jobs N ...
redukt decompress --input F.azth --output F [--strata-weights W] [--primer P]   # tier, ids from the header
```

- `--tier`: default = `min(DEFAULT_TIER, highest tier present in the file)`: a property of the file (a truncated
  file keeps the header of the full file), identical on every machine; tested by C9.
  `auto` is opt-in (section 6.8). The tier is always written to the header.
- For `--backbone strata` the flags `--quant`, `--numeric`, `--reset-every`, `--lr`, `--ngram-scale`,
  `--mix-eta`, `--emb-surgery`, `--adaptive-skip`, `--skip-threshold` are **errors** (not warnings, not silently
  ignored): each of them would otherwise suggest a setting that does not exist.
- No preprocessing in v1: for `--backbone strata` the automatic content detection is skipped and
  `--e8e9`, `--shuffle`, `--delta` with any value other than `off` are errors (5.6).
- `--primer P`: the primer file; its CRC-32 goes to the header. Decompression needs `--primer` with a file of the
  same CRC-32 exactly when the header declares one.
- Decompression never takes a tier or a plan from the command line.

### 6.5 Engine (single file, lockstep, API)

One generic routine codes a stream; the three entry points share it.

```
per stream:  boot; for each byte:  [begin_byte]  ->  8 x (bit_p1 -> encode/decode -> bit_update)  ->  commit_byte
single file: the loop above on one stream.
lockstep:    Phase A (one pool unit per active stream): the 8 bit steps + commit of the current byte, coder included.
             Phase B (one call): begin_byte for all active streams.
             Finished streams are replaced from the source (continuous batching), slots are reused after boot().
```

- Lockstep groups are keyed by `(Strata, TIER_ID, PRIMER_ID)`: one group = one tier of one weight file with one
  primer (or none). Streams of different groups never share a `begin_byte` call. The primer snapshot is
  computed once per group and cloned into a slot at every `boot()`.
- Batch == single holds by construction (integer sums; private per-stream state); it is still gated (C2).
- `redukt api`: `--backbone strata[,mamba,...]` loads the weight file (and `--primer P`, one per server in v1)
  at start-up; `--tier` selects the tier
  used for compression (default as the CLI); decompression accepts any tier the file provides. `/v1/status`
  lists, per tier: name, width, `TIER_ID`, MAC/byte, bytes of state per stream. Unknown or mismatching
  `(TIER, WEIGHTS_ID)` answer 422 with the same text as the CLI. API and CLI MUST call the same header builder
  (today they do not for other backbones: `01-integration-map.md` trap 12).
- The engine stops after exactly `TOTAL_BYTES` decoded bytes and then verifies the CRC-32 of 5.6; empty input is
  rejected (5.6).

### 6.6 Kernels

Required kernels: (1) `i8 x i8 -> i32` dot product / GEMV / N-stream GEMM over rows of a stratum-triangular
matrix; (2) the accumulator update (subtract, shift, subtract, shift; vectorisable with per-lane variable
shifts); (3) clamps. Each has a scalar implementation that is the definition, and optional x86-64 AVX2
(`vpmaddubsw` + `vpmaddwd`, exact because of the +-127 rule), AVX-VNNI (`vpdpbusd`) and aarch64 NEON
(`smull`/`sdot`) implementations. All of them MUST return exactly the integer of section 2; none may use
saturating intermediate arithmetic that can trigger. Unit tests compare each SIMD kernel with the scalar one on
adversarial inputs (all +-127, alternating signs, every length mod 64).

### 6.7 Execution autoscaler (axiom A4a: always on, never changes a bit)

The autoscaler chooses an **execution plan**; the plan is not recorded anywhere because it cannot influence the
output.

| Decision | Choices | Chosen by |
|---|---|---|
| kernel | scalar / AVX2 / AVX-VNNI / NEON | CPU feature detection (existing `REDUKT_NO_SIMD` honoured) |
| stream placement | P1 "one stream per thread" (each thread runs whole streams, no cross-stream batching) or P2 "lockstep GEMM" (N streams share every weight fetch) | calibration |
| lockstep width | N in {1, 2, 4, 8, 16, 32, 64} | calibration |
| per-matrix row threading | on / off per layer | calibration |
| GEMM row block | {8, 16, 32, 64} rows | calibration |

Memory cap: the lockstep width is limited so that `width x state bytes per stream <= REDUKT_STRATA_MEM`
(default 2 GiB; 39 MB per stream at the default tables gives at most 52 streams). Streams are booted lazily
(tables cleared by generation counters, not by copying) where an implementation can do so bit-exactly.

Calibration: the first time a tier is used in a process, the dock times each candidate plan on the real
weights with pseudo-random input bytes for a fixed budget (default 30 ms per tier, `REDUKT_STRATA_CALIB_MS`),
keeps the plan with the highest bytes per second for the number of streams actually requested, and prints it
under `--verbose` together with the measured MAC/s. Nothing is hard-coded from the cost model of section 7:
the model predicts, the calibration decides. `REDUKT_STRATA_PLAN=<plan string>` forces a plan (used by gate C2
to prove every plan gives the same sha256); `REDUKT_THREADS` keeps its meaning.

### 6.8 Tier autoscaling (axiom A4b: declared, opt-in)

`--tier auto` picks, at compression time, the largest tier whose calibrated throughput on this machine is at
least `--target-kbps` (default 16 KB/s per thread), and writes it to the header like any explicit tier. Same
input + same flags then gives the same bytes only on machines that resolve to the same tier, which is why it is
never the default and why gates 2b / 2c / 3b run with explicit tiers.

---

## 7. Cost model and crossover table

### 7.1 Formulas (per input byte, tier `k`, width `d = d_k`)

```
MAC_layer(k) = sum over strata s <= k of (d_s - d_{s-1}) * d_s            (+ 1/8 of it again when GEARED)
MAC_trunk    = L * MAC_layer(k)
MAC_head     = 8 * d (node dots)  +  NORD * d (gains)
MAC_mixer    = 8 * 2 * N_IN       (64-bit products: dot + update)
MAC(k)       = MAC_trunk + MAC_head + MAC_mixer
weight bytes touched per byte = MAC_trunk + MAC_head + d   (int8; the whole model is touched once per byte)
random table accesses per byte = NORD reads + NORD writes (4 B slots), 8 * NORD calibration-cell updates in a
                                 7 KB array, 8 * NBC bit-cell updates (up to 8 cache lines per hashed order)
state bytes per stream = 4 * (L + 1) * d + 4 * sum_i 2^tb[i] + 4 * NORD * 32 * 8 + 4 * NSET * N_IN
                         + 4 * sum_m 2^btb[m]
time per byte ~= MAC(k) / R_mac + t_tables + t_bits,   R_mac = integer MAC rate of one core
time per message += t_boot = state bytes / copy bandwidth (cloning a 39 MB boot or primer snapshot costs ~4-8 ms,
                    which dominates a 128-byte message unless lazy boot or messaging-size tables are used)
```

### 7.2 Tiers of v1 (L = 3, geared, NORD = 8, successor table bits 8/16/16/16/18/18/18/18, bit cells 8/16/22/22)

G0 fixed the v1 tiers: a general weight file holds the width-0 tier only (`K = 1`, `DEFAULT_TIER = 0`); a messaging
weight file holds widths 0 and 88 (`K = 2`, `DEFAULT_TIER = 1`) with the messaging table sizes chosen by the sweep of
gate V3 (section 9.2). All columns except the last are **DERIVED** (and the MAC and parameter counts were also
printed by the G0 trainer); the last column is a **HYPOTHESIS**, `1 / (MAC / R_mac + 1 us)` with `R_mac` = 10 and 20 G
MAC/s per core and 1 us per byte for tables, bit cells and bit steps, to be replaced by the autoscaler's calibration
(6.7) and gate V4.

| Tier | Name | Width `d_k` | MAC / byte | Parameters | Weight file prefix, bytes | Weight bytes touched / byte | State / stream, bytes (default tables) | KB/s per core (hypothesis) |
|---|---|---|---|---|---|---|---|---|
| 0 | `cm` | 0 | 208 | 1,199 | 34,530 (33,816 as a stand-alone general file, `K = 1`) | 0 | 38,811,296 | ~1,000 |
| 1 | `msg88` | 88 | 27,752 | 73,392 | 112,510 | 27,632 | 38,812,704 | 260 - 420 |

- Wider tiers (the 64-1,024 ladder of the draft, and the measured width 208) are **not part of v1**: G0 found that
  width 208 buys 0.3-0.5 % on long streams and about 3 % per message cell over width 88, below the 5 % per 100K
  MAC/byte rule. They remain legal weight files for a later version, after their own measurement.
- The state is dominated by the bit cells (33.8 MB) and the successor tables (4.98 MB) at the default table bits; a
  messaging file is expected to use much smaller tables (V3 sweep). Table bits are a property of the weight file.
- Reference: Mamba-130M touches 153.6 MB of weights per token in Q8 and its file is 516.6 MB; `msg88` is 0.11 MB.
- Budgets, per vCPU (not per core): with Strata's own measured margins over the classical codecs (prototype,
  `03-bakeoff.md` 7.3) text had no break-even speed and logs need 250-890 KB/s per vCPU at $0.014 per vCPU-hour
  (review 04, m-15); the G0 margins (`cm` 5.5 % below the best classical codec on the selection macro) do not change
  the storage verdict of Fase 20. Path 3 has no throughput budget.

### 7.3 Crossover table

"Pays" means: lowers cost per byte at equal bits, or lowers bits at equal cost. Status: **M** measured in the
bake-off (`03-bakeoff.md`), **D** derived from the formulas, **H** hypothesis to calibrate (named gate or
calibration).

| # | Dimension or technique | Pays when | Stops paying when | Status | Evidence or calibration |
|---|---|---|---|---|---|
| X1 | Byte vocabulary + binary tree (vs BPE + softmax), at MAC parity (not at equal bits) | per byte, `L d^2 + 8d` MACs (bytes) < `(L d^2 + V d) / T` (BPE) while `d < (V - 8T) / (L (T - 1))` = 6,700 for V = 50,280, T = 3.5 B/token, L = 3: every tier of 7.2. Host work O(8) per byte instead of O(V) per token (the measured R-110 bottleneck) | `d` above ~6,700 (no tier here) | D | 7.1; `02-evidence` B.5, B.6 |
| X2 | Binary-tree head vs 256-way softmax, both static | neutral in ratio; 8 node dot products per byte instead of 256 | — | M | static LSTM: tree head 4.78, softmax head 4.87 bpb (selection macro) |
| X3 | Online components (tables, calibration, mixer) | always on unseen sources: cost / 6.3-6.6 | not observed | M | selection set, every trunk |
| X4 | Weight of the trunk against the online statistics (G0, bit cells in the loop: 0.0-0.9 % on long streams, 10 % per short-message cell for width 88; M) | seen-source text (14-18 % with bit cells, 29 % without; training-domain evidence, not held-out) and short messages (4-27 % with a primer on held-out sources) | long streams of unseen sources once bit cells exist (0-15 %; costs 4-8 % on `xml`) | M (bit cells: post hoc) | `03-bakeoff.md` 7.3, 7.6, 8; G0c decides per data type |
| X5 | Width (tier; G0: 88 -> 208 buys 0.3-0.5 % on long streams and ~3 % per message cell, marginal 3.4-3.7 % per 100K MAC/byte: stops paying above 88; M) | text: tier B (4.1x MACs) 2.6 % better than tier A on held-out `dickens` (4.9 % on the seen-source `enwik8`) | unseen long streams: 0.4 % for 4.1x MACs after 33.5 MB of training | M; H beyond | `03-bakeoff.md` 7.4; the crossover moves with training data |
| X6 | Gears (input-driven choice of time constant; +1/8 of trunk MACs for gates) | -3.0 % on the selection set at equal budget | frozen test: +0.4 % macro (better on text, worse on the synthetic XML) | M | D1 |
| X7 | int8 integer arithmetic vs float | always: -0.18 % to +0.35 % bpb buys determinism by construction, 4x smaller weights, exact SIMD | — | M | `03-bakeoff.md` 7.4 |
| X8 | Nesting (tiers as prefixes, one run) | one file and one run for every operating point; tier A ties with controls trained alone | the cost against the same tier trained alone was not measured | H (D3 was written after the test set was visible) | D3 |
| X9 | Bit cells (adaptive order-0..3 bit statistics) | unseen sources: -10 % to -21 % (post hoc, float); in the loop (G0) the width-0 tier reaches 0.619 bpb on the selection macro against 0.748-0.771 without cells | seen-source text: ~0 %; 33.8 MB per stream at the default sizes | M (post hoc) / H (in the loop: G0) | 3.3 |
| X10 | Primer | short messages against zstd + dictionary: payload + 4 B 10 -> 30 of 30 cells; real framing 22 / 20 of 30 (17 / 15 of 25 held-out) | long streams, where the stream's own statistics take over (not measured) | M (same-source primer) / H | 3.4; V3 |
| X11 | Tight flush | always: -4 B per stream | — | M | 4.3 |
| X12 | Row threading inside one matrix | a layer's int8 matrix (triangular, gates included) is large enough to split (`base` 193.5 KiB, `large` 769.5 KiB per layer) | `cm` .. `small` (<= 49.5 KiB per layer): one thread per stream | H | calibration (6.7) |
| X13 | Lockstep GEMM (N streams share each weight fetch) vs one stream per core | weights touched per byte exceed the core's L2: `large` (2.4 MB) on the dev laptop (L2 1.25-2 MB) | `cm` .. `base` (<= 0.6 MB) fit in L2: one stream per core avoids synchronisation; the 39 MB of tables per stream are private either way | H | calibration (6.7) |
| X14 | Weights below 8 bits | — | not measured (RWKV Q4 cost +12.7 % in the repo) | H | out of v1 |
| X15 | Table sizes (successor 2^16-2^18, bit cells 2^8-2^22) | long and diverse inputs | short messages, small terminals; accesses beyond L2/L3 add latency per byte | H | not swept; hook H5 |
| X16 | Online update cost per byte | fixed: 8 bit steps x `N_IN` mixer inputs + `NORD` table writes, independent of `d` | dominates the per-byte time of `cm` .. `micro` (hypothesis ~1 us/byte) | H | calibration prints the split |

---

## 8. Training recipe

### 8.1 Principle

The trainer is a PyTorch program whose forward pass **is** the integer model: every tensor that the runtime
holds as an integer holds the same integer in the trainer (as an integer-valued float below 2^24, or as int64 in
the sequential parts), obtained with straight-through estimators for rounding, flooring and clamping. The
exported file is read back by the reference implementation and must reproduce the trainer's probabilities with
zero mismatches before any number is reported. A component that cannot be trained this way is not in v1.

### 8.2 Parametrisation (quantisation-aware from step 0)

| Runtime tensor | Latent float parameter | Forward value |
|---|---|---|
| `EMB` | `e` (init N(0, 1)) | `clamp(round(32 e), -127, 127)` |
| `W[l]` | `w` (init N(0, 1 / fan_in of the row)) | `clamp(round(128 w), -127, 127)`, masked to the stratum triangle |
| `B[l]`, `BG[l]` | `b` (init 0) | `round(4096 b)` |
| `HW`, `GW`, `WG` | init N(0, (0.5 / sqrt d)^2), N(0, (0.1 / sqrt d)^2), as `W` | `clamp(round(64 .), -127, 127)` (`WG`: 128) |
| `HB_k`, `GB_k` | init 0, 1 | `clamp(round(2048 .), -(2^24 - 1), 2^24 - 1)` (every bias is clamped in the forward pass exactly as the exporter clamps it; `B`, `BG`: `round(4096 .)` likewise) |
| `W0_k` | init 1.0 for the trunk input, 0.25 for the orders and for the bit-cell inputs | `clamp(round(65536 .), -W_MAX, W_MAX)` |

Gradients: rounding and flooring are identity; clamps pass gradient inside the range; an accumulator is
differentiated as the linear filter `r_t = (1 - a) r_{t-1} + a u_t`, `a = 2^-shift`; a gate uses the hard
decision forward and a hard-sigmoid slope backward. The online mixer weights and the calibration cells are
computed exactly (sequentially) and treated as constants by the backward pass. Loss per bit = the exact cost
`-log2(P / 65536)` of the integer probability with the gradient of the logistic cost at the same stretch value;
loss of a step = mean over tiers of the mean cost per byte.

### 8.3 Data handling

- Streams are chunks of consecutive bytes cut from one source; a chunk start is moved to the next line start
  when one is within 512 bytes. Every chunk starts from the boot state with empty tables (this is also how
  short messages are seen: the beginning of every chunk).
- Chunks are fed as consecutive windows with the trunk state, tables, cells and mixer carried across windows
  (truncated back-propagation, exact forward). Long chunks MUST be at least `(L + 1) * M` bytes, with `M` of
  section 2.6 computed from the largest shift of the file's schedules (21.6-28.4 KB for the v1 schedules); the
  trainer checks it. Message-sized chunks (128-4,096 B, 25 % of the long run) are exempt: they train the stream
  start, not the length regime. The bake-off and G0 used 8,192-byte chunks, below this rule (review 04, m-3); their
  length curves are measured, not covered by the argument of 2.6.
- The tables and cells are pure functions of the bytes, so they are computed in the data pipeline (vectorised,
  exact; `tools/strata/bakeoff/common.py` is the design-stage implementation and has a sequential twin in
  `ref_int.py`).
- All tiers are trained together: one forward pass at full width yields every tier's activations (they are
  prefixes), then one head per tier.

### 8.4 Boot calibration `Q0`

Counted before training on 256 training chunks: for each (order, confidence, bit position), the fraction of
live bits that equalled the expected bit, with add-one smoothing (formula in 5.4). Not trained by gradient.

### 8.5 Hyper-parameters

| Item | Bake-off (MEASURED configuration) | Long run (plan) |
|---|---|---|
| Optimiser | AdamW, betas (0.9, 0.99), weight decay 0.01, clip 1.0 | same |
| Learning rate | peak 2e-3, 100 warm-up steps, cosine to 5 % | peak chosen among {1e-3, 2e-3, 4e-3} in the smoke run by training loss at 300 steps |
| Step | 64 streams x 256 bytes | 256 streams x 256 bytes |
| Steps | 2,048 (33.55 MB) | until the hour budget ends |
| Tiers | 2 (widths 96, 224), L = 3 | the ladder of 7.2 |
| Checkpoint | at chunk boundaries; atomic rename; same command resumes | every 10 minutes and at exit |

### 8.6 Data and split for the long run (by source; leak rules of `02-evidence` E.3)

| Role | Sources |
|---|---|
| Train | Wikipedia: enwik9 bytes [1e8, 1e9) (the first 1e8 bytes are enwik8 and are verified and dropped); Loghub full logs of BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack, HealthApp (48 MB cap each, as R-112); public source code, JSON and CSV corpora (list frozen in the trainer before the run). **Excluded by source**, whatever the date or the file: GH Archive (any hour; the JSON source of V3), the Samba project (any code, not only the Silesia file), Wikipedia beyond the slice above is still Wikipedia (`enwik8` stays seen-source), Project Gutenberg (any etext, including Canterbury's `world192.txt`, `lcet10.txt`, `alice29.txt`, `asyoulik.txt`, `plrabn12.txt`: their boilerplate occurs in `webster` and `dickens`), anything produced by `scripts/prepare-benchmark-data.py`, Silesia (every file), and the Loghub systems used for selection or test (Zookeeper, Proxifier, Apache, Linux, OpenSSH) |
| Validation (model selection, early stopping) | Silesia `webster`, `xml`, `reymont`, `osdb`; Loghub Zookeeper, Proxifier |
| Frozen test (report-only) | `dickens`, the three Loghub 2k samples of the repo, both synthetic XML files, `samba`, `nci`, `mozilla`, `mr`, `ooffice`, `sao`, `x-ray` |
| Seen-source (reported separately, never called held-out) | `enwik8` (any prefix): Wikipedia is in the training set |

A pretrained initialisation is not used. The trainer MUST fail (not fall back) if a source is missing — the
silent fallback to random bytes of an earlier notebook is failure mode B.2 / B.13.

### 8.7 Export

`export.py` writes the file of section 5 from a checkpoint (all tiers), then runs the reference implementation
on 64 KB of validation data per tier and compares every `p1` with the trainer's: zero mismatches or the export
fails. It prints `TIER_ID`s and the SHA-256 of the file.

### 8.8 GPU plan (ceiling 12 h of the 18.5 h left; charter 5b)

Order (mandatory): gate G0 (9.0) -> local CPU end-to-end pass (train -> export -> reference -> Rust bit match)
-> 30-minute GPU smoke -> long run. If G0 leaves only the width-0 tier, no GPU hour is spent.

| Run | GPU hours | Purpose | Exit criteria |
|---|---|---|---|
| Smoke | 0.5 | measure bytes/s of the full ladder on the T4, test checkpoint + resume inside the session, export, bit match of the exported file against the reference on the Kaggle CPU, pick the learning rate | resume reproduces the loss curve; 0 mismatches; throughput recorded |
| Long run | 9.5 (one session, self-stopping at 9.3 h) | train the ladder | checkpoints every 10 min; export + bit match at the end |
| Reserve | 2.0 | resume after a session failure, or a second pass at a lower learning rate | — |

Throughput is a **HYPOTHESIS** until the smoke run measures it. Reference points from the repo (T4, byte-level
models, `02-evidence` A.3 row 15): 0.107 MB/s for a 7.1 MFLOP/byte LSTM, 0.53 MB/s for a 1.65 MFLOP/byte
transformer with AMP. The top tier of 7.2 costs about 4.2 MFLOP/byte forward; the sequential parts (accumulator
scans, mixers of six tiers) run on the 4 host vCPUs. Planning figure: 60-120 KB/s, i.e. **0.2-0.4 GB per hour,
2-4 GB in the long run** (about one pass over 2-4 GB of the corpus). Decision rule fixed now: if the smoke run
measures less than 40 KB/s with the `large` tier, the long run drops `large` and trains `cm .. base` (expected
3-4 times faster); if it measures more than 200 KB/s, the freed hours are not spent (they stay in reserve).
What each hour buys in ratio is not known in advance; the bake-off's learning curves (`03-bakeoff.md`) are the
only measured indication and they are at 33.55 MB.

**What to train, given G0.** The long-input tier `cm` has no trunk; its trained parameters (node biases, gain
biases, boot weights, `Q0`) are trained on CPU in minutes and need no GPU hour. The only trunk of v1 is the width-88
tier of the messaging file, which G0 kept for short messages. The GPU hours are therefore conditional on gate V3's
CPU sweep (section 9.2): no GPU hour before the messaging file `M` (table sizes, bit-cell orders) is chosen and its
short-message result is estimated on CPU; then the 30-minute smoke run and at most 9.5 hours training the width-88
tier on message-like traffic (logs, JSON, short text; the exclusions of 8.6), with chunks that respect 8.3. Expected
yield per hour is a HYPOTHESIS: the trunk's 10 % per message cell was measured after 33.5 MB of training (83 minutes
of a shared CPU); one T4 hour at the planning rate sees 6-12 times that set.

---

## 9. Pre-registered gates

Written before the Rust implementation, the reference implementation and the long training run exist. A gate
is changed only through `05-gate-amendments.md` (rule T1). `W` = `weights/strata-v1`; `REF` =
`python -I tools/strata/reference/strata_ref.py`; test inputs are files of `data/benchmark/` named in
`02-evidence` E.1.

### 9.0 Design gate G0 (local CPU; before the stage-3 review closes)

G0 is defined in one place only: `docs/internal/research/strata-backbone/05-gate-amendments.md` (pre-registration of
2026-11-13 08:44 and its amendment 1 of 09:16; the file's sha256 is registered in `00-charter.md`). In short: the
bake-off is repeated with the bit cells of 3.3 in the training loop and a trained width-0 tier; only sources never
seen in training vote; a trunk tier is kept only if, on both seeds of the candidate, it saves at least 5 % over the
width-0 tier and at least 5 % per 100,000 MAC/byte (long streams, or short messages measured per cell); otherwise v1
is the width-0 tier alone. Nothing in this section overrides that file.

**Result (2026-11-13 11:05): PASS** — `03-bakeoff.md` section 12: G0b pass on both seeds; G0c keeps the width-88
tier on short messages only (v1 = `cm` for long inputs, width 88 for messaging files); G0a TIE.

### 9.1 Correctness gates (must pass; a failure blocks release)

| Id | Gate | Threshold | Command |
|---|---|---|---|
| C1 | Rust == reference, probabilities | 0 mismatching `p1` over every bit of 12 inputs x every tier, for the synthetic weight files and the trained file | `cargo test strata_golden` (reads `tests/data/strata/*.gold`); `REF dump --weights W --tier T --input F` vs `redukt compress ... --strata-dump FILE` (diagnostic flag: every `p1` as u16 LE in coding order), sha256 |
| C1b | Rust == reference, archives | identical sha256 of the `.azth` for the same 12 inputs x tiers, default, `--strata-compact` and with a primer | `REF compress ...` vs `redukt compress --backbone strata --tier T ...`; `sha256sum` |
| C2a | SIMD independence | identical sha256 with and without `REDUKT_NO_SIMD=1` | 6 files x 3 tiers |
| C2b | Thread independence | identical sha256 with `REDUKT_THREADS` = 1, 2, 5, 12 | same |
| C2c | Plan independence | identical sha256 for every plan string accepted by `REDUKT_STRATA_PLAN` | same |
| C2d | Batch == single | every archive of `compress --input-dir ... --jobs N`, N in {1, 2, 3, 8, 32}, mixed lengths 1 B .. 2 MB, with and without a primer, equals the single-file archive; both directions | `tests/roundtrip.rs` pattern |
| C2e | API == CLI | sha256 of `/v1/compress` output == CLI output, 3 files x 3 tiers | `tests/api_parity.rs` pattern |
| C3 | Round trip | decompress(compress(x)) == x and CRC OK for lengths 1, 2, 7, 128, 4,096, 10,240, 1 MiB, 2 MiB of 6 files, every tier, with and without a primer; empty input rejected (5.6) | CLI |
| C4a | Length, real data | for every tier with `d > 0`, on `dickens` and `enwik8`: bpb of the first 2 MiB <= 1.03 x bpb of the first 10,240 B, **and** bpb of bytes [1.5 MiB, 2 MiB) <= 1.10 x bpb of bytes [0.5 MiB, 1 MiB) (was 1.25; review 04 m-12, `05-gate-amendments.md`); no reset exists | `redukt compress --bytes N`, sizes; window costs from `--strata-dump` |
| C4b | Length, stress | 256 MiB stream (64 KiB period text with 5 % random substitutions): bpb of the last 16 MiB within 5 % of the bpb of MiB 16-32; zero internal bound errors | `tools/strata/gates/stress.py` |
| C5 | Legacy invariants (T6) | sha256 of the Mamba / RWKV reference archives unchanged: enwik8 10 KB Q8 `d550671b...`, 100 KB Q8 `d907ffc0...`, the nine rows of `data/benchmark/results/determinism-cross-machine-2026-11-10.tsv`; all pre-existing tests pass; lint passes | existing commands of `02-evidence` F.1 run before and after |
| C6 | Old decoder | the binary built before Strata, given a Strata archive, exits non-zero with `unknown backbone: 4` and loads no model | saved pre-Strata `redukt` binary |
| C7 | Wrong weights | decoding with a different weight file fails before the first byte is decoded, naming both ids; with `--strata-compact` it fails at the CRC | CLI |
| C8 | No float on the codec path | no `f32` / `f64` token in `src/infrastructure/strata/` outside `plan.rs` (timing only); `tests/libm_lint.rs` passes | new lint test |
| C9 | Weight file validation | every corruption of a fuzzed set (flipped byte in each block, truncation inside a block, -128 weight, wrong CRC) and every row of the validation table of 5.5 is rejected at load; truncation at each stratum boundary loads, offers exactly the tiers present, and its default tier is `min(DEFAULT_TIER, highest tier present)` | unit tests |
| C9b | Weight-file layout across teams | from the synthetic parameters of `make_synth.py` (team O), team T's exporter writes a file whose SHA-256 equals O's synthetic file; team R loads O's file and reproduces O's golden `p1` (a layout misreading cannot pass both) | sha256 of the two files; C1 on O's file |
| C10 | Trainer == reference | 0 mismatching `p1` between the trainer's forward pass and the reference on 64 KB per tier of the exported file | `export.py` (8.7) |
| C11 | Cross-architecture (deferred until hardware exists) | identical sha256 x86-64 <-> aarch64 for the C1b set | same commands on an ARM64 host |

### 9.2 Viability gates (GO / NO-GO information; a NO-GO is reported, it does not block correctness work)

Sizes are real `.azth` bytes including header and CRC (rule T3), next to xz -9e, zstd -19 and brotli -11
(lgwin 24) on the identical bytes.

| Id | Gate | Threshold (GO) | Command / protocol |
|---|---|---|---|
| V1 | Held-out ratio vs classical | per tier: `.azth` <= best classical on at least 3 of the 4 real held-out sources (`dickens` 500 KB, Apache, Linux, OpenSSH), and total bytes over the 4 <= the best classical total; the two synthetic invoice files are reported separately and do not vote (`02-evidence` E.1) | `tools/strata/gates/heldout.py` |
| V2 | Held-out ratio vs Mamba-130M | top tier: total `.azth` bytes over the five whole industry files <= 1.30 x Mamba's (reference bpb 0.2408 / 0.2392 / 0.1849 / 0.2449 / 0.2014) and `dickens` 500 KB <= 1.30 x 1.0620, on identical bytes (Mamba re-run on the local copies, or Strata on the LF copies behind those figures) | same script; Mamba numbers of `02-evidence` A.1 |
| V3 | Short messages | R-113 protocol on the **25 held-out cells** (Wikipedia excluded: a training source of every Strata model), with the messaging weight file `M` (named below): the real archives (compact header + `PRIMER_ID` + CRC = 24 B of framing) beat zstd-19 + dictionary in >= 20 of 25 cells, in **both** pairings: primer and dictionary from the first half of the message source, and primer and dictionary from another source that is neither a training source nor the message source, named before the run. Reported: payload + 4 B (the R-114 target, labelled as such), no primer, the Wikipedia cells | `notebooks/short-message-bench.py` with `AZTH_OVERHEAD` corrected to the real header |
| V4 | Speed vs budget | there is a tier with `R >= 13 KB/s per vCPU` and `bpb <= bpb_classical - 16.1 / R[KB/s]` on `dickens` 500 KB (text) — and one with `R >= 110 KB/s per vCPU` and the same inequality on the three logs (formula of `02-evidence` C.4 at $0.014 per vCPU-hour) | `tools/strata/gates/speed.py`: wall bytes/s of `compress --input-dir` on 12 x 100 KB, divided by logical CPUs |
| V5 | Integer cost | bpb of the integer model <= 1.05 x bpb of the same architecture trained without the integer simulation (the repo's +5 % budget) | bake-off: MEASURED; repeated on the long run only if hours remain |
| V6 | Terminal footprint | on the same messaging file `M` as V3 and in the same run: peak RSS of a 256-byte compress <= 32 MB; file <= 1.5 MB | `--strata-compact` |

The messaging weight file `M` (tier, successor table bits, bit-cell orders) is chosen before V3 by a pre-registered sweep at message scale on selection sources (successor table bits {12, 16, 18} beyond order 2, bit-cell orders {0-1, 0-3}, 64 KB primer, 128-2,048 B messages); V3 and V6 are then measured on `M` only (review 04, G-2). The prototype's 30 / 30 used 5 MB of tables and no bit cells: it says nothing about `M`.

V4 uses the classical codec as the competitor because the value of a byte saved is measured against it
(`scripts/breakeven.py`). With the Fase 20 verdict in force, a NO-GO on V4 is the expected outcome for storage
and does not, by itself, stop the path-3 use.

---

## 10. Test plan and ownership (rule T2)

| Team | Reads | Writes | Must not |
|---|---|---|---|
| **R** — Rust runtime | this document | `src/infrastructure/strata/{mod,weights,model,tables,mixer,kernels,stream,plan}.rs`, `src/application/strata_engine.rs`; additive edits to `src/domain/backbone.rs`, `src/domain/coder.rs` (`encode_bit`, `decode_bit`, `finish_tight`), `src/application/{codec,lockstep,api,model_loader}.rs`, `src/main.rs`; in-crate unit tests of kernels (SIMD == scalar), loader rejection, header tables, engine with a synthetic file; `tests/strata_cli.rs` (round trip, batch == single, old-decoder message) | compute or edit any golden value; read `tools/strata/reference/` |
| **O** — oracle | this document only | `tools/strata/reference/strata_ref.py` (numpy, integers only: `dump`, `compress`, `decompress`), `make_synth.py` (three synthetic weight files: not geared / geared / with a width-0 tier and 10-bit tables), `tests/data/strata/*.bin` and `*.gold` (inputs, `p1` sha256 per tier, archive sha256, with and without a primer) | read any Rust source of Strata or the bake-off's `ref_int.py` |
| **T** — trainer | this document | `tools/strata/train/{train,export,data}.py`, `notebooks/strata-train.py`; gates C10 and C9b (its exporter on O's synthetic parameters) | change arithmetic to make training easier without an amendment here |
| **A** — audit (T5) | everything | `04-design-review.md`, `06-verification.md`: checks that no test asserts only on values produced by the code under test, that thresholds equal section 9, that the synthetic files exercise saturation, collisions, clamped mixer weights, gates both ways, order inactivity at stream start | write production code |
| **V** — verification (stage 6) | everything | `tools/strata/gates/*.py`, the gate table with measured values and commands | change thresholds |

Disagreements between R and O are resolved against this text; if the text is ambiguous it is amended first.
The bake-off's `ref_int.py` is a design-stage check by the architect and is **not** the oracle.

Dock tests (owned by R, expected values from O where bytes are involved): header build / parse table for
every legal and illegal combination of 5.6; flag rejection of 6.4; group keying of 6.5; plan forcing of 6.7;
`/v1/status` content; 422 paths; slot reuse after `boot()` equals a fresh stream.

Files touched outside code: `docs/api/FORMAT-SPEC.md` (backbone id table, STRATA block, history),
`weights/MANIFEST.md`, `README.md`, `CLAUDE.md` facts, `.gitignore` (`data/strata/` added by the bake-off),
`deploy/gcp/entrypoint.sh` (weight set per backbone).

---

## 11. Risks, and what result would make us abandon the design

| # | Risk | Evidence so far | What we do |
|---|---|---|---|
| 1 | **The trunk adds little over the online tables** — materialised on long streams: with bit cells in the loop it buys 0.0-0.9 % (G0), so long inputs use `cm`; on short messages it buys 10 % per cell. Original wording: on the data where bytes are cheapest (logs), and classical codecs are already near the tables' level there | MEASURED (prototype): online components off x6.3-6.6, trunk zeroed +11-14 % on the selection set; with post-hoc bit cells the trunk is worth 0-15 % on three unseen sources and costs 4-8 % on `xml`; it is worth 14-18 % on seen-source text and 4-27 % on held-out short messages with a primer (30-33 % on the seen-source Wikipedia messages). **Partly materialised**: the trunk's value depends on domain similarity and message length | G0c keeps a trunk tier only for data types where it buys >= 5 %; elsewhere the `cm` tier is the honest operating point (crossover X4) |
| 2 | **Small pretrained trunks do not generalise to unseen text** (R-105, R-112 prior) and 33 MB of bake-off training says little about convergence | MEASURED: the prior holds with the online component present: on long unseen streams the trained trunks are within a few % of no trunk once bit cells exist (post hoc); Mamba-130M is 35-164 % better per test file | long run with by-source validation; V1/V2 decide |
| 3 | **Fixed fading memory is weaker than gated recurrence** at equal compute | MEASURED: tie on the selection set (0.995 against the GRU); on in-domain text the candidate is 5-7 % worse (seen `enwik8` 1.9277 vs 1.7996, `dickens` 2.2502 vs 2.1516) and fits the training data worse (1.60 vs 1.44); gears helped (-3.0 %) | accepted for v1 (bit-exact determinism, bounded state and nesting at a tie); revisited if G0 or the long run widens the in-domain gap beyond 5 % |
| 4 | Trainer throughput: the exact sequential parts (scans, mixers) are slow in PyTorch | bake-off MEASURED: about 3x the wall time per step of `nn.LSTM` at equal MAC/byte on a shared CPU | smoke run measures it on the T4 before any long run; drop the top tier by the rule of 8.8 |
| 5 | Integer kernels slower than assumed (the per-core MAC rate is a hypothesis) | no Rust kernel exists yet | calibration prints MAC/s; V4 uses wall bytes/s only |
| 6 | Table state (39 MB per stream at the default sizes: successor tables 5 MB, bit cells 33.8 MB) limits lockstep width and small terminals | derived | table bits are per weight file; messaging files use small tables (V6) |
| 7 | The dock is larger than "register a backbone" (A7 deviation) and touches `coder.rs` and the three engines' dispatch | 6.1 | additive edits only; C5 before and after every merge |
| 8 | Resemblance to prior art (section 12) may not satisfy the owner's reading of A0 | documented | owner decision; the measured comparison stands either way |
| 9 | The primer result is a best case (primer from the same source as the messages, like the zstd dictionary) and both ends must hold the same primer | measured same-source only | V3 requires the different-source pairing; the header carries `PRIMER_ID` |

**Abandon (or go back to design) if any of these is measured:**

1. C4 fails with correct code (ratio degrades with length): the central structural claim would be false.
2. C1/C2 cannot be met without constraining kernels to one accumulation order: the integer contract would be
   broken somewhere this document did not foresee.
3. V5 fails by more than the budget (integer model > 5 % worse than its float twin) after a full training run.
4. After the long run, every tier with `d > 0` loses V1 on the held-out text source **and** the `cm` tier is
   within 2 % of the `large` tier on every frozen test source and on the short-message protocol with a primer:
   the trunk would be buying nothing a classical context mixer does not already give, and the right product would be the `cm` tier alone (which needs no
   training infrastructure at all). G0 applied this logic early: long inputs use `cm`; the trunk survives only for
   messaging files, where it measured 10 % per cell.
5. V2 fails by more than 2x on the path-3 traffic types (text-like messages): then Mamba-130M remains the only
   candidate for that segment and Strata's footprint advantage does not compensate.
6. With real framing, the primed width-88 messaging file does not beat a same-source zstd dictionary on held-out
   messages in gate V3: path 3, the reason the trunk survived, would then not need it either.

---

## 12. Relation to prior work (axiom A0: documented, not hidden)

| Ingredient here | Closest prior work | Honest statement |
|---|---|---|
| Binary decomposition, stretch / squash, logistic mixing with online integer SGD, weight sets selected by a small context, adaptive probability cells | PAQ / lpaq (Mahoney, public domain) | Directly inherited. Redukt already credits context mixing to this lineage |
| "Which byte followed this context last time", with run-length confidence | LZP (Bloom, 1996); the match model of PAQ | Same idea, eight orders, stored without positions |
| Network trained as a component of an online mixer; trunk-driven per-context gains | cmix / lstm-compress (the other way round: a network as one mixer input, trained online); gated linear networks (Veness et al., 2017-2019: neurons that are context-gated geometric mixers); dynamic evaluation (Mikolov 2010, Krause 2018); neural cache (Grave et al., 2017) | The specific arrangement — offline quantisation-aware training with the tables and the mixer running exactly in the loop, and gains emitted by the state — was not found in this form; we did not do a literature search beyond what is cited in the repo, so "not found" is weak |
| Trunk: fixed power-of-two leaky accumulators after saturating affine layers, no recurrent weights | diagonal linear recurrences (IndRNN, LRU, S4D), QRNN / minGRU (state = convex combination driven by the input only), multi-time-scale leaky units, TDNN delay taps | **This is the degenerate, un-gated corner of the QRNN / minGRU family with gates frozen to constants.** It was derived here from the requirement of a provable contraction in integers, not by shrinking those models, but an informed reader will recognise it. The geared variant is closer still: a minGRU-like update with a binary, input-only gate restricted to two powers of two |
| Binary-tree output over bytes | hierarchical softmax (Morin & Bengio, 2005) | Same structure; used here because the coder is binary |
| Nested tiers as prefixes, one training pass | slimmable networks (Yu et al., 2018), nested dropout (Rippel et al., 2014), Matryoshka representations (2022), progressive networks (Rusu et al., 2016: new columns read old ones, never the reverse) | The block-triangular constraint that makes smaller tiers exact prefixes *in activations* is the progressive-network topology used for size instead of tasks |
| Integer-only inference by quantisation-aware training with straight-through estimators | Bengio et al. 2013; Jacob et al. 2018 | Standard technique |

Under a strict reading of A0 ("not a recombination of a published cell") the trunk is defensible as a design
derived from the codec's constraints; under a stricter reading ("nothing a reviewer would file next to minGRU")
it is not. The bake-off's measured comparison is what the decision should rest on.

---

## Appendix A. Spot values (for self-checking an implementation; the weight file remains the authority)

Computed with `tools/strata/bakeoff/common.py` and `ref_int.py` on 2026-11-12 (MEASURED: the command is in
`03-bakeoff.md` section 6).

```
asr(-5, 2) = -2      asr(-1, 7) = -1      asr(5, 2) = 1      bit_length(0) = 0      bit_length(255) = 8

accumulator, shift 3, starting from S = 0:
    u = 100  -> S = 51200,  read-out 12
    u = 100  -> S = 96000,  read-out 23
    u = 100  -> S = 135200, read-out 33
    u = -127 -> S = 53276,  read-out 13

SQ[0] = 1   SQ[1] = 1   SQ[2815] = 17625   SQ[3070] = 32704   SQ[3071] = 32768   SQ[3072] = 32832
SQ[3327] = 47911   SQ[5886] = 65535   SQ[6142] = 65535
ST[0] = -2307   ST[1] = -2025   ST[1023] = -281   ST[2047] = 0   ST[2048] = 0   ST[3072] = 281
ST[4094] = 2025   ST[4095] = 2307
QSHIFT[0] = 2   [1] = 2   [2] = 3   [5] = 3   [6] = 4   [13] = 4   [14] = 5   [29] = 5   [30] = 6   [61] = 6
QSHIFT[62] = 7   [255] = 7

SHA-256 of the blocks as stored in the file:
    SQ      (6143 x u16 LE)  2ac2e07ba085a55d746069983f10fcf7ff86f255f757ab07fa6444d5840486ca
    ST      (4096 x i16 LE)  40ca9c9b85fd4c847389863042d72f437f50877df71bce9fd6b0b95625c01733
    QSHIFT  (256 x u8)       85e8ca3a14da54f342fe882df92d6bcc64c2078a85f25bbafdedcc6eac689965

ctx_hash (context bytes oldest first -> slot, check):
    "abc"            len 3,  tb 16  ->  49292, 111
    "abcd"           len 4,  tb 16  ->  26255, 44
    "hello world!"   len 12, tb 18  ->  25353, 153
    bytes 0x00..0x0F len 16, tb 18  ->  214843, 18
    0x00 0x00 0x00   len 3,  tb 16  ->  25381, 60
```

CRC-32 everywhere in this document is the one of the AZTH container: IEEE 802.3, reflected polynomial
0xEDB88320, initial value and final XOR 0xFFFFFFFF.

## Appendix B. What the bake-off prototype is, relative to this document

The prototype in `tools/strata/bakeoff/` implements sections 2.3-2.5, 3 and 4.2-4.3 with these differences, none
of which changes a probability:

| Item | Prototype | This document |
|---|---|---|
| Node matrix | 255 rows indexed `node - 1` | 256 rows indexed `node`, row 0 zero |
| Mixer state | deltas from the boot weights | the weights themselves |
| Weight file | none: integer tensors are exported in memory (`models.export_int`) | section 5 |
| Tiers | two, widths (96, 224) fixed variant or (88, 208) geared variant, L = 3 | the ladder of 7.2 |
| Width-0 tier | only as an ablation (`--trunk off`: trunk output zeroed in a model trained with a trunk) | a trained tier |
| Container | none: bits are ideal code lengths; `ref_int.py` produces real range-coder bytes without a header | AZTH v5 with the STRATA block |
| Training chunks | 8,192 bytes, all from the boot state | 32,768 bytes plus 25 % short chunks |
| Primer | `shortmsg.py --primer N`, same semantics as 3.4, measured on short messages | section 3.4 (v1) |
| Bit cells (3.3) | absent from the bake-off's training; measured post-hoc in float, then in the training loop by G0 (`strata_g_bc`, exact integer simulation, 0 mismatches against the reference) | part of v1, pending gate G0 |

## Appendix C. Resolutions of the adversarial review (`docs/internal/research/strata-backbone/04-design-review.md`)

Fixed by the author on 2026-11-13; not yet re-checked by the reviewer.

| Finding | Severity | Resolution | Where |
|---|---|---|---|
| B-1 | blocking | one normative G0 text in `05-gate-amendments.md` (amendment 1, registered before any G0 result was read): no seen source votes, per-cell `S`, two seeds of the candidate, worse-of-two verdicts; G0 run and decided | 9.0; `05`; `03-bakeoff.md` 12 |
| B-2 | blocking | explicit byte sequences for GLOBALS and STRATUM, row-major rule, width-0 stratum; cross-team layout gate C9b | 5.1, 5.3, 5.5, 9.1 |
| B-3 | blocking | big-endian `code`, STREAM = archive[header_size .. length - 4), zeros past STREAM, `lo` of the decoded bit | 4.2, 4.3 |
| B-4 | blocking | no preprocessing in v1 (FLAGS exactly 0x40, flags are errors), CRC-32 of the message bytes (primer excluded), empty input rejected, 4.2-4.3 take precedence over `coder.rs` | 5.6, 6.4, 6.5, C3 |
| G-1 | grave | real framing (24 / 28 B with a primer): 22 / 20 of 30 cells, 17 / 15 of 25 held-out; V3 on the 25 held-out cells, threshold 20 | 3.4, 4.4, X10, V3; `03-bakeoff.md` 0, 7.6 |
| G-2 | grave | V3 and V6 on one messaging file `M`, chosen by a pre-registered message-scale sweep | 9.2 |
| G-3 | grave | default tier = `min(DEFAULT_TIER, highest tier present)`, tested in C9 | 6.4, C9 |
| G-4 | grave | long-run exclusions listed by source (GH Archive, Samba project, Wikipedia, Project Gutenberg, Silesia, generated XML, selection and test Loghub systems) | 8.6 |
| m-1, m-2, m-4, m-6, m-7, m-11 to m-17, m-19 to m-21 | minor | fixed in the text (m-16: the clashes of `n` and `k` remain, the header byte is now `SFLAGS`) | 1, 2.2, 2.5, 2.6, 5.2, 5.5, 5.6, 6.3-6.7, 7.1-7.3, 8.2, 8.3, 9 |
| m-3 | minor | chunk rule restated for long chunks; G0 and the bake-off below it, disclosed | 8.3; `05` |
| m-5, m-8, m-10, m-18 | minor | corrected or disclosed in `03-bakeoff.md` (held-out macro +1.9 %, Gutenberg text, D1 caveat, frozen copy of the measured sources) | `03-bakeoff.md` 0, 2, 3, 7 |
| m-9 | minor | G0b labelled a same-author consistency check | `05`; `03-bakeoff.md` 12 |
