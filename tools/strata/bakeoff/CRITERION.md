# Strata bake-off — selection criterion (pre-registered)

Written 2026-11-12 22:05 local time, **before any full training run was started**. This file is not edited after
the runs start; any later change goes to `docs/internal/research/strata-backbone/05-gate-amendments.md` with a
date and a reason (rule T1 of the charter). Its sha256 is quoted in `03-bakeoff.md`.

## What had been seen when this was written (disclosure)

Only pipeline smoke tests, 64 training steps each (1.05 MB of data, i.e. essentially untrained models), run to
debug the code. Their selection-set macro bpb were visible: `strata_f` tier B 0.7847, `lstm` 0.7832,
`lstm_sm` 6.0929, `strata_f` with the trunk zeroed 0.8641, `strata_f` with the online components off 5.7549.
No threshold below was derived from them; the thresholds are the ones planned before writing any code.

## Equal budget

| Item | Value (same for every model) |
|---|---|
| Compute cap, tier B / tier A | <= 128,000 / <= 32,000 integer multiply-accumulates per input byte at inference (encoder = decoder), counted analytically: every weight x activation product of the trunk, the head (8 node evaluations per byte), the trunk-driven gains and the online mixer (dot product + update). Table lookups, shifts, adds and compares are not MACs and are reported separately in the specification's cost model |
| Parameter cap, tier B / tier A | <= 256,000 / <= 96,000 trainable scalars |
| Data | the same chunks in the same order: sampler = pure function of seed 1204 and the step |
| Steps | 2,048 steps of 64 streams x 256 bytes = 33.55 MB seen once; stateful windows over 8,192-byte chunks |
| Optimiser | AdamW, peak lr 2e-3, betas (0.9, 0.99), weight decay 0.01, 100 warm-up steps, cosine to 5 %, clip 1.0 |
| Seeds | one (1204) for initialisation and data |

Wall-clock time is not a budget (another process is using this machine); it is reported for information only.

## Models

| Name | What | Arithmetic in the evaluated model |
|---|---|---|
| `strata_f` | original candidate, fixed time scales; nested tiers A and B from one training run | integer (simulated exactly) |
| `strata_g` | original candidate, geared time scales (input-driven choice between two shifts); nested tiers A and B | integer (simulated exactly) |
| `lstm`, `gru` | published cells (PyTorch `nn.LSTM`, `nn.GRU`, one layer) as the trunk of the same coder-domain head, online tables and online mixer; tier B | float (their best case) |
| `lstm_a`, `gru_a` | same at tier A | float |
| `lstm_sm` | textbook form: LSTM + 256-way softmax, static, no online component; tier B | float |
| `strata_f_float`, `strata_g_float` | the candidates without the integer simulation | float (reported only) |

## Data roles (by source; no source has two roles)

- Train: Wikipedia = bytes [1e8, 3e8) of enwik9 (the first 1e8 bytes, which are enwik8, are verified by sha256 and
  dropped); 11 Loghub systems (BGL, HDFS_1, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird, OpenStack,
  HealthApp; first 16 MB of each); six non-literary Canterbury files.
- Selection (the only data used for the decision below): first 300,000 bytes of `webster` and of `xml` (Silesia),
  and of the Loghub systems Zookeeper and Proxifier. None of these sources is in the training set.
- Frozen test (report-only, read after the decision is written): `dickens` first 500,000 B, the three Loghub
  2k samples of the repo (Apache, Linux, OpenSSH), the two synthetic invoice XML files.
- Seen-source (never called held-out): `enwik8` first 500,000 B — Wikipedia is a training source here.

## Metric

`M(model, tier)` = unweighted mean over the four selection sources of bits per byte, each source coded as one
stream from the boot state, online components on, no resets. Bits = ideal code length of the probabilities handed
to the coder (16-bit integer probabilities for the integer models).

## Decision rule

- **D1 — which original candidate.** The one with the lower `M` at tier B. If the two are within 2 % of each
  other, `strata_f` (fewer operations, no gates).
- **D2 — freeze or not.** Let `S` = `M` of the D1 winner at tier B in integer arithmetic and `C` = the lower of
  `M(lstm)` and `M(gru)` (float). **WIN** if `S < 0.98 C`; **TIE** if `0.98 C <= S <= 1.02 C`; **LOSS** if
  `S > 1.02 C`. The trunk is frozen as designed on WIN or TIE (on a TIE the reason is that only the original design
  is integer, bounded and nested by construction). On LOSS it is **not** frozen: the loss is reported as such, with
  a proposal, and the specification is marked as not frozen in the part that lost.
- **D3 — tier A.** Same comparison between tier A of the D1 winner and the lower of `M(lstm_a)`, `M(gru_a)`.
  Reported with the same WIN / TIE / LOSS words. A LOSS at tier A does not block D2 but must be stated as the cost
  of nesting or of the design at that size.

Nothing else decides: not the frozen test set, not the seen-source slice, not the ablations, not the float
variants, not wall-clock time.

## Declared limitations (known before running)

- One seed; the 2 % band is a judgement of single-seed noise, not a measured variance.
- 33.55 MB of training data seen once: every model is far from converged; ranking at convergence may differ.
- The controls are evaluated in float, the candidates in integer arithmetic: the comparison is biased against the
  candidates on purpose. A quantised control is not part of the decision.
- All models share the coder-domain head, the online tables and the online mixer, so D2 isolates the trunk. The
  textbook control `lstm_sm` shows what the published form does without them.
