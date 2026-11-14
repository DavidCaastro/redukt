# Strata backbone — charter and orchestration state

> Working name: **Strata** (nested operating points = strata). Rename freely before release.
> Opened 2026-11-12. This file is the durable state of the effort: any session or agent can resume from it.
> Every agent working on Strata reads this file first and obeys sections 2, 3 and 5.

## 1. Purpose and honest framing

Build a neural backbone from scratch that is co-designed with the Redukt codec instead of adapting a research
model (RWKV-7, Mamba-130M) to it. The codec's needs are known and measured; the backbone should be shaped by them.

What this effort is NOT allowed to assume (CLAUDE.md, Fase 20 verdict, all measured):
- Redukt is not viable as a storage-saving product today (cost 100-170x the value).
- No published small backbone keeps the ratio (R-105 step 1); small static specialists trained by us do not
  generalize to unseen systems (R-112). A new small backbone inherits that prior until it measures otherwise.

Why it is still justified (it must keep justifying itself against these):
1. Path 3 (narrowband satellite messaging) is the one candidate segment, and its open conditions are exactly
   backbone properties: a model far smaller than 130 MB on the sender terminal, ARM <-> x86 bit-identity (R-115),
   compact framing (R-114).
2. Portability principle: bit-identity across tiers and accelerators (R-104) is a stated goal that a float
   research model makes expensive and an integer co-designed model can give by construction.
3. It produces new evidence on the closed fast-tier question under a different design (integer, bounded state,
   codec co-design, elastic tiers), with gates fixed before measuring. A NO-GO is an acceptable, reportable outcome.

## 1b. Objective (owner, 2026-11-14)

Asked what objective the effort serves, the owner stated it: **Strata is the project's intended backbone — built
from scratch on the lessons learned, for a codec whose input, formulas and output are known — and it should reach
what the product reaches today.** This widens reason 3 above from "new evidence" to the effort's goal, and it does
not replace the measured framing of section 1.

Operational form (so that it can be measured and can fail): on held-out data, find whether some Strata size has a
compute cost per raw GB below the storage value of the bytes it saves against the best classical codec, and how
close to Mamba-130M's ratio it gets. Gate V7 (`05-gate-amendments.md` amendment 4; plan and position at
registration in `08-capacity-ladder.md`). Three things the owner was told the same day and that stay true:
- the storage saving per GB is linear in time and already counted over 7 years; only the compute per byte can move;
- knowing the codec's interface fixes the engineering (integer, bounded state, size), not the ratio: the ratio is
  what the model knows about the data, and at equal budget the original trunk ties with GRU and LSTM controls;
- at registration nothing meets the objective: with PPMd in the classical set every trained tier loses on text, the
  width-0 tier is viable only on logs, and Mamba-130M is 1.2-2.0x smaller.

Path 3 remains the scope of v1 (the frozen specification); the objective is pursued through V7 and through
amendments, never by loosening a gate.

## 2. Design axioms (binding; deviation needs written justification in the spec and must still pass the gates)

- **A0 Original design, derived from the codec.** (Owner's instruction, 2026-11-12.) Strata is not a port, a
  shrunken copy or a recombination of a published cell (LSTM, GRU, minGRU, RWKV, Mamba, transformer). The
  architecture is derived from first principles from what the codec needs: integer arithmetic, bounded state,
  a probability interface in the coder's own domain, deterministic online adaptation, nested tiers. Published
  cells appear ONLY as equal-budget controls in the bake-off: the original design must be measured against
  them, and the spec must state plainly where it wins, ties or loses. Novelty is not a gate; measured ratio per
  unit of compute on held-out data is. Any resemblance to prior art found along the way is documented, not hidden.
- **A1 Deterministic by construction.** Inference is integer / fixed-point only (integer weights, integer
  activations, table-driven nonlinearities, exact integer accumulation). Output must be bit-identical across
  ISA (scalar, AVX2, NEON), thread count, batch size and accelerator, without relying on float operation order.
- **A2 Stable at any length by construction.** The recurrent state is bounded by the arithmetic, not by a reset.
  Gate: no ratio collapse from 10 KB to >= 2 MB with resets disabled.
- **A3 One code path for every dimension.** Width, depth, vocabulary and state size are configuration read from
  the weight file. A single weight file carries nested operating points (tiers); a smaller tier is a prefix
  (in width and/or depth) of a larger one.
- **A4 Two kinds of autoscaling, never confused.**
  (a) Execution autoscaling (threads, SIMD path, lockstep width, cache-aware blocking) is automatic, always on,
  and NEVER changes a single output bit.
  (b) Tier autoscaling (which nested model is used) changes the ratio, so it is declared in the AZTH header and
  the decoder reproduces it on any hardware. The default tier is fixed and reproducible across machines;
  hardware-driven tier selection is opt-in (`auto`), because same input + same flags must give the same sha256
  everywhere (gates 2b, 2c, 3b).
- **A5 Scale philosophy.** Nothing is enabled "because it is an optimization". The spec carries a cost model
  (integer MACs and bytes of weight traffic per input byte as a function of width, depth, vocabulary, bits) and a
  crossover table stating at which dimensions each technique pays or stops paying (vocabulary type, per-matrix
  threading, lockstep vs one stream per core, bit width, head factorization, online adaptation, CM weight).
  Crossovers used at runtime are calibrated by measurement on the host, not hard-coded folklore.
- **A6 Co-designed with the codec.** Vocabulary is chosen by the cost model, not inherited (byte-level unless the
  cost model says otherwise at that scale). The backbone emits exactly what the mixer consumes and exposes the
  hooks the online components need. No 50K-logit host copy.
- **A7 CPU is the reference.** Pure Rust, zero external crates, plugs in through the `Backbone` trait; the core is
  touched only to register the backbone and its header declaration.
- **A8 Format discipline.** Everything that changes the coded stream is in the header; older decoders must fail
  early and cleanly; the new decoder must keep decoding v2-v5 byte-identically. Existing Mamba/RWKV outputs must
  not change by one byte.

## 3. Test independence rules (green must mean "works", not "agrees with itself")

- **T1 Pre-registration.** Gates and thresholds are written in the spec BEFORE the measurement. Changing one after
  seeing a result requires a dated amendment in `05-gate-amendments.md` with the reason. No silent retuning.
- **T2 Independent oracle.** Golden vectors come from a reference implementation written from the spec alone by an
  agent that has not read the Rust implementation. The Rust backbone must match it bit-for-bit. Disagreements are
  resolved against the spec text; if the spec is ambiguous, the spec is fixed first.
- **T3 Held-out truth.** Ratio is reported on sources never used for training or tuning, as actual `.azth` bytes
  including header, next to xz / zstd / brotli on the identical bytes. Never report on a training file.
- **T4 No self-goldens, no weakening.** A test may not assert only on values produced by the code under test. A
  failing gate is reported as failing; it is never loosened, skipped or special-cased to get green.
- **T5 Audit.** A separate agent audits the test suite for tautologies and for thresholds fitted after the fact.
- **T6 Legacy invariants.** Existing reference outputs (Mamba and RWKV archives, sha256 and sizes) are identical
  before and after the change.

## 4. Stages and files

| Stage | Output | File |
|---|---|---|
| 1a Integration map (code) | how a backbone plugs in | `01-integration-map.md` |
| 1b Evidence digest | measured constraints, data, hardware | `02-evidence-and-constraints.md` |
| 2 Architecture + bake-off | spec, cost model, gates | `docs/api/BACKBONE-STRATA-SPEC.md`, `03-bakeoff.md` |
| 3 Adversarial design review | findings and resolutions | `04-design-review.md` |
| 4a Rust implementation | backbone, loader, header, autoscaler | `src/infrastructure/strata/` + registration |
| 4b Independent reference + goldens | spec-only oracle | `tools/strata/reference/`, `tests/data/strata/` |
| 4c Trainer + exporter | QAT trainer, resumable, Kaggle-ready | `tools/strata/train/`, `notebooks/` |
| 5 Local training (functional proof) | small real weights | `weights/strata-*/` (gitignored) |
| 6 Independent verification + audit | gate results, honest numbers | `06-verification.md` |
| 7 Docs and roadmap | user docs, ROADMAP, CLAUDE.md facts | `docs/` |
| V7 Capacity ladder (objective 1b) | frontier tool, PPMd baseline, plan; then results | `tools/strata/viability/`, `08-capacity-ladder.md`, `09-capacity-ladder-results.md` |

Paths in the table are the plan; the spec may refine them.

## 5. Rules for every agent

- No `git commit`, `push`, `stash`, `reset`, `checkout`, `clean`. Never delete `weights/` or `weights/MANIFEST.md`.
- Shell commands in the foreground only; never `run_in_background`. Anything longer than ~9 minutes must be
  resumable in chunks. `taskkill //F //IM redukt.exe` only for stray Redukt processes.
- Zero external crates. Rust edition 2021, lines <= 120, nesting <= 3, no dead code, warnings are errors,
  no new `unwrap()` / `panic!()` on codec or coder paths.
- Privacy: no external request may contain names, e-mails, user paths, project or client identifiers. No pushes
  to Kaggle or any other service.
- Write results to disk; reports back to the orchestrator are short. Measured facts are labelled MEASURED with
  the command that produced them; everything else is labelled as inference.

## 5b. GPU training authorization and budget (owner, 2026-11-12)

- The owner authorized launching Strata training on Kaggle GPU from this effort (overrides the "no pushes to
  Kaggle" line in section 5 for Strata training notebooks only; nothing else is pushed anywhere, no `git push`).
- Quota: 26.50 of 45 GPU hours already consumed -> **18.5 h remaining**. Planned ceiling for Strata: 12 h, keeping
  6.5 h in reserve. Every launch is logged below with its expected and actual hours.
- Order is mandatory: (1) the trainer passes locally on CPU end to end (train -> export -> reference -> Rust, bit
  match); (2) a GPU smoke run of at most 30 min; (3) only then the long run. No long run on untested code.
- Runs must checkpoint and resume, and must stop themselves before the Kaggle session limit.
- Pushed notebooks carry code and public corpora only: no personal data, no client data, no secrets.

| Date | Run | Expected h | Actual h | Result |
|---|---|---|---|---|

## 6. Status ledger

| Date | Stage | State | Notes |
|---|---|---|---|
| 2026-11-12 | 1a, 1b | done | `01-integration-map.md`, `02-evidence-and-constraints.md` (22 contradictions in its section I) |
| 2026-11-12 | legacy fixes | running | owner asked to fix the pre-existing defects found in 1a -> `07-legacy-fixes.md` |
| 2026-11-12 | 2 | running | architect launched: original design (A0), bake-off vs published controls, spec |
| 2026-11-13 | 2 | draft done, NOT frozen | bake-off complete (`03-bakeoff.md`): D1 geared trunk, D2 TIE vs GRU, D3 TIE; no model beats the best classical codec on the selection set; spec `docs/api/BACKBONE-STRATA-SPEC.md` adds bit cells (3.3) and primer (3.4) and waits for design gate G0; no GPU hour used |
| 2026-11-13 | 2 / G0 | rule registered | `05-gate-amendments.md` with G0 amendment 1 (review B-1: one definition, no seen source votes, second seed of the candidate) has sha256 `30a8cb51287c6d5e3ed21f5e2c70c0b6ba7e9f5f6e1fe475388b4df74b8aae7c`, registered at 09:16 before any G0 selection or test result was read |
| 2026-11-13 | 2 / review | gates amended | amendment 2 of `05-gate-amendments.md` (V1, V2, V3, V6, C3, C4a, C9, C9b after review 04); file sha256 now `c9efaf148bd106e3f79d0f95906614c8f4d8b54c14f34e7108e53b2660a16025` (amendment 1, the G0 rule, is unchanged) |
| 2026-11-13 | 2 / G0 | done: PASS, spec frozen for v1 | G0 decided 11:05 by the registered rule (`03-bakeoff.md` 12): v1 = width-0 tier `cm` for long inputs, width-88 trunk tier for messaging files, width 208 dropped; G0a TIE; G0b pass on both seeds. Review 04 blocking B-1..B-4 and grave G-1..G-4 fixed in the spec text (Appendix C), reviewer re-check pending; a reopened blocker withdraws the freeze. Report-only test set and R-113 messages in `03-bakeoff.md` 12.2. No GPU hour used |
| 2026-11-14 | 2 / amendment 3 | done, freeze stands | owner asked to reserve the accelerator path (portability principle) before implementation: spec 6.9 (trunk-only offload, exact and undeclared, encoder windows, probe, reserved ABI, seam for 4a), hook H7, deferred gate C12, X17, risk 10, Appendix D. Arithmetic, format, v1 scope and thresholds unchanged. `05-gate-amendments.md` sha256 now `218cc8f82f97fc08f42b4bb4ee130c1f7b6b7be729451c04f0c7cf22fb8378dd` (amendments 1 and 2 byte-identical, append-only). No GPU hour used |
| 2026-11-14 | legacy fixes | done, committed | full re-verification (`07-legacy-fixes.md` section 11): 0 warnings, 318 + 3 + 5 tests and all 15 ignored integration tests pass, 23/23 legacy archives byte-identical (T6), batch == single 10/10, `REDUKT_NO_SIMD=1` identical. Pre-existing fixture `data/test-fixtures/golden.azth` fails identically at the previous HEAD (not caused here, no test reads it). Committed as 4b92829; the owner authorized commit and push of this effort's work on 2026-11-14 (section 5's no-commit rule still binds sub-agents) |
| 2026-11-14 | objective 1b / amendment 4 | registered, freeze stands | owner stated the objective (section 1b). PPMd measured and added to the classical set (stricter): Strata loses on text (`dickens` +3.7 % to +8.1 %), still wins on logs (-15 % to -22 %); projected V1 FAIL for both v1 tiers (`03-bakeoff.md` 13). Gate V7 (capacity ladder) registered before any ladder model exists; position at registration: logs "CM only" passes, text O3, nothing O1. `05-gate-amendments.md` sha256 now `9b0b61f44ea442f25a883565b9be29a613c7cbdd179fe39a20b1425f9da2a7d6` (amendments 1-3 byte-identical: the first 160 lines still hash to `218cc8f8…`). `08-capacity-ladder.md` is a plan: its sections 6.2 and 6.3 are incomplete, and the sha256 that counts is the one logged here before the first ladder evaluation. No GPU hour used |
