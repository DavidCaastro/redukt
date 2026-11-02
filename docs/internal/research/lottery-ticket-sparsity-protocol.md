# Lottery Ticket Sparsity Protocol — Backbone Acceleration via Pruning & Dynamic Sparsity

> Research: 2026-11-02
> Origin: Frankle & Carbin (MIT, 2018) — "The Lottery Ticket Hypothesis"
> Context: Redukt backbone at 34 tok/s (Mamba SIMD), 140M FLOPs/token
> Status: RESEARCH — viable techniques identified, implementation pending

---

## 1. Problem Recap

The backbone forward pass dominates compression time:
```
128.4M FLOPs per token breakdown:
  in_proj  (24 layers): 56.6M FLOPs  44.1%  ← MATMULS
  out_proj (24 layers): 28.3M FLOPs  22.0%  ← MATMULS
  LM head  (1x):        38.6M FLOPs  30.1%  ← MATMUL
  SSM recurrence (24x):  2.4M FLOPs   1.9%  ← ACTUAL STATE UPDATE
  conv + dt + norms:      2.5M FLOPs   2.0%
```

96.2% of FLOPs are matmuls. The SSM state update (the "memory") is only 1.9%.
Previous approaches (R-049 PredCache, R-056 CM-selective) failed because they
froze the state, causing compounding divergence through 24 layers.

## 2. Foundational Paper: The Lottery Ticket Hypothesis

**Citation**: Frankle, J. & Carbin, M. (2018). "The Lottery Ticket Hypothesis:
Finding Sparse, Trainable Neural Networks." ICLR 2019. arXiv:1803.03635.

**Core thesis**: Dense, randomly-initialized networks contain sparse subnetworks
("winning tickets") that reach comparable accuracy at 10-20% of the original size.

**Original method**: Iterative magnitude pruning (IMP) — train, prune lowest weights,
reset remaining weights to initialization, retrain. Requires full training loop.

**Why it doesn't apply directly**: Redukt uses pretrained Mamba-130M weights with
no training infrastructure. IMP requires iterative retraining.

**What DOES apply**: The 2023-2026 derivative works that achieve pruning
WITHOUT retraining, specifically for SSMs and LLMs.

## 3. Derivative Papers — Direct Applicability to Redukt

### 3.1 SparseSSM (Tuo & Wang, 2025) — One-Shot SSM Pruning

**Paper**: "Efficient Selective Structured State Space Models Can Be Pruned
in One-Shot." arXiv:2506.09613. ICML 2026.

**Method**: Extends Optimal Brain Surgeon (OBS) to SSM architectures.
Uses closed-form second-order saliency from SSM state-transition matrix diagonals.
Structured column pruning on A matrix — removes entire state dimensions.

**Key results**:
- Mamba-370M: 50% SSM sparsity, NO zero-shot accuracy loss
- Mamba-1.4B: 50% SSM sparsity, negligible degradation
- Structured column pruning → 1.72x SSM module speedup
- One-shot (no retraining, no fine-tuning)

**Applicability to Redukt**: DIRECT. Mamba-130M uses same architecture.
Prune A matrix columns → reduce d_state from 16 to 8-12 per layer.
BUT: SSM recurrence is only 1.9% of FLOPs → 1.72x on SSM = ~1.7% total.

### 3.2 LAST (Gwak et al., NeurIPS 2024) — Layer-Adaptive State Pruning

**Paper**: "Layer-Adaptive State Pruning for Deep State Space Models."
arXiv:2411.02824. NeurIPS 2024.

**Method**: Scores state dimensions using H-infinity norms of subsystems.
Layer-adaptive: different layers lose different state dimensions.
Upper layers pruned more aggressively than lower layers.

**Key results**:
- 33% state pruning → 0.52% accuracy loss, NO retraining
- 60% pruning → 1.6x inference speedup
- Layer-adaptive outperforms uniform pruning significantly

**Applicability to Redukt**: DIRECT. Use weight norms as proxy for H-infinity
(avoid dependency on control theory library). d_state 16 → 12 (25%).
Small absolute savings (~0.5% total FLOPs) but improves cache locality.

### 3.3 Simba (Song et al., TMLR 2025) — Hierarchical Sparsification

**Paper**: "Sparsified State-Space Models are Efficient Highway Networks."
arXiv:2505.20698. TMLR 2025.

**Key insight**: Tokens in SSMs are highly redundant in upper layers.
Dense recurrence in upper layers BLOCKS information flow rather than helping.
Upper layers = highways where residual stream passes through almost unchanged.

**Key results**:
- 90% token sparsity at final layer retains performance
- Up to 9x FLOPs reduction with improved accuracy
- Hierarchical: lower layers dense (local details), upper layers sparse (global)

**Applicability to Redukt**: INSIGHT only (not direct implementation).
Token pruning doesn't apply (we need logits for every token in compression).
BUT the highway insight means: layers 18-23 contribute little to output →
can be dynamically skipped when residual change is small.

### 3.4 Wanda (Sun et al., ICML 2023) — Pruning by Weights and Activations

**Paper**: "A Simple and Effective Pruning Approach for Large Language Models."
arXiv:2306.11695. ICML 2023.

**Method**: Weight importance = |W[i,j]| * ||X_col[j]||_2.
One-shot, single forward pass. No retraining, no Hessian computation.
Matches SparseGPT quality at fraction of compute cost.

**Key results**:
- 50% unstructured sparsity with <1% perplexity increase
- Competitive with SparseGPT (which needs Hessian inverse)
- Works on models from 125M to 175B parameters

**Applicability to Redukt**: ADAPTED. For CPU speedup we need STRUCTURED
pruning (column removal), not unstructured. Use column L2 norms as
deterministic proxy (no calibration data needed for deterministic
compress/decompress). Prune d_inner columns from in_proj/out_proj.

### 3.5 Activity Sparsity (Mukherji et al., NeurIPS-W 2023)

**Paper**: "Activity Sparsity Complements Weight Sparsity for
Efficient RNN Inference." arXiv:2311.07625. NeurIPS 2023 Workshop.

**Key insight**: Activations that are near-zero after nonlinearities (SiLU, ReLU)
contribute nothing to downstream computation. Skipping them is near-free.
Activity sparsity COMPOSES MULTIPLICATIVELY with weight sparsity.

**Key results**:
- GRU-based RNN: 20x computation reduction
- Maintains perplexity on Penn Treebank
- Multiplicative composition: 5x weight * 4x activity = 20x total

**Applicability to Redukt**: DIRECT. After silu_inplace(conv_out) in Mamba,
channels with |conv_out[j]| < epsilon contribute negligibly. Skip their
SSM recurrence AND out_proj column. Extension of R-057 (dt-sparse).

## 4. Forward Pass Analysis — What Each Technique Attacks

```
                                                TECHNIQUE
Embedding lookup (768)         ──── [not optimizable]
FOR each of 24 layers:
  RMSNorm (768)                ──── [not optimizable]
  in_proj (3072,768) = 2.36M   ──── [A1] Wanda column pruning
  Conv1d (1536×4)   = 6K       ──── [not optimizable]
  SiLU (1536)                  ──── [B1] Activity sparsity gate
  x_proj (80,1536)  = 122K     ──── [A1] Column pruning (follows d_inner)
  dt_proj (1536,48) = 73K      ──── [A1] Column pruning (follows d_inner)
  SSM recurrence (1536×16) 180K ──── [A2] LAST state pruning (d_state)
                               ──── [B1] Activity sparsity (per-channel skip)
  Gate (1536)                  ──── [B1] Activity sparsity
  out_proj (768,1536) = 1.18M  ──── [A1] Wanda column pruning
                               ──── [B1] Activity sparsity (column skip)
  Residual add (768)           ──── [B2] Highway check ||delta||/||x||
Final RMSNorm (768)            ──── [not optimizable]
LM head (50280,768) = 38.6M   ──── [C1] Split-Head Pipeline
```

## 5. Protocol: 3-Phase Composable Acceleration

### PHASE A: One-Shot Model Surgery (load-time, persistent)

#### A1: Deterministic Column Pruning (Wanda-inspired)

Reduce d_inner from 1536 to ~1230 per layer (20% reduction).

**Importance criterion** (deterministic from weights alone):
```
column_importance[j] = L2_norm(in_proj_w[:, j]) * L2_norm(out_proj_w[:, j])
```

Both compress and decompress compute identical norms from identical weights →
prune identical columns → deterministic. No calibration data needed.

**Implementation**:
1. At model load, compute column importance for each layer
2. Sort columns by importance, identify bottom 20%
3. Create pruned weight matrices (physically remove columns)
4. Adjust d_inner to pruned size for SSM, conv, x_proj, dt_proj
5. Cache pruned model to disk for future loads

**Savings**: d_inner 1536 → 1230:
- in_proj: 56.6M × 0.80 = 45.3M → saves 11.3M (8.8%)
- out_proj: 28.3M × 0.80 = 22.6M → saves 5.7M (4.4%)
- x_proj + dt_proj: proportional → saves ~1M
- **Total A1: -18M FLOPs = -14.0%**

**BPB risk**: 1-3% without fine-tuning (Wanda on similar-size LLMs).
Needs empirical validation on enwik8, apache, dickens.

**Determinism**: column norms computed from identical model weights → identical
pruning decisions on both sides → bit-exact within same SIMD tier. ✓

#### A2: Layer-Adaptive State Pruning (LAST-inspired)

Reduce d_state from 16 to 10-12 per layer.

**Importance criterion** (deterministic):
```
state_importance[layer][n] = sum(|a_log[j * d_state + n]| for j in 0..d_inner)
```

Higher-magnitude A entries → stronger decay → more important for prediction.
Remove lowest-importance state dimensions per layer.

**Savings**: d_state 16 → 12 average:
- SSM recurrence: 2.4M × 0.75 = 1.8M → saves 0.6M (0.5%)
- Improves SIMD utilization (12 = 8+4, still vectorizable)

**BPB risk**: ~0.5% (LAST reports 0.52% at 33% pruning).

### PHASE B: Runtime Dynamic Sparsity (per-token, adaptive)

#### B1: Activation Sparsity — SiLU Channel Skip

Extension of R-057 (dt-sparse). After SiLU activation, channels with
|conv_out[j]| < epsilon contribute negligibly to SSM and out_proj.

```
const ACT_SPARSE_EPS: f32 = 1e-4;

for j in 0..d_inner {
    let x_j = conv_out[j];  // post-SiLU
    if x_j.abs() < ACT_SPARSE_EPS {
        // Skip SSM recurrence for this channel
        // Skip out_proj column for this channel
        y_out[j] = d_vec[j] * x_j;  // near-zero
        channel_sparse[j] = true;
        continue;
    }
    // ... existing dt-sparse check (R-057) ...
    // ... normal SSM recurrence ...
}

// Sparse out_proj: skip columns where channel_sparse[j] = true
// result[i] += out_proj_w[i * d_inner + j] * gated[j]  only if !sparse[j]
```

**Savings estimate** (SiLU produces ~20-30% near-zero activations):
- SSM: 25% of 2.4M = 0.6M
- out_proj: 25% of 28.3M = 7.1M (5.5% of total)
- **Total B1: ~7.7M = 6.0%**

**BPB risk**: ~0% if epsilon is small enough. Activations below 1e-4 after
SiLU contribute < 1e-8 to output (SiLU(x) ≈ x*sigmoid(x) → 0 for x→0).
Needs empirical measurement of actual sparsity in Mamba-130M.

**Composition with weight sparsity** (Activity Sparsity paper):
Multiplicative — 20% column pruning × 25% activity sparsity = 40% effective.

#### B2: Highway Layer Skip (Simba-inspired)

For layers 18-23 (upper layers), check if the layer contribution is negligible:

```
const HIGHWAY_START: usize = 18;
const RESIDUAL_THR: f32 = 0.01;

for layer_idx in 0..n_layer {
    // ... normal layer processing ...
    // After residual add: scratch.x[j] += scratch.result[j]

    if layer_idx >= HIGHWAY_START {
        let delta_norm = l2_norm(&scratch.result[..d]);
        let x_norm = l2_norm(&scratch.x[..d]);
        if delta_norm < RESIDUAL_THR * x_norm {
            // Layer contributed < 1% — skip remaining layers
            // State for layers (layer_idx+1)..24 NOT updated
            break;
        }
    }
}
```

**Savings estimate** (2-3 layers skipped on average):
- 2.5 layers × 3.6M FLOPs/layer = 9M FLOPs (7.0%)

**BPB risk**: 1-2%. Upper layers contribute little per Simba (90% sparsity OK).
State divergence in skipped layers bounded by their "highway" nature.
Mitigation: force full forward every N tokens to resync upper-layer state.

**Determinism**: ||delta||/||x|| computed identically on both sides → same skip
decision. Force-sync counter also identical. ✓

### PHASE C: Compute Overlap (pipeline, 0% BPB)

#### C1: Split-Head Pipeline

The LM head (30.1% FLOPs) only READS scratch.x — doesn't modify state.
Pipeline it with the next token's layer processing:

```
Token t:   [--- layers 0-23 (20ms) ---][-- LM head (9ms) --]
Token t+1:                              [--- layers 0-23 (20ms) ---]
```

After warm-up: 20ms/token instead of 29ms = 1.45x theoretical.
DRAM bandwidth limits to ~1.20x on DDR4-2667 (80% saturated at baseline).

**BPB risk**: 0%. Identical computation, different temporal scheduling.

## 6. Combined Projection

```
Baseline: 128.4M FLOPs/token, 34 tok/s (Mamba-130M, SIMD)

PHASE A (one-shot, load-time):
  A1 Column pruning (20% d_inner)     : -14.0%
  A2 State pruning (25% d_state)      : -0.5%
  Subtotal Phase A                    : -14.5% → 109.8M FLOPs

PHASE B (runtime, per-token):
  B1 Activation sparsity (25% channels): -6.0%
  B2 Highway layer skip (2.5 layers)   : -7.0%
  Subtotal Phase B                     : -13.0% → 95.5M FLOPs

PHASE C (pipeline overlap):
  C1 Split-Head Pipeline               : LM head overlapped with next layers
  Effective latency                    : ~71.5M FLOPs equivalent

PROJECTED RESULTS:
  Total FLOPs reduction  : 128.4M → 95.5M = -25.6%
  Pipeline latency       : ~71.5M effective
  Throughput             : 34 → ~55-61 tok/s (x1.6-1.8)
  BPB overhead           : +2-5% (dominated by A1 column pruning)
```

### With budget constraint (BPB ≤ +3%):

If A1 causes >3% BPB, restrict to 10% pruning (d_inner 1536→1382):
```
Conservative protocol (B1 + C1 only):
  B1 Activation sparsity              : -6.0%
  C1 Split-Head Pipeline              : -15 to 25% latency
  Total                               : ~1.22-1.31x, ~0% BPB

Moderate protocol (B1 + B2 + C1):
  Add B2 Highway skip                 : -7.0% additional
  Total                               : ~1.30-1.45x, +1-2% BPB

Full protocol (A1 + B1 + B2 + C1):
  Add A1 Column pruning               : -14% additional
  Total                               : ~1.6-1.8x, +2-5% BPB
```

## 7. Implementation Priority (by ROI)

| Priority | Technique | Lines | Speedup | BPB | ROI |
|----------|-----------|-------|---------|-----|-----|
| 1 | B1: Activation sparsity | ~30 | 6% | ~0% | BEST |
| 2 | C1: Split-Head Pipeline | ~50 | 15-25% | 0% | EXCELLENT |
| 3 | A1: Wanda column pruning | ~150 | 14% | 1-3% | GOOD if BPB OK |
| 4 | B2: Highway layer skip | ~50 | 7% | 1-2% | GOOD |
| 5 | A2: LAST state pruning | ~80 | 0.5% | 0.5% | LOW |

**Recommended first step**: B1 + C1 (activation sparsity + pipeline).
~80 lines of code, ~22-31% speedup, ~0% BPB risk.
Empirically measure activation sparsity in Mamba-130M first.

## 8. What Does NOT Work

| Technique | Why Not |
|-----------|---------|
| Classic Lottery Ticket (IMP) | Requires iterative retraining — no training infra |
| Unstructured sparsity | CPU dense matmul faster than sparse (cache-unfriendly) |
| Token pruning (Simba full) | Need logits for EVERY token in compression |
| Pruning + Q4 | Q4 already +12.7% BPB; combining = catastrophic |
| State caching (R-049) | State divergence compounds through 24 layers |

## 9. Determinism Guarantee

All techniques maintain deterministic compress/decompress because:

1. **Column pruning (A1)**: Importance scores computed from model weights alone.
   Both sides have identical weights → identical pruning decisions. ✓
2. **State pruning (A2)**: Importance from A matrix norms. Same weights → same dims. ✓
3. **Activation sparsity (B1)**: Threshold is hardcoded constant.
   Same input → same activations → same sparse channels. ✓
4. **Highway skip (B2)**: Residual norm computed identically.
   Same state + same token → same norm → same skip decision. ✓
5. **Split-Head Pipeline (C1)**: Identical computation, different scheduling. ✓

Cross-tier (SIMD vs scalar): same-tier determinism maintained.
Cross-tier may diverge (pre-existing, documented in R-013).

## 10. Validation Protocol

Before committing any implementation:

1. **Measure activation sparsity**: Run Mamba-130M on enwik8 10KB, count
   channels with |conv_out[j]| < 1e-4 per layer. Need >15% to justify B1.

2. **Measure residual norms**: For layers 18-23, compute ||delta||/||x|| per token.
   Need >30% of tokens below 0.01 to justify B2.

3. **A1 BPB measurement**: Prune 10%, 20%, 30% d_inner columns. Measure BPB on
   enwik8, apache, dickens (3 representative files). Accept only if BPB < +3%.

4. **Roundtrip verification**: All techniques must pass compress→decompress→diff
   (byte-identical, CRC-32 match).

5. **Controlled A/B**: Same session, alternating pruned/unpruned, 3+ rounds.
   Report tok/s median and BPB delta.

## 11. References

- Frankle & Carbin (2018). The Lottery Ticket Hypothesis. arXiv:1803.03635
- Tuo & Wang (2025). SparseSSM: One-Shot SSM Pruning. arXiv:2506.09613
- Gwak et al. (2024). LAST: Layer-Adaptive State Pruning. arXiv:2411.02824
- Song et al. (2025). Simba: Sparsified SSMs as Highway Networks. arXiv:2505.20698
- Sun et al. (2023). Wanda: Pruning Without Retraining. arXiv:2306.11695
- Mukherji et al. (2023). Activity Sparsity for RNN Inference. arXiv:2311.07625
- S4oP (2026). Operator-level SSM Pruning. arXiv:2606.18096
- Liou et al. (2026). ReplaySSM: Cache SSM Inputs. arXiv:2610.13778
- Behrouz et al. (2026). Memory Caching: RNNs with Growing Memory. arXiv:2602.24281
- Marconi (2025). Prefix Caching for Hybrid LLMs. arXiv:2411.19379
- Compiler-First SSD (2026). O(1) Autoregressive Caching. arXiv:2603.09555
