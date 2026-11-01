# CPU Parallelism Strategies — Breaking the Sequential Bottleneck

> Research: 2026-11-01
> Context: Redukt at 24 tok/s (Mamba) / 95 tok/s (RWKV) — 46,000x slower than brotli-11

## Problem Statement

CPU utilization during compression is ~8-16% (1 of 12 threads active).
The 1 active thread spends ~40% stalled on DRAM reads.
The sequential token dependency prevents naive multi-core parallelism.

## Mathematical Properties to Exploit

### 1. Linear Diagonal Recurrence (Mamba SSM)

```
s_d[t] = a_d[t] · s_d[t-1] + b_d[t]    (scalar, per dimension d)
```

24,576 independent scalar recurrences per layer (1536 × 16).
Each parallelizable across cores; only intra-chain dependency is sequential.

### 2. Compression Knows All Tokens (Asymmetry vs Generation)

In compression, tokens = [t₁, ..., tₙ] are known upfront.
Enables layer-by-layer processing: all tokens through layer 1, then all through layer 2.
Reduces sequential depth from L×N to L with N-parallel work per layer.

### 3. dt Selectivity Sparsity

`dt[t]` controls state update magnitude. When dt ≈ 0, state unchanged:
```
s[t] = exp(-softplus(dt)·A) · s[t-1] + dt · B · x
dt → 0 ⟹ s[t] ≈ s[t-1]
```
~30-50% of tokens in natural text have negligible dt.

### 4. CM-Backbone Factorization

CM predicts 60-80% of tokens correctly in structured data.
State divergence after K skips bounded by:
```
|s_true - s_approx| ≤ Σ |a|^i · |b| ≤ |b| / (1 - |a|)
```
For |a| ≈ 0.95: K=2 → error ≤ 1.95|b|, K=8 → 6.90|b| (R-049 failure point).

### 5. Low Effective State Dimensionality

24,576 state dimensions, but visited subspace is ~100-500 effective dimensions
for structured data. LSH over states could enable approximate memoization.

## Strategies

### R-054: Multi-threaded matmul (0% BPB risk)

Split large matmuls (rows > 4096) across available cores using `std::thread::scope`.
Target: LM head (50,280×768 Mamba, 65,536×768 RWKV).
Within-layer matmuls (768×1536 Mamba in_proj) also eligible.
Expected: 1.5-2x overall throughput.
Implementation: zero-dep, std::thread only.

### R-055: Layer-by-layer parallel scan (0% BPB risk, compression-only)

Process all N tokens through each layer sequentially across layers:
```
for layer in 0..24:
    pre_compute_all_tokens(layer)     // parallel across tokens
    parallel_scan_ssm(all_tokens)     // O(log N) depth
    post_compute_all_tokens(layer)    // parallel across tokens
```
Requires buffering all intermediate activations: N × d_model × 4 bytes.
For 3000 tokens: ~9.2 MB per buffer — fits easily in RAM.
Expected: 3-5x on 4 cores (compression only).

### R-056: CM-selective backbone, K≤3 (1-3% BPB risk)

Use CmEnsemble confidence to skip backbone forward pass:
```
if cm_confidence > θ AND consecutive_skips < 3:
    use CM-only logits (no backbone forward)
else:
    full backbone forward
```
Bounded divergence: K≤3 with |a|≈0.95 → error ≤ 2.85|b|.
Expected: 2-4x on repetitive data (logs, XML). ~1x on random text.

### R-057: dt-sparse state skip (~0% BPB risk)

Skip SSM state update when |dt[t]| < ε for a channel:
```
if dt[j] < ε:  skip s[j][t] update (reuse s[j][t-1])
```
Per-channel decision, no global skip. Exact when ε → 0.
Expected: 1.05-1.15x (SSM is 25% of layer time, 30-50% sparse).

### R-058: LSH state memoization (research, 1-5% BPB risk)

Locality-sensitive hashing on model states.
Similar states → similar logits. Cache and reuse.
Higher hit rate than exact-match PredCache (R-049).
Status: research only, needs empirical validation.

## Combined CPU Projection

| Component | Speedup | Cumulative | BPB Risk |
|---|---|---|---|
| Baseline | 1.0x (24 tok/s) | 24 tok/s | 0% |
| R-054 (parallel matmul) | 1.5-2x | 36-48 tok/s | 0% |
| R-055 (layer scan, compress) | 2.5-3x | 90-144 tok/s | 0% |
| R-056 (CM-selective, logs) | 2-3x | 180-430 tok/s | 1-3% |
| **Total (logs/XML)** | **~10-18x** | **~240-430 tok/s** | **1-3%** |
| **Total (text)** | **~4-6x** | **~96-144 tok/s** | **0%** |

From ~43 B/s to ~420-1500 B/s (structured data) or ~170-500 B/s (text).
Gap vs brotli: from 46,000x to ~1,300-4,700x (CPU-only).

## Implementation Priority

1. R-054 (parallel matmul) — universal, 0% risk, immediate gain
2. R-057 (dt-sparse) — simple, ~0% risk, small gain
3. R-056 (CM-selective) — moderate complexity, biggest gain on target data
4. R-055 (layer scan) — complex restructuring, compression-only
5. R-058 (LSH memo) — research, future
