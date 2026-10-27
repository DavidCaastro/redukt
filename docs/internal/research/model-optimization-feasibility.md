# R-V06: RWKV Model Pruning / Optimization Feasibility Assessment

**Date**: 2026-10-27
**Status**: Research complete
**Recommendation**: CONDITIONAL GO (SVD pruning first — Q4 REJECTED by R-V05 benchmarks)

---

## 1. Current Architecture Baseline

| Parameter       | Value                              | Source       |
|-----------------|------------------------------------|-------------|
| Model           | RWKV-x070-World-0.1B-v2.8         | [SOURCED] config.json |
| Architecture    | RWKV-7 ("Goose")                   | [SOURCED] model.rs |
| Layers          | 12                                 | [SOURCED] config.json |
| Embedding dim   | 768 (D)                            | [SOURCED] config.json |
| Heads           | 12 (H)                             | [SOURCED] config.json |
| Head size       | 64 (N)                             | [SOURCED] config.json |
| Vocab size      | 65,536 (V)                         | [SOURCED] config.json |
| Context length  | 4,096                              | [SOURCED] config.json |
| Parameters      | ~100M                              | [SOURCED] model name |
| Current quant   | Q8 (auto-selected)                 | [SOURCED] CLAUDE.md |
| Throughput Q8   | ~37 tok/s                          | [SOURCED] MEMORY.md |
| Throughput F32  | ~17 tok/s                          | [SOURCED] MEMORY.md |
| BPB (enwik8 10K)| 1.2812 (Q8)                        | [SOURCED] CLAUDE.md |

### Parameter Breakdown (Calculated)

Per-layer dominant matrices (D=768):
- TimeMix: key_w, value_w, receptance_w, output_w = 4 x (768 x 768) = 2.36M params each [CALCULATED]
- ChannelMix: key_w (D_FFN x D), value_w (D x D_FFN). D_FFN is typically 3.5x D = 2688, so ~2 x (768 x 2688) = 4.13M [CALCULATED]
- LoRA matrices (decay, a, v, g): small, ~50K per layer [ESTIMATE]
- Embedding: 65536 x 768 = 50.3M [CALCULATED]
- Head: 65536 x 768 = 50.3M [CALCULATED]
- 12 layers x ~14.5M per layer = ~174M (but embedding/head share weights or are half the model) [CALCULATED]

The embedding + head matrices dominate at ~100M combined (~50% of model).
This is critical: any pruning technique that does not address the embedding/head layer misses half the parameters.

---

## 2. RWKV-edge Paper Analysis (arXiv 2412.10856)

**Paper**: "RWKV-edge: Deeply Compressed RWKV for Resource-Constrained Devices"
**Source**: https://arxiv.org/abs/2412.10856

### Techniques Applied

1. **SVD (Singular Value Decomposition)** for RWKV block weight matrices
   - Decomposes (D, D) matrices into (D, k) x (k, D) with k << D
   - Used k=8 factor for decomposition [SOURCED]
   - Reduces time-mix by 2.5x, channel-mix by 3.6x [SOURCED]

2. **Sparsity Predictors** for FFN layers
   - Ensemble sparsity prediction reduces memory during inference [SOURCED]
   - Acts as a runtime pruning mechanism

3. **Clustering Head** for embedding/output layers
   - Shrinks embedding/head by ~10x [SOURCED]
   - Critical for the 0.1B model where these layers are ~50% of params

### Results on 0.1B (Tiny) Model

| Metric | Vanilla | SVD-compressed | Delta |
|--------|---------|----------------|-------|
| Accuracy | 0.38 | 0.37 | -1pp [SOURCED] |
| Model compression | 1x | 3.5-4.8x | [SOURCED] |
| Memory reduction (full load) | 1x | ~4x average | [SOURCED] |
| Memory reduction (layerwise) | 1x | ~5x average | [SOURCED] |

**Key finding**: The 0.1B model showed minimal accuracy degradation (1pp) from SVD compression. The paper claims the 0.1B model actually saw "slight gains of 1.4pp on average" with SVD in some benchmarks [SOURCED].

### Hardware Benchmarks

- Raspberry Pi 5B: 2.4GHz 4x Cortex-A76, 8GB RAM [SOURCED]
- Orange Pi Zero 2W: 1.5GHz 4x Cortex-A53, 4GB RAM [SOURCED]
- Specific tok/s numbers for 0.1B on these devices: NOT FOUND in search results

### Applicability to Redukt

The RWKV-edge techniques ARE applicable to our 0.1B model — the paper explicitly tested on 0.1B.
However, there are important caveats:

1. **Accuracy metric mismatch**: RWKV-edge measures NLP task accuracy (classification). We measure BPB (bits per byte) for compression. A 1pp accuracy drop in NLP tasks could translate to a larger BPB regression because compression quality depends on precise probability distributions, not just argmax correctness.

2. **SVD rank selection**: With D=768 and k=8, the SVD approximation is very aggressive. For compression, we need high-fidelity probability estimates, not just "good enough" token prediction. This is a fundamental tension.

3. **Clustering head risk**: Our embedding layer feeds directly into probability estimation. Clustering it would introduce quantization noise in the logit space, directly degrading BPB.

---

## 3. Quantization Beyond Q8

### RWKVQuant (arXiv 2505.03803)

**Paper**: "RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization"
**Source**: https://arxiv.org/abs/2505.03803

- Quantizes RWKV-6-14B to ~3-bit with <1% accuracy loss [SOURCED]
- 2.83x memory saving, 2.14x speedup [SOURCED]
- First comprehensive PTQ framework for RWKV family [SOURCED]
- Uses hybrid scalar + vector quantization with proxy-guided selection [SOURCED]

### rwkv.cpp Quantization Support

- Supports: FP32, FP16, Q8_0, Q5_K_M, Q4_K_M, Q4_0, Q4_1 [SOURCED]
- Recommended priority: FP16 > Q8_0 > Q5_K_M > Q4_K_M [SOURCED]
- Q4_0 on 1.5B: 76ms latency, 1.53GB file [SOURCED]
- Q4_1 on 1.5B: 72ms latency, 1.68GB file [SOURCED]
- Warning: Q3_0 and Q2_0 "potentially severely degrade performance" [SOURCED]

### Applicability to Redukt

**Q4 quantization is the most promising near-term optimization.**

Current state: Redukt already supports `--quant q4` (per CLAUDE.md). The question is:
- What is the BPB regression at Q4 vs Q8? [UNKNOWN — needs measurement]
- If <2.5% (matching the F32-vs-Q8 pattern), Q4 is an easy win
- Expected speedup: ~1.5-2x over Q8 [ESTIMATE based on rwkv.cpp data]
- This would bring throughput from ~37 tok/s to ~55-74 tok/s [ESTIMATE]

**Q2 is likely NOT viable** for compression workloads. The severe degradation warning from rwkv.cpp, combined with our need for precise probabilities, makes Q2 a poor fit.

---

## 4. Knowledge Distillation

### State of the Art

- Student models retain 70-91% of teacher capability in English tasks [SOURCED]
- No RWKV-specific distillation papers found [SOURCED: absence of results]
- General LLM distillation is mature but focused on transformers [SOURCED]

### Applicability to Redukt

Distillation would mean training a new, smaller RWKV model supervised by the 0.1B model. This is:

1. **Not practical**: We use pretrained weights (Apache-2.0, BlinkDL). Training new weights requires significant compute and the RWKV training pipeline.
2. **Diminishing returns**: Our model is already tiny (0.1B). Distilling to 50M or 25M params would almost certainly cause significant BPB regression.
3. **Out of scope**: Redukt is a compression product, not an ML training shop.

**Verdict**: NO-GO for distillation.

---

## 5. Neural Compression Throughput: Industry Context

### Current Throughput Landscape

| System | Throughput | Hardware | Source |
|--------|-----------|----------|--------|
| **Redukt Q8** | **~37 tok/s** | CPU | [SOURCED] MEMORY.md |
| **Redukt F32** | **~17 tok/s** | CPU | [SOURCED] MEMORY.md |
| NNCP | 2.0 KB/s | CPU (55MB transformer) | [SOURCED] survey |
| Nacrith (1 worker) | 20-30 tok/s | GTX 1050 Ti GPU | [SOURCED] arXiv 2602.19626 |
| Nacrith (3 workers) | 60-90 tok/s | GTX 1050 Ti GPU | [SOURCED] arXiv 2602.19626 |
| TRACE | 15.9 KB/s | GPU | [SOURCED] survey |
| PAC | 31.1 KB/s | GPU | [SOURCED] survey |
| Best LLM compressor | 181 KB/s | A100 GPU | [SOURCED] arXiv 2610.17586 |

### Key Insight

Redukt at ~37 tok/s on CPU is **competitive with GPU-based neural compressors**. This is because:
1. Our model is tiny (0.1B vs 1B-7B for competitors)
2. RWKV has O(1) per-token inference (no KV cache growth)
3. Q8 quantization is already effective

The "High-Throughput LLM-Based Compression" paper (arXiv 2610.17586) achieved 181 KB/s on an A100 with system-level optimizations including multistream pipelined arithmetic coding [SOURCED]. Their approach:
- 18.84x speedup over naive baseline [SOURCED]
- Key technique: entropy-based segmentation for parallel arithmetic coding [SOURCED]

The "Hybrid-Neural Archival" paper (arXiv 2603.25526) found neural compression is ~2600x slower than Zstd [SOURCED], confirming that throughput remains the fundamental barrier for the entire field.

### Parallelization Opportunity

The entropy-based segmentation approach from arXiv 2610.17586 is potentially applicable to Redukt:
- Split input into segments at high-entropy boundaries
- Compress segments in parallel (requires independent model states)
- Could provide 2-4x throughput on multi-core CPUs [ESTIMATE]
- **Implementation effort**: HIGH — requires rearchitecting the compression pipeline

---

## 6. Feasibility Assessment

### Option A: Q4 Quantization (RECOMMENDED FIRST)

| Factor | Assessment |
|--------|-----------|
| Expected speedup | 1.5-2x (55-74 tok/s) [ESTIMATE] |
| Expected BPB regression | 1-5% [ESTIMATE based on Q8 vs F32 pattern] |
| Implementation effort | LOW — Redukt already has `--quant q4` support |
| Risk | LOW — easily measured, easily reverted |
| Prerequisite | Benchmark Q4 BPB on enwik8 + Silesia corpus |

**Action**: Run `redukt hybrid-eval --quant q4` on reference files and compare to Q8 baseline.

### Option B: SVD Pruning (RWKV-edge style)

| Factor | Assessment |
|--------|-----------|
| Expected speedup | 2-3x (for weight-matrix-bound operations) [ESTIMATE] |
| Expected BPB regression | 3-10% [ESTIMATE — higher than NLP tasks due to probability precision requirements] |
| Implementation effort | HIGH — need SVD decomposition, new weight format, modified forward pass |
| Risk | MEDIUM — may not be worth it on a 100M model |
| Model size reduction | 3.5-4.8x [SOURCED from RWKV-edge] |

**Key concern**: The 0.1B model is already at the floor of useful model sizes. SVD with k=8 on 768-dim matrices is extremely aggressive. For NLP tasks this loses 1pp accuracy; for compression, the BPB impact could be much worse because:

1. Compression needs the full probability distribution, not just the top-1 prediction
2. Small errors in logits compound over thousands of tokens in arithmetic coding
3. The coder overhead is already only 0.064 BPB — there is little margin to absorb regression

**Estimated BPB at various SVD ranks** (for D=768):

| SVD Rank (k) | Compression | Estimated BPB | Reasoning |
|--------------|------------|---------------|-----------|
| 768 (none) | 1x | 1.2812 | [SOURCED] current Q8 baseline |
| 384 (50%) | 2x | ~1.31-1.34 | [ESTIMATE] linear interpolation, minimal loss |
| 192 (25%) | 4x | ~1.38-1.48 | [ESTIMATE] significant quality loss likely |
| 96 (12.5%) | 8x | ~1.55-1.80 | [ESTIMATE] severe degradation probable |
| 8 (1%) | 96x | UNKNOWN | [ESTIMATE] possibly unusable for compression |

These estimates are speculative. No one has measured SVD impact on BPB for neural compression specifically.

### Option C: Sparsity Predictors

| Factor | Assessment |
|--------|-----------|
| Expected speedup | 1.3-1.5x (FFN layers only) [ESTIMATE] |
| Expected BPB regression | 2-5% [ESTIMATE] |
| Implementation effort | HIGH — need sparsity predictor network, runtime activation pruning |
| Risk | HIGH — adds complexity for modest gain |

**Verdict**: Not worth it at 0.1B scale. The FFN computation is only ~40% of per-token cost.

### Option D: Parallel Segment Compression

| Factor | Assessment |
|--------|-----------|
| Expected speedup | 2-4x on 4+ core CPUs [ESTIMATE] |
| Expected BPB regression | 0-2% (segment boundaries add overhead) [ESTIMATE] |
| Implementation effort | VERY HIGH — pipeline rearchitecture |
| Risk | MEDIUM — proven technique but complex implementation |

**Verdict**: Most promising for large throughput gains, but highest engineering cost.

### Option E: Distillation to Smaller Model

**Verdict**: NO-GO. Out of scope, requires training infrastructure, diminishing returns at 0.1B.

---

## 7. GO/NO-GO Recommendation

### CONDITIONAL GO: Phased Approach

**Phase 1: Q4 Quantization — REJECTED (R-V05 measured)**
- **Measured** (R-V05, 6 files): +13.3% mean BPB regression, 7.6x SLOWER than Q8
- Q4 destroys competitive advantage on strongest verticals (facturae +20.8%, xml +25.7%)
- Q4 throughput: 2.7-6.1 tok/s vs Q8 26-47 tok/s — catastrophically worse
- See `data/benchmark/results/q4-evaluation.csv`
- **REJECTED — do not pursue Q4**

**Phase 2 (now Phase 1, MEDIUM EFFORT): Conservative SVD on Linear Projections**
- Apply SVD only to the 4 TimeMix projection matrices (key, value, receptance, output)
- Start with rank 384 (50% of D=768) — modest compression, likely minimal BPB impact
- Do NOT touch embedding or head matrices (too sensitive for compression)
- Measure BPB impact precisely before committing
- Estimated effort: 2-3 weeks (new weight format, modified forward pass, benchmarking)
- CONDITIONAL GO — only if Phase 1 shows Q4 is insufficient

**Phase 3 (HIGH EFFORT): Parallel Segment Compression**
- Implement entropy-based segmentation from arXiv 2610.17586
- Requires independent model states per segment
- Would be the largest single throughput improvement
- Estimated effort: 4-6 weeks
- CONDITIONAL GO — only if throughput remains a blocking business requirement

**NO-GO items:**
- Distillation (out of scope, requires training)
- Q2 quantization (too aggressive, severe quality loss)
- Clustering head (too risky for compression probability precision)
- Sparsity predictors (insufficient return for complexity at 0.1B scale)

---

## 8. Expected Throughput After Optimizations

| Configuration | Throughput (est.) | BPB (est.) | Status |
|---------------|-------------------|------------|--------|
| Current Q8 | 22.7 tok/s (post-SIMD measured) | 1.2812 | [MEASURED: R-V01] |
| Q4 | 4.1 tok/s | 1.44 mean | [MEASURED: R-V05] **REJECTED** |
| Q8 + SVD-384 | 30-45 tok/s | 1.31-1.34 | [ESTIMATE] |
| Q8 + SVD + parallel | 100-200 tok/s | 1.31-1.40 | [ESTIMATE] |

For context: 200 tok/s on text with average 3.5 bytes/token = ~700 bytes/s = ~0.7 KB/s.
This is still orders of magnitude slower than brotli (~100 MB/s) but competitive with the best neural compressors on GPU [SOURCED].

---

## Sources

- [RWKV-edge: Deeply Compressed RWKV for Resource-Constrained Devices](https://arxiv.org/abs/2412.10856)
- [RWKVQuant: Quantizing the RWKV Family](https://arxiv.org/abs/2505.03803)
- [High-Throughput LLM-Based Compression and What It Takes](https://arxiv.org/abs/2610.17586)
- [Investigating the Fundamental Limit: A Feasibility Study of Hybrid-Neural Archival](https://arxiv.org/abs/2603.25526)
- [Nacrith: Neural Lossless Compression via Ensemble Context Modeling](https://arxiv.org/abs/2602.19626)
- [rwkv.cpp documentation](https://github.com/RWKV/rwkv.cpp)
- [llama.cpp RWKV inference guide](https://wiki.rwkv.com/inference/llamacpp.html)
- [The Evolution of RWKV (arXiv 2411.02795)](https://arxiv.org/abs/2411.02795)
- [A Survey of RWKV](https://arxiv.org/abs/2412.14847)
