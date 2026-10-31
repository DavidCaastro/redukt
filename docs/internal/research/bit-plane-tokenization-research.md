# Bit-Plane Tokenization & Information-Theoretic Enhancements — Research

> Research for Fase 6. Source: PDF "Matrices y Alternativas Vectoriales".
> 3 parallel research agents. Date: 2026-10-31.

## 1. Concepts from PDF

### 1.1 Bit-Plane Projection (BPP)
Replace `nn.Embedding(V, d)` with `W_proj ∈ R^{d×8} · bits(byte) + bias`.
Decomposes each byte into 8 bits, projects directly to latent space.
Parameter reduction: 96.5% of embedding layer (from 256*d to 8*d + d).

### 1.2 Entropy-Adaptive Segmentation
Segment boundaries where surprisal H(x_t | x_{<t}) exceeds threshold.
Variable-length "information-coherent chunks" pooled into latent vectors.
Formula: τ_k = arg min_t (H(x_t | x_{<t}) - λ·len(x_t))

### 1.3 Domain-Optimal Symbol Width
k-bit symbols instead of fixed 8-bit bytes:
- DNA: k=2 (4 states), Audio: k=16/24, Float: k=32, Medical: k=10/12

### 1.4 Operator Folding Ω(k)
Fuse bit extraction + embedding into single matrix: Ω(k) = (I_S ⊗ E) · S_k
SVD factorization for efficient SIMD execution.

### 1.5 Walsh-Hadamard Spectral Correction
Phase alignment via WHT for non-byte-aligned symbols.
z_s = c_δs ⊙ (H_W · x_reg)

## 2. Published Work Found

### Bit-Plane / Byte Decomposition Embeddings
- **Kronecker Embeddings** (2605.29459): Byte decomposition + Kronecker products + learned projection.
  -2.5% val loss vs BPE-tied, 91-94% embedding param reduction. Requires full training. [SOURCED]
- **EmByte** (EMNLP 2025 Findings): DeComp — decompose subwords to bytes, aggregate, project.
  94% embedding param reduction. Requires training. [SOURCED]
- **Neural Projections** (Ravi, ICML 2019): LSH-based binary projections replacing embeddings.
  Fixed low-memory footprint. Learned projection. [SOURCED]

### Entropy-Adaptive Segmentation
- **BLT** (2412.09871, ACL 2025): Byte Latent Transformer — entropy-based patching.
  Matches Llama 3 at 8B scale. Full training required. [SOURCED]
- **Fast BLT** (2605.08044): >50% lower memory-bandwidth. [SOURCED]
- **ByteFlow** (2603.03583, ICLR 2026): Learned coding-rate segmentation.
  Outperforms BPE+Transformers. Full training required. [SOURCED]
- **Proxy Compression** (2602.04289, ICML 2026): Entropy-based segmentation for
  parallel compression. 95th-percentile entropy threshold. [SOURCED]

### CDF Precision (Critical Finding)
- **Nacrith** (2602.19626, DCC 2026): 135M transformer + ensemble CM.
  enwik8 0.9389 BPB (100MB). CDF-24 upgrade alone: **-0.52 BPB (-28%)**. [SOURCED]
  Key insight: CDF precision is the dominant bottleneck in arithmetic coding.
  Redukt uses TOP=2^24 — but min_freq floors and rounding strategy may differ.

### Variable Symbol Width
- **No published system** uses domain-adaptive variable-width tokenization [SOURCED: survey of cmix, PAQ8, NNCP, Nacrith, zpaq, DZip]
- cmix/PAQ8: bit-level (k=1 always) — universal but extremely slow
- NNCP: byte or 16-bit, selected by preprocessing, not by domain content

### Operator Folding
- **DNNFusion** (2108.13342): Up to 9.3x speedup from operator fusion. [SOURCED]
- **WHT Projections** (2603.08343): Walsh-Hadamard as parameter-free dense layer replacement. [SOURCED]
- **FJLT** (1712.01774): Fast Johnson-Lindenstrauss via Hadamard + random diagonal. [SOURCED]

## 3. Feasibility Assessment for Redukt

### Feasible WITHOUT Retraining (Fase 6 Tasks)

| Technique | Mechanism | Expected BPB Impact | Effort |
|-----------|-----------|---------------------|--------|
| Entropy-adaptive mixing | Modulate CM weights by backbone CE | 0.2-1.0% [ESTIMATE] | Low |
| SegmentModel predictor | Per-segment byte histogram as CM | 0.3-1.5% [ESTIMATE] | Medium |
| Bit-plane CM hash | MSB/nibble in ngram hash context | 0-0.5% [ESTIMATE] | Low |
| CDF precision audit | Verify min_freq, rounding vs Nacrith | 0-5% coder_eff [ESTIMATE] | Low |

### NOT Feasible Without Retraining

| Technique | Blocker | Alternative |
|-----------|---------|-------------|
| Bit-plane embedding | Destroys pretrained weights (0.05% of model) | Future: R-047 after R-041 |
| BLT entropy patching | Full architectural change | SegmentModel approximation |
| Variable k-bit symbols | RWKV pretrained on fixed 65K vocab | Content-aware preprocessing (already exists) |
| Operator folding Ω(k) | Embedding <1% inference time | No benefit for current bottleneck |
| WHT spectral correction | Requires non-standard input layer | Future research |

## 4. Integration Points (from Codebase Analysis)

### Embedding Lookup
- RWKV-7: `src/infrastructure/rwkv7/model.rs:656` — `scratch.x.copy_from_slice(&emb[start..start+d])`
- Mamba: `src/infrastructure/mamba/model.rs:232` — same pattern
- Shape: RWKV (65536, 768), Mamba (50280, 768)

### Mixer
- `src/domain/mixer.rs:41-55` — `combine_with_match()` — 5 components
- `src/domain/mixer.rs:58-81` — `update_with_match()` — SGD update
- Adding 6th component: add `w_segment: f32`, extend combine/update signatures

### CM Predictors
- Established pattern: struct + `compute_bias_into(&self, buf)` + `observe(&mut self, token)`
- Files: ngram.rs, bias_head.rs, match_model.rs, run_model.rs, word_model.rs
- New predictor follows same pattern (SegmentModel, BitPlaneModel)

### Arithmetic Coder
- `src/domain/coder.rs` — RangeEncoder/RangeDecoder
- TOP = 2^24 = 16,777,216 (24-bit CDF precision)
- CDF construction in codec.rs — `cdf_from_logits()`

### Hot Loop
- `src/application/codec.rs:600-850` — per-token compression loop
- Integration: add CM calls at lines 709-711 (instantiate), 745-751 (compute_bias), 753-768 (mixer)

## 5. References

| Paper | ArXiv/Venue | Key Finding |
|-------|-------------|-------------|
| BLT | 2412.09871 (ACL 2025) | Entropy patching matches Llama 3 at 8B |
| Fast BLT | 2605.08044 | >50% lower bandwidth |
| ByteFlow | 2603.03583 (ICLR 2026) | Learned coding-rate segmentation |
| Kronecker Embeddings | 2605.29459 | Byte decomposition, -2.5% loss, 91-94% param reduction |
| EmByte | EMNLP 2025 Findings | Byte DeComp, 94% param reduction |
| Proxy Compression | 2602.04289 (ICML 2026) | Entropy segmentation for parallel compression |
| Nacrith | 2602.19626 (DCC 2026) | CDF-24 = -28% BPB; 0.9389 BPB enwik8 100MB |
| DNNFusion | 2108.13342 | 9.3x speedup from operator fusion |
| WHT Projections | 2603.08343 | Parameter-free WHT replacing dense layers |
| FJLT | 1712.01774 | O(d log d) JL embedding via Hadamard |
| Neural Projections | ICML 2019 | Binary projection replacing embeddings |
| Entropy Pre-Tokenization | 2506.15889 | PMI/entropy for BPE boundary selection |
