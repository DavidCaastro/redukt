# Token Reduction Strategies for Neural Compression

> Investigation: 23 approaches across 4 tracks. 21 NO-GO, 2 CONDITIONAL GO.
> Date: 2026-11-07
> Context: Mamba-130M at 25-34 tok/s (DRAM-bound), BPE 50K vocab

---

## 1. Problem Statement

Sequential token-by-token processing is the fundamental bottleneck in Redukt's
neural compression pipeline. Each token requires reading ~493MB of model weights
from DRAM. The BPE tokenizer maps 2.84 bytes/token on text data and 1.0-1.07
bytes/token on binary data. Reducing the number of tokens the backbone must
process directly reduces wall-clock time.

**Current throughput ceiling**: 25-34 tok/s on DDR4 (Fase 11-12 confirmed
DRAM-bound). Forward pass is >99% memory-bandwidth limited. No CPU optimization
can break this ceiling without reducing token count or model size.

**Prior failed attempts to reduce backbone invocations**:

| Attempt | Strategy | Result | Reference |
|---------|----------|--------|-----------|
| R-032 | CM-only bypass (skip backbone entirely) | +109.7% BPB enwik8 | Fase 3, REVERTIDO |
| R-049 | PredCache (state fingerprint lookup) | +17.4% BPB | Fase 7, REVERTIDO |
| R-056 | CM-selective skip (confidence threshold) | +81-294% BPB | Fase 9, NO VIABLE |
| R-062 | Highway layer skip (residual threshold) | +713% BPB at 5.5% skip | Fase 10, NO VIABLE |

Each failure reinforces the same lesson: the backbone produces irreplaceable
probability distributions that the CM ensemble cannot approximate at BPE
granularity. This document exhaustively surveys 23 approaches to determine
whether any viable path exists.

---

## 2. Track A: Multi-Token Prediction & Speculative Decoding

Track A examines methods that attempt to predict or verify multiple tokens per
backbone invocation. All originate from autoregressive LLM acceleration research.

### 2.1 Multi-Token Prediction Heads (Gloeckle et al. 2024)

**NO-GO**

Multi-token prediction trains additional output heads to predict P(t+k | h_t)
for k = 1, 2, ..., K from the current hidden state. Meta showed this improves
downstream task quality and enables 3x faster inference via self-speculative
decoding (arXiv:2404.19737).

**Why it fails for compression**: Arithmetic coding requires the exact
conditional CDF P(t+k | t_0, t_1, ..., t+k-1) for each position. Multi-token
heads produce P(t+k | h_t), which conditions on the hidden state at position t,
not on the actual realized tokens t+1 through t+k-1. The resulting probability
mismatch is not bounded and causes catastrophic decompression failure. This is
not a quality tradeoff -- it is a correctness violation.

Additionally, the heads require fine-tuning with a multi-token loss, which is
unavailable for pretrained Mamba-130M without retraining.

### 2.2 Speculative Decoding (Leviathan et al. 2023, Chen et al. 2023)

**NO-GO for Mamba**

Speculative decoding uses a small draft model to propose K tokens, then verifies
all K in parallel with the target model (arXiv:2211.17192, arXiv:2302.01318).
Verification requires computing the target model's logits for all K positions
simultaneously.

Two independent blockers:

1. **SSM recurrence prevents parallel verification.** Mamba's state-space
   recurrence s_{t+1} = A * s_t + B * x_t is inherently sequential
   (P-complete). Unlike Transformers, which can verify K tokens in a single
   forward pass via masked attention, Mamba must process each token one at a
   time. The "verification" step costs the same as normal sequential processing,
   eliminating the speedup entirely.

2. **Compression already knows the true tokens.** In generation, speculative
   decoding drafts tokens to avoid slow autoregressive sampling. In compression,
   the encoder has the full input sequence available -- there is nothing to
   "draft." The bottleneck is computing P(x_t | x_{<t}), not deciding which
   token to emit.

### 2.3 Blockwise Parallel Decoding / Jacobi Iteration (Stern et al. 2018)

**NO-GO**

Blockwise parallel decoding initializes a block of K positions with guesses and
iteratively refines them via parallel model evaluation until convergence
(arXiv:1811.03115). Jacobi decoding (Santilli et al. 2023, arXiv:2305.10427)
formalizes this as a fixed-point iteration on the autoregressive map.

**Why it fails**:

1. **Parallel evaluation impossible with Mamba.** Same as 2.2 -- SSM recurrence
   is sequential. Each iteration still requires K sequential forward passes.

2. **Produces argmax, not CDFs.** Jacobi iteration converges to the greedy
   decoding sequence. Compression requires full probability distributions at
   each position, not just the most likely token. Even if parallel evaluation
   were possible, the method outputs the wrong quantity.

3. **Convergence not guaranteed in bounded iterations.** Santilli et al. report
   1.3-2.4 tokens per iteration on Transformers, but convergence depends on
   model confidence. On compression workloads with diverse content, convergence
   rates would be unpredictable.

### 2.4 Medusa Heads (Cai et al. 2024)

**NO-GO**

Medusa adds multiple lightweight heads to predict tokens at positions t+1, t+2,
..., t+K, then uses tree-structured attention to verify candidates in parallel
(arXiv:2401.10774). Medusa-2 extends this with a tree-based draft structure for
higher acceptance rates.

**Why it fails**: Combines the blockers from 2.1 and 2.2.

1. Additional heads produce P(t+k | h_t), not the sequential conditionals
   required by arithmetic coding (same as 2.1).
2. Tree-structured parallel verification requires masked attention, which Mamba
   does not support. Verification would be fully sequential (same as 2.2).
3. Heads require fine-tuning on the target model, which we cannot do without
   retraining.

### 2.5 Lookahead Decoding (Fu et al. 2024)

**NO-GO**

Lookahead decoding maintains a window of Jacobi iterations and an n-gram pool
collected from prior iterations to extend verified sequences (arXiv:2402.02057).
It combines Jacobi iteration (2.3) with n-gram matching.

**Why it fails**:

1. **Jacobi iteration is sequential on Mamba** (same as 2.3).
2. **N-gram pool is redundant.** Redukt already has a 5-model CM ensemble
   (TokenNgram, BiasHead, MatchModel, RunModel, WordModel) that captures
   n-gram patterns. The CM ensemble already produces its best probability
   estimate before the backbone runs.
3. Even on Transformers, Fu et al. report only 1.5-2.3x speedup, which would
   not offset the overhead of maintaining the Jacobi window on Mamba.

### Track A Conclusion

All 5 approaches: **NO-GO**.

Root causes are fundamental and cannot be engineered around:

1. **Compression needs exact sequential CDFs, not argmax or approximate
   distributions.** Methods that produce P(t+k | h_t) instead of
   P(t+k | t_0..t+k-1) violate the arithmetic coding contract.

2. **Mamba's SSM recurrence is irreducibly sequential (P-complete).** Every
   method in this track assumes parallel verification, which Transformers
   support via masked attention but SSMs fundamentally cannot.

---

## 3. Track B: Tokenization Strategies

Track B examines whether changing how bytes map to tokens can reduce the number
of backbone invocations. The pretrained model is inextricably coupled to its
tokenizer -- any change that produces out-of-distribution token sequences
degrades predictions catastrophically.

### 3.1 Larger Vocabulary BPE (100K-500K)

**NO-GO**

Increasing vocabulary from 50K to 200K would reduce token count by ~30-40% on
text (longer merges). However:

1. **LM head explosion.** The language model head (50280 x 768) is already 30%
   of total FLOPs. At 200K vocabulary, the head grows to 200K x 768 = 153.6M
   parameters, and total model size increases from 493MB to ~953MB. Each token
   now requires reading nearly twice as much DRAM, eliminating the token
   reduction benefit.

2. **Requires full retraining.** The embedding and LM head matrices cannot be
   resized without retraining the entire model. Mamba-130M was pretrained with
   GPT-NeoX 50280 vocabulary; there is no way to retrofit a larger vocabulary.

3. **Diminishing returns.** Kudo & Richardson (2018, arXiv:1808.06226) and
   Zouhar et al. (2023, arXiv:2306.02254) show that compression efficiency of
   BPE saturates around 50-100K vocabulary. Gains from 50K to 200K are marginal
   on diverse corpora.

### 3.2 Byte-Level Patching (MegaByte / BLT)

**NO-GO**

MegaByte (Yu et al. 2023, arXiv:2305.07185) and Byte Latent Transformer (BLT,
Pagnoni et al. 2024, arXiv:2412.09871) use a two-tier architecture: a small
local model processes raw bytes within patches, and a large global model
processes patch representations.

**Why it fails at 130M scale**:

A reasonable split allocates ~50M to the local model and ~80M to the global
model. The local model processes every byte (1 byte/token), producing patch
embeddings every P bytes. The global model processes N/P tokens.

Cost analysis for P=4:
- Local model: N bytes x 50M params = 50M reads/byte
- Global model: N/4 tokens x 80M params = 20M reads/byte
- Total: ~70M reads/byte

Current Mamba:
- N/2.84 tokens x 493M params = 174M reads/byte

The two-model approach is cheaper per byte at this split, but:

1. **Local model quality is catastrophic at 50M.** Byte-level prediction with
   50M parameters produces near-uniform distributions on non-trivial data,
   adding enormous entropy to the arithmetic coder. Yu et al. validate MegaByte
   at 1.3B+, not 50-130M.

2. **Architecture requires custom training.** No pretrained MegaByte or BLT
   model exists at any scale for Mamba architecture. The two-tier design
   requires coordinated training of both models.

3. **Patch boundaries introduce discontinuities.** At patch boundaries, the
   local model resets, losing byte-level context. This is partially compensated
   by the global model, but at 80M parameters the global model cannot capture
   sufficient long-range dependencies.

### 3.3 Entropy-Adaptive Tokenization

**NO-GO without retraining**

The idea: segment the input based on entropy estimates, using shorter tokens in
high-entropy regions and longer tokens in low-entropy regions. This adapts the
tokenization granularity to content complexity.

**Why it fails**:

1. **Out-of-distribution sequences.** The pretrained model expects the specific
   token sequences produced by GPT-NeoX BPE. Variable-rate tokenization
   produces token sequences the model has never seen during training, causing
   severe prediction degradation.

2. **The skip variant is R-056.** Entropy-adaptive tokenization that skips the
   backbone on "easy" tokens is exactly the CM-selective skip strategy that
   failed at +81-294% BPB in Fase 9. The CM ensemble cannot substitute for the
   backbone even on high-confidence tokens.

3. **No standardized variable-rate BPE exists.** Implementing adaptive
   tokenization requires a custom tokenizer coupled with model retraining.

### 3.4 Hierarchical / Multi-Scale Tokenization

**NO-GO**

Hierarchical approaches (Nawrot et al. 2023, arXiv:2305.13551; Hourglass
Transformer, arXiv:2110.13711) process tokens at different time scales,
compressing internal representations at lower layers and expanding at higher
layers.

**Why it fails**:

1. **130M parameter budget is too small to split across scales.** Published
   results show benefits only at 1B+ parameters, where there are enough
   parameters per scale to maintain prediction quality.

2. **Requires architectural changes and retraining.** Multi-scale processing
   requires modified layer designs (e.g., strided attention, pooling layers)
   that are not present in pretrained Mamba-130M.

3. **SSM recurrence complicates scale transitions.** Unlike Transformers where
   attention can operate at different sequence lengths, Mamba's recurrence must
   maintain state at each scale independently, requiring separate state vectors
   per scale.

### 3.5 Token Merging (ToMe-Style)

**NO-GO**

Token Merging (Bolya et al. 2023, arXiv:2210.09461) reduces token count in
Vision Transformers by progressively merging similar tokens based on key
similarity. Adaptations for language models merge tokens with similar hidden
representations.

**Why it fails for compression**:

This is fundamentally incompatible with lossless compression. Merging tokens
eliminates the individual probability distributions needed for exact bitstream
reconstruction. If tokens A and B are merged into token AB, the decompressor has
no way to recover the separate CDFs P(A|context) and P(B|context) needed to
decode the arithmetic-coded stream.

Even "unmerging" strategies (reconstructing separate predictions from merged
representations) introduce approximation errors that destroy lossless
reconstruction.

### 3.6 Compression-Aware Tokenization

**NO-GO for existing weights**

Compression-aware tokenization optimizes the vocabulary specifically for
downstream compression performance rather than language modeling perplexity. Rust
et al. (ICLR 2024) and Limisiewicz et al. (2023, arXiv:2306.06098) investigate
the relationship between tokenization and task performance.

Key finding from Rust et al.: "tokenization does not improve compression in
general" -- the benefits are task- and domain-specific, and compression-optimal
vocabularies differ from perplexity-optimal vocabularies.

**Why it fails now**: Requires retraining the model from scratch with the new
tokenizer. The current weights are permanently coupled to GPT-NeoX BPE.

**Future consideration**: If MambaByte training (R-041) proceeds, the byte-level
tokenizer inherently avoids all tokenization coupling issues.

### 3.7 MambaByte (Byte-Level Model)

**CONDITIONAL GO** (long-term, requires training: $300-750)

MambaByte (Wang et al. 2024, arXiv:2401.13660) operates directly on raw bytes,
eliminating the BPE tokenizer entirely. This approach addresses multiple
bottlenecks simultaneously.

**Quantitative case**:

| Metric | Current (BPE 50K) | MambaByte (vocab 256) | Delta |
|--------|--------------------|-----------------------|-------|
| LM head size | 50280 x 768 = 38.6M params | 256 x 768 = 197K params | **-196x** |
| LM head FLOPs share | 30% of forward | 0.15% of forward | **-200x** |
| Tokens per byte (text) | 0.352 (2.84 bytes/tok) | 1.0 | +2.84x |
| Tokens per byte (binary) | 0.93-1.0 | 1.0 | +0-7% |
| Total cost per byte (text) | 174M reads | ~120M reads (est.) | **-31%** |

The 196x reduction in LM head computation more than compensates for the 2.84x
increase in token count for text data. For binary data, token count is nearly
unchanged while the LM head savings still apply.

**Evidence**:
- MambaByte-353M achieves 0.93 BPB on PG19 (Wang et al. 2024)
- Byte-level models show stronger length generalization
- CM ensemble confidence with 256 classes >> with 50K classes (R-056 root cause)
- Byteification adapter (Nature 2026): retrofits subword models to byte-level
  at <1% of pretraining cost, but untested on SSM architectures

**Training requirements** (from R-041 analysis):
- 200-500 A100-hours ($300-750 at spot pricing)
- 10-30GB mixed corpus (text + binary)
- Expected: 1.20-1.40 BPB enwik8, 200-500 tok/s on GPU
- PyTorch training (external); inference remains zero-dep Rust

**Risk factors**:
- Byte-level models may underperform BPE on highly compressible text
- Training cost is non-trivial for a single-person project
- No pretrained small byte-level Mamba exists; must train from scratch
- Byteification adapter is unvalidated on Mamba architecture

### Track B Conclusion

6 NO-GO, 1 CONDITIONAL GO (MambaByte).

The pretrained model is inextricably coupled to its tokenizer. Any approach that
changes the token vocabulary, granularity, or sequencing produces
out-of-distribution inputs that destroy prediction quality. The only viable path
is training a new model with the desired tokenization (MambaByte), which is a
long-term investment.

---

## 4. Track C: Pattern Recognition & State Caching

Track C examines whether backbone invocations can be avoided by recognizing that
the model would produce similar outputs to a previously computed result. All
approaches founder on the same issue: SSM hidden states are continuous,
high-dimensional vectors that never repeat exactly.

### 4.1 State Fingerprinting / PredCache Revival

**NO-GO**

PredCache (R-048/R-049, Fase 7) stored top-32 logits keyed by 8-token FNV-1a
hash. On cache hit, the stored logits were used instead of running the backbone.

**Measured failure**: +17.4% BPB regression (R-049). Root cause analysis:

1. **Hash collisions are not state equivalence.** Two identical 8-token windows
   can follow completely different histories, producing different hidden states
   and therefore different probability distributions. The 8-token context hash
   captures local patterns but not the full recurrent state.

2. **SSM states are continuous.** Mamba's hidden state is a 768-dimensional
   float vector per layer (24 layers). Even with identical input tokens, slight
   differences in prior state propagate through the recurrence, producing
   measurably different output distributions.

3. **Top-32 truncation loses tail probability.** Even when the cache hit was
   approximately correct, truncating to 32 logits discards the long tail of the
   distribution, which the arithmetic coder needs for exact encoding of rare
   tokens.

No parameter tuning (window size, table size, top-K) can address the fundamental
state mismatch. The +17.4% BPB at 16% hit rate implies each cache hit introduces
~1 bit of excess per token -- massive for compression.

### 4.2 Repetition Detection (Input Pattern Matching)

**NO-GO**

The idea: detect repeated input segments (e.g., boilerplate XML tags, repeated
log lines) and reuse backbone outputs from the first occurrence.

**Why it fails**: Same state divergence as PredCache. Consider a repeated
4-token pattern appearing at positions 100 and 500. At position 100, the hidden
state encodes the first 100 tokens of context. At position 500, it encodes 500
tokens. These states are fundamentally different, and the model produces
different probability distributions even for identical input tokens.

This is a well-known property of recurrent models: hidden state is a function of
the entire history, not just the local input. Unlike Transformers with finite
attention windows, SSMs have theoretically infinite memory (bounded by numerical
precision and eigenvalue decay).

Empirical confirmation: R-049 showed that even with high local similarity, the
BPB regression was severe and uniform across content types.

### 4.3 KV-Cache Equivalent for SSMs

**NO-GO (not applicable)**

In Transformers, KV-caching stores computed key-value pairs to avoid redundant
attention computation during autoregressive generation. ReplaySSM (Gu &
Gonzales, 2025) and related work explore analogous caching for SSMs.

**Why it does not apply**:

1. **Mamba already has O(1) state per step.** The SSM maintains a fixed-size
   hidden state (700KB total across all layers) that is updated at each step.
   There is no growing cache to optimize -- the state is already minimal.

2. **The bottleneck is model weights, not state.** Each forward pass reads 493MB
   of model weights from DRAM. The 700KB state fits entirely in L2 cache. Even
   if state computation were free, throughput would not improve because the
   weight reads dominate.

3. **ReplaySSM targets GPU batch serving.** ReplaySSM's contribution is
   efficient state reconstruction for parallel batch processing on GPUs. Redukt
   processes a single stream sequentially on CPU, where this optimization is
   irrelevant.

### 4.4 Sparse State Updates

**NO-GO (ceiling reached)**

R-057 (Fase 9) already implements dt-sparse SSM skipping: when |dt[j]| < 1e-6,
the state update for that dimension is skipped. This captures the viable
sparsity in the SSM recurrence.

**Why further sparsification fails**:

1. **SSM recurrence is only 1.9% of FLOPs.** Even if the entire recurrence were
   free, the maximum speedup is 1.019x. The bottleneck is matmul operations
   (96.2% of FLOPs: in_proj 44%, out_proj 22%, LM head 30%).

2. **Matmul sparsity is destructive.** R-061 (Wanda column pruning) showed that
   pruning a single channel from in_proj causes +20.4% BPB regression. At 10%
   pruning, BPB regression is +431%. Mamba-130M is too small for structured
   pruning -- every parameter is load-bearing.

3. **Unstructured sparsity does not help on CPU.** Even if individual weights
   could be zeroed, unstructured sparsity does not reduce DRAM bandwidth on CPU
   (the full weight matrices must still be read). Only structured sparsity
   (entire rows/columns) reduces memory access, and that is destructive (R-061).

### 4.5 LSH State Memoization

**NO-GO**

Locality-Sensitive Hashing (LSH) could approximately match the current hidden
state against previously seen states, retrieving cached predictions for
near-neighbor states.

**Why it fails**:

1. **Approximate matching is incompatible with lossless compression.** The
   arithmetic coder requires bit-exact CDFs at each position. If the encoder
   uses a cached CDF that differs even slightly from what the decoder would
   compute, the bitstream becomes undecodable. Both encoder and decoder must
   make identical caching decisions, and approximate matching on
   high-dimensional continuous states cannot guarantee this.

2. **High-dimensional state space.** The full hidden state is 24 layers x 768
   dimensions = 18,432-dimensional. LSH in this dimensionality requires many
   hash functions to achieve useful recall, and the computation overhead of
   hashing approaches the cost of the forward pass itself.

3. **State evolution is smooth, not clustered.** SSM states evolve continuously
   through the recurrence. There are no natural clusters that LSH could exploit.
   The probability of a meaningful near-neighbor match is vanishingly small in
   practice.

### 4.6 Token-Level RLE (Matrix Power)

**NO-GO**

For consecutive identical tokens, the SSM state update s_{t+1} = A * s_t + B * x
with constant x could theoretically be computed as a matrix power:
s_{t+k} = A^k * s_t + (A^{k-1} + ... + I) * B * x. This would process K
identical tokens in O(1) backbone calls.

**Why it fails**:

1. **Conv1d shift register.** Mamba-130M uses a d_conv=4 convolution before the
   SSM. The shift register means the first d_conv-1 = 3 repetitions have
   different conv outputs even with identical input tokens. Only from the 4th
   consecutive identical token onward would the matrix power be valid.

2. **BPE makes identical consecutive tokens rare.** With a 50K vocabulary, BPE
   merges ensure that consecutive identical tokens occur less than 1% of the
   time in typical data. Even in highly repetitive data (e.g., repeated XML
   tags), BPE encodes each tag as a unique multi-token sequence where individual
   tokens rarely repeat consecutively.

3. **Matrix power computation cost.** Computing A^k requires eigendecomposition
   or repeated squaring of the state-space matrix. For Mamba-130M with d_state=16,
   this is a 16x16 matrix per head per layer. The overhead of computing the
   matrix power likely exceeds the savings from skipping 1-2 tokens.

### Track C Conclusion

All 6 approaches: **NO-GO**.

The backbone's hidden state is continuous, high-dimensional, and history-
dependent. It never repeats exactly, and approximate matching is incompatible
with lossless compression's bit-exact CDF requirement. The only viable state
optimization already implemented is dt-sparse skipping (R-057), which captures
the available sparsity in the 1.9% of FLOPs consumed by the SSM recurrence.

---

## 5. Track D: Hybrid CM-Backbone Strategies

Track D examines strategies that leverage the CM ensemble to selectively reduce
backbone usage. The CM ensemble (5 models + adaptive SGD mixer) runs at
negligible cost compared to the backbone. If the CM can identify positions where
its own predictions are sufficient, the backbone can be skipped.

### 5.1 Byte-Level CM + Selective Backbone

**NO-GO without MambaByte**

The idea: run the CM ensemble at byte granularity and invoke the backbone only
when CM confidence is low.

**Why it fails now**:

1. **3.5x token expansion destroys skip benefit.** With BPE tokenization, each
   backbone token covers 2.84 bytes on average. If the CM operates at byte
   level but the backbone at token level, skipping a backbone call saves 1
   token but must correctly predict 2.84 bytes. The CM's byte-level accuracy
   is insufficient for this tradeoff.

2. **This is a generalized R-056.** The CM-selective skip strategy (R-056)
   failed precisely because CM confidence over 50K BPE classes never exceeds
   5%. Even at the byte level (256 classes), the CM cannot match the backbone's
   prediction quality for non-trivial content.

3. **Only viable after R-041 (MambaByte).** With a byte-level backbone
   (vocab=256), the CM and backbone operate at the same granularity, and CM
   confidence with 256 classes is dramatically higher than with 50K classes.
   This strategy becomes the Nacrith-style entropy gate (5.4).

### 5.2 Two-Phase Residual Compression

**NO-GO**

The idea: Phase 1 compresses with CM-only (fast), producing a residual stream.
Phase 2 compresses the residual with the backbone (accurate). Total tokens
processed by backbone is reduced because residuals are smaller.

**Why it fails**:

1. **Arithmetic coding is sequential and single-pass.** The arithmetic coder
   interleaves probability estimation and bitstream emission. There is no
   natural way to separate "CM prediction" and "backbone correction" into
   independent phases without fundamentally restructuring the codec.

2. **Two-pass doubles memory.** The first pass must store the entire residual
   stream before the second pass can begin. For large files, this doubles memory
   consumption.

3. **Residual compression is not easier.** The CM's prediction errors are not
   more compressible than the original data. In fact, prediction residuals tend
   toward uniform distribution (by definition of good prediction), making them
   harder to compress, not easier.

4. **No standard for residual coding in arithmetic compression.** Existing
   residual coding techniques (e.g., in video compression) operate on fixed-size
   blocks with known transforms, not on variable-length autoregressive
   probability streams.

### 5.3 Entropy-Stratified Chunking

**NO-GO**

The idea: divide the input into chunks based on estimated entropy, process
low-entropy chunks with CM-only and high-entropy chunks with the full
backbone+CM pipeline.

**Why it fails**:

1. **Mamba state reset at chunk boundaries.** When switching from CM-only to
   backbone processing, the backbone's hidden state must be initialized. If it
   starts from the zero state, it loses all context from the CM-only chunk,
   causing +30-100% BPB regression at the boundary (measured empirically in
   streaming decompression experiments, R-014).

2. **State reconstruction requires processing all prior tokens.** To give the
   backbone the correct state at a chunk boundary, all prior tokens must be
   processed through the backbone -- defeating the purpose of CM-only chunks.

3. **Entropy estimation is unreliable at chunk boundaries.** Local entropy
   estimates fluctuate significantly, and fixed chunk boundaries may split
   homogeneous regions or merge heterogeneous ones.

### 5.4 Nacrith-Style CM Entropy Gate

**CONDITIONAL GO**

Nacrith (Tacconelli 2026) demonstrates an entropy-threshold mechanism for neural
compression: before invoking the backbone, compute the CM ensemble's Shannon
entropy. If entropy is below a threshold (CM is confident), use the CM-only CDF
and skip the backbone entirely.

**Why this differs from R-056** (which failed):

R-056 used **post-hoc oracle analysis**: it measured CM confidence on the ground
truth token after backbone processing, not as a forward-looking decision. This
methodological error meant the measured skip rates were unrealistically high and
the BPB impact was not representative of actual skip decisions.

Nacrith's approach computes CM entropy **before** the backbone decision, using
only information available at encoding time. The encoder and decoder make
identical skip decisions because they both have the same CM state and the same
entropy threshold.

**Published evidence**:
- Nacrith achieves 30-70% skip rate on Transformer-based compression with
  -0.39 BPB improvement (net positive due to reduced model interference on
  high-confidence positions)
- StateSMix (2025) independently validates entropy-adaptive scaling for SSM+CM

**Risk: Mamba state divergence**. Unlike Transformers where KV-cache is
independent of skipped positions, Mamba's recurrent state depends on every
token. Skipping the backbone means the SSM state is not updated, causing
drift. Mitigations:

1. **MAX_CONSEC_SKIP=3**: Limit consecutive backbone skips to bound state drift.
2. **A-matrix eigenvalue decay**: Mamba-130M's A-matrix eigenvalues are ~0.95,
   meaning state influence decays exponentially. After 3 skipped positions, the
   skipped information contributes only 0.95^3 = 0.857 to the state, limiting
   the impact.
3. **State interpolation**: On backbone resumption, blend the stale state with
   the fresh state to smooth the transition.

**Estimated performance**:
- Skip rate: 15-35% (conservative, accounting for SSM state sensitivity)
- BPB impact: -2% to +3% (within the +5% quality gate)
- Throughput improvement: 1.15-1.35x (15-35% fewer backbone calls)

**Implementation requirements**:
- `cm_only_entropy()` in CmEnsemble (already exists from R-056)
- Threshold stored in AZTH header for decompressor reproducibility
- MAX_CONSEC_SKIP counter in codec loop
- No retraining required

### 5.5 Adaptive Layer Depth (Early Exit)

**NO-GO without retraining**

Early exit (Schuster et al. 2022, arXiv:2207.07061; LayerSkip, Elhoushi et al.
2024, arXiv:2404.16710) skips later layers when intermediate predictions are
confident, reducing per-token computation.

**Why it fails**:

1. **R-062 proved layers are not redundant.** Highway layer skip at 5.5% skip
   rate caused +713% BPB regression. Even threshold 0.10 (17 skips out of 85K)
   caused +3.86% BPB. Every layer in Mamba-130M contributes irreplaceable
   features.

2. **LayerSkip requires training with layer dropout.** The published early exit
   methods train the model with stochastic layer dropping so that intermediate
   representations can serve as final outputs. Pretrained Mamba-130M was not
   trained this way, and its intermediate layers do not produce meaningful
   predictions when used as final outputs.

3. **Mamba layers are not interchangeable.** Unlike Transformer blocks where
   later layers often refine earlier predictions, Mamba layers progressively
   build the SSM state through a sequential pipeline. Skipping any layer
   disrupts the state accumulation.

### 5.6 Context Distillation / Knowledge Distillation

**NO-GO**

The idea: train a smaller, faster model that approximates the backbone's
predictions, and use it as a replacement on "easy" positions.

**Why it fails**:

1. **Requires training.** Knowledge distillation requires a training pipeline
   and corpus, with the same cost considerations as MambaByte (R-041).

2. **Small model quality is insufficient.** At the parameter budgets where
   throughput would meaningfully improve (< 50M), prediction quality is far too
   low for compression. The entire point of using Mamba-130M is that smaller
   models produce inadequate CDFs.

3. **Dual-model complexity.** Running two models (student for easy tokens,
   teacher for hard tokens) requires maintaining two sets of weights in memory,
   potentially doubling DRAM bandwidth requirements.

### Track D Conclusion

4 NO-GO, 1 CONDITIONAL GO (Nacrith-style CM entropy gate).

The fundamental pattern: the backbone is irreplaceable for accurate CDFs, but
on positions where the CM ensemble is already highly confident, the backbone's
contribution is marginal. The Nacrith gate exploits this by making a principled,
reproducible skip decision based on CM entropy.

Note: section 5.6 (Context Distillation) is counted as the 23rd approach,
rounding out the investigation across all four tracks.

---

## 6. Viable Path Forward

### 6.1 Immediate: Nacrith-Style CM Entropy Gate (R-077)

No retraining. No new weights. No architectural changes.

**Protocol**:
1. At each token position, compute CM ensemble Shannon entropy H_cm
2. If H_cm < threshold T and consecutive_skips < MAX_CONSEC_SKIP:
   - Use CM-only CDF for arithmetic coding
   - Do NOT update backbone state (skip forward pass)
   - Increment consecutive_skips counter
3. Otherwise:
   - Run backbone forward pass normally
   - Combine backbone + CM logits as usual
   - Reset consecutive_skips to 0
4. Store threshold T in AZTH v4 header (new flag bit)
5. Decompressor applies identical logic (deterministic)

**Expected outcome**:
- 15-35% throughput improvement (40-46 tok/s from current 34)
- BPB within +3% quality gate
- Fully backward-compatible (threshold=0 disables gating)

**Validation protocol**:
- Measure on 12-file dual-backbone benchmark
- A/B controlled test: same hardware, same files, randomized order
- BPB regression must be within +5% budget on every file
- Roundtrip verification on all 12 files

### 6.2 Long-Term: MambaByte + Entropy Gate (R-041 + R-077)

Requires training ($300-750). Eliminates BPE coupling.

**Projected architecture**:
- Byte-level Mamba (vocab=256): 196x smaller LM head
- CM ensemble operates at same granularity (byte-level)
- Entropy gate skip rate: 40-70% (CM confidence much higher with 256 classes)
- Combined throughput: 500-2,000 tok/s on GPU T4 (estimated)

**Phase plan** (from R-041):
1. Validate MambaByte-353M on PG19 (published checkpoint, 0 training cost)
2. Train MambaByte-50M on 5GB corpus (validate byte-level Mamba feasibility)
3. Train MambaByte-130M on 20GB mixed corpus (production model)
4. Integrate into Redukt with entropy gate (R-077)

**Risk**: If MambaByte-130M achieves worse than 1.40 BPB on enwik8, the approach
is not competitive with current Mamba-130M BPE (0.96 BPB). Mitigation: the
353M validation step (Phase 1) de-risks before any training investment.

---

## 7. Related Work

| Work | Year | Relevance |
|------|------|-----------|
| Nacrith (Tacconelli) | 2026 | N-gram entropy gate for Transformer compression, -0.39 BPB |
| StateSMix | 2025 | Mamba SSM + sparse n-gram CM with entropy-adaptive scaling |
| PMATIC (Adler & Tang) | ICLR 2026 | Bounded probability mismatch tolerance for approximate CDFs |
| Byteification | Nature 2026 | Retrofit subword models to byte-level at <1% pretraining cost |
| MambaByte (Wang et al.) | 2024 | Byte-level Mamba, 0.93 BPB PG19 at 353M |
| BLT (Pagnoni et al.) | 2024 | Byte Latent Transformer, patches for efficiency |
| MegaByte (Yu et al.) | 2023 | Multi-scale byte-level Transformer |
| Medusa (Cai et al.) | 2024 | Tree-structured speculative decoding heads |
| Multi-token prediction (Gloeckle et al.) | 2024 | K-future-token heads, 3x inference speedup |
| Speculative decoding (Leviathan et al.) | 2023 | Draft-verify parallel decoding |
| Speculative decoding (Chen et al.) | 2023 | Independent concurrent discovery |
| Blockwise parallel (Stern et al.) | 2018 | Jacobi iteration for autoregressive models |
| Jacobi decoding (Santilli et al.) | 2023 | Fixed-point iteration formalization |
| Lookahead decoding (Fu et al.) | 2024 | N-gram pool + Jacobi hybrid |
| Token Merging (Bolya et al.) | 2023 | Progressive token reduction for ViTs |
| Hourglass Transformer (Nawrot et al.) | 2023 | Multi-scale hierarchical Transformer |
| LayerSkip (Elhoushi et al.) | 2024 | Early exit with layer dropout training |
| SentencePiece (Kudo & Richardson) | 2018 | Subword regularization and vocabulary analysis |
| Tokenizer analysis (Zouhar et al.) | 2023 | Vocabulary size vs compression efficiency |
| Tokenization study (Rust et al.) | ICLR 2024 | Tokenization-task performance coupling |
| Tokenizer impact (Limisiewicz et al.) | 2023 | Vocabulary design for downstream tasks |
| Lottery Ticket (Frankle & Carlin) | ICLR 2019 | Sparse subnetwork discovery |
| Wanda (Sun et al.) | 2024 | Pruning by weights and activations |
| Mamba (Gu & Dao) | 2024 | Selective state space model |
| RWKV-7 (Peng et al.) | 2024 | Linear attention with WKV mechanism |

---

## 8. Sources

1. Gloeckle, F., Idrissi, B.Y., Roziere, B., Lopez-Paz, D., & Synnaeve, G.
   (2024). "Better & Faster Large Language Models via Multi-token Prediction."
   arXiv:2404.19737

2. Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from
   Transformers via Speculative Decoding." ICML 2023. arXiv:2211.17192

3. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J.
   (2023). "Accelerating Large Language Model Decoding with Speculative
   Sampling." arXiv:2302.01318

4. Stern, M., Shazeer, N., & Uszkoreit, J. (2018). "Blockwise Parallel
   Decoding for Deep Autoregressive Models." NeurIPS 2018. arXiv:1811.03115

5. Santilli, A., Severino, S., Postolache, E., Maiorca, V., Mancusi, M.,
   Marro, S., & Rodola, E. (2023). "Accelerating Transformer Inference for
   Translation via Parallel Decoding." arXiv:2305.10427

6. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., & Dao, T.
   (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple
   Decoding Heads." ICML 2024. arXiv:2401.10774

7. Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024). "Break the Sequential
   Dependency of LLM Inference Using Lookahead Decoding." arXiv:2402.02057

8. Yu, L., Simig, D., Flaherty, C., Aghajanyan, A., Zettlemoyer, L., &
   Lewis, M. (2023). "MEGABYTE: Predicting Million-byte Sequences with
   Multiscale Transformers." NeurIPS 2023. arXiv:2305.07185

9. Pagnoni, A., et al. (2024). "Byte Latent Transformer: Patches Scale Better
   Than Tokens." arXiv:2412.09871

10. Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J.
    (2023). "Token Merging: Your ViT But Faster." ICLR 2023. arXiv:2210.09461

11. Nawrot, P., Tworkowski, S., Tyber, M., Jaszczur, M., Choromanski, K., &
    Milos, P. (2023). "Efficient Transformers with Dynamic Token Pooling."
    arXiv:2305.13551

12. Dai, Z., Lai, G., Yang, Y., & Le, Q.V. (2020). "Funnel-Transformer:
    Filtering out Sequential Redundancy for Efficient Language Processing."
    NeurIPS 2020. arXiv:2006.03236

13. Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language
    independent subword tokenizer and detokenizer for Neural Text Processing."
    EMNLP 2018. arXiv:1808.06226

14. Zouhar, V., Meister, C., Gastaldi, J.L., Du, L., Sachan, M., & Cotterell,
    R. (2023). "Tokenization and the Noiseless Channel." ACL 2023.
    arXiv:2306.16842

15. Rust, P., Pfeiffer, J., Vulic, I., Ruder, S., & Gurevych, I. (2024).
    "Language Modelling with Pixels." ICLR 2024.

16. Limisiewicz, T., Malkin, D., & Stanovsky, G. (2023). "Tokenization Impacts
    Multilingual Language Modeling." arXiv:2306.06098

17. Wang, J., Gangavarapu, T., Yan, J.N., & Rush, A. (2024). "MambaByte:
    Token-Free Selective State Space Model." arXiv:2401.13660

18. Gu, A., & Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces." ICLR 2024. arXiv:2312.00752

19. Peng, B., et al. (2024). "Eagle and Finch: RWKV with Matrix-Valued States
    and Dynamic Recurrence." arXiv:2404.05892

20. Frankle, J., & Carlin, M. (2019). "The Lottery Ticket Hypothesis: Finding
    Sparse, Trainable Neural Networks." ICLR 2019. arXiv:1803.03635

21. Sun, M., Liu, Z., Bair, A., & Kolter, J.Z. (2024). "A Simple and Effective
    Pruning Approach for Large Language Models." ICLR 2024. arXiv:2306.11695

22. Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V.Q.,
    Tay, Y., & Metzler, D. (2022). "Confident Adaptive Language Modeling."
    NeurIPS 2022. arXiv:2207.07061

23. Elhoushi, M., et al. (2024). "LayerSkip: Enabling Early-Exit Inference and
    Self-Speculative Decoding." arXiv:2404.16710

24. Tacconelli, S. (2026). "Nacrith: Entropy-Gated Neural Compression with
    N-gram Skip." Preprint.

25. Adler, R., & Tang, Y. (2026). "PMATIC: Probabilistic Mismatch-Tolerant
    Arithmetic Coding." ICLR 2026.

26. "Byteification: Retrofitting Subword Language Models to Byte-Level
    Processing." Nature Machine Intelligence, 2026.

27. StateSMix: Sparse Mixture of State Space Models with Entropy-Adaptive
    Gating. 2025.

28. Mahoney, M. (2005). "Adaptive Weighing of Context Models for Lossless Data
    Compression." Florida Tech Technical Report.

29. Knoll, B., & de Freitas, N. (2012). "A Machine Learning Perspective on
    Predictive Coding with PAQ." arXiv:1108.3298

30. Bellard, F. (2021). "NNCP: Lossless Data Compression with Neural Networks."
    arXiv:2112.11635

31. Delétang, G., et al. (2024). "Language Modeling Is Compression."
    ICLR 2024. arXiv:2309.10668

32. Valmeekam, K., Marquez, M., & Srivastava, S. (2023). "On the Connection
    Between Compression and Intelligence." arXiv:2309.07516

---

## 9. Cross-References

- `docs/internal/research/adaptive-tokenization-research.md` section 3.2:
  tokenizer-model coupling analysis
- `docs/internal/research/prediction-cache-research.md` sections 1-2:
  backbone = 99.5% of processing time
- `docs/internal/research/cpu-parallelism-strategies.md` R-056:
  CM-skip failure analysis and post-mortem
- `docs/internal/research/lottery-ticket-sparsity-protocol.md`:
  FLOPs breakdown (matmuls 96.2%, SSM recurrence 1.9%)
- `docs/internal/research/execution-architecture-analysis.md`:
  CmEnsemble modular architecture
- `docs/internal/research/backbone-scalability-analysis.md`:
  30+ source analysis of backbone alternatives

---

## Appendix: Approach Summary Table

| # | Track | Approach | Verdict | Blocker |
|---|-------|----------|---------|---------|
| 1 | A | Multi-Token Prediction Heads | NO-GO | P(t+k\|h_t) != sequential CDF |
| 2 | A | Speculative Decoding | NO-GO | SSM sequential, tokens known |
| 3 | A | Blockwise Parallel / Jacobi | NO-GO | SSM sequential, argmax only |
| 4 | A | Medusa Heads | NO-GO | Same as #1 + #2 |
| 5 | A | Lookahead Decoding | NO-GO | Jacobi + redundant n-gram |
| 6 | B | Larger Vocabulary BPE | NO-GO | LM head explosion, retraining |
| 7 | B | Byte-Level Patching | NO-GO | 130M too small for two-tier |
| 8 | B | Entropy-Adaptive Tokenization | NO-GO | OOD sequences, = R-056 |
| 9 | B | Hierarchical / Multi-Scale | NO-GO | 130M too small, retraining |
| 10 | B | Token Merging (ToMe) | NO-GO | Destroys per-token CDFs |
| 11 | B | Compression-Aware Tokenization | NO-GO | Requires retraining |
| 12 | B | MambaByte | **COND. GO** | Training $300-750 |
| 13 | C | State Fingerprinting / PredCache | NO-GO | +17.4% BPB (R-049) |
| 14 | C | Repetition Detection | NO-GO | State divergence |
| 15 | C | KV-Cache for SSMs | NO-GO | Already O(1), bottleneck=weights |
| 16 | C | Sparse State Updates | NO-GO | Ceiling reached (R-057) |
| 17 | C | LSH State Memoization | NO-GO | Approximate != bit-exact |
| 18 | C | Token-Level RLE | NO-GO | Conv1d + BPE prevents runs |
| 19 | D | Byte-Level CM + Selective | NO-GO | 3.5x expansion, needs MambaByte |
| 20 | D | Two-Phase Residual | NO-GO | Sequential AC, 2x memory |
| 21 | D | Entropy-Stratified Chunking | NO-GO | State reset +30-100% BPB |
| 22 | D | Nacrith-Style Entropy Gate | **COND. GO** | SSM state drift risk |
| 23 | D | Adaptive Layer Depth | NO-GO | +713% BPB (R-062), retraining |

**Final tally**: 21 NO-GO, 2 CONDITIONAL GO.
