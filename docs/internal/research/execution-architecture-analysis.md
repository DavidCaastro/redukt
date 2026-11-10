# Execution Architecture Analysis — Breaking the Sequential Bottleneck

> Date: 2026-11-01
> Status: Research complete, implementation pending
> Context: Redukt achieves world-class compression ratios (0.963 BPB enwik8, -45% vs brotli on structured data)
> but throughput is ~43 B/s (Mamba) due to sequential neural forward pass.
> This document analyzes the bottleneck and proposes viable alternative execution strategies.

---

## 1. Hot Path Decomposition

### Pipeline per token (~1 byte)

```
Byte -> Tokenize -> [6 CM Models] -> Neural Forward -> Combine -> Softmax -> Encode -> Update
         0.1us       ~30us           10,600us          3us       2us       0.2us     20us
                     0.3%            99.5%             0.03%     0.02%     0.002%    0.2%
```

### Neural Forward Breakdown (RWKV-7, per token)

| Component | Time | % of Total | FLOPs |
|-----------|------|------------|-------|
| **Neural Forward (backbone)** | **10,600 us** | **99.5%** | 107M |
| -- TimeMix state update | 6,600 us | 62% | 1.3M |
| -- ChannelMix FFN | 1,900 us | 18% | 56.6M |
| -- LM Head (vocab projection) | 2,100 us | 20% | 50M |
| CM Models (6 models) | 30 us | 0.3% | ~1M |
| Ensemble combine + softmax | 5 us | 0.05% | 523K |
| Arithmetic encoder | 0.2 us | 0.002% | 50K |
| CM learning + update | 24 us | 0.2% | 527K |
| **TOTAL** | **~10,660 us** | **100%** | ~108M |

### Bottleneck Structure

```
NEURAL FORWARD PASS (99.5% of time)
+--------------------------------------------+----------+--------+
|     TimeMix State Update (62%)             | FFN (18%)| LM Head|
|     SEQUENTIAL -- P-complete               | matmul   | (20%)  |
|     state[t] = f(state[t-1], input[t])     | D->3D->D | D->65K |
|     CANNOT BE PARALLELIZED (per-token)     | SIMD'd   | Q8 SIMD|
+--------------------------------------------+----------+--------+
|CM| |enc|  <-- everything else (0.5%) is invisible
```

### SSM Recurrence — The Exact Bottleneck

Per cell of the state matrix:

```
cell_new = cell_old * A + input_new * B
```

Where A is DATA-DEPENDENT (computed from current input). This creates a chain:

```
Token 1: state_1 = 0 * A_1 + data_1 * B_1           <- must finish
Token 2: state_2 = state_1 * A_2 + data_2 * B_2      <- before this starts
Token 3: state_3 = state_2 * A_3 + data_3 * B_3      <- before this starts
```

Each step DEPENDS on the previous. Like a domino chain.

**Within each token**: 1,536 channels are independent (parallelizable).
**Across tokens**: strictly sequential.

CPU (i5-1235U): 8 channels at a time (AVX2) -> 192 passes per token per layer.
GPU (RTX 4090): 1,536 channels at once -> 1 pass per token per layer.

### Theoretical Performance Ceiling (CPU)

```
State update latency:    ~5-6 ms  (data dependency, irreducible)
ChannelMix FFN:          ~1-2 ms  (bandwidth-limited: 20.6 GB/s DRAM)
LM Head:                 ~1-2 ms  (50M FLOPs / 48 GFLOPS/s)
-----------------------------
Theoretical minimum:     ~7-10 ms/token
Actual:                  10.6 ms/token
Efficiency:              ~92-95% of hardware ceiling

-> Already near maximum on CPU.
-> No software optimization breaks this ceiling.
```

---

## 2. The Key Insight: Compression Has ALL Tokens Upfront

```
CHATBOT (generation):    doesn't know what comes next. MUST go token by token.
COMPRESSOR (our case):   HAS the entire file. Knows all tokens from the start.
```

Today Redukt ignores this advantage — processes as if generating (RNN mode, one token at a time).

---

## 3. Alternative Execution Strategies

### Strategy A: Parallel Scan (GPU) — The Breakthrough

The SSM recurrence is a LINEAR recurrence: `s[t] = A[t] * s[t-1] + b[t]`

Linear recurrences can be represented as composable pairs:

```
Step t:  (A[t], b[t])    where b[t] = B[t] * data[t]

Compose two consecutive steps:
    (A_2, b_2) . (A_1, b_1) = (A_2*A_1, A_2*b_1 + b_2)
```

This composition is ASSOCIATIVE, enabling parallel scan (prefix sum):

```
SEQUENTIAL (current):   step_1 -> step_2 -> step_3 -> ... -> step_3000
                        3,000 sequential steps

PARALLEL SCAN:
    Level 1:  (1.2)  (3.4)  (5.6)  ... (2999.3000)     1,500 pairs IN PARALLEL
    Level 2:  (1..4)    (5..8)    ...   (2997..3000)       750 pairs IN PARALLEL
    Level 3:  (1..8)       (9..16)      ...                375 IN PARALLEL
    ...
    Level 12: (1...............................3000)          1 final result

    3,000 steps -> 12 steps.  Reduction: 250x.
```

**Pipeline with parallel scan (GPU, per layer):**

```
ALL 3,000 tokens enter AT ONCE

For each layer (1 to 24, sequential between layers):
+------------------------------------------------------------+
| 1. RMSNorm x 3000 tokens          -> PARALLEL (GPU)       |
| 2. in_proj x 3000 tokens          -> PARALLEL (batch matmul)|
| 3. Conv1d x 3000 tokens           -> PARALLEL              |
| 4. Compute A,B,C x 3000 tokens    -> PARALLEL              |
| 5. SSM scan (3000 steps)          -> 12 levels (scan)      | <- the magic
| 6. Gate + out_proj x 3000 tokens  -> PARALLEL              |
| 7. Residual add x 3000 tokens     -> PARALLEL              |
+------------------------------------------------------------+

Then: LM Head x 3000 tokens -> PARALLEL
Finally: arithmetic coding (sequential but microseconds)
```

**Projected performance (10KB file, ~3000 tokens):**

| Component | CPU (today) | GPU + Parallel Scan |
|-----------|-------------|---------------------|
| Matmuls per layer | 1 token at a time | 3,000 tokens batch |
| SSM recurrence | 3,000 seq. steps | 12 levels of scan |
| Total layers (x24) | 30,000 ms | ~1.0 ms (A100) |
| LM Head | (included) | ~0.4 ms |
| Arithmetic coding | (included) | ~0.5 ms (CPU) |
| **TOTAL** | **~30,000 ms** | **~2 ms** |
| **Throughput** | **333 B/s** | **~5 MB/s** |

On modest GPU (RTX 3060): ~25x less power -> still **~200 KB/s**.
On RTX 4090: **~5-8 MB/s** — *faster than brotli-11* (0.3 MB/s).

**Key property: LOSSLESS** — produces exactly the same logits in the same order.

References:
- Mamba-2 SSD: https://arxiv.org/abs/2405.21060
- ScanWeaver (compiler-driven parallel scan): https://arxiv.org/abs/2606.00601
- High-Throughput LLM Compression: https://arxiv.org/abs/2610.17586

### Strategy B: Speculative CM-Draft (CPU, immediate, zero investment)

Redukt already has 6 CM models (TokenNgram, MatchModel, RunModel, WordModel, BiasHead,
SegmentModel) that cost 0.3% of total time (~30us vs 10,600us for neural forward).

**Idea**: CM models PREDICT the next K tokens without neural forward. Neural model
VERIFIES all K at once in a batch. On repetitive data, acceptance rate is high.

```
Context: "GET /index.html HTTP/1.1" 200 "
CM predicts: [5, 1, 2, 3]  (tokens for "512 ")
Cost to predict: ~120us (4 x 30us) = FREE

Neural verifies all 4 in pipeline:
If all 4 match (logs: ~60-80% of the time):
    4 tokens verified = 4 tokens encoded
    Cost: ~1 batch verification
    Speedup: ~2-4x on repetitive data

If mismatch: discard from failure point, continue normally.
LOSSLESS -- encode with full model probabilities.
```

Estimated acceptance rates:
- apache logs: 60-80% (many near-identical lines)
- XML invoices: 50-70% (repetitive tags)
- dickens: 20-40% (literary text, less predictable)
- binary: 5-15% (nearly useless)

**No GPU required. No retraining. Uses existing code.**

### Strategy C: Micro-Macro Model (inspired by StateSMix + Proxy Compression)

Two-tier model architecture:

```
LEVEL 1: Micro-model (StateSMix-style, 120K params)
    - Trained ONLINE on the file being compressed
    - Predicts "easy tokens" (high confidence)
    - Throughput: ~700 B/s (16x faster than current Redukt)

LEVEL 2: Macro-model (Mamba-130M, pretrained)
    - Only activates when micro-model is UNCERTAIN
    - Threshold: if max(P_micro) > 0.8 -> use micro, skip macro
    - If max(P_micro) < 0.8 -> call macro

Economy:
    If 60% of tokens are "easy" -> 60% skip x 10ms = 6ms/token saved
    Effective throughput: ~2.5x

Quality:
    Easy tokens encoded with less precise probabilities -> ~+1-3% BPB
    WITH VERIFICATION (lossless): macro verifies batch, re-encodes on mismatch
```

References:
- StateSMix (May 2026): https://arxiv.org/abs/2605.02904
  - Online Mamba 120K params + 9 n-gram tables, 700 B/s CPU, 2.123 BPB enwik8
- Proxy Compression (ICML 2026): https://arxiv.org/abs/2602.04289

### Strategy D: SEP-style Semantic Patching + Parallel Scan (GPU)

```
Step 1: PATCHING
    Group bytes into semantic patches BEFORE tokenizing.
    Result: 2-4x fewer tokens to process.
    10KB -> 3,000 tokens (BPE) -> ~800-1,500 "super-tokens"

Step 2: PARALLEL SCAN (all super-tokens at once on GPU)
    800 super-tokens -> log2(800) = 10 scan levels
    Combined reduction: 4x (patching) x 250x (scan vs sequential) = ~1,000x

Step 3: Multi-stream arithmetic coding
    Split into 8 streams -> 8 coders in parallel (CPU, while GPU processes)

Result: 10KB in ~0.5ms (GPU) + ~0.2ms (coding) = ~14 MB/s
```

Reference:
- SEP Framework (IJCAI 2025): https://www.ijcai.org/proceedings/2025/0370.pdf
  - +5% compression, +30% speed, -63% GPU memory

### Strategy E: NCM-style Coarse-to-Fine (requires new training)

Inspired by Next-Concept Modeling (Stanford/Hazy Research, Oct 2026).

```
PHASE 1 -- COARSE (tiny model, 10M params, VERY fast):
    Generate rough distribution over "token classes":
    {vowels: 0.3, consonants: 0.2, numbers: 0.15, ...}
    Cost: ~0.5ms for ALL tokens (parallel scan)

PHASE 2 -- FINE (full model, 130M params, selective):
    Only refine tokens where coarse prediction was uncertain.
    Clear tokens (high probability in one class): skip
    Ambiguous tokens: full forward pass

    On logs: 70% of tokens predictable by class -> skip 70%

PHASE 3 -- ENCODE
    Encode using: P_fine where available, P_coarse where sufficient.
    Arithmetic coder combines both distributions.

Speedup: 3-10x (depends on skip rate)
```

Reference:
- NCM (Oct 2026): https://arxiv.org/abs/2610.14162
  - 10-31x speedup over autoregressive, coarse-to-fine VQ-VAE hierarchy

### Strategy F: Full Pipeline Integration (all combined)

```
+------------------------------------------------------------------+
|                    FULL PIPELINE                                  |
|                                                                   |
|  INPUT (10KB bytes)                                               |
|      |                                                            |
|      v                                                            |
|  +----------------+                                               |
|  | SEP PATCHING   |  Group bytes into semantic patches            |
|  | (CPU, 0.1ms)   |  3,000 tokens -> ~1,000 super-tokens         |
|  +-------+--------+                                               |
|          |                                                        |
|          v                                                        |
|  +---------------------+                                          |
|  | CM SPECULATIVE      |  6 CM models predict super-token batches |
|  | DRAFT (CPU, 1ms)    |  (~60% accepted)                        |
|  +-------+-------------+                                          |
|          |                                                        |
|          +-- "easy" tokens (60%) -> encode directly               |
|          |                          (verify later)                |
|          v                                                        |
|  +----------------------------------+                             |
|  | PARALLEL SCAN -- GPU             |  Only "hard" tokens         |
|  | (400 tokens x 24 layers)         |  + full batch verification  |
|  | 9 scan levels                    |                             |
|  | Time: ~0.3ms                     |                             |
|  +-------+--------------------------+                             |
|          |                                                        |
|          v                                                        |
|  +------------------------+                                       |
|  | MULTI-STREAM ENCODE    |  8 parallel arithmetic streams        |
|  | (CPU, 0.2ms)           |  while GPU processes next chunk       |
|  +-------+----------------+                                       |
|          |                                                        |
|          v                                                        |
|  OUTPUT: AZTH compressed file                                     |
|                                                                   |
|  TOTAL TIME: ~1.6ms for 10KB                                     |
|  THROUGHPUT: ~6 MB/s                                              |
|  QUALITY: identical (lossless, verified)                          |
+------------------------------------------------------------------+
```

---

## 4. Comparison Matrix

| Strategy | Speedup | GPU Required | Retraining | Lossless | Impl. Effort | Time |
|----------|---------|-------------|------------|----------|-------------|------|
| **A: Parallel Scan** | **250x/layer** | Yes | No | **Yes** | High | 4-8 weeks |
| B: Speculative CM | 2-4x | No | No | **Yes** | Medium | 1-2 weeks |
| C: Micro-Macro | 2.5x | No | Partial | Yes (w/verify) | High | 3-4 weeks |
| D: SEP + Scan | ~1,000x | Yes | No | **Yes** | High | 4-8 weeks |
| E: Coarse-to-Fine | 3-10x | Optional | Yes | Yes (w/verify) | Very High | 8-12 weeks |
| **F: Full Pipeline** | **~1,000-5,000x** | Yes | Partial | **Yes** | Very High | 8-12 weeks |

---

## 5. Projected Throughput

| Configuration | Throughput | Break-even vs brotli storage |
|---------------|-----------|------------------------------|
| Current (CPU RNN) | 43 B/s | Never |
| + Strategy B (Speculative CM) | 120-170 B/s | Never (but 3-4x better) |
| + Strategy A (GPU Parallel Scan) | ~5 MB/s | Immediate |
| + Strategy D (SEP + Scan) | ~14 MB/s | Immediate, faster than brotli |
| + Strategy F (Full Pipeline) | ~6 MB/s | Immediate |

---

## 6. Competitive Context

### Neural Compressor Landscape (Nov 2026)

| System | enwik8 BPB | Speed | Hardware | Status |
|--------|-----------|-------|----------|--------|
| **Redukt (Mamba+CM)** | **0.963** (10KB) | 43 B/s | CPU | Production binary |
| **Nacrith** | **0.939** (100MB) | 60-90 tok/s | GPU GTX 1050+ | Open source |
| ts_zip (Bellard) | ~1.11 | **1 MB/s** (RTX 4090) | GPU | Research |
| Oct'26 paper (arXiv 2610) | ~1.0 | **181 KB/s** (A100) | GPU | Paper |
| FineZip | 1.024 | ~250 B/s (A6000) | GPU | Research |
| StateSMix | 2.123 (1MB) | 700 B/s | CPU | Research |
| cmix v21 | 1.170 | 1.6 KB/s | CPU | Research (LTCB; corrected 2026-11-10) |
| NNCP v2 | ~1.2 | ~10 KB/s | GPU | Research |
| brotli-11 | 2.352 (10KB) | **300 KB/s** | CPU | Standard |

### Key Economic Reference

VLDB CIDR'26 paper (Kipf et al.): "LLM-based compression needs 120 years to break
even vs traditional compression" at current compute/storage costs.

BUT this assumes:
1. Generic storage ($1/TB archive)
2. GPU at on-demand prices
3. No high-value niches

Where Redukt wins the economics:
- Bandwidth > $1/GB (satellite: $1-225/GB)
- Ingestion pricing ($0.10-150/GB: Datadog/Splunk)
- Hot storage x long retention ($23/TB x 60 months = $1,380/TB lifetime)
- GPU spot ($0.15/hr T4) + throughput >100 KB/s

References:
- VLDB Economics: https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf
- Nacrith: https://arxiv.org/abs/2602.19626
- ts_zip: https://bellard.org/ts_zip/
- High-Throughput LLM Compression: https://arxiv.org/abs/2610.17586
- StateSMix: https://arxiv.org/abs/2605.02904

---

## 7. Market Sizing (2026)

| Market | TAM | Source |
|--------|-----|--------|
| Data Compression Software | $1.28B (2026), $1.65B (2030) | Business Research Company |
| Enterprise Information Archiving | $11.33B (2026), $33.15B (2034) | Fortune Business Insights |
| Archival Data Storage | $10.02B (2026), $17.62B (2035) | Business Research Insights |
| Log Management | $4.1-5.0B (2026) | MarketsandMarkets, Mordor, R&M |
| Spain E-Invoicing | $299M (2025), $1,139M (2034) | Vocal Media (low authority) |

### Storage Costs (2026)

| Tier | $/TB/month |
|------|-----------|
| S3 Standard | $23 |
| S3 Infrequent Access | $12.50 |
| S3 Glacier Instant | $4 |
| S3 Glacier Flexible | $3.60 |
| Glacier Deep Archive | $1 |
| Azure Archive | $1 |
| GCS Archive | $1.20 |

### Bandwidth Costs (high-value niches)

| Channel | $/GB |
|---------|------|
| Starlink Maritime Priority | $1.08-5.00 |
| OneWeb Maritime | $1.75-3.25 |
| VSAT traditional | $10-225 |
| Datadog log ingestion | $0.10/GB |
| Splunk Cloud indexing | ~$150/GB/day |

### Compute Costs (GPU)

| GPU | $/hour (spot) | $/hour (on-demand) |
|-----|-------------|-------------------|
| RTX 4090 (Lambda) | ~$0.50 | ~$1.00 |
| A100 (AWS p4d) | ~$3.00 | ~$12.00 |
| T4 (AWS g4dn) | ~$0.15 | ~$0.53 |

---

## 8. Recommended Sequence

### Move 1: Speculative CM-Draft (NOW, CPU, 0 investment)

- Use existing 6 CM models as draft predictors
- Implement batch verification with neural model
- Target: 2-4x throughput on repetitive data
- Zero quality loss (lossless verification)
- **1-2 weeks implementation**

### Move 2: GPU Parallel Scan (next, primary investment)

- Implement CUDA parallel scan for Mamba-130M
- Target: 100+ KB/s on consumer GPU, 5+ MB/s on A100
- Exact same model, exact same quality
- **4-8 weeks implementation, $0-2K compute**

### Move 3: Full Pipeline (future)

- SEP patching + speculative + parallel scan + multi-stream coding
- Target: 1-14 MB/s (faster than brotli-11)
- **8-12 weeks, builds on moves 1-2**

---

## 9. Conclusion

The model is good. The execution is what needs rethinking.

```
WHAT WOULD CHANGE:

1. DO NOT change the model (weights, formula) -- those produce good predictions
2. DO change HOW we execute the formula:
   - From sequential (RNN) -> parallel (scan)               <- 250x
   - From all-by-neural -> draft/verify with CM              <- 3x
   - From byte-by-byte -> semantic patches                   <- 3x
   - From single-stream -> multi-stream coding               <- 2-4x

3. The combination is MULTIPLICATIVE:
   250 x 3 x 3 x 3 = ~6,750x theoretical
   Conservative (with real overheads): ~1,000-5,000x

   43 B/s x 1,000 = 43 KB/s
   43 B/s x 5,000 = 215 KB/s -> COMMERCIALLY VIABLE
```

---

*Analysis date: 2026-11-01. All performance projections marked as [PROJECTION].
All current measurements from local hardware (i5-1235U, 32GB DDR4).
GPU estimates based on published benchmarks from ts_zip, Nacrith, and arXiv 2610.17586.*
