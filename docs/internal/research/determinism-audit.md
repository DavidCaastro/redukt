# Cross-Platform Determinism Audit — R-013

> Date: 2026-10-29
> Platform: Windows x86_64, i5-1235U, AVX2, Q8 auto

---

## Determinism Guarantees

| Scenario | Guarantee | Verified |
|---|---|---|
| Same binary, same platform, multiple runs | **DETERMINISTIC** | 5 runs SHA-256 identical |
| Same binary, SIMD path | **DETERMINISTIC** | Golden reference SHA-256 match |
| Same binary, scalar path (REDUKT_NO_SIMD=1) | **DETERMINISTIC** | 3 runs identical |
| SIMD compress → scalar decompress | **INCOMPATIBLE** | CRC mismatch (expected) |
| Scalar compress → SIMD decompress | **INCOMPATIBLE** | CRC mismatch (expected) |
| Windows x86 → Linux x86 (same SIMD tier) | **NOT GUARANTEED** | Untested (different libm) |
| x86_64 → ARM64 | **NOT GUARANTEED** | Untested (no SIMD, different libm) |
| Debug build → Release build | **INCOMPATIBLE** | Different .azth output observed |

## Integer-Only Paths (Platform-Deterministic)

These operations use only integer arithmetic and produce identical results everywhere:

| Component | File | Operations |
|---|---|---|
| Range coder (encode/decode) | `coder.rs` | u32, u64, i64 arithmetic |
| CRC-32 | `crc32.rs` | GF(2) polynomial (u32) |
| E8E9 preprocessing | `preprocess.rs` | Byte-level bitwise ops |
| Q8 matmul accumulation | `tensor.rs` | i16→i32 integer SIMD |
| AZTH header/format | `codec.rs` | Byte layout, LE integers |

## Float Operations Audit

### HIGH risk (hot path, platform-dependent)

| Operation | File | Lines | SIMD | Scalar | Risk |
|---|---|---|---|---|---|
| fast_exp_ps (Cephes) | tensor.rs | 764-796 | AVX2 polynomial | n/a | x86_64 only |
| fast_sigmoid_ps | tensor.rs | 801-805 | via fast_exp | n/a | x86_64 only |
| fast_tanh_ps | tensor.rs | 810-814 | 2*sigmoid(2x)-1 | n/a | x86_64 only |
| sigmoid_inplace | tensor.rs | 833-845 | fast_sigmoid_ps | 1/(1+exp(-v)) libm | Different results |
| tanh_inplace | tensor.rs | 827 | fast_tanh_ps | v.tanh() libm | Different results |
| sigmoid_add_into | tensor.rs | 866-895 | SIMD | scalar | Different results |
| decay_sigmoid_into | tensor.rs | 899-910 | SIMD | scalar | Different results |
| sigmoid_mix_inplace | tensor.rs | 936-974 | SIMD | scalar | Different results |

### MEDIUM risk (accumulation order)

| Operation | File | Lines | Notes |
|---|---|---|---|
| F32 mat_vec_mul | tensor.rs | 1025-1184 | AVX2: 8-lane parallel vs scalar sequential |
| softmax_into (sum) | tensor.rs | 1337-1353 | Sequential accumulation, uses exp() |
| mixer gradient (grad sum) | mixer.rs | 62-64 | Sequential over Vec (deterministic order) |

### LOW risk / NOT in hot path

| Operation | File | Notes |
|---|---|---|
| FMA matmul (dead code) | tensor.rs | Disabled, not dispatched |
| Box-Muller normal | spectral.rs | Profiling only |
| Entropy calculation | telemetry.rs | Logging only |
| -prob.log2() | benchmark.rs | Reporting only |

## False Positive: HashMap in ngram.rs

The audit initially flagged `HashMap<u32, u32>` iteration in `ngram.rs:124` as CRITICAL.

**Analysis**: Each iteration updates `logits[tok as usize]` where `tok` is a unique key.
Since each key maps to a **different** index, iteration order is irrelevant — the final
logits are identical regardless of HashMap traversal order.

**Verdict**: NOT a determinism issue. No code change needed.

## Why Cross-Tier Fails

Encoder and decoder must produce **identical** probability distributions at each step.
The arithmetic coder then selects symbols based on exact CDF boundaries.

When SIMD (Cephes approximation) and scalar (libm) produce different sigmoid/tanh/exp
values, the probability distributions diverge. After a few tokens, the decoder selects
a wrong symbol, cascade-corrupting all subsequent bytes. CRC-32 detects this.

**Measured divergence** (per operation):
- sigmoid: SIMD vs scalar ~1e-6 per element [MEASURED]
- tanh: SIMD vs scalar ~1e-6 per element [MEASURED]
- exp: SIMD vs scalar ~1e-4 per element [MEASURED]
- After softmax normalization: ~1e-5 per probability
- After 10+ tokens: accumulated divergence exceeds coder precision

## Test Infrastructure (R-013)

4 new tests in `tests/roundtrip.rs`:

1. **golden_reference_determinism**: 39B input → 35B AZTH, SHA-256 hardcoded
2. **multi_process_determinism**: 5 separate process runs, SHA-256 identical
3. **scalar_determinism**: 3 runs with REDUKT_NO_SIMD=1, identical
4. **cross_tier_roundtrip_documents_divergence**: SIMD→scalar CRC mismatch, same-tier PASS

SHA-256 implemented zero-dep in test file (no external crate).

## Golden Reference Values [MEASURED 2026-10-29]

- **Input**: `b"Determinism: the quick brown fox jumps!"` (39 bytes)
- **Platform**: Windows x86_64, AVX2, Q8 auto
- **AZTH output**: 35 bytes
- **SHA-256**: `736a2024822ef712bf9847cffd2b1065f03f7f1de152329a02b816f71b6c3e21`
- **CRC-32**: `D124164C`

## Recommendations

1. **Compress and decompress on same SIMD tier** — document in README
2. **REDUKT_NO_SIMD=1 for maximum portability** — scalar path uses only libm
3. **Future: portable transcendentals** — replace libm with fixed-precision polynomial
   approximations in scalar path (would make scalar cross-platform deterministic)
4. **CI**: Add Linux x86_64 + ARM64 runners, record platform-specific golden hashes
