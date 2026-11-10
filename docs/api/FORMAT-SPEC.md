# AZTH Format Specification

## Purpose

AZTH is a lossless compression format that encodes data using neural network
probability predictions combined with arithmetic coding. Decompression requires
the identical model and weights; hyperparameters are stored in the header.

## Format Layout (v4 — read-only since R-104 F1)

Since R-104 F1 the encoder writes **v5** for every archive (see below): identical
layout plus one PROFILE byte after FLAGS, which carries the weight profile and the
numeric contract. v4 is still fully decoded; a v4 archive implies the F32 weight
profile and **numeric contract 0 (legacy libm)**.

```
Offset  Size  Field          Description
------  ----  -----          -----------
0x00    8B    MAGIC          "AZTH\x04\x00\x00\x00" (version 4)
0x08    4B    TOTAL_BYTES    Original file size (little-endian u32)
0x0C    1B    FLAGS          Bit flags (see below)
--- if FLAGS & 0x01 (custom hyperparams): ---
0x0D    4B    BIAS_LR        BiasHead learning rate (little-endian f32)
0x11    4B    NGRAM_SCALE    N-gram logit bias scale (little-endian f32)
0x15    4B    MIX_ETA        Adaptive mixer learning rate (little-endian f32)
--- endif ---
--- if FLAGS & 0x02 (emb-surgery): ---
+5B     1B    EMB_METHOD     Surgery method (0=none, 1=center, 2=norm, 3=spread)
+5B     4B    EMB_ALPHA      Alpha parameter (little-endian f32, meaningful for center)
--- endif ---
var     var   STREAM         Arithmetic coded probability stream
EOF-4   4B    CRC32          CRC-32 (IEEE) of original uncompressed data
```

Header size: **13 bytes** (default), +12 (custom HP), +5 (emb-surgery), +1 (shuffle), +1 (delta),
+1 (adaptive-skip), +1 (backbone). Max header: 34 bytes (all flags set). CRC-32 trailer: 4 bytes.
Optional fields always appear in this order: HP, EMB, SHUFFLE, DELTA, SKIP, BACKBONE.

### FLAGS byte

| Bit | Mask   | Name            | Description                                |
|-----|--------|-----------------|--------------------------------------------|
| 0   | `0x01` | CUSTOM_HP       | Hyperparameters follow FLAGS (12 bytes)    |
| 1   | `0x02` | EMB_SURGERY     | Embedding surgery method follows FLAGS (5 bytes) |
| 2   | `0x04` | E8E9            | E8/E9 x86 preprocessing was applied (R-027) |
| 3   | `0x08` | SHUFFLE         | Byte-plane shuffle was applied (R-029, 1B stride follows) |
| 4   | `0x10` | DELTA           | Delta coding was applied (R-030, 1B stride follows) |
| 5   | `0x20` | ADAPTIVE_SKIP   | Adaptive RWKV skip threshold follows (1B, R-033) |
| 6   | `0x40` | BACKBONE        | Backbone ID follows (1 byte)               |
| 7   | `0x80` | ENHANCED_CM     | Entropy-adaptive mixer + SegmentModel (R-043/R-044); no payload |

All 8 FLAGS bits are assigned — new header fields need a new version (see v5).

When CUSTOM_HP is clear (FLAGS = 0x00), the decoder uses built-in defaults:
- BIAS_LR = 0.30
- NGRAM_SCALE = 0.50
- MIX_ETA = 0.01

### Embedding Surgery (EMB_SURGERY flag, R-010)

When FLAGS bit 1 (0x02) is set, 5 bytes follow (after custom HP if both set):

| Field      | Size | Description                                  |
|------------|------|----------------------------------------------|
| EMB_METHOD | 1B   | 0=none, 1=center, 2=norm, 3=spread          |
| EMB_ALPHA  | 4B   | f32 LE alpha parameter (default 0.3 for center) |

On decompression, the decoder reads `EMB_METHOD` and `EMB_ALPHA` from the header
and applies the matching embedding surgery automatically. CLI `--emb-surgery` flag
is overridden by the header value when present.

If the flag is absent (files without emb-surgery), the decoder uses embeddings as-is
unless the user explicitly passes `--emb-surgery` on the CLI (backward compatibility).

### Byte-Plane Shuffle (SHUFFLE flag, R-029)

When FLAGS bit 3 (0x08) is set, byte-plane shuffling was applied before encoding.
One additional byte follows in the header:

| Field  | Size | Description                         |
|--------|------|-------------------------------------|
| STRIDE | 1B   | Shuffle stride (2, 4, or 8 bytes)   |

**Transform**: For stride N, input bytes are transposed so that all byte-0's of
each N-byte group come first, then all byte-1's, etc. For float32 arrays (N=4),
this clusters exponent bytes (low entropy) together, improving compression.

**Auto-detection** (`--shuffle auto`, default): Applied when the content-type
detector (R-028) classifies input as `FloatArray`. Override with `--shuffle on`,
`--shuffle on:N` (custom stride), or `--shuffle off`.

**CRC note**: CRC covers the post-shuffle bytes (after all preprocessing).

### Delta Coding (DELTA flag, R-030)

When FLAGS bit 4 (0x10) is set, delta coding was applied before encoding.
One additional byte follows in the header:

| Field  | Size | Description                       |
|--------|------|-----------------------------------|
| STRIDE | 1B   | Delta stride (typically 1)        |

**Transform**: `out[i] = in[i] - in[i-stride]` (wrapping subtraction).
First `stride` bytes are copied unchanged.

**Auto-detection** (`--delta auto`, default): Applied when byte-plane shuffle
is also applied (FloatArray pipeline). Override with `--delta on`, `--delta on:N`,
or `--delta off`.

**Decode order**: delta decode → unshuffle → E8E9 decode.

### Backbone ID (BACKBONE flag)

When FLAGS bit 6 (0x40) is set, 1 byte follows identifying the neural backbone
used during compression:

| Field       | Size | Description                              |
|-------------|------|------------------------------------------|
| BACKBONE_ID | 1B   | Neural backbone identifier (see table)   |

| ID     | Backbone | Notes                                     |
|--------|----------|-------------------------------------------|
| `0x00` | RWKV-7   | Default backbone, backward compatible     |
| `0x01` | Mamba    | Mamba SSM backbone (R-035)                |
| `0x02` | MambaByte | Byte-level Mamba (R-082)                 |
| `0x03` | Plugin   | External GPU plugin backend (R-090)       |
| `0x04`–`0xFF` | (reserved) | Decoder rejects unknown IDs      |

When the flag is not set, RWKV-7 is assumed. This maintains full backward
compatibility with existing v4 files — all files produced before this flag
was introduced used RWKV-7 exclusively.

**Decompression**: The decoder selects the matching backbone and weights based
on `BACKBONE_ID`. Using the wrong backbone produces incorrect predictions and
a CRC-32 mismatch.

### E8E9 Preprocessing (E8E9 flag, R-027)

When FLAGS bit 2 (0x04) is set, the compressor applied E8/E9 x86 preprocessing
before encoding. No additional data is stored in the header — the flag merely
signals the decompressor to apply the inverse transform after decoding.

**Compression**: x86 CALL (0xE8) and JMP (0xE9) relative addresses are converted
to absolute addresses, clustering targets to common offsets and improving
compression of executable binaries.

**Decompression**: After CRC-32 verification (on the transformed data), the
inverse transform converts absolute addresses back to relative. The final output
is byte-identical to the original input.

**Auto-detection** (`--e8e9 auto`, default): The compressor scans the first 8KB
for E8/E9 density ≥0.5%, ≥3 occurrences, and >50% in-range targets. Override
with `--e8e9 on` (force) or `--e8e9 off` (disable).

**CRC note**: CRC-32 covers the post-E8E9-transformed bytes (what enters the
arithmetic coder), NOT the original input. The decompressor verifies CRC before
applying the inverse transform.

## Format Layout (v5 — weight profile, R-099)

v5 is v4 plus a 1-byte **PROFILE** immediately after FLAGS. It records the
numeric weight profile of the Mamba-family backbone and the **numeric contract**
(R-104) — both change the probabilities and therefore the coded stream.

- R-099 encoders wrote v5 only for non-F32 profiles (contract bits = 0).
- Since R-104 F1 the encoder writes v5 for **every** archive (F32 included) with
  `NUMERIC_CONTRACT = 1`. F32 archives are therefore one byte larger than the
  former v4 output.
- A v4-only decoder fed a v5 file fails up front with `unsupported AZTH version 5`;
  an R-099 decoder fed a contract-1 file fails with `PROFILE has reserved bits set`
  (never with a CRC mismatch).

```
Offset  Size  Field          Description
------  ----  -----          -----------
0x00    8B    MAGIC          "AZTH\x05\x00\x00\x00" (version 5)
0x08    4B    TOTAL_BYTES    Original file size (little-endian u32)
0x0C    1B    FLAGS          Same bits and semantics as v4
0x0D    1B    PROFILE        Weight profile (see below)
[0x0E]  1B    RESET_LOG2     Only if PROFILE bit 6 is set (R-108): backbone state reset period = 2^RESET_LOG2
                             forwards, 8 <= RESET_LOG2 <= 20 (256 .. 1,048,576). Written by default for the
                             Mamba family (12 = 4096); never for RWKV/plugin.
var     var   [optional]     Exactly the v4 optional fields, same order:
                             [HP 12B] [EMB 5B] [SHUFFLE 1B] [DELTA 1B] [SKIP 1B] [BACKBONE 1B]
var     var   STREAM         Arithmetic coded probability stream
EOF-4   4B    CRC32          CRC-32 (IEEE) of the bytes that entered the coder
```

Header size: **14 bytes** minimum (16 for a default Mamba archive: +1 RESET_LOG2 +1 BACKBONE), 36 bytes maximum.

### PROFILE byte

| Bits | Field             | Values |
|------|-------------------|--------|
| 0–3  | QUANT             | `0` = F32, `1` = Q8 (recipe below), `2` = Q6 (reserved, rejected), `3`–`15` reserved (rejected) |
| 4–5  | NUMERIC_CONTRACT  | `0` = legacy libm, `1` = fmath-v1 (default since R-104 F1), `2` = fma-v1 (R-107, Mamba family only, opt-in), `3` reserved (rejected) |
| 6    | RESET_PERIOD      | `1` = a RESET_LOG2 byte follows PROFILE (R-108). Pre-R-108 decoders reject it as a reserved bit, never with a CRC mismatch |
| 7    | —                 | Reserved, must be 0 (decoder rejects) |

Values written by the current encoder: Mamba family `0x50` (F32) and `0x51` (Q8) by default (bit 6 set, RESET_LOG2 = 12),
`0x60` / `0x61` with `compress --numeric fma`; `0x10` / `0x11` with `--reset-every 0`; RWKV `0x10`.

### RESET_LOG2 (R-108)

Mamba-130M's recurrent state degenerates to near-uniform logits beyond ~16-18K tokens of context on some inputs
(reproduced with the HF reference implementation, 2026-11-10; `data/benchmark/results/kaggle-fullsize/`). The
encoder zeroes the backbone conv + SSM state every 2^RESET_LOG2 forwards (the context-mixing models keep running)
and records the period here so the decoder replays it exactly. Measured: enwik8 100 KB 1.6327 -> 0.9951 bpb with 4096;
all five whole industry files go from losing to beating brotli-11/xz. Inputs shorter than the period produce the same
coded stream as before R-108 (+1 header byte). `compress --reset-every N` (0 = never, power of two 256..1048576)
overrides the default; the flag applies to the Mamba family only. In lockstep batches every stream carries its own
period (from its header on decode), so batch output stays byte-identical to single-file output.
A non-F32 profile is only valid for the Mamba / MambaByte backbones;
the decoder rejects it for RWKV (RWKV's own `--quant` is not part of the format).

### Numeric contract (R-104)

The contract names the implementation of the transcendental functions whose
results enter the coded probabilities. All other arithmetic on that path is
IEEE `+ − × ÷ sqrt` (correctly rounded, identical everywhere) or the portable
f32 kernels of R-013b.

| Code | Name        | `exp`/`ln` in `cdf_from_logits` and the CM entropy gate | Portability |
|------|-------------|------------------------------------------|-------------|
| 0    | legacy libm | `f64::exp` / `f64::ln` of the platform libm (MSVC UCRT, glibc, …) | bit-exact only where the decoder's libm equals the encoder's (UCRT and glibc differ in 0.5% of `exp` values by 1 ulp; masked in practice, latent risk ~1e-8–1e-6 per token) |
| 1    | fmath-v1    | `src/domain/fmath.rs`: port of musl `exp.c`/`log.c` (ARM optimized-routines, MIT), non-FMA path, pure IEEE f64 arithmetic, < 0.52 ulp | identical on every IEEE-754 platform; golden values and a golden sweep hash in the unit tests |
| 2    | fma-v1      | same as contract 1 (fmath-v1), **plus** fused multiply-add GEMV/GEMM kernels in the Mamba backbone (below) | as contract 1; the GEMV/GEMM is in addition SIMD/scalar tier-invariant (scalar `f32::mul_add`) |

Interpretation by version: v2, v3, v4 and v5 with bits 4–5 = 0 → contract 0.
A decoder selects the implementation from the header, so legacy archives keep
being decoded with the platform libm (the only bit-exact choice on the platform
that wrote them). An unknown contract (3) is rejected while parsing the
header, before the model is loaded; so is contract 2 with a non-Mamba backbone
(RWKV / plugin). Decoders older than R-107 reject contract 2 the same way
(`numeric contract 2 not supported by this build`), never with a CRC mismatch.

Contracts 0 and 1 do **not** cover the SIMD tier: F32 GEMV / SSM accumulation
order still differs between the AVX2 and scalar kernels (R-104 F2).

### Numeric contract 2: fma-v1 GEMV/GEMM (R-107)

Covers every matrix-vector product of the Mamba backbone: `in_proj`, `x_proj`,
`dt_proj`, `out_proj` and the embedding / LM head (F32 and Q8 weights, single
and lockstep multi-stream). Each `acc + w·x` of contract 1 becomes one IEEE 754
`fusedMultiplyAdd(w, x, acc)` (single rounding); the order is unchanged:

1. per output row (F32) or per (row, column group) (Q8): 8 lanes; lane `l`
   accumulates columns `c ≡ l (mod 8)` of the 8-aligned body in column order,
   `acc_l = fma(w_c, x_c, acc_l)`, from +0;
2. horizontal sum `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` (plain adds);
3. sequential tail `s = fma(w_c, x_c, s)` over the remaining `K mod 8` columns;
4. Q8: `int8 → f32` exactly, row = `sum = fma(dot_g, scale_g, sum)` over the
   groups in order from +0 (same recipe as QUANT = 1).

Register tiling (rows × streams), thread count and lockstep stream count never
change the per-output order, hence never the bits. Not covered (unchanged
contract-1 code): conv1d, selective scan (`exp`), softplus, SiLU, norms,
residual adds — so contract 2 keeps the SIMD-tier rule for those phases.

Implementation: `src/domain/gemv_fma.rs` (AVX2+FMA `vfmadd231ps` kernels and a
scalar `f32::mul_add` emulation; unit-tested bit-identical, including 10^6
random triples with subnormals and exact cancellation, against both the hardware
scalar `vfmadd` and the MSVC UCRT `fmaf`). A decoder on a CPU with AVX2 but
without FMA (or with `REDUKT_NO_FMA=1`) uses the scalar kernels — bit-exact,
much slower — and prints a warning. Selection: `compress --numeric fma`
(default `fmath` = contract 1); decompression always follows the header. A
`decompress --input-dir` batch must not mix contract 2 with contracts 0/1 for
Mamba archives (rejected before loading the model). The HTTP API (`api`) writes
and loads contract 1 only.

### Q8 profile recipe (QUANT = 1)

Part of the format contract — any change needs a new QUANT code.

- **Quantized** (weight-only, W8A32): every layer's `in_proj` and `out_proj`, and
  the embedding / LM head (tied in Mamba-130M: one int8 matrix; the embedding
  lookup dequantizes its row). Untied heads (MambaByte) quantize both.
- **Scheme**: symmetric int8, one f32 scale per 64 consecutive columns of a row
  (whole row if 64 does not divide the row length):
  `scale = max|w| / 127` (1.0 for an all-zero group),
  `q = clamp(round(w * (1/scale)), -127, 127)`.
- **Kept f32**: `A_log`, `D`, `dt_proj` (weight and bias), `x_proj`, `conv1d`,
  all norms. Activations are never quantized.
- **GEMV order**: per (row, group), 8 f32 lanes (lane `l` takes columns
  `c ≡ l mod 8`) accumulate `f32(q) * x` with separate multiply and add (no FMA),
  then horizontal sum `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`, then a sequential
  tail; the row result is `0 + Σ_groups dot_g * scale_g` in group order. The
  AVX2 and scalar Q8 GEMV kernels implement this order exactly (bit-identical).
  The archive as a whole still needs the same SIMD tier on both sides, because
  the f32 parts (x_proj/dt_proj GEMV, SSM `exp`, SiLU) keep the F32 tier rule.

Measured (Mamba-130M, 10 KB, vs F32): enwik8 +0.6%, facturae XML +1.2%,
apache log +1.1% BPB; weight traffic 516.5 → 153.6 MB/token.

### Decoder rules (v5)

1. `validate_magic` accepts v2–v5 (v5 minimum 14 bytes).
2. PROFILE is validated (reserved bits, reserved/unknown quant codes, unknown
   numeric contracts) while parsing.
3. The header profile is authoritative. `decompress --quant` is optional; if
   given and different from the header (e.g. `--quant f32` on a Q8 archive) the
   decoder stops with `weight profile mismatch ...` **before** loading the model
   or decoding — no output file is written.

### Key change from v3: TOKEN_COUNT removed

v4 eliminates TOKEN_COUNT from the header. The decoder terminates by byte count:
it decodes tokens and detokenizes them interleaved, stopping when
`output_bytes.len() >= TOTAL_BYTES`. This saves 4 bytes in the header and
simplifies the format (token count was redundant — deterministic tokenization
means byte count implies token count).

### Version History

| Version | Magic byte 4 | Header bytes | Changes |
|---------|-------------|-------------|---------|
| v2      | `\x02`      | 28          | Added BIAS_LR, NGRAM_SCALE, MIX_ETA to header |
| v3      | `\x03`      | 28          | Added CRC-32 trailer, hyperparameter validation |
| v4      | `\x04`      | 13–34       | Removed TOKEN_COUNT, FLAGS (HP, emb-surgery, E8E9, shuffle, delta, skip, backbone, CM) |
| v5      | `\x05`      | 14–36       | v4 + PROFILE byte after FLAGS (Mamba weight profile, R-099; numeric contract, R-104; contract 2 fma-v1, R-107; RESET_LOG2 byte when PROFILE bit 6, R-108) |

The decoder accepts v2, v3, v4 and v5 files:
- **v2**: no CRC check (backward compatible)
- **v3**: CRC-32 verified after decompression
- **v4**: CRC-32 verified, compact header, byte-count termination
- **v5**: as v4, plus the weight profile and the numeric contract are taken from PROFILE

The encoder writes **v5** for every archive since R-104 F1 (PROFILE `0x10` for
F32, `0x11` for Q8; `0x20` / `0x21` with `--numeric fma`, R-107; since R-108 the Mamba
family sets bit 6 and appends RESET_LOG2, i.e. `0x50` / `0x51` by default). Before that,
F32 archives were written as v4. A timing-based
auto-selection of the weight profile is never used: it would make the bitstream
machine-dependent.

## Format Layout (v3 — legacy)

```
Offset  Size  Field          Description
------  ----  -----          -----------
0x00    8B    MAGIC          "AZTH\x03\x00\x00\x00" (version 3)
0x08    4B    TOTAL_BYTES    Original file size (little-endian u32)
0x0C    4B    TOKEN_COUNT    Number of tokens after tokenization (little-endian u32)
0x10    4B    BIAS_LR        BiasHead learning rate (little-endian f32)
0x14    4B    NGRAM_SCALE    N-gram logit bias scale (little-endian f32)
0x18    4B    MIX_ETA        Adaptive mixer learning rate (little-endian f32)
0x1C    var   STREAM         Arithmetic coded probability stream
EOF-4   4B    CRC32          CRC-32 (IEEE) of original uncompressed data
```

Total header size: 28 bytes (0x1C). CRC-32 trailer: 4 bytes.

### TOTAL_BYTES

Original file size in bytes, stored as unsigned 32-bit little-endian.
Maximum supported file size: 4,294,967,295 bytes (~4 GB).

**Validation**: The compressor rejects inputs exceeding `u32::MAX` with an
explicit error before encoding begins.

### Hyperparameters (BIAS_LR, NGRAM_SCALE, MIX_ETA)

Three IEEE 754 single-precision floats stored as little-endian.
These are the online-learning rates used during compression:

| Field       | Default | Description                          |
|-------------|---------|--------------------------------------|
| BIAS_LR     | 0.30    | BiasHead per-symbol learning rate    |
| NGRAM_SCALE | 0.50    | N-gram logit bias scaling factor     |
| MIX_ETA     | 0.01    | AdaptiveMixer weight learning rate   |

**Validation**: All three values must be finite (`is_finite()`).
NaN or Infinity values cause immediate rejection.

### STREAM

Schindler-style carry-propagation range coder output.
- 24-bit CDF precision (TOP = 2^24 = 16,777,216)
- u64 low register (encoder), u32 code register (decoder)
- Each symbol encoded with CDF derived from model predictions
- First token encoded with uniform CDF (1/vocab_size per symbol)
- Subsequent tokens: ensemble logits -> softmax -> CDF

### CRC32 (v3+ only)

CRC-32 (IEEE 802.3, polynomial 0xEDB88320 reflected) of the **original
uncompressed data**, stored as 4-byte little-endian after STREAM.

Located at: `file_size - 4` bytes from the start.
On decompression, the CRC is recomputed from the restored output and
compared against the stored value. Mismatch → error, no output written.

This detects:
- Bit flips or corruption in the compressed stream
- Truncated or extended files
- Model/weight mismatch during decompression
- Arithmetic decoder state divergence

## Compression Pipeline

```
1. Read input file (raw bytes)
2. E8E9 preprocessing (if auto-detected or --e8e9 on): relative→absolute addresses
3. Compute CRC-32 of (possibly transformed) bytes
4. Tokenize with World tokenizer (65K vocab)
5. For token t = 0:
     Encode with uniform CDF
6. For token t > 0:
     a. RWKV-7 forward pass -> logits[vocab_size]
     b. TokenNgram -> n-gram bias[vocab_size]
     c. BiasHead -> adaptive bias[vocab_size]
     d. AdaptiveMixer -> ensemble_logits = logits + w_ng*ngram + w_b*bias
     e. softmax(ensemble_logits) -> probabilities
     f. CDF from probabilities (min freq 1 per symbol)
     g. RangeEncoder.encode_symbol(token, CDF)
7. Update all models with observed token
8. RangeEncoder.finish() -> compressed bytes
9. Write: MAGIC + TOTAL_BYTES + FLAGS [+ HP] [+ EMB_SURGERY] + STREAM + CRC32
```

## Decompression Pipeline

### v4 (byte-count driven)

```
1. Read AZTH file, validate MAGIC (accept v2, v3, v4 or v5)
2. Parse header (TOTAL_BYTES, FLAGS, [v5: PROFILE], optional HP, optional emb-surgery, ...)
2a. Resolve weight profile from the header (reject a conflicting `--quant`)
2b. If emb-surgery stored in header: apply matching surgery to model
3. Validate hyperparameters if custom (reject NaN/Inf)
4. Extract STREAM and CRC32
5. Initialize RangeDecoder with STREAM
6. Loop:
     a. If t == 0: decode with uniform CDF
     b. If t > 0: ensemble logits -> CDF -> decode symbol
     c. Detokenize decoded token -> append bytes
     d. If output_bytes.len() >= TOTAL_BYTES: stop
     e. Update all models with decoded token
7. Verify output length == TOTAL_BYTES (error on mismatch)
8. Verify CRC-32 of output == stored CRC (error on mismatch)
9. If E8E9 flag set: apply inverse transform (absolute→relative addresses)
10. Write output file
```

### Streaming Mode (R-014)

When `--streaming` is specified (v4 files only):
- Output bytes are written progressively to disk via `BufWriter`
- CRC-32 is computed incrementally (not from final buffer)
- Peak memory reduced from O(file_size) to O(buffer_size)
- On CRC mismatch: output file is deleted, error reported
- Without `--streaming`: behavior unchanged (full buffer + verify + write)

### v2/v3 (token-count driven, legacy)

```
1. Read AZTH file, validate MAGIC
2. Parse TOTAL_BYTES, TOKEN_COUNT, hyperparameters
3. Extract STREAM and CRC32 (v3) or STREAM only (v2)
4. Decode TOKEN_COUNT symbols sequentially
5. Detokenize all decoded tokens -> bytes
6. Verify length and CRC
7. Write output file
```

## Determinism Guarantee

Decompression produces byte-identical output if and only if:
1. Same RWKV-7 model weights (exact floating-point values)
2. Same World tokenizer vocabulary file
3. Same hyperparameters (stored in header; ngram order=4 is fixed)
4. Same forward pass implementation (floating-point order matters)
5. Same embedding surgery method (stored in v4 header since R-010)
6. Same weight profile (stored in the v5 PROFILE byte since R-099; v2–v4 imply F32)
7. Same numeric contract (v5 PROFILE bits 4–5 since R-104). Contract 1 (fmath-v1)
   is OS/libm-independent; contract 0 (v2–v4 and R-099 v5) relies on the
   platform libm `exp`; contract 2 (fma-v1, R-107) is contract 1 plus FMA
   GEMV/GEMM kernels, whose scalar fallback (`f32::mul_add`) is bit-identical
8. Same SIMD tier (`REDUKT_NO_SIMD=1` must match on both sides), for every
   profile: the AVX2 (no-FMA) and scalar f32 kernels (GEMV accumulation order,
   SSM `exp`, SiLU) are NOT bit-compatible. In the Q8 profile the int8 GEMV
   itself is tier-invariant, but x_proj/dt_proj/SSM remain f32 and tier-bound.

The number of GEMV threads (`REDUKT_THREADS`) never affects the output: work is
partitioned by output rows / SSM channels and every row or channel is computed
by the same single-thread kernel (R-098).

This is inherent to ALL neural compression schemes (cmix, NNCP, etc.).

## Limitations

- Maximum file size: ~4 GB (u32 TOTAL_BYTES)
- No random access (sequential decode only)
- No multi-file archive support
- Weights must be distributed separately (~400 MB for RWKV-7 0.1B)

## Security Considerations

- No encryption (compress BEFORE encrypt; encrypted data is incompressible)
- No authentication (wrap with authenticated encryption if needed)
- MAGIC validation prevents processing random files
- CRC-32 detects accidental corruption (NOT cryptographic — use HMAC for tamper detection)
- TOTAL_BYTES capped at u32 prevents memory exhaustion
- Hyperparameter validation rejects NaN/Inf (prevents numeric corruption)
- FLAGS validation rejects unknown bits (forward compatibility)
- PROFILE validation rejects reserved bits, unknown/reserved quant codes and
  unknown numeric contracts (v5)
- RAM monitor aborts if available memory is insufficient for model

## MIME Type (Proposed)

```
application/x-azth
```

File extension: `.azth`
