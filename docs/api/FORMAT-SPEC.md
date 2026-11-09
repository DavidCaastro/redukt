# AZTH Format Specification

## Purpose

AZTH is a lossless compression format that encodes data using neural network
probability predictions combined with arithmetic coding. Decompression requires
the identical model and weights; hyperparameters are stored in the header.

## Format Layout (v4 — current default)

v4 is still what the encoder writes for the default **F32** weight profile.
Archives using another weight profile (R-099, Mamba Q8) are written as **v5**
(see below): identical layout plus one PROFILE byte after FLAGS.

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
numeric weight profile of the Mamba-family backbone, which changes the logits
and therefore the coded stream. The encoder writes v5 **only** when the profile
is not F32, so default (F32) archives stay byte-identical v4 and remain
readable by v4-only decoders; a v4-only decoder fed a v5 file fails up front
with `unsupported AZTH version 5` (never with a CRC mismatch).

```
Offset  Size  Field          Description
------  ----  -----          -----------
0x00    8B    MAGIC          "AZTH\x05\x00\x00\x00" (version 5)
0x08    4B    TOTAL_BYTES    Original file size (little-endian u32)
0x0C    1B    FLAGS          Same bits and semantics as v4
0x0D    1B    PROFILE        Weight profile (see below)
0x0E    var   [optional]     Exactly the v4 optional fields, same order:
                             [HP 12B] [EMB 5B] [SHUFFLE 1B] [DELTA 1B] [SKIP 1B] [BACKBONE 1B]
var     var   STREAM         Arithmetic coded probability stream
EOF-4   4B    CRC32          CRC-32 (IEEE) of the bytes that entered the coder
```

Header size: **14 bytes** minimum (15 for a Mamba archive: +1 BACKBONE), 35 bytes maximum.

### PROFILE byte

| Bits | Field  | Values |
|------|--------|--------|
| 0–3  | QUANT  | `0` = F32, `1` = Q8 (recipe below), `2` = Q6 (reserved, rejected), `3`–`15` reserved (rejected) |
| 4–7  | —      | Reserved, must be 0 (decoder rejects) |

`QUANT = 0` in a v5 header is valid (equivalent to v4) but the encoder never
writes it. A non-F32 profile is only valid for the Mamba / MambaByte backbones;
the decoder rejects it for RWKV (RWKV's own `--quant` is not part of the format).

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
2. PROFILE is validated (reserved bits, reserved/unknown codes) while parsing.
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
| v5      | `\x05`      | 14–35       | v4 + PROFILE byte after FLAGS (Mamba weight profile, R-099) |

The decoder accepts v2, v3, v4 and v5 files:
- **v2**: no CRC check (backward compatible)
- **v3**: CRC-32 verified after decompression
- **v4**: CRC-32 verified, compact header, byte-count termination
- **v5**: as v4, plus the weight profile is taken from PROFILE

The encoder writes **v4** for the F32 profile (default) and **v5** otherwise
(`compress --backbone mamba --quant q8`). `--quant auto` / unspecified selects
F32 for Mamba: a timing-based auto-selection would make the bitstream
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
7. Same SIMD tier (`REDUKT_NO_SIMD=1` must match on both sides), for every
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
- PROFILE validation rejects reserved bits and unknown/reserved quant codes (v5)
- RAM monitor aborts if available memory is insufficient for model

## MIME Type (Proposed)

```
application/x-azth
```

File extension: `.azth`
