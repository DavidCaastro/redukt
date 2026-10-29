# AZTH Format Specification

## Purpose

AZTH is a lossless compression format that encodes data using neural network
probability predictions combined with arithmetic coding. Decompression requires
the identical model and weights; hyperparameters are stored in the header.

## Format Layout (v4 — current)

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

Header size: **13 bytes** (default) or **18 bytes** (emb-surgery) or **25 bytes** (custom HP) or **30 bytes** (both).
CRC-32 trailer: 4 bytes. Total fixed overhead: **17 to 34 bytes**.

### FLAGS byte

| Bit | Mask   | Name            | Description                                |
|-----|--------|-----------------|--------------------------------------------|
| 0   | `0x01` | CUSTOM_HP       | Hyperparameters follow FLAGS (12 bytes)    |
| 1   | `0x02` | EMB_SURGERY     | Embedding surgery method follows FLAGS (5 bytes) |
| 2-7 | `0xFC` | (reserved)      | Must be 0; decoder rejects unknown flags   |

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
| v4      | `\x04`      | 13–30       | Removed TOKEN_COUNT, added FLAGS, conditional HP (-15 bytes), emb-surgery (R-010) |

The decoder accepts v2, v3, and v4 files:
- **v2**: no CRC check (backward compatible)
- **v3**: CRC-32 verified after decompression
- **v4**: CRC-32 verified, compact header, byte-count termination

The encoder always writes **v4**.

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
2. Compute CRC-32 of input
3. Tokenize with World tokenizer (65K vocab)
4. For token t = 0:
     Encode with uniform CDF
5. For token t > 0:
     a. RWKV-7 forward pass -> logits[vocab_size]
     b. TokenNgram -> n-gram bias[vocab_size]
     c. BiasHead -> adaptive bias[vocab_size]
     d. AdaptiveMixer -> ensemble_logits = logits + w_ng*ngram + w_b*bias
     e. softmax(ensemble_logits) -> probabilities
     f. CDF from probabilities (min freq 1 per symbol)
     g. RangeEncoder.encode_symbol(token, CDF)
6. Update all models with observed token
7. RangeEncoder.finish() -> compressed bytes
8. Write: MAGIC + TOTAL_BYTES + FLAGS [+ HP] [+ EMB_SURGERY] + STREAM + CRC32
```

## Decompression Pipeline

### v4 (byte-count driven)

```
1. Read AZTH file, validate MAGIC (accept v2, v3, or v4)
2. Parse header (TOTAL_BYTES, FLAGS, optional HP, optional emb-surgery)
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
9. Write output file
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
- RAM monitor aborts if available memory is insufficient for model

## MIME Type (Proposed)

```
application/x-azth
```

File extension: `.azth`
