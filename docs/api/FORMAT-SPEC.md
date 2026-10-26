# AZTH Format Specification v3

## Purpose

AZTH is a lossless compression format that encodes data using neural network
probability predictions combined with arithmetic coding. Decompression requires
the identical model and weights; hyperparameters are stored in the header.

## Format Layout (v3)

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

### Version History

| Version | Magic byte 4 | Changes |
|---------|-------------|---------|
| v2      | `\x02`      | Added BIAS_LR, NGRAM_SCALE, MIX_ETA to header |
| v3      | `\x03`      | Added CRC-32 trailer, hyperparameter validation |

The decoder accepts both v2 and v3 files:
- **v2**: no CRC check (backward compatible, warns about missing integrity)
- **v3**: CRC-32 verified after decompression (error on mismatch)

### Magic Bytes

```
41 5A 54 48 03 00 00 00
A  Z  T  H  v3 (reserved)
```

Byte 4 is the format version. Decoder accepts `02` or `03`.

### TOTAL_BYTES

Original file size in bytes, stored as unsigned 32-bit little-endian.
Maximum supported file size: 4,294,967,295 bytes (~4 GB).

### TOKEN_COUNT

Number of tokens produced by the World tokenizer (65,536 vocab).
Required for decompression to know when to stop decoding.

### Hyperparameters (BIAS_LR, NGRAM_SCALE, MIX_ETA)

Three IEEE 754 single-precision floats stored as little-endian.
These are the online-learning rates used during compression:

| Field       | Default | Description                          |
|-------------|---------|--------------------------------------|
| BIAS_LR     | 0.30    | BiasHead per-symbol learning rate    |
| NGRAM_SCALE | 0.50    | N-gram logit bias scaling factor     |
| MIX_ETA     | 0.01    | AdaptiveMixer weight learning rate   |

Decompression reads these values from the header and initializes the
ensemble with the exact same parameters used during compression.
This guarantees lossless roundtrip even with non-default hyperparameters.

**Validation (v3)**: All three values must be finite (`is_finite()`).
NaN or Infinity values cause immediate rejection of the file.

### STREAM

Schindler-style carry-propagation range coder output.
- 24-bit CDF precision (TOP = 2^24 = 16,777,216)
- u64 low register (encoder), u32 code register (decoder)
- Each symbol encoded with CDF derived from model predictions
- First token encoded with uniform CDF (1/vocab_size per symbol)
- Subsequent tokens: ensemble logits -> softmax -> CDF

### CRC32 (v3 only)

CRC-32 (IEEE 802.3, polynomial 0xEDB88320 reflected) of the **original
uncompressed data**, stored as 4-byte little-endian after STREAM.

Located at: `file_size - 4` bytes from the start.
The CRC is computed over the exact bytes that were input to compression.
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
8. Write: MAGIC + header + STREAM + CRC32
```

## Decompression Pipeline

```
1. Read AZTH file, validate MAGIC (accept v2 or v3)
2. Parse TOTAL_BYTES, TOKEN_COUNT
3. Validate hyperparameters (reject NaN/Inf)
4. Extract STREAM and CRC32 (v3) or STREAM only (v2)
5. Initialize RangeDecoder with STREAM
6. For token t = 0:
     Decode with uniform CDF
7. For token t > 0:
     a. RWKV-7 forward pass (using previously decoded token) -> logits
     b. TokenNgram, BiasHead, AdaptiveMixer -> ensemble_logits
     c. CDF from ensemble_logits
     d. RangeDecoder.decode_symbol(CDF) -> token
     e. Update all models with decoded token
8. Detokenize: World tokenizer decode all tokens -> bytes
9. Verify output length == TOTAL_BYTES (error on mismatch)
10. If v3: verify CRC-32 of output == stored CRC (error on mismatch)
11. Write output file
```

## Determinism Guarantee

Decompression produces byte-identical output if and only if:
1. Same RWKV-7 model weights (exact floating-point values)
2. Same World tokenizer vocabulary file
3. Same hyperparameters (stored in header since v2; ngram order=4 is fixed)
4. Same forward pass implementation (floating-point order matters)

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
- RAM monitor aborts if available memory is insufficient for model

## MIME Type (Proposed)

```
application/x-azth
```

File extension: `.azth`
