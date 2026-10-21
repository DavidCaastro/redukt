# AZTH Format Specification v1

## Purpose

AZTH is a lossless compression format that encodes data using neural network
probability predictions combined with arithmetic coding. Decompression requires
the identical model, weights, and hyperparameters used during compression.

## Format Layout

```
Offset  Size  Field          Description
------  ----  -----          -----------
0x00    8B    MAGIC          "AZTH\x01\x00\x00\x00" (version 1)
0x08    4B    TOTAL_BYTES    Original file size (little-endian u32)
0x0C    4B    TOKEN_COUNT    Number of tokens after tokenization (little-endian u32)
0x10    var   STREAM         Arithmetic coded probability stream
```

### Magic Bytes

```
41 5A 54 48 01 00 00 00
A  Z  T  H  v1 (reserved)
```

Bytes 4-7 are reserved for future format versions. Currently must be `01 00 00 00`.

### TOTAL_BYTES

Original file size in bytes, stored as unsigned 32-bit little-endian.
Maximum supported file size: 4,294,967,295 bytes (~4 GB).

### TOKEN_COUNT

Number of tokens produced by the World tokenizer (65,536 vocab).
Required for decompression to know when to stop decoding.

### STREAM

Schindler-style carry-propagation range coder output.
- 24-bit CDF precision (TOP = 2^24 = 16,777,216)
- u64 low register (encoder), u32 code register (decoder)
- Each symbol encoded with CDF derived from model predictions
- First token encoded with uniform CDF (1/vocab_size per symbol)
- Subsequent tokens: ensemble logits -> softmax -> CDF

## Compression Pipeline

```
1. Read input file (raw bytes)
2. Tokenize with World tokenizer (65K vocab)
3. For token t = 0:
     Encode with uniform CDF
4. For token t > 0:
     a. RWKV-7 forward pass -> logits[vocab_size]
     b. TokenNgram -> n-gram bias[vocab_size]
     c. BiasHead -> adaptive bias[vocab_size]
     d. AdaptiveMixer -> ensemble_logits = logits + w_ng*ngram + w_b*bias
     e. softmax(ensemble_logits) -> probabilities
     f. CDF from probabilities (min freq 1 per symbol)
     g. RangeEncoder.encode_symbol(token, CDF)
5. Update all models with observed token
6. RangeEncoder.finish() -> compressed bytes
7. Write: MAGIC + TOTAL_BYTES + TOKEN_COUNT + compressed bytes
```

## Decompression Pipeline

```
1. Read AZTH file, validate MAGIC
2. Parse TOTAL_BYTES, TOKEN_COUNT
3. Initialize RangeDecoder with STREAM
4. For token t = 0:
     Decode with uniform CDF
5. For token t > 0:
     a. RWKV-7 forward pass (using previously decoded token) -> logits
     b. TokenNgram, BiasHead, AdaptiveMixer -> ensemble_logits
     c. CDF from ensemble_logits
     d. RangeDecoder.decode_symbol(CDF) -> token
     e. Update all models with decoded token
6. Detokenize: World tokenizer decode all tokens -> bytes
7. Verify output length == TOTAL_BYTES
8. Write output file
```

## Determinism Guarantee

Decompression produces byte-identical output if and only if:
1. Same RWKV-7 model weights (exact floating-point values)
2. Same World tokenizer vocabulary file
3. Same hyperparameters: ngram order=4, scale=0.5, bias_lr=0.30, mix_eta=0.01
4. Same forward pass implementation (floating-point order matters)

This is inherent to ALL neural compression schemes (cmix, NNCP, etc.).

## Limitations

- Maximum file size: ~4 GB (u32 TOTAL_BYTES)
- No random access (sequential decode only)
- No multi-file archive support
- No built-in integrity check (consider external SHA-256)
- Weights must be distributed separately (~400 MB for RWKV-7 0.1B)

## Security Considerations

- No encryption (compress BEFORE encrypt; encrypted data is incompressible)
- No authentication (wrap with authenticated encryption if needed)
- MAGIC validation prevents processing random files
- TOTAL_BYTES capped at u32 prevents memory exhaustion
- RAM monitor aborts if available memory is insufficient for model

## MIME Type (Proposed)

```
application/x-azth
```

File extension: `.azth`
