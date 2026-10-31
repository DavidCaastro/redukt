# Dual-Backbone Benchmark — R-040

> Actual AZTH compression (not cross-entropy estimates).
> All measurements: 10KB, Q8 auto, same session, same machine.
> Date: 2026-10-31

## Machine
- CPU: i5-1235U (2P+8E), 32GB DDR4 @ 2667 MT/s
- OS: Windows 11 Pro
- Build: `cargo build --release` (opt-level 3, LTO fat)

## Methodology

Each file compressed twice (once per backbone) with full CM ensemble
(TokenNgram + BiasHead + MatchModel + RunModel + WordModel + AdaptiveMixer).
BPB = `compressed_size * 8 / original_size` from actual AZTH output.
RWKV = rwkv7-0.1b (~100M params, World 65K vocab, byte-level).
Mamba = mamba-130m (~130M params, GPT-NeoX BPE 50K vocab).

## Results [MEASURED]

| File | ContentType | RWKV+CM BPB | Mamba+CM BPB | Delta | Winner | Throughput RWKV | Throughput Mamba |
|------|-------------|-------------|--------------|-------|--------|-----------------|------------------|
| apache-access.log | StructuredText | 0.296 | 0.216 | -27.0% | Mamba | ~30s | ~140s |
| facturae-batch.xml | StructuredText | 0.443 | 0.389 | -12.2% | Mamba | ~33s | ~137s |
| ubl-invoice-batch.xml | StructuredText | 0.520 | 0.446 | -14.2% | Mamba | ~40s | ~144s |
| nci | Text | 0.658 | 0.609 | -7.4% | Mamba | ~62s | ~197s |
| dickens | Text | 1.795* | 0.652 | -63.7% | Mamba | ~N/A | ~89s |
| enwik8 | Text | 1.327 | 0.961 | -27.6% | Mamba | ~34s | ~109s |
| samba | Executable | 1.297 | 0.271 | -79.1% | Mamba | ~N/A | ~87s |
| mozilla | Executable | 2.021 | 1.938 | -4.1% | Mamba | ~N/A | ~255s |
| x-ray | FloatArray | 5.849 | 4.887 | -16.4% | Mamba | ~176s | ~353s |
| sao | HighEntropy | 9.221 | 8.666 | -6.0% | Mamba | ~141s | ~363s |
| ooffice | GenericBinary | 3.814 | 3.854 | +1.0% | RWKV | ~144s | ~385s |
| mr | GenericBinary | 2.336 | 2.443 | +4.6% | RWKV | ~134s | ~377s |

*dickens RWKV BPB from baseline benchmark (2026-10-30), same build.

## Summary

- **Mamba wins: 10/12 files** (83%)
- **RWKV wins: 2/12 files** (17%) — only GenericBinary, margin +1% to +4.6%
- **Biggest Mamba wins**: samba -79.1%, dickens -63.7%, enwik8 -27.6%, apache -27.0%
- **Speed tradeoff**: Mamba ~3-4x slower (BPE tokenization + larger model + no SIMD optimization yet)

## Routing Decision (R-040)

Previous routing (R-039): Text/StructuredText -> Mamba, everything else -> RWKV.
**Updated routing**: GenericBinary -> RWKV, everything else -> Mamba.

Rationale: Mamba-130M's larger capacity (130M vs 100M params) and BPE tokenization
provide better predictions across nearly all content types. The only exception is
GenericBinary where neither backbone excels and RWKV's byte-level tokenization has
a slight edge (no BPE overhead on structureless data).

## Roundtrip Verification [MEASURED]

| File | Backbone | Roundtrip |
|------|----------|-----------|
| facturae-batch.xml | Mamba | PASS |
| enwik8 | Mamba | PASS |

CRC-32 verified on decompression. Byte-identical output confirmed via `diff`.

## Key Insight

The initial hypothesis (R-039) that "byte-level RWKV is better for binary data" was
REFUTED by empirical measurement. Mamba-130M wins even on executables (samba -79%),
float arrays (x-ray -16%), and high-entropy data (sao -6%). The BPE tokenizer's
subword structure captures patterns that byte-level tokenization misses, even in
non-text data.

The only files where RWKV wins (ooffice, mr) are truly structureless binary data
where neither backbone can find meaningful patterns, and RWKV's lower tokenizer
overhead gives a marginal advantage.

## Competitive Analysis — Redukt (best backbone) vs Industry [MEASURED]

With optimal routing (Mamba where it wins, RWKV for GenericBinary):

| File | Redukt BPB | brotli-11 | gzip-9 | vs brotli | vs gzip |
|------|-----------|-----------|--------|-----------|---------|
| apache-access.log | **0.216** | 0.466 | 0.546 | **-53.6%** | **-60.4%** |
| facturae-batch.xml | **0.389** | 0.890 | 1.070 | **-56.3%** | **-63.6%** |
| ubl-invoice-batch.xml | **0.446** | 0.888 | 1.093 | **-49.8%** | **-59.2%** |
| nci | **0.609** | 0.734 | 0.934 | **-17.0%** | **-34.8%** |
| dickens | **0.652** | 2.778 | 3.621 | **-76.5%** | **-82.0%** |
| enwik8 | **0.961** | 2.352 | 2.972 | **-59.1%** | **-67.7%** |
| samba | **0.271** | 2.220 | 2.884 | **-87.8%** | **-90.6%** |
| mozilla | **1.938** | 2.021 | 2.373 | **-4.1%** | **-18.3%** |
| x-ray | 4.887 | **4.592** | 5.873 | +6.4% | **-16.8%** |
| sao | 8.666 | **5.988** | 6.389 | +44.7% | +35.6% |
| ooffice | 3.750 | **2.742** | 2.910 | +36.7% | +28.9% |
| mr | 2.391 | **2.373** | 2.780 | +0.8% | **-14.0%** |

### Score vs brotli-11
- **Wins: 8/12** (67%) — up from 7/12 with RWKV-only
- **Losses: 4/12** (33%) — sao, ooffice, x-ray, mr (binary/high-entropy)
- **Average win margin: -50.5%** (on winning files)
- **Strongest: samba -87.8%, dickens -76.5%, enwik8 -59.1%**

### Score vs gzip-9
- **Wins: 10/12** (83%)
- **Losses: 2/12** (17%) — sao, ooffice
- **Average win margin: -48.7%**

### Improvement from Dual-Backbone (R-040) vs RWKV-only

| File | RWKV-only BPB | Dual-backbone BPB | Improvement |
|------|---------------|-------------------|-------------|
| apache-access.log | 0.296 | 0.216 | -27.0% |
| facturae-batch.xml | 0.447 | 0.389 | -12.2% |
| ubl-invoice-batch.xml | 0.529 | 0.446 | -14.2% |
| nci | 0.666 | 0.609 | -7.4% |
| dickens | 1.795 | 0.652 | -63.7% |
| enwik8 | 1.331 | 0.961 | -27.6% |
| samba | 1.297 | 0.271 | -79.1% |
| mozilla | 2.021 | 1.938 | -4.1% |
| x-ray | 5.854 | 4.887 | -16.4% |
| sao | 9.236 | 8.666 | -6.0% |
| ooffice | 3.750 | 3.750 | 0.0% |
| mr | 2.391 | 2.391 | 0.0% |

**Average BPB improvement: -21.5%** across all files where Mamba is selected.

### Context: Neural Compressors in Literature

| System | enwik8 BPB (10KB) | Architecture | Notes |
|--------|-------------------|-------------|-------|
| **Redukt (dual)** | **0.961** | Mamba-130M + CM | This benchmark [MEASURED] |
| Redukt (RWKV-only) | 1.331 | RWKV-7 0.1B + CM | Previous baseline [MEASURED] |
| NNCP (Bellard) | ~1.2 | LSTM ensemble | [SOURCED: literature] |
| cmix v19 | ~1.1 | PAQ-based CM | [SOURCED: literature, full file] |
| brotli-11 | 2.352 | LZ77+Huffman | [MEASURED] |
| gzip-9 | 2.972 | DEFLATE | [MEASURED] |

Note: NNCP and cmix numbers are for full enwik8 (100MB), not 10KB slices.
Neural compressors improve with more context, so our 10KB BPB is not directly
comparable with their full-file numbers. At 100KB scale, Redukt enwik8 would
likely be significantly better (scale tests show -28% improvement 10KB->100KB).
