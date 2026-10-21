# Limitations & Mitigation Strategies

## Critical: Throughput Constraint

### The Problem

Redukt processes at **~62 B/s** on the benchmark hardware (i5-1235U).
This is **~6,000x slower than brotli-11** and **~800,000x slower than gzip-9**.

### Impact by Use Case

| Use case | Daily volume | Time at 62 B/s | Cores needed (24h) |
|---|---|---|---|
| E-invoicing (50M docs/yr) | 41 GB | 7.7 days | 8 |
| E-invoicing (Edicom) | 82 GB | 15.3 days | 16 |
| PACS (SAS 50 hospitals) | 2.05 TB | 383 days | 384 |
| Log archival (bank) | 2-5 TB | 373-933 days | 373-933 |

### Mitigation: Two-Tier Architecture

Real-time compression is not viable. The product operates as a **batch archival tier**:

```
Ingest -> zstd (fast, inline) -> hot storage
                                      |
                              nightly batch job
                                      |
                              Redukt re-compress -> cold/archive storage
```

This is the same model used by:
- AWS S3 Intelligent-Tiering (auto-moves to cheaper tiers)
- Cribl (reduces data before archival)
- PetaGene (batch recompression of genomic archives)

### Throughput Optimization Roadmap

| Optimization | Expected speedup | Effort |
|---|---|---|
| SIMD vectorization (AVX2) | 2-4x | Medium |
| Multi-threaded block processing | Linear with cores | Medium |
| Quantized RWKV (Q4/Q8 instead of F32) | 1.5-2x | Low |
| GPU offload (future) | 10-50x | High |
| Target: 1 KB/s | 16x total | Phase 1 |

Current target: **1 KB/s** (16x improvement) would make the product viable for
mid-tier e-invoicing without excessive parallelism.

## OSI Protocol: Not Viable

Neural compression is inherently **sequential and stateful** — each byte's probability
depends on all prior bytes. This conflicts with packet-based transport protocols
where packets must be independently decodable and can arrive out of order.

**Conclusion**: Redukt is a storage/archival product, NOT a transport protocol.

## Proprietary Format Lock-In

AZTH format requires our exact model + weights for decompression. Mitigation:
- Transparent Proxy model (client never sees AZTH)
- Open-source the format spec + reference decoder
- Escrow weights with trusted third party for enterprise contracts
- Contractual guarantee of decompression service availability

## High-Entropy Data

Redukt performs **at par or worse** on near-random data (databases, astronomy, encrypted):
- osdb: +32.9% worse than xz
- sao: +3.6% worse than brotli
- random: +0.2% (essentially identical — no compressor beats entropy)

**Mitigation**: Content-type detection in the proxy. Route high-entropy data to
standard compressors (zstd), route compressible data to Redukt.

## Scale Limitations

- Single-file processing (no multi-file archive format yet)
- No streaming API (full file must be buffered)
- Weights must be identical for compress/decompress (version management required)
- Max tested file size: 100 MB (enwik8). Larger files untested.

## What We Don't Know Yet

- Performance on FASTQ/BAM (genomics) — not benchmarked
- Performance on DICOM (real medical images) — only proxy via x-ray/mr samples
- Convergence curves (does advantage grow or shrink with file size?)
- Multi-threaded throughput on server-grade hardware (Ryzen 9, Xeon)
- GPU inference speed for RWKV-7 0.1B
