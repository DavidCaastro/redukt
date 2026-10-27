# Limitations & Mitigation Strategies

## Critical: Throughput Constraint

### The Problem

**Audited 2026-10-27** — throughput numbers corrected with measured B/s per domain.

Redukt throughput varies significantly by data type (pre-SIMD, from CSV):

| Domain | B/s range | Mean B/s | tok/s range | Source |
|---|---|---|---|---|
| Text (EN, PL, dict, code) | 75-183 | 123 | 33-46 | [CALCULATED from f32-vs-q8.csv] |
| Logs (apache, syslog, ssh) | 66-75 | 69 | 26-27 | [CALCULATED from f32-vs-q8.csv] |
| XML/markup | 59-92 | 71 | 26-27 | [CALCULATED from f32-vs-q8.csv] |
| Binary (exe, db, medical) | 27-60 | 47 | 27-47 | [CALCULATED from f32-vs-q8.csv] |
| **All 17 files** | **27-183** | **80** | **26-47** | [CALCULATED from f32-vs-q8.csv] |

Post-SIMD (R-011): ~1.3x end-to-end [MEASURED on enwik8 5KB only]. Full 17-file
post-SIMD CSV not yet collected — see R-V01.

This is **~4,000-8,000x slower than brotli-11** (~0.5-1 MB/s) [ESTIMATE for brotli speed].

### Impact by Use Case

[CALCULATED: days = volume_bytes / (domain_B_s × 86400). Cores = ceil(days).]

| Use case | Domain B/s | Daily volume | Time (1 core) | Cores needed (24h) |
|---|---|---|---|---|
| E-invoicing XML (50M/yr) | 71 B/s | 41 GB | 6.7 days | 7 |
| E-invoicing XML (Edicom) | 71 B/s | 82 GB | 13.4 days | 14 |
| PACS (SAS 50 hospitals) | 47 B/s | 2.05 TB | 505 days | 505 |
| Log archival (bank, 2 TB) | 69 B/s | 2 TB | 336 days | 336 |
| Log archival (bank, 5 TB) | 69 B/s | 5 TB | 839 days | 839 |

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

| Optimization | Expected speedup | Effort | Status |
|---|---|---|---|
| SIMD vectorization (AVX2/VNNI) | **1.3x** end-to-end | Medium | COMPLETED (R-011) |
| Multi-threaded block processing | Linear with cores | Medium | PENDING (R-012) |
| RWKV model pruning (RWKV-edge) | 3-5x | High | RESEARCH |
| Q4 quantization | 1.5-2x | Low | PARTIAL (flag exists) |
| GPU offload (CUDA) | 10-50x | High | FUTURE |
| **Combined (no GPU)** | **~31x** (8 cores) | | **~3.6 KB/s** |

SIMD results (R-011): Q8 matmul 6.7x AVX2 / 14.5x VNNI, but end-to-end only 1.3x
due to memory-bound head matmul (48MB exceeds L3) and non-SIMD overhead (~20% of time).
F32 FMA disabled for cross-path determinism. See `data/benchmark/results/simd-benchmark.md`.

Current target: **1 KB/s** (8.5x improvement) would make the product viable for
mid-tier e-invoicing without excessive parallelism.

See `GO-TO-MARKET-STRATEGY.md` for full strategic analysis including throughput economics.

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

Redukt performs **at par or worse** on high-entropy data [MEASURED]:

| File | Redukt Q8 BPB | brotli-11 BPB | Delta | Source |
|---|---|---|---|---|
| ooffice | 3.704 | 2.742 | +35.0% | [MEASURED: gzip9-brotli11-comparison.csv] |
| osdb | 5.643 | 4.676 | +20.7% | [MEASURED] |
| sao | 9.241 | 5.988 | +54.3% | [MEASURED] |
| x-ray | 6.524 | 4.592 | +42.1% | [MEASURED] |

**Mitigation**: Content-type detection in the proxy. Route high-entropy data to
standard compressors (zstd), route compressible data to Redukt.

## Scale Limitations

- Single-file processing (no multi-file archive format yet)
- No streaming API (full file must be buffered)
- Weights must be identical for compress/decompress (version management required)
- Max tested file size: 100 MB (enwik8). Larger files untested.

## Data Integrity — RESOLVED

AZTH v3 includes comprehensive integrity protection (all items completed):

- **R-001**: CRC-32 (IEEE) appended to every v3 file, verified on decompression
- **R-002**: Size mismatch is a hard error (not warning)
- **R-003**: Hyperparameter validation rejects NaN/Inf
- **R-004**: Arithmetic decoder bounds checking with structured errors
- **R-006**: All panics replaced with Result propagation
- **R-007**: u32 overflow validated at compression time (>4GB rejected)

Backward compatible: decoder accepts v2 (no CRC) and v3 (CRC verified).

## What We Don't Know Yet

- Performance on FASTQ/BAM (genomics) — not benchmarked
- Performance on DICOM (real medical images) — only proxy via x-ray/mr samples
- Convergence curves (does advantage grow or shrink with file size?)
- Multi-threaded throughput on server-grade hardware (Ryzen 9, Xeon)
- GPU inference speed for RWKV-7 0.1B
