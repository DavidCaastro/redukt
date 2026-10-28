# Limitations & Mitigation Strategies

## Critical: Throughput Constraint

### The Problem

**Audited 2026-10-27, updated 2026-10-28** — throughput now from two post-SIMD measurements.

Redukt throughput varies significantly by data type and system load.
Two independent post-SIMD measurements available:

**R-V01 systematic benchmark** (2026-10-27, 18 files, `hybrid-eval`):

| Domain | B/s range | Mean B/s | tok/s range | Source |
|---|---|---|---|---|
| Text (EN, PL, dict, code) | 30-93 | 64 | 18-26 | [MEASURED: post-simd-benchmark.csv] |
| Logs (apache, syslog, ssh) | 50-62 | 55 | 20-22 | [MEASURED: post-simd-benchmark.csv] |
| XML/e-invoicing | 56-66 | 59 | 19-25 | [MEASURED: post-simd-benchmark.csv] |
| Binary (exe, db, medical, astro) | 25-33 | 28 | 20-26 | [MEASURED: post-simd-benchmark.csv] |
| **All 18 files** | **25-93** | **51** | **18-26** | [MEASURED: post-simd-benchmark.csv] |

**Actual-compression benchmark** (2026-10-28, `benchmark --verify`, different system state):
Reported ~25-47 tok/s, ~27-164 B/s. Apache specifically: ~132 B/s [MEASURED 2026-10-28].
R-V01 notes: "tok/s menor que mediciones anteriores, probablemente por carga del sistema".

**Discrepancy note**: R-V01 measured 62 B/s on apache, 2026-10-28 run measured ~132 B/s.
Difference likely due to system load / thermal state. Conservative estimate: use R-V01 means.
Optimistic estimate: use 2x R-V01 means. **Both are post-SIMD Q8 on i5-1235U.**

This is **~3,300-8,000x slower than brotli-11** (~0.5-1 MB/s) [ESTIMATE for brotli speed].

### Impact by Use Case

[CALCULATED: days = volume_bytes / (domain_B_s × 86400). Cores = ceil(days).]
Using R-V01 conservative B/s.

| Use case | Domain B/s | Daily volume | Time (1 core) | Cores needed (24h) |
|---|---|---|---|---|
| E-invoicing XML (50M/yr) | 59 B/s [MEASURED] | 41 GB | 8.1 days | 9 |
| E-invoicing XML (Edicom) | 59 B/s [MEASURED] | 82 GB | 16.1 days | 17 |
| Log archival (bank, 2 TB) | 55 B/s [MEASURED] | 2 TB | 421 days | 421 |
| Log archival (bank, 5 TB) | 55 B/s [MEASURED] | 5 TB | 1,052 days | 1,052 |

> **PACS (medical imaging)**: REMOVED — vertical invalidated (Redukt +42% WORSE on x-ray).
> See BUSINESS-CASES.md Case 3.

With optimistic throughput (~2x, based on 2026-10-28 run):

| Use case | Domain B/s (opt.) | Time (1 core) | Cores needed (24h) |
|---|---|---|---|
| E-invoicing XML (50M/yr) | ~118 B/s | 4.0 days | 5 |
| Log archival (bank, 2 TB) | ~110 B/s | 211 days | 211 |
| Log archival (bank, 5 TB) | ~110 B/s | 526 days | 526 |

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

> Sequence ordered by BPB degradation risk. Budget: **+3% max** on logs.
> See ROADMAP.md P3 for full quality gate protocol.

| Step | Optimization | Expected speedup | BPB risk | Status |
|---|---|---|---|---|
| 0a | SIMD vectorization AVX2/VNNI (R-011) | 1.3x [MEASURED] | 0% | **COMPLETED** |
| 0b | Coder efficiency (R-018) | ~1x (BPB gain) [ESTIMATE] | 0% (gain) | PENDING |
| 1 | KV Batch Caching (R-015) | 2-3x [ESTIMATE] | 0% | PENDING |
| 2 | Multi-file parallelism (R-012) | Nx cores [PROJECTION] | 0% | PENDING |
| 3 | SVD model pruning (R-016) | 3-5x [SOURCED: RWKV-edge] | +2-3% | CONDITIONAL |
| 4 | Parallel segments (R-017) | 2-4x [ESTIMATE] | variable | STRICTLY CONDITIONAL |
| ~~X~~ | ~~Q4 quantization~~ | ~~REJECTED~~ | ~~+13.3%~~ | **REJECTED** [MEASURED: R-V05] |
| Future | GPU offload (CUDA) | 10-50x [ESTIMATE] | 0% | FUTURE |

Q4 quantization tested and REJECTED: +13.3% BPB regression AND 7.6x slower [MEASURED: R-V05].

**Projected combined (no GPU, 8 cores, conservative R-V01 baseline)**:
- R-018 + R-015 + R-012: ~51 × 2.5 × 8 = **~1020 B/s** (~1 KB/s) [PROJECTION, 0% BPB loss]
- + R-016 (SVD): ~1020 × 4 = **~4 KB/s** [PROJECTION, +2-3% BPB loss]

SIMD results (R-011): Q8 matmul 6.7x AVX2 / 14.5x VNNI, but end-to-end only 1.3x
due to memory-bound head matmul (48MB exceeds L3) and non-SIMD overhead (~20% of time).
F32 FMA disabled for cross-path determinism. See `data/benchmark/results/simd-benchmark.md`.

Current target: **~2-3 KB/s** (R-018 + R-015 + R-012, 0% BPB loss) would make the product
viable for mid-tier e-invoicing and small-scale log archival without model degradation.

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

Redukt performs **at par or worse** on high-entropy data (actual AZTH compression) [MEASURED]:

| File | Redukt BPB | brotli-11 BPB | Delta | Source |
|---|---|---|---|---|
| ooffice | 3.779 | 2.742 | +37.8% | [MEASURED: actual-compression-benchmark.csv] |
| osdb | 5.717 | 4.676 | +22.3% | [MEASURED: actual-compression-benchmark.csv] |
| sao | 9.254 | 5.988 | +54.5% | [MEASURED: actual-compression-benchmark.csv] |
| x-ray | 6.608 | 4.592 | +43.9% | [MEASURED: actual-compression-benchmark.csv] |

> **Correction 2026-10-28**: Previous values used cross-entropy BPB (5-20% lower than actual).
> Updated to actual AZTH compression BPB. Deltas are slightly larger than previously reported.

**Mitigation**: Content-type detection in the proxy. Route high-entropy data to
standard compressors (zstd), route compressible data to Redukt.

## Scale Limitations

- Single-file processing (no multi-file archive format yet)
- No streaming API (full file must be buffered)
- Weights must be identical for compress/decompress (version management required)
- Max tested file size: 100 MB (enwik8) [MEASURED]. Larger files untested.

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
- Convergence curves (does advantage grow or shrink with file size?) — R-V04 partial
- Multi-threaded throughput on server-grade hardware (Ryzen 9, Xeon)
- GPU inference speed for RWKV-7 0.1B
- Exact throughput under stable conditions (two measurements differ ~2x)
