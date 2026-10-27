# Go-to-Market Strategy — Redukt

> Analysis date: 2026-10-27.
> Based on verified benchmarks (18 files), SIMD results (R-011), and market research.
> All EUR figures. Throughput numbers from i5-1235U (Alder Lake), Q8 quantization.

---

## Executive Summary

Redukt compresses 13/18 benchmark files better than brotli-11, with decisive advantages
in server logs (-48%) and e-invoicing XML (-54%). The compression technology works.
The blocking constraint is **throughput**: ~117 B/s makes compute cost exceed storage
savings in most conventional scenarios. This document defines the optimal path to a
first viable, quantifiable, auditable, and reproducible product.

---

## Current Position

### Strengths (verified, reproducible)

| Metric | Value | Source |
|---|---|---|
| BPB enwik8 10KB (Q8) | 1.2812 | `data/benchmark/results/f32-vs-q8.csv` |
| vs brotli-11 on logs | **-48%** | apache-access.log benchmark |
| vs brotli-11 on XML | **-54%** | facturae-batch.xml benchmark |
| Wins vs brotli-11 | 13/18 files | Silesia + industry corpus |
| Coder efficiency | 95.27% | CoderMetrics (R-009) |
| Integrity | CRC-32 + bounds check | AZTH v3 (R-001 to R-004) |
| Error handling | Result<> propagation | R-006, zero panics in codec |
| SIMD speedup (Q8 matmul) | 6.7x AVX2, 14.5x VNNI | R-011 micro-benchmark |
| Cross-path determinism | Bit-exact (Q8 SIMD) | R-011 roundtrip validation |
| Test suite | 74 pass (70 unit + 4 integration) | `cargo test` |

### Weaknesses (quantified)

| Constraint | Value | Impact |
|---|---|---|
| Throughput | ~117 B/s (text), ~20 B/s (binary) | ~50,000x slower than brotli-11 |
| End-to-end SIMD speedup | ~1.3x (memory-bound) | Insufficient alone |
| Max file size | ~4 GB (u32 TOTAL_BYTES) | Limits large file use cases |
| High-entropy data | +1% to +66% worse than brotli | Medical imaging, astronomy |
| Format lock-in | Requires identical model+weights | Mitigation needed |

---

## Strategic Options Evaluated

### Option A: Batch Archival SaaS (PetaGene model)

**Model**: Price per TB saved. Client sends data, we recompress in batch.

**Economics** (log archival, 1 TB/day):
- Compute: ~100 cores x $0.04/hr = ~$96/day to process 1 TB at 117 B/s
- Storage savings: 1 TB x 48% extra reduction x $0.023/GB/mo = ~$11.50/month
- **ROI: negative** — compute cost far exceeds storage savings at current throughput

**Verdict**: Not viable without ~10x throughput improvement.

### Option B: Bandwidth-Constrained Niches (satellite/maritime/mining)

**Context**: Satellite bandwidth costs $1-225/GB depending on provider.
- Starlink maritime: ~$6/GB
- Traditional VSAT (maritime/mining): $10-225/GB
- LEO providers (OneWeb): $1-5/GB

**Economics** (vessel transmitting 10 GB/day of logs):
- Bandwidth cost: 10 GB x $6/GB = $60/day = **$21,900/year**
- With Redukt (-48%): saves $10,512/year per vessel
- Fleet of 100 vessels: **$1.05M/year savings**

**Throughput required**: 10 GB/day at 117 B/s = 988 days on 1 core (impossible).
- With 8 cores: 124 days (still impossible)
- Need ~1 KB/s minimum (8.5x improvement) + 116 cores for 10 GB/day

**Verdict**: Highest value per byte compressed. Viable only after throughput phase.

### Option C: SDK/Library Licensing

**Model**: Sell Rust library as embeddable component to archival/invoicing platforms.

**Advantage**: Client provides hardware, we provide technology.
**Problem**: No integrator adopts a library that processes at 117 B/s.

**Verdict**: Not viable without throughput improvement.

### Option D: Transparent Proxy (cloud middleware)

**Model**: Proxy that recompresses data in transit to cold storage (Cribl model).

**Same problem**: Throughput insufficient for production volumes.

**Verdict**: Not viable without throughput improvement.

### Option E: Research Publication + Benchmark Tool

**Model**: Publish results, open-source, monetize via consulting/contracts/credibility.

**Advantage**: Does NOT require production throughput. BPB results are already verified,
reproducible, and auditable. Establishes credibility for future commercial product.

**Verdict**: **Immediately viable.** Lowest cost, builds foundation for all other options.

---

## Recommended Strategy: 3 Phases

```
PHASE 1: Auditable Product (NOW — minimal new code)
│
├── Reproducible benchmark suite (any user can verify claims)
├── Whitepaper with verified BPB results
├── Open-source Apache-2.0 (already done)
├── `redukt benchmark` command for prospect self-evaluation
└── Goal: CREDIBILITY + market feedback
│
PHASE 2: Minimum Viable Throughput (~10x → 1 KB/s)
│
├── R-012: Multi-file parallelism (linear with cores)
├── RWKV model pruning/distillation (RWKV-edge: 3.4-5x compression)
├── Q4 quantization evaluation (BPB impact vs speed gain)
├── Combined target: SIMD 1.3x × model 3x × 8 cores = ~31x → ~3.6 KB/s
└── Goal: demonstrate viability in target vertical
│
PHASE 3: First Customer (vertical = log archival)
│
├── PoC with real enterprise: 30-day log archive, measured vs brotli/zstd
├── Pricing: $/TB saved (PetaGene-validated model)
├── Two-tier architecture: zstd inline → Redukt batch nightly
└── Goal: revenue + case study
```

---

## Phase 1 Detail: Auditable Benchmark Product

### Deliverable: `redukt benchmark` command

```bash
# Prospect runs this with THEIR data
redukt benchmark --input /var/log/archive/ \
                 --reference brotli-11 \
                 --weights weights/rwkv7-0.1b \
                 --output report.json
```

**Output** (JSON + human-readable):
- BPB per file: Redukt vs brotli-11 vs zstd-19
- Compression ratio comparison
- Throughput measured on prospect's hardware
- Storage savings estimate (configurable $/GB/month)
- Projected ROI over configurable retention period

**This converts the project from "research" to "product"**: the prospect runs a command
and gets a specific ROI number for their data.

### Deliverable: Reproducible benchmark suite

Current state: `scripts/bench-f32-vs-q8.sh` exists but targets internal validation.
Need: public-facing benchmark that any third party can audit.

Requirements:
1. Deterministic output (already guaranteed — CRC-32 verified roundtrip)
2. Documented methodology (bytes processed, BPB formula, timing protocol)
3. Reference files included or downloadable (Silesia corpus is public domain)
4. Comparison against at least brotli-11, zstd-19, gzip-9
5. JSON output for automated analysis

### Deliverable: Whitepaper / Technical Report

Structure:
1. Problem: archival storage costs growing faster than capacity
2. Approach: neural compression (RWKV-7 + Context Mixing + Arithmetic Coding)
3. Results: 13/18 files beat brotli-11, -48% on logs, -54% on XML
4. Limitations: throughput (~117 B/s), high-entropy data, format lock-in
5. Architecture: two-tier (fast compressor inline → Redukt batch archival)
6. Reproducibility: all benchmarks runnable with public tools

---

## Phase 2 Detail: Throughput Optimization

### Throughput Improvement Stack

| Optimization | Expected speedup | Cumulative | Status |
|---|---|---|---|
| SIMD AVX2/VNNI (R-011) | 1.3x | 1.3x | COMPLETED |
| Multi-file parallelism (R-012) | Nx (cores) | 1.3 × N | PENDING |
| RWKV model pruning (RWKV-edge) | 3-5x | 3.9-6.5 × N | RESEARCH |
| Q4 quantization | 1.5-2x | 5.9-13 × N | PARTIAL |
| GPU offload (CUDA) | 10-50x | — | FUTURE |

### Key Research: RWKV-edge

Paper demonstrates 3.4-5x RWKV model compression maintaining accuracy:
- Structured pruning + knowledge distillation
- ARM NEON INT8 kernels for edge deployment
- Applied to our 0.1B model: ~20-30M effective parameters
- Fewer parameters = fewer matmuls = proportionally faster

### Throughput Targets

| Target | B/s | Enables |
|---|---|---|
| Current | 117 | Benchmark tool only |
| Phase 2a (8 cores) | ~936 | Small batch jobs (<1 GB/day) |
| Phase 2b (+model opt) | ~3,600 | Mid-tier e-invoicing (41 GB/day with 132 cores) |
| Phase 2c (GPU) | ~6,000-60,000 | Log archival at scale |

### Combined Estimate: SIMD (1.3x) × model pruning (3x) × 8 cores (8x) = ~31x

At 31x: ~3.6 KB/s per core. Processing 10 GB/day requires ~32 cores.
This is the **minimum viable throughput** for a commercial product.

---

## Phase 3 Detail: First Customer

### Why Log Archival is the Optimal First Vertical

1. **Strongest competitive advantage**: -48% vs brotli (largest measured delta)
2. **Low-entropy data**: text logs play to our strength (repetitive patterns, natural language)
3. **Predictable volume**: X GB/day, easy to calculate ROI
4. **Mandatory retention**: regulations (SOX, GDPR, BCE, CNMV) force multi-year storage
5. **Latency-tolerant**: logs are not re-read in real-time; batch nightly is acceptable
6. **Quantifiable market**: any enterprise with >1 TB/month of logs is a prospect
7. **Proven business case**: 198% ROI over 5 years (see BUSINESS-CASES.md Case 4)

### Target Customer Profile

| Attribute | Value |
|---|---|
| Industry | Banking, telco, utilities, SaaS |
| Log volume | >1 TB/month |
| Retention | 5+ years (regulatory) |
| Current compression | gzip-9 or zstd (inline, fast) |
| Storage | S3/GCS/Azure Blob (compliance tier) |
| Pain point | Storage costs growing faster than budget |

### Pricing Model (PetaGene-validated)

PetaGene charges per TB saved in genomics (55% storage cost reduction).
Our model: **fixed price per TB of input processed**, regardless of savings achieved.

| Tier | Volume | Price/TB | Annual |
|---|---|---|---|
| Starter | <10 TB/month | EUR 12/TB | EUR 1,440 |
| Growth | 10-100 TB/month | EUR 8/TB | EUR 9,600-96,000 |
| Enterprise | >100 TB/month | EUR 5/TB | Negotiated |

Alternative: annual license (see BUSINESS-CASES.md for detailed models).

### PoC Structure

1. **Duration**: 30 days
2. **Scope**: single log pipeline (e.g., Apache access logs, syslog)
3. **Data**: 30 days of production logs (~30-150 GB)
4. **Comparison**: Redukt vs client's current compression (gzip/zstd/brotli)
5. **Deliverable**: JSON report with measured BPB, ratio, throughput, projected savings
6. **Success criteria**: >30% better compression ratio than client's current solution
7. **Hardware**: client-provided server (8+ cores, 32 GB RAM)

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Throughput insufficient for Phase 3 | High | Critical | GPU offload as backup; focus Phase 1-2 first |
| Model pruning degrades BPB | Medium | High | Benchmark BPB at each pruning level; accept ≤5% regression |
| No market interest (Phase 1) | Low | Medium | Validates early, pivots cheap |
| Competitor ships neural compressor | Low | High | Microsoft MLVC (video only), NVIDIA NTC (early); we're ahead on text |
| Format lock-in deters enterprise | Medium | Medium | Open-source spec + weight escrow + contractual guarantees |
| Cross-platform determinism fails | Low | High | R-013 addresses; Q8 integer math is inherently portable |

### Competitive Intelligence

| Player | Status | Domain | Threat Level |
|---|---|---|---|
| Microsoft MLVC | Shipping (Teams) | Video calling | None (different domain) |
| NVIDIA NTC | Early stage | Neural texture compression | None (GPU-only, images) |
| cmix | Academic | General compression | Low (slower, no product) |
| NNCP | Research | General neural compression | Low (no product) |
| PetaGene | Commercial | Genomics (FASTQ/BAM) | Low (different vertical) |
| Brotli/zstd | Ubiquitous | General purpose | Incumbent (we beat on ratio) |

No direct competitor in **neural compression for text/structured data archival**.
Microsoft and NVIDIA validate the approach but target different verticals.

---

## Timeline Estimate

| Phase | Duration | Dependencies | Exit Criteria |
|---|---|---|---|
| Phase 1 | 4-6 weeks | None | Benchmark command ships, whitepaper published |
| Phase 2 | 3-6 months | Phase 1 feedback | ≥1 KB/s sustained throughput |
| Phase 3 | 3-6 months | Phase 2 throughput | First paying customer or PoC signed |

---

## Decision: Optimal First Investment

**Phase 1 is the correct first investment** because:

1. **Zero throughput dependency**: works with current 117 B/s
2. **Lowest cost**: minimal new code (benchmark command + documentation)
3. **Maximum information gain**: real market feedback before heavy engineering
4. **Builds foundation**: every future phase benefits from credibility established here
5. **Reversible**: if market says no, we've spent weeks not months
6. **Auditable and reproducible**: BPB numbers are deterministic, CRC-verified, third-party verifiable

The product at Phase 1 is: **"Run this command on your data. See how much smaller we make it.
When throughput improves, this becomes your archival tier."**

This is an honest, quantifiable pitch that lets the market validate (or invalidate) the
value proposition before we invest in throughput optimization.

---

## References

- `docs/internal/business-case/BUSINESS-CASES.md` — Detailed EUR projections (4 verticals)
- `docs/internal/business-case/MARKET-ANALYSIS.md` — TAM, competitive landscape, Spain context
- `docs/internal/business-case/LIMITATIONS.md` — Throughput constraint analysis
- `data/benchmark/results/f32-vs-q8.csv` — Complete 18-file benchmark data
- `data/benchmark/results/simd-benchmark.md` — SIMD optimization results (R-011)
- `data/benchmark/results/emb-surgery-benchmark.csv` — Embedding surgery results (R-010)
- `docs/ROADMAP.md` — Engineering roadmap (R-001 to R-014)
- RWKV-edge paper: structured pruning + distillation for edge deployment (3.4-5x compression)
- PetaGene: per-TB pricing model validated in genomics vertical
- Microsoft MLVC: neural compression shipping in Teams (video domain)
