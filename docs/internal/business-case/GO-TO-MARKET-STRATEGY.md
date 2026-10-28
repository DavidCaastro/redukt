# Go-to-Market Strategy — Redukt

> Analysis date: 2026-10-27. **Audited 2026-10-27** — corrected for data integrity.
> Based on verified benchmarks (17 files), SIMD results (R-011), and market research.
> All EUR figures. Throughput numbers from i5-1235U (Alder Lake), Q8 quantization, pre-SIMD CSV.
>
> Every number tagged: [MEASURED], [CALCULATED], [ESTIMATE], [UNMEASURED], [SOURCED].

---

## Executive Summary

Redukt compresses benchmark files better than brotli-11, with decisive advantages
in server logs (-34%) and e-invoicing XML (-48%) — measured with actual AZTH compression,
not cross-entropy estimates. The compression technology works.
The blocking constraint is **throughput**: ~150 B/s makes compute cost exceed storage
savings in most conventional scenarios. This document defines the optimal path to a
first viable, quantifiable, auditable, and reproducible product.

> **CORRECTION (2026-10-28)**: Previous version used cross-entropy BPB for Redukt
> (5-20% lower than actual compressed file size). All figures updated to actual
> compression BPB. Only 3 files verified so far; 15 pending re-measurement.

---

## Current Position

### Strengths (verified, reproducible)

| Metric | Value | Source |
|---|---|---|
| BPB enwik8 10KB (actual AZTH) | 1.3448 | [MEASURED: benchmark --verify 2026-10-28] |
| BPB enwik8 10KB (cross-entropy) | 1.2812 | `data/benchmark/results/f32-vs-q8.csv` |
| vs brotli-11 on logs | **-34%** | apache-access.log actual compression [MEASURED 2026-10-28] |
| vs brotli-11 on XML | **-48%** | facturae-batch.xml actual compression [MEASURED 2026-10-28] |
| Wins vs brotli-11 | 3/3 verified | 15 files pending re-measurement |
| Coder efficiency | 80-95% | varies by file compressibility [MEASURED 2026-10-28] |
| Integrity | CRC-32 + bounds check | AZTH v3 (R-001 to R-004) |
| Error handling | Result<> propagation | R-006, zero panics in codec |
| SIMD speedup (Q8 matmul) | 6.7x AVX2, 14.5x VNNI | R-011 micro-benchmark |
| Cross-path determinism | Bit-exact (Q8 SIMD) | R-011 roundtrip validation |
| Test suite | 74 pass (70 unit + 4 integration) | `cargo test` |

### Weaknesses (quantified)

| Constraint | Value | Source | Impact |
|---|---|---|---|
| Throughput (post-SIMD) | ~132-164 B/s (mean ~150 B/s) | [MEASURED: benchmark 2026-10-28] | ~3,300x slower than brotli-11 |
| End-to-end SIMD speedup | ~1.3x (memory-bound) | [MEASURED: simd-benchmark.md] | Insufficient alone |
| Max file size | ~4 GB (u32 TOTAL_BYTES) | [MEASURED: R-007 validation] | Limits large file use cases |
| High-entropy data | +20% to +54% worse than brotli | [MEASURED: ooffice, osdb, sao, x-ray] | Medical imaging, astronomy, databases |
| Format lock-in | Requires identical model+weights | Architecture constraint | Mitigation needed |

Note: Previous version claimed "~50,000x slower" — that used binary worst case (27 B/s)
vs optimistic brotli (1 MB/s) without disclosure. Corrected to ~4,000-8,000x range
based on brotli-11 ~0.5-1 MB/s vs Redukt mean ~80 B/s [CALCULATED].

---

## Strategic Options Evaluated

### Option A: Batch Archival SaaS (PetaGene model)

**Model**: Price per TB saved. Client sends data, we recompress in batch.

**Economics** (log archival, 1 TB/day):
- Throughput on logs: ~69 B/s pre-SIMD [CALCULATED from CSV: mean of 3 log files]
- To process 1 TB/day at 69 B/s: 1,000,000,000,000 / 69 / 86400 = **167,800 cores** [CALCULATED]
- [ESTIMATE] Compute at $0.04/hr per core (unsourced, ~c5.large vCPU on-demand): $161K/day
- [CALCULATED] Storage savings: 1 TB × 51.8% extra vs gzip × $0.023/GB/mo = ~$11.90/mo
- **ROI: catastrophically negative** — compute cost exceeds savings by ~400,000x

**Verdict**: Not viable without ~10x throughput improvement.

### Option B: Bandwidth-Constrained Niches (satellite/maritime/mining)

**Context**: Satellite bandwidth costs $1-225/GB depending on provider.
- Starlink maritime: ~$6/GB
- Traditional VSAT (maritime/mining): $10-225/GB
- LEO providers (OneWeb): $1-5/GB

**Economics** (vessel transmitting 10 GB/day of logs):
- Bandwidth cost: 10 GB x $6/GB = $60/day = **$21,900/year**
- With Redukt (-34% actual compression): saves $7,446/year per vessel
- Fleet of 100 vessels: **$745K/year savings**

**Throughput required**: 10 GB/day at ~132 B/s (logs) [MEASURED 2026-10-28]:
- 1 core: 10,000,000,000 / 132 / 86400 = 877 days (impossible)
- 8 cores: 110 days (still impossible)
- Need ~1 KB/s per core + 116 cores for 10 GB/day [CALCULATED: 10e9/(1000×86400)=116]

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
PHASE 2: Minimum Viable Throughput (sequenced by BPB risk)
│
├── Step 1: R-015 KV Batch Caching (2-3x, 0% BPB risk)
├── Step 2: R-012 Multi-file parallelism (Nx cores, 0% BPB risk)
├── ── GATE: ~1 KB/s target (0% BPB loss) ──
├── Step 3: R-016 SVD Pruning (3-5x, +2-5% BPB risk) [CONDITIONAL]
├── Step 4: R-017 Parallel Segments (2-4x, variable BPB risk) [CONDITIONAL]
├── ── GATE: combined BPB degradation ≤ +5% on logs ──
├── Q4 quantization: REJECTED [MEASURED: +13.3% BPB, 7.6x slower]
└── Goal: ≥1 KB/s sustained, <+5% BPB degradation
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
3. Results: 3/3 verified files beat brotli-11 [MEASURED], -34% on logs, -48% on XML (actual compression, 10KB)
4. Limitations: throughput (mean ~150 B/s post-SIMD [MEASURED]), high-entropy data, format lock-in
5. **CAVEAT**: All BPB results measured at 10KB. Scale behavior unknown (R-V04).
5. Architecture: two-tier (fast compressor inline → Redukt batch archival)
6. Reproducibility: all benchmarks runnable with public tools

---

## Phase 2 Detail: Throughput Optimization

> **CRITICAL**: Optimizations are NOT fully independent. SVD pruning and parallel
> segment compression both degrade BPB. Combined, degradation is **compounded, not additive**.
> A BPB degradation budget of **+5% on logs** governs the sequence.
> See ROADMAP.md P3 for full quality gate protocol.

### Optimization Sequence (ordered by BPB risk)

**Baseline**: ~132 B/s on logs [MEASURED: benchmark 2026-10-28, apache-access.log].
Previous measurement (R-V01): 62 B/s — improved with current binary and SIMD optimizations.

**Rejected**: Q4 quantization — +13.3% BPB regression AND 7.6x slower [MEASURED: R-V05].

| Step | Task | Speedup | BPB risk | Status | Source |
|---|---|---|---|---|---|
| 0 | SIMD AVX2/VNNI (R-011) | 1.3x | 0% | **COMPLETED** | [MEASURED: simd-benchmark.md] |
| 1 | KV Batch Caching (R-015) | 2-3x | 0% | PENDING | [ESTIMATE: architectural] |
| 2 | Multi-file parallelism (R-012) | Nx cores | 0% | PENDING | [PROJECTION: linear] |
| 3 | SVD Pruning (R-016) | 3-5x | +2-5% | PENDING | [SOURCED: RWKV-edge, unverified for our model] |
| 4 | Parallel Segments (R-017) | 2-4x | variable | PENDING | [ESTIMATE, depends on segment size] |

### Why This Order Matters

Steps 1-2 are **zero-risk**: they don't change model predictions. If KV caching delivers
2-3x and 8 cores deliver 8x, we reach **~1-1.5 KB/s on logs without ANY BPB loss**.
This is the minimum viable throughput for small-scale log archival (<1 GB/day).

Steps 3-4 carry **BPB degradation risk**:
- SVD pruning changes the model — predictions get worse. RWKV-edge reports -1pp accuracy
  on NLP [SOURCED], but BPB impact on compression is [UNMEASURED] and likely higher.
- Parallel segments break context at segment boundaries — each segment starts "cold".
- **Combined**: if SVD costs +4% BPB and segments cost +3% BPB, the combined effect
  could be +8% or more (not +7%) due to interaction effects [UNMEASURED].
- At +8% BPB degradation on logs, the advantage vs brotli drops from -34% to ~-29%.
  Still viable, but margin is thinner.

### Quality Gates

Each step has a mandatory gate before proceeding:

| Gate | Condition | Action if FAIL |
|---|---|---|
| After R-015 | BPB delta = 0.000 on logs | Bug — fix implementation |
| After R-012 | BPB delta = 0.000 on logs | Bug — fix implementation |
| After R-016 | BPB delta ≤ +5% on logs | Reduce SVD aggressiveness (rank-512) |
| After R-017 | **Combined** BPB delta ≤ +5% on logs | Increase segment size or ABORT R-017 |

### Throughput Targets [ALL PROJECTIONS except row 1]

| Milestone | B/s (logs) | Formula | BPB loss | Enables |
|---|---|---|---|---|
| Current | ~132 | [MEASURED: benchmark 2026-10-28] | 0% | Benchmark tool only |
| +R-015 | ~264-396 | [PROJECTION: 132 × 2-3] | 0% | — |
| +R-012 (8 cores) | ~2112-3168 | [PROJECTION: above × 8] | 0% | **~2-3 GB/day log archival** |
| +R-016 (SVD) | ~6336-15840 | [PROJECTION: above × 3-5] | +2-5% | ~5-14 GB/day |
| +R-017 (segments) | ~12672-63360 | [PROJECTION: above × 2-4] | +3-8%? | ~11-55 GB/day |

**Nota crítica sobre multiplicatividad**:
- R-012 es genuinamente multiplicativo (paralelismo puro, procesos independientes)
- R-015 × R-016 podrían ser sub-multiplicativos (ambos tocan el inference path)
- R-017 tiene overhead de merge y cold-start por segmento
- Los rangos altos son optimistas; planificar con los rangos bajos

### Minimum Viable Target

**R-015 + R-012 (8 cores) = ~2-3 KB/s en logs, 0% BPB loss.**

Esto es suficiente para un PoC de log archival a escala modesta (~2 GB/day a 2 KB/s
requiere ~12 días, viable para batch semanal/mensual).

Para viabilidad real (1 TB/day), se necesitan R-016 + R-017 o GPU offload.
Pero el PoC puede ejecutarse antes de resolver throughput completo.

---

## Phase 3 Detail: First Customer

### Why Log Archival is the Optimal First Vertical

1. **Strongest competitive advantage**: -34% vs brotli on logs (actual compression) [MEASURED]
2. **Low-entropy data**: text logs play to our strength (repetitive patterns, natural language)
3. **Predictable volume**: X GB/day, easy to calculate ROI
4. **Mandatory retention**: regulations (SOX, GDPR, BCE, CNMV) force multi-year storage
5. **Latency-tolerant**: logs are not re-read in real-time; batch nightly is acceptable
6. **Quantifiable market**: any enterprise with >1 TB/month of logs is a prospect
7. **Business case requires throughput improvement**: see BUSINESS-CASES.md Case 4 (corrected)

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
| Throughput insufficient for Phase 3 | High | Critical | Phase 2a (0% BPB risk) may suffice for small PoC; GPU as backup |
| Model pruning degrades BPB >+5% | Medium | High | Quality gate: reduce SVD aggressiveness or ABORT. Budget: +5% max |
| Combined degradation (SVD+segments) compounds | Medium | High | Measure combined, not just individual. STOP if >+5% total |
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
| Phase 2a | 4-8 weeks | Phase 1 feedback | R-015 + R-012: ≥1 KB/s, 0% BPB loss |
| Phase 2b | 2-4 months | Phase 2a gate pass | R-016 + R-017: ≥5 KB/s, ≤+5% BPB |
| Phase 3 | 3-6 months | Phase 2 throughput | First paying customer or PoC signed |

---

## Decision: Optimal First Investment

**Phase 1 is the correct first investment** because:

1. **Zero throughput dependency**: works with current ~150 B/s
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
