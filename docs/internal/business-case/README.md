# Redukt — Business Case

> **SUPERSEDED (2026-11-10)** — The pricing (EUR 5-12/TB), throughput targets (>=2 KB/s), "no direct competitor",
> "best-in-class" and cold/archive claims in this folder are contradicted by measured data (compute ~$50-60/GB on CPU,
> ~$9-17/GB on T4; R* = 18-100 KB/s per vCPU) and by external evidence (Nacrith, Hutter 2026 lineage, cmix at ~$3/GB,
> CIDR 2026). Kept for traceability only. Authoritative analysis: `docs/internal/research/competitive-position-2026-11-10/REPORT.md`.


Commercial viability analysis for Redukt's compression engine as a standalone product.

## Contents

### Business Analysis
| Document | Description |
|---|---|
| [MARKET-ANALYSIS.md](MARKET-ANALYSIS.md) | Target verticals, TAM, competitors, market sizing |
| [BENCHMARK-RESULTS.md](BENCHMARK-RESULTS.md) | Comparative benchmark vs standard compressors (18 file types) |
| [BUSINESS-CASES.md](BUSINESS-CASES.md) | 4 concrete cases with invoice-level EUR figures |
| [PRODUCT-ARCHITECTURE.md](PRODUCT-ARCHITECTURE.md) | API design, deployment model, pricing strategy |
| [LIMITATIONS.md](LIMITATIONS.md) | Throughput constraint, mitigation strategies |

### Migration & Legal
| Document | Description |
|---|---|
| [MIGRATION-GUIDE.md](MIGRATION-GUIDE.md) | Complete port instructions: dependency map, file list, validation |
| [LEGAL-ANALYSIS.md](LEGAL-ANALYSIS.md) | Full legal review: licenses, patents, training data risk, EU AI Act, jurisdictions |
| [CLAUDE-TEMPLATE.md](CLAUDE-TEMPLATE.md) | CLAUDE.md for the new repo (includes legal compliance matrix) |
| [FORMAT-SPEC.md](FORMAT-SPEC.md) | AZTH v1 binary format specification |
| [LICENSE-TEMPLATE](LICENSE-TEMPLATE) | Apache-2.0 full text (fill in copyright holder) |
| [NOTICE-TEMPLATE](NOTICE-TEMPLATE) | Third-party attributions (RWKV-7, algorithms) |

## Key Findings

- **Best vertical**: Enterprise log archival (ROI 198%, breakeven 2.5 years)
- **Best compression advantage**: XML/markup — 44.7% better than brotli-11
- **Critical constraint**: ~62 B/s throughput limits product to batch/archival tier
- **Recommended model**: Transparent Proxy (compress/decompress over client's S3/GCS)
- **Legal**: No blockers. Apache 2.0 (no royalties, no restrictions). All patents expired. EU AI Act does not apply.
- **One risk**: Training data copyright (The Pile/Books3) — mitigated by SaaS model + indemnification clause.

## Date

Analysis conducted: 2026-10-20 / 2026-10-21
Benchmark data: `data/benchmark/` (reproducible)
