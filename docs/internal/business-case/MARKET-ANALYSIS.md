# Market Analysis — Redukt

## Global Compression Market

- Data compression market: **$1.28B** (2026), CAGR 9.2% [UNSOURCED — no market research firm cited]
- Drivers: cloud storage costs, regulatory retention, data growth

> **AUDIT NOTE (2026-10-27)**: Multiple TAM figures in this document are UNSOURCED.
> Tagged below. Do not use unsourced figures in external communications or decisions.

## Target Verticals

### 1. Enterprise Log Archival (TAM: $4.0-5.0B)

TAM sources: MarketsandMarkets $4.1B (2026), Mordor Intelligence $4.35B, Research and Markets $4.99B.
Note: this is log MANAGEMENT market, not log ARCHIVAL specifically (narrower).

**Pain**: 2-5 TB/day of logs [UNSOURCED — industry typical, not verified], 5-year retention (BCE, CNMV, GDPR).
**Validation**: Cribl ($390M ARR Aug 2026 [SOURCED: Sacra/GlobeNewsWire], $3.5B valuation Series E Aug 2024 [SOURCED: Yahoo Finance/GlobeNewsWire], $600M raised, 43 Fortune 100 customers).
**Our advantage** (actual AZTH compression, all [MEASURED 2026-10-28]):
-44% vs gzip-9 mean on logs (apache -43%, syslog -40%, openssh -46%).
-31% vs brotli-11 mean on logs (apache -34%, syslog -27%, openssh -31%).
**CAVEAT**: measured at 10KB only. At 100KB, gzip-9 improves from 0.55 to 0.44 BPB, brotli from 0.47 to 0.33. Redukt at larger sizes is [UNMEASURED] — see R-V04.
**Target clients**: Banks (Bankinter, Abanca, Kutxabank), telcos (Telefonica, MasMovil), utilities.

### 2. E-Invoicing Platforms ($1.1B TAM Spain by 2034 [SOURCED: Vocal Media — low authority])

**Context**: Spain mandatory e-invoicing from July 2026 (Ley Crea y Crece, RD 238/2026).
- 500M+ e-invoices B2B in Spain (2024) [SOURCED: Silicon.es]. Note: previous version said 557M — inflated.
- Format: XML Facturae/UBL, 6-year retention
- Market leaders: SERES (1000M+ docs/year), Edicom (EUR 252.9M revenue 2024 [SOURCED: Valencia Plaza/El Economista], EUR 276.6M in 2025 [SOURCED: Valencia Plaza], 60% net margin [SOURCED: El Economista Jul 2025], 15K-19K clients [SOURCED: GetLatka 15K, ZoomInfo 19K])

**Our advantage on XML** (actual compression) [MEASURED 2026-10-28]:
-48% vs brotli-11 on Facturae (0.462 vs 0.890 BPB), -37% on UBL (0.558 vs 0.888 BPB).
**BLOCKER**: PDF = 67% of e-invoice storage. PDF compression is [UNMEASURED] — see R-V03.

### 3. Medical Imaging / PACS (TAM: $3-4B [UNSOURCED])

> **INVALIDATED** (2026-10-27). Redukt LOSES on medical imaging.

**Measured results** [MEASURED: gzip9-brotli11-comparison.csv]:
- x-ray: Redukt 6.524 vs brotli 4.592 → **+42.1% WORSE**
- mr: Redukt 2.283 vs brotli 2.373 → -3.8% (marginal, within noise)

**Conclusion**: Do NOT pursue this vertical. Hospital volume estimates (20 TB/year per
500-bed hospital [UNSOURCED], 30%/year growth [UNSOURCED]) are moot.

### 4. Cold/Archive Storage (TAM: [UNSOURCED — $49B figure could not be verified])

**Pain**: Long-term retention of compliance data, backups, historical records.
**Our advantage**: Best-in-class compression means less data to store/replicate/transfer.

### 5. Bandwidth-Constrained Niches (satellite/maritime/mining)

**Pain**: Satellite bandwidth costs $1-225/GB [UNSOURCED — from web search, not verified against provider pricing pages].
Starlink maritime ~$6/GB [ESTIMATE: ~$5000/mo for 1TB plan, varies by region].
**Use case**: Vessels, mining sites, oil rigs transmitting logs/telemetry over expensive links.
**Economics** [CALCULATED from unsourced bandwidth price]:
- Fleet of 100 vessels × 10 GB/day × $6/GB × 365 = $21.9M/year bandwidth
- With Redukt (-34% actual compression on logs at 10KB [MEASURED 2026-10-28]): saves $7.4M/year on bandwidth [CALCULATED]
- CAVEAT: -34% is apache-access.log at 10KB. Mean of 3 log types: -31%. At larger sizes: [UNMEASURED] (R-V04).
**Blocker**: Throughput (~69 B/s on logs [CALCULATED]) insufficient for 10 GB/day.
**Status**: Highest value per byte compressed. Viable after throughput Phase 2.

### 6. Genomics (TAM: $2-5B [UNSOURCED])

**Pain**: FASTQ/BAM files, 100GB+ per genome, growing exponentially.
**Validation**: PetaGene, Genozip exist in this space.
**Status**: Not yet benchmarked with genomic formats.

## Competitive Landscape

| Competitor | Model | Strength | Weakness |
|---|---|---|---|
| gzip/zstd/brotli | Open source, free | Fast, ubiquitous | Lower compression ratio |
| PetaGene | Per-TB license | Genomics specialist | Narrow vertical |
| Cribl | SaaS, per-GB | Log pipeline leader | Reduction, not compression |
| NNCP/cmix | Academic | High compression | Slower than Redukt on CPU |

## Spain-Specific Market Context

- AWS investing EUR 33.7B in Spain data centers (eu-south-2 Madrid) [UNSOURCED — verifiable via press]
- Spanish cloud adoption: 41.65% paid cloud, only 13.68% migrated software [UNSOURCED]
- Key consultancies for channel: Indra/Minsait (EUR 5.46B), NTT Data Spain (15K+), Inetum (27K) [UNSOURCED]
- E-invoicing market growth: $299M (2025) -> $1,139M (2034), CAGR 16.03% [SOURCED: Vocal Media — low authority]

## Sources (verified 2026-10-27)

### Verified with primary sources
- MarketsandMarkets: Log Management Market $4.1B by 2026 (press release)
- Mordor Intelligence: Log Management Market $4.35B in 2026
- Research and Markets: Log Management Market $4.99B in 2026
- Valencia Plaza (Jul 2025): Edicom EUR 252.9M revenue 2024, EUR 276.6M in 2025
- El Economista (Jul 2025): Edicom 60% net margin, EUR 151.4M profit
- GlobeNewsWire (Feb 2026): Cribl surpasses $300M ARR
- Sacra (Aug 2026): Cribl $390M ARR, $3.5B valuation
- Silicon.es: Spain 500M+ e-invoices (2024)

### Low authority / unverified
- Vocal Media: Spain e-invoicing $1,139M by 2034 (user-generated content platform)
- Purview: PACS storage costs $180K/300TB (vendor blog)
- QSS Technosoft: Hospital PACS $2M overspend analysis (vendor)
- Cloud Storage Explorer: 2026 pricing comparison

### Not found / could not verify
- $49B cold/archive data storage TAM — no credible source found
- $2-5B genomics TAM — no credible source found
- 2-5 TB/day enterprise log generation — industry typical but no specific source
- Hospital 20 TB/year DICOM — no specific source
- 30%/year medical imaging growth — no specific source
