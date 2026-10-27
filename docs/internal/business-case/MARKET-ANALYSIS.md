# Market Analysis — Redukt

## Global Compression Market

- Data compression market: **$1.28B** (2026), CAGR 9.2% [UNSOURCED — no market research firm cited]
- Drivers: cloud storage costs, regulatory retention, data growth

> **AUDIT NOTE (2026-10-27)**: Multiple TAM figures in this document are UNSOURCED.
> Tagged below. Do not use unsourced figures in external communications or decisions.

## Target Verticals

### 1. Enterprise Log Archival (TAM: [UNSOURCED])

**Pain**: 2-5 TB/day of logs [UNSOURCED], 5-year regulatory retention (BCE, CNMV, GDPR).
**Validation**: Cribl ($390M ARR [UNSOURCED — private company], $3.5B valuation [UNSOURCED]) proves the market pays for data reduction.
**Our advantage**: -55% vs gzip-9 on apache logs, -49% syslog, -52% openssh [MEASURED: gzip9-brotli11-comparison.csv]. -48% vs brotli-11 on apache, -38% syslog, -39% openssh [MEASURED].
**Target clients**: Banks (Bankinter, Abanca, Kutxabank), telcos (Telefonica, MasMovil), utilities.

### 2. E-Invoicing Platforms ($1.1B TAM Spain by 2034 [SOURCED: Vocal Media — low authority])

**Context**: Spain mandatory e-invoicing from July 2026 (Ley Crea y Crece, RD 238/2026).
- 500M+ e-invoices B2B in Spain (2024) [SOURCED: Silicon.es]. Note: previous version said 557M — inflated.
- Format: XML Facturae/UBL, 6-year retention
- Market leaders: SERES (1000M+ docs/year), Edicom (EUR 252.9M revenue [SOURCED: El Economista], 60% margin [UNSOURCED], 17K clients [UNSOURCED])

**Our advantage on XML**: -54% vs brotli-11 on Facturae (0.406 vs 0.890 BPB), -45% on UBL (0.490 vs 0.888) [MEASURED].
**BLOCKER**: PDF = 67% of e-invoice storage. PDF compression is [UNMEASURED] — see R-V03.

### 3. Medical Imaging / PACS (TAM: $3-4B [UNSOURCED])

> **INVALIDATED** (2026-10-27). Redukt LOSES on medical imaging.

**Measured results** [MEASURED: gzip9-brotli11-comparison.csv]:
- x-ray: Redukt 6.524 vs brotli 4.592 → **+42.1% WORSE**
- mr: Redukt 2.283 vs brotli 2.373 → -3.8% (marginal, within noise)

**Conclusion**: Do NOT pursue this vertical. Hospital volume estimates (20 TB/year per
500-bed hospital [UNSOURCED], 30%/year growth [UNSOURCED]) are moot.

### 4. Cold/Archive Storage (TAM: $49B [UNSOURCED])

**Pain**: Long-term retention of compliance data, backups, historical records.
**Our advantage**: Best-in-class compression means less data to store/replicate/transfer.

### 5. Bandwidth-Constrained Niches (satellite/maritime/mining)

**Pain**: Satellite bandwidth costs $1-225/GB (traditional VSAT $10-225/GB, Starlink maritime ~$6/GB).
**Use case**: Vessels, mining sites, oil rigs transmitting logs/telemetry over expensive links.
**Economics**: Fleet of 100 vessels transmitting 10 GB/day of logs at $6/GB = $21.9M/year bandwidth.
With Redukt (-48%): saves **$10.5M/year** on bandwidth alone.
**Blocker**: Throughput (~117 B/s) insufficient to process 10 GB/day even with 8 cores.
**Status**: Highest value per byte compressed. Viable after throughput Phase 2.

### 6. Genomics ($2-5B TAM)

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

- AWS investing EUR 33.7B in Spain data centers (eu-south-2 Madrid)
- Spanish cloud adoption: 41.65% paid cloud, only 13.68% migrated software
- Key consultancies for channel: Indra/Minsait (EUR 5.46B), NTT Data Spain (15K+), Inetum (27K)
- E-invoicing market growth: $299M (2025) -> $1,139M (2034), CAGR 16.03%

## Sources

- Silicon.es: Spain 500M+ e-invoices (2024)
- El Economista: Edicom EUR 252.9M revenue
- Valencia Plaza: Edicom international expansion
- Vocal Media: Spain e-invoicing $1,139M by 2034
- Purview: PACS storage costs $180K/300TB
- QSS Technosoft: Hospital PACS $2M overspend analysis
- Cloud Storage Explorer: 2026 pricing comparison
