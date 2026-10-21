# Market Analysis — Redukt

## Global Compression Market

- Data compression market: **$1.28B** (2026), CAGR 9.2%
- Drivers: cloud storage costs, regulatory retention, data growth

## Target Verticals

### 1. Enterprise Log Archival ($5B+ TAM)

**Pain**: 2-5 TB/day of logs, 5-year regulatory retention (BCE, CNMV, GDPR).
**Validation**: Cribl ($390M ARR, $3.5B valuation) proves the market pays for data reduction.
**Our advantage**: 61.7% better than gzip-9 on structured text (benchmark proven).
**Target clients**: Banks (Bankinter, Abanca, Kutxabank), telcos (Telefonica, MasMovil), utilities.

### 2. E-Invoicing Platforms ($1.1B TAM Spain by 2034)

**Context**: Spain mandatory e-invoicing from July 2026 (Ley Crea y Crece, RD 238/2026).
- 557M+ e-invoices B2B in Spain (2024), growing 21%/year
- Format: XML Facturae/UBL, 6-year retention
- Market leaders: SERES (1000M+ docs/year), Edicom (EUR 252.9M revenue, 60% margin, 17K clients)

**Our advantage**: 44.7% better than brotli-11 on XML (0.492 vs 0.89 BPB).

### 3. Medical Imaging / PACS ($3-4B TAM)

**Pain**: Hospital 500 beds = 20 TB/year DICOM, growing 30%/year. 10-year retention.
**Validation**: PetaGene charges per TB saved in genomics, similar model applicable.
**Our advantage**: 12-20% better than brotli on medical imaging data.
**Target**: Servicio Andaluz de Salud (50 hospitals, EUR 10B budget), Sermas (Madrid).

### 4. Cold/Archive Storage ($49B TAM global)

**Pain**: Long-term retention of compliance data, backups, historical records.
**Our advantage**: Best-in-class compression means less data to store/replicate/transfer.

### 5. Genomics ($2-5B TAM)

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
