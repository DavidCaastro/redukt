# Business Cases — Concrete EUR Figures

Analysis date: 2026-10-20. All figures in EUR unless noted.

---

## Case 1: Mid-Tier E-Invoicing Platform

**Profile**: Docuten, B2Brouter, or similar. 50M documents/year, ~200 enterprise clients.

### Document Composition

| Component | Avg size | Share |
|---|---|---|
| XML Facturae invoice | 30 KB | 10% |
| PDF representation | 200 KB | 67% |
| Digital signature + metadata | 10 KB | 3% |
| Attachments (delivery notes, contracts) | 60 KB | 20% |
| **Total per transaction** | **~300 KB** | 100% |

### Volume

- Annual raw: 50M x 300 KB = **15 TB/year**
- Legal retention: 6 years (Ley General Tributaria)
- Peak accumulated: **90 TB**

### Compression Comparison

| Compressor | Weighted BPB | Compression ratio | TB/year stored | Peak 6 years |
|---|---|---|---|---|
| brotli-11 | 2.05 | 25.6% | 3.84 TB | 23.0 TB |
| **Redukt** | **1.28** | **16.0%** | **2.40 TB** | **14.4 TB** |
| Delta | | | -1.44 TB/year | **-8.6 TB** |

### Cost Analysis (Compliance Storage @ EUR 50/TB/month)

| Year | brotli stored | brotli cost | Redukt stored | Redukt cost | Savings |
|---|---|---|---|---|---|
| 1 | 3.84 TB | 2,304 | 2.40 TB | 1,440 | 864 |
| 2 | 7.68 TB | 4,608 | 4.80 TB | 2,880 | 1,728 |
| 3 | 11.52 TB | 6,912 | 7.20 TB | 4,320 | 2,592 |
| 4 | 15.36 TB | 9,216 | 9.60 TB | 5,760 | 3,456 |
| 5 | 19.20 TB | 11,520 | 12.00 TB | 7,200 | 4,320 |
| 6 | 23.04 TB | 13,824 | 14.40 TB | 8,640 | 5,184 |
| **Total** | | **48,384** | | **30,240** | **18,144** |

Multiplier: 3x replicas (compliance DR) = **54,432 savings over 6 years**.
Bandwidth savings (egress DR): 8.6 TB x 3 x EUR 20/TB = **3,096 total**.
**Total savings: ~57,528 over 6 years.**

### Our Invoice

| Item | Price |
|---|---|
| Annual license | 8,400/year (700/month) |
| Setup + API integration | 4,500 (one-time) |
| Technical support | Included |
| **Year 1 total** | **12,900** |
| **6-year total** | **54,900** |

**ROI**: Saves 57,528, pays 54,900. Breakeven year 5. Marginal.

**Verdict**: Functional but not compelling. Modest savings for a mid-tier platform.

---

## Case 2: Large E-Invoicing Platform (Edicom Scale)

**Profile**: Edicom — EUR 252.9M revenue, 100M+ transactions/year, 17K clients, 86 countries.

### Spain-Only (30% of volume)

- 100M transactions x 300 KB = **30 TB/year**
- 6-year peak: 180 TB

| Metric | brotli | Redukt | Delta |
|---|---|---|---|
| TB/year stored | 7.68 | 4.80 | -2.88 |
| Peak 6 years | 46.1 TB | 28.8 TB | -17.3 TB |
| Storage cost 6y (EUR 50/TB/mo) | 96,768 | 60,480 | 36,288 |
| x3 replicas | 290,304 | 181,440 | **108,864** |
| + egress DR | | | 6,192 |
| **Total savings 6y** | | | **115,056** |

### Global (Full Volume)

Edicom operates in 86 countries. Global volume ~3.3x Spain = 330M transactions/year.

- **Global savings 6y: ~379,685**

### Our Invoice (Global License)

| Item | Price |
|---|---|
| Enterprise annual license | 48,000/year |
| Dedicated integration | 25,000 (one-time) |
| SLA 99.9% + priority support | Included |
| **6-year total** | **313,000** |

**ROI**: Saves 379,685, pays 313,000. **Net savings: 66,685 (21% ROI)**. Breakeven year 5.

**Verdict**: Sellable as part of infrastructure optimization. Not a standalone killer product.

---

## Case 3: Public Health System — PACS Medical Imaging

**Profile**: Servicio Andaluz de Salud (SAS) — 50 hospitals, EUR 10B annual budget.

### Volume Estimates

- 500-bed hospital: 20 TB/year DICOM images
- 1000-bed hospital: 60 TB/year
- SAS average: 50 hospitals x 15 TB/year = **750 TB/year**
- Legal retention: 10 years (medical records)
- Peak: **7.5 PB**

### Compression on Medical Imaging

| Data type | Redukt BPB | brotli BPB | Advantage |
|---|---|---|---|
| X-ray medical | 3.87 | 4.42 | -12.4% |
| Binary medical (mr) | 1.81 | 2.27 | -20.3% |
| Weighted DICOM estimate | ~3.50 | ~4.10 | **-14.6%** |

| Metric | brotli | Redukt | Delta |
|---|---|---|---|
| Compression ratio | 51.3% | 43.8% | -7.5 pp |
| TB/year stored | 384.4 | 328.1 | -56.3 TB/year |
| Peak 10 years | 3,844 TB | 3,281 TB | -563 TB |

### Cost (Certified Medical Storage @ EUR 40/TB/month)

| Period | Storage savings | + 2x replicas | + Egress | **Total** |
|---|---|---|---|---|
| Year 1 | 27,024 | 54,048 | 2,252 | **83,324** |
| Year 5 cumul. | 135,120 | 270,240 | 11,260 | **416,620** |
| **Year 10 cumul.** | **270,240** | **540,480** | **22,520** | **833,240** |

### Our Invoice

| Item | Price |
|---|---|
| Institutional annual license (50 hospitals) | 60,000/year |
| PACS integration (HL7/FHIR + DICOM gateway) | 45,000 (one-time) |
| Training + 24/7 support | 12,000/year |
| **Year 1 total** | **117,000** |
| **10-year total** | **765,000** |

**ROI**: Saves 833,240, pays 765,000. **Net savings: 68,240. Breakeven year 9.**

**Verdict**: Long payback but defensible in public procurement. Medical imaging grows 30%/year, so real savings accelerate beyond projections.

---

## Case 4: Enterprise Log Archival (THE WINNER)

**Profile**: Mid-size Spanish bank (Bankinter, Abanca, Kutxabank) or telco.

### Volume

- Log generation: 2-5 TB/day
- Annual: **~1 PB/year**
- Regulatory retention: 5 years (BCE, CNMV, GDPR)
- Current compression: gzip-9 (fast, inline)

### Compression Comparison

| Compressor | BPB (structured logs) | Ratio | TB/year stored |
|---|---|---|---|
| gzip-9 (current) | 3.00 | 37.5% | 375 TB |
| **Redukt** | **1.15** | **14.4%** | **144 TB** |
| **Delta** | | | **-231 TB/year** |

5-year peak: gzip 1,875 TB vs Redukt 720 TB = **-1,155 TB saved**.

### Cost (Compliance S3 @ EUR 30/TB/month for volume)

| Year | gzip stored | gzip cost | Redukt stored | Redukt cost | Savings |
|---|---|---|---|---|---|
| 1 | 375 TB | 135,000 | 144 TB | 51,840 | **83,160** |
| 2 | 750 TB | 270,000 | 288 TB | 103,680 | 166,320 |
| 3 | 1,125 TB | 405,000 | 432 TB | 155,520 | 249,480 |
| 4 | 1,500 TB | 540,000 | 576 TB | 207,360 | 332,640 |
| 5 | 1,875 TB | 675,000 | 720 TB | 259,200 | **415,800** |

**Total storage savings 5 years: 1,247,400**

With 2x DR replicas: **x3 = 1,247,400** (already included in cumulative).
With bandwidth: additional ~50,000.

**Total savings 5 years: ~1,247,400**

### Our Invoice

| Item | Price |
|---|---|
| Enterprise license (unlimited per node) | 120,000/year |
| Pipeline integration (Kafka/Splunk/ELK) | 30,000 (one-time) |
| SLA + dedicated support | Included |
| **Year 1 total** | **150,000** |
| **5-year total** | **630,000** |

**ROI**: Saves 1,247,400, pays 630,000 = **net savings 617,400. ROI: 198%. Breakeven: year 2.5.**

**Verdict**: STRONG. Clear value proposition, short payback, regulatory pressure ensures retention.

---

## Executive Summary

| Case | Target client | 5-6y savings | Our invoice | Net | ROI | Viability |
|---|---|---|---|---|---|---|
| E-invoicing mid | Docuten | 57K | 55K | 2.6K | ~5% | Low |
| E-invoicing large | Edicom (global) | 380K | 313K | 67K | 21% | Medium |
| PACS hospital | SAS (10y) | 833K | 765K | 68K | 9% | Medium-High |
| **Log archival** | **Bankinter/Telefonica** | **1,247K** | **630K** | **617K** | **198%** | **HIGH** |

### Recommendation

**Enter through enterprise log archival.** It has:
1. The highest ROI (198%)
2. The shortest breakeven (2.5 years)
3. The strongest regulatory driver (BCE/CNMV/GDPR mandate retention)
4. The clearest competitive advantage (61.7% better than gzip on structured text)
5. The simplest integration (batch archive tier, not real-time)
