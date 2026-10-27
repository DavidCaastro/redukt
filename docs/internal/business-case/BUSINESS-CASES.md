# Business Cases — Concrete EUR Figures

Analysis date: 2026-10-20. **Audited 2026-10-27** — corrected with verified benchmark data.
All figures in EUR unless noted.

> **DATA INTEGRITY NOTE**: Every BPB number in this document is tagged with its source.
> [MEASURED] = from actual benchmark run, traceable to CSV.
> [UNMEASURED] = no benchmark exists for this data type. Case is blocked until measured.
> [CALCULATED] = derived from measured values with formula shown.

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

### Measured BPB (XML component only — 10% of mix)

Source: `data/benchmark/results/gzip9-brotli11-comparison.csv`

| File | Redukt Q8 [MEASURED] | brotli-11 [MEASURED] | gzip-9 [MEASURED] | Redukt vs brotli |
|---|---|---|---|---|
| facturae-batch.xml | 0.406 | 0.890 | 1.070 | **-54.4%** |
| ubl-invoice-batch.xml | 0.490 | 0.888 | 1.093 | **-44.9%** |

### BLOCKER: 90% of document mix is UNMEASURED

**PDF, digital signatures, and attachments have NEVER been benchmarked with Redukt.**
PDF represents 67% of transaction volume. Without PDF BPB data, any weighted projection
is speculation. The previous version of this document used "weighted BPB" of 1.28/2.05
that included unmeasured components — those numbers have been removed.

**Task required**: R-V03 (benchmark PDF compression) before this case can be quantified.

### What We Know (XML-only subset)

If the platform stored ONLY XML Facturae (hypothetical):
- Volume: 50M × 30 KB = **1.5 TB/year** (not 15 TB — XML is 10% of mix)
- [CALCULATED] brotli-11 stored: 1.5 TB × (0.890/8) = 0.167 TB/year
- [CALCULATED] Redukt stored: 1.5 TB × (0.406/8) = 0.076 TB/year
- Delta: -0.091 TB/year — marginal absolute savings at this scale

**Verdict**: BLOCKED — cannot assess until PDF compression is measured.

---

## Case 2: Large E-Invoicing Platform (Edicom Scale)

**Profile**: Edicom — EUR 252.9M revenue [SOURCE: El Economista], 100M+ transactions/year.

**Same BLOCKER as Case 1**: 90% of document mix (PDF) is UNMEASURED. All previous EUR
projections were based on unverifiable weighted BPB and have been removed.

XML-only advantage is strong (-54% vs brotli), but XML is only 10% of storage volume.
The business case depends entirely on whether Redukt can beat brotli on PDF.

**Verdict**: BLOCKED — requires R-V03 (PDF benchmark).

---

## Case 3: Public Health System — PACS Medical Imaging

> **INVALIDATED** (2026-10-27 audit). Previous BPB numbers were fabricated.
> Actual benchmarks show Redukt is WORSE than brotli on medical imaging.

**Profile**: Servicio Andaluz de Salud (SAS) — 50 hospitals.

### Actual Measured BPB (10KB, Q8)

Source: `data/benchmark/results/gzip9-brotli11-comparison.csv`

| Data type | Redukt Q8 [MEASURED] | brotli-11 [MEASURED] | gzip-9 [MEASURED] | Redukt vs brotli |
|---|---|---|---|---|
| x-ray | **6.524** | 4.592 | 5.873 | **+42.1%** (WORSE) |
| mr | **2.283** | 2.373 | 2.780 | **-3.8%** (marginal) |

Previous document claimed x-ray=3.87 (actual: 6.524, off by 68%) and mr=1.81 (actual: 2.283, off by 26%).
Previous document claimed -12.4% and -20.3% advantages — reality is +42.1% and -3.8%.

### Conclusion

Redukt is **not competitive** on medical imaging data. High-entropy binary data
(near-random pixel values) defeats neural prediction models.

All previous EUR projections (833K savings) were based on inverted data and are **void**.

**Verdict**: DEAD vertical. Do not pursue.

---

## Case 4: Enterprise Log Archival

**Profile**: Mid-size Spanish bank (Bankinter, Abanca, Kutxabank) or telco.

### Measured BPB (10KB, Q8)

Source: `data/benchmark/results/gzip9-brotli11-comparison.csv`

| Log type | Redukt Q8 [MEASURED] | gzip-9 [MEASURED] | brotli-11 [MEASURED] | vs gzip-9 | vs brotli-11 |
|---|---|---|---|---|---|
| apache-access.log | 0.245 | 0.546 | 0.466 | **-55.2%** | **-47.5%** |
| linux-syslog.log | 0.362 | 0.714 | 0.582 | **-49.3%** | **-37.8%** |
| openssh-auth.log | 0.475 | 0.990 | 0.783 | **-52.0%** | **-39.4%** |
| **Mean (3 log files)** | **0.361** | **0.750** | **0.610** | **-51.8%** | **-40.8%** |

### Volume [ESTIMATE — industry typical, not verified for specific client]

- Log generation: 2-5 TB/day
- Annual: **~1 PB/year**
- Regulatory retention: 5 years (BCE, CNMV, GDPR)
- Current compression: gzip-9 (fast, inline) [ASSUMPTION]

### Compression Projection

[CALCULATED from measured BPB, formula: stored = raw × (BPB/8)]

| Compressor | Mean log BPB [MEASURED] | Ratio | TB/year stored (1 PB raw) |
|---|---|---|---|
| gzip-9 | 0.750 | 9.38% | 93.8 TB |
| brotli-11 | 0.610 | 7.63% | 76.3 TB |
| **Redukt Q8** | **0.361** | **4.51%** | **45.1 TB** |
| **Delta vs gzip-9** | | | **-48.7 TB/year** |

5-year peak: gzip 469 TB vs Redukt 225.5 TB = **-243.5 TB saved**.

> **CAVEAT**: These BPB numbers are from 10KB samples. At GB scale, gzip-9 has larger
> dictionary windows and may perform better. Redukt also improves with more context.
> The relative advantage at production scale is UNKNOWN — requires R-V04 (scale test).

### Cost [CALCULATED from measured BPB, storage price is ESTIMATE]

Storage: EUR 30/TB/month [ESTIMATE — S3 Infrequent Access or compliance tier, unsourced]

| Year | gzip stored | gzip cost/yr | Redukt stored | Redukt cost/yr | Annual savings |
|---|---|---|---|---|---|
| 1 | 93.8 TB | 33,768 | 45.1 TB | 16,236 | **17,532** |
| 2 | 187.6 TB | 67,536 | 90.2 TB | 32,472 | 35,064 |
| 3 | 281.4 TB | 101,304 | 135.3 TB | 48,708 | 52,596 |
| 4 | 375.2 TB | 135,072 | 180.4 TB | 64,944 | 70,128 |
| 5 | 469.0 TB | 168,840 | 225.5 TB | 81,180 | **87,660** |

[CALCULATED] cost/yr = accumulated_TB × EUR 30/TB/month × 12

**Total storage savings 5 years: 262,980** [CALCULATED]

With 2x DR replicas (×3 total): **788,940** [CALCULATED, assumes same $/TB]

### Compute Cost Constraint [ESTIMATE]

Redukt throughput on logs: ~69 B/s pre-SIMD, ~90 B/s post-SIMD [CALCULATED from CSV]
Processing 1 PB/year (2.74 TB/day) at 90 B/s requires:
- Per core: 90 B/s × 86400 s = 7.78 MB/day
- Cores needed: 2,740,000 MB / 7.78 MB = **352,000 cores** [CALCULATED]

**This is not viable at current throughput.** Compute cost would vastly exceed savings.

### Verdict

**Direction correct — Redukt strongly beats gzip-9 and brotli-11 on logs (-52%, -41%).**
But the absolute savings are much smaller than previously claimed (263K vs 1,247K over 5y)
because gzip-9 is already very effective on logs (9.4% ratio, not 37.5% as previously stated).
And compute cost at current throughput makes the case **economically negative**.

**Required for viable case**: throughput ≥1 KB/s (R-012 + model optimization).

---

## Executive Summary (Corrected 2026-10-27)

| Case | Status | Key data | Blocker |
|---|---|---|---|
| E-invoicing mid | **BLOCKED** | XML: -54% vs brotli [MEASURED] | PDF never benchmarked (67% of mix) |
| E-invoicing large | **BLOCKED** | Same | Same |
| PACS hospital | **DEAD** | x-ray: +42% WORSE [MEASURED] | Redukt loses on medical imaging |
| Log archival | **DIRECTION CORRECT** | Logs: -52% vs gzip [MEASURED] | Throughput (~90 B/s) makes compute cost prohibitive |

### What Changed from Previous Version

1. **Case 3 (PACS)**: BPB numbers were fabricated (claimed 3.87, actual 6.524 for x-ray).
   Entire case invalidated — Redukt LOSES on medical imaging.
2. **Cases 1-2 (E-Invoicing)**: Weighted BPB removed — 90% of mix (PDF) was never measured.
3. **Case 4 (Logs)**: gzip-9 BPB corrected from 3.00 to 0.750 [MEASURED]. Redukt BPB
   corrected from 1.15 to 0.361 [MEASURED]. Savings reduced from 1,247K to 263K.
   Compute cost constraint added.
4. **All projections now traceable** to `gzip9-brotli11-comparison.csv`.
