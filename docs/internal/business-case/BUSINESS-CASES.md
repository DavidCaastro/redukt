# Business Cases — Concrete EUR Figures

> **SUPERSEDED (2026-11-10)** — The pricing (EUR 5-12/TB), throughput targets (>=2 KB/s), "no direct competitor",
> "best-in-class" and cold/archive claims in this folder are contradicted by measured data (compute ~$50-60/GB on CPU,
> ~$9-17/GB on T4; R* = 18-100 KB/s per vCPU) and by external evidence (Nacrith, Hutter 2026 lineage, cmix at ~$3/GB,
> CIDR 2026). Kept for traceability only. Authoritative analysis: `docs/internal/research/competitive-position-2026-11-10/REPORT.md`.


Analysis date: 2026-10-20. **Audited 2026-10-27, corrected 2026-10-28** with actual AZTH compression.
All figures in EUR unless noted.

> **DATA INTEGRITY NOTE**: Every BPB number in this document is tagged with its source.
> [MEASURED] = from actual benchmark run, traceable to JSON output.
> [UNMEASURED] = no benchmark exists for this data type. Case is blocked until measured.
> [CALCULATED] = derived from measured values with formula shown.
>
> **CORRECTION (2026-10-28)**: Previous Redukt BPB was cross-entropy (hybrid-eval), 5-20% lower
> than actual compressed file size. Updated to actual AZTH compression BPB where available.
> Only 3 files verified so far; other files' Redukt BPB are cross-entropy (pending re-measurement).

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

| File | Redukt BPB [MEASURED] | brotli-11 [MEASURED] | gzip-9 [MEASURED] | Redukt vs brotli | Note |
|---|---|---|---|---|---|
| facturae-batch.xml | **0.462** | 0.890 | 1.070 | **-48.1%** | actual AZTH compression |
| ubl-invoice-batch.xml | 0.490 (ce_bpb) | 0.888 | 1.093 | pending | cross-entropy, needs re-measurement |

### BLOCKER: 90% of document mix is UNMEASURED

**PDF, digital signatures, and attachments have NEVER been benchmarked with Redukt.**
PDF represents 67% of transaction volume. Without PDF BPB data, any weighted projection
is speculation. The previous version of this document used "weighted BPB" of 1.28/2.05
that included unmeasured components — those numbers have been removed.

**R-V03 result** [MEASURED]: Redukt on PDF = **11.11 BPB** vs brotli 7.90 BPB (+40.7% WORSE).
PDF binary content is near-incompressible. Redukt loses badly on PDF.

With 67% PDF in the mix, the blended BPB would be dominated by PDF losses,
negating the XML advantage. **E-invoicing is only viable with selective XML-only compression.**

CAVEAT: tested with W3C dummy.pdf (13KB). Real invoice PDFs with text may differ slightly.

### What We Know (XML-only subset)

If the platform stored ONLY XML Facturae (hypothetical):
- Volume: 50M × 30 KB = **1.5 TB/year** (not 15 TB — XML is 10% of mix)
- [CALCULATED] brotli-11 stored: 1.5 TB × (0.890/8) = 0.167 TB/year
- [CALCULATED] Redukt stored: 1.5 TB × (0.462/8) = 0.087 TB/year
- Delta: -0.080 TB/year — marginal absolute savings at this scale

**Verdict**: BLOCKED — cannot assess until PDF compression is measured.

---

## Case 2: Large E-Invoicing Platform (Edicom Scale)

**Profile**: Edicom — EUR 252.9M revenue [SOURCE: El Economista], 100M+ transactions/year.

**Same BLOCKER as Case 1**: 90% of document mix (PDF) is UNMEASURED. All previous EUR
projections were based on unverifiable weighted BPB and have been removed.

XML-only advantage is strong (-48% vs brotli, actual compression), but XML is only 10% of storage volume.
The business case depends entirely on whether Redukt can beat brotli on PDF.

**Verdict**: E-invoicing only viable if XML is separated from PDF for selective compression.
The storage savings would be limited to the XML portion (~10% of volume).

**R-V04 good news**: Scale test shows Redukt advantage on XML GROWS at larger sizes:
dickens -30.6% at 1KB → -46.0% at 100KB [MEASURED]. Apache stable at -43% to -48%.

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

| Log type | Redukt BPB [MEASURED] | gzip-9 [MEASURED] | brotli-11 [MEASURED] | vs gzip-9 | vs brotli-11 |
|---|---|---|---|---|---|
| apache-access.log | **0.309** | 0.546 | 0.466 | **-43.4%** | **-33.8%** |
| linux-syslog.log | **0.426** | 0.714 | 0.582 | **-40.3%** | **-26.7%** |
| openssh-auth.log | **0.538** | 0.990 | 0.783 | **-45.7%** | **-31.3%** |
| **Mean (3 log files)** | **0.424** | **0.750** | **0.610** | **-43.5%** | **-30.5%** |

All values from actual AZTH compression, measured 2026-10-28. Roundtrip-verified.

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
| **Redukt Q8** | **0.424** | **5.30%** | **53.0 TB** |
| **Delta vs gzip-9** | | | **-40.8 TB/year** |

5-year peak: gzip 469 TB vs Redukt 265 TB = **-204 TB saved**.

All Redukt BPB from actual AZTH compression (mean of 3 log files), measured 2026-10-28.

> **R-V04 RESULT** [MEASURED]: Scale test on apache-access.log shows:
> - 10KB: Redukt 0.245 vs brotli 0.466 = -47.5%
> - 50KB: Redukt 0.185 vs brotli 0.353 = -47.4%
> - 100KB: Redukt 0.186 vs brotli 0.328 = -43.4%
> Advantage persists at larger sizes. On dickens, advantage GROWS: -36% (10KB) → -46% (100KB).
> **Caveat 2026-11-10**: that scale test used the RWKV backbone. With Mamba (current default) enwik8 got WORSE from
> 10 KB to 50 KB (0.963 → 1.054); whole-file brotli-11 reaches 0.247 on UBL vs Redukt 0.446 at 10 KB.
> gzip-9 also improves (0.55→0.44 at 100KB) but Redukt maintains strong lead.
> See `data/benchmark/results/scale-test-complete.csv`.

### Cost [CALCULATED from measured BPB, storage price is ESTIMATE]

Storage: EUR 30/TB/month [ESTIMATE — S3 Infrequent Access or compliance tier, unsourced]

| Year | gzip stored | gzip cost/yr | Redukt stored | Redukt cost/yr | Annual savings |
|---|---|---|---|---|---|
| 1 | 93.8 TB | 33,768 | 53.0 TB | 19,080 | **14,688** |
| 2 | 187.6 TB | 67,536 | 106.0 TB | 38,160 | 29,376 |
| 3 | 281.4 TB | 101,304 | 159.0 TB | 57,240 | 44,064 |
| 4 | 375.2 TB | 135,072 | 212.0 TB | 76,320 | 58,752 |
| 5 | 469.0 TB | 168,840 | 265.0 TB | 95,400 | **73,440** |

[CALCULATED] cost/yr = accumulated_TB × EUR 30/TB/month × 12

**Total storage savings 5 years: 220,320** [CALCULATED]

With 2x DR replicas (×3 total): **660,960** [CALCULATED, assumes same $/TB]

### Compute Cost Constraint [ESTIMATE]

Redukt throughput on logs: ~132 B/s [MEASURED 2026-10-28, apache-access.log]
Processing 1 PB/year (2.74 TB/day) at 132 B/s requires:
- Per core: 132 B/s × 86400 s = 11.4 MB/day
- Cores needed: 2,740,000 MB / 11.4 MB = **240,000 cores** [CALCULATED]

**This is not viable at current throughput.** Compute cost would vastly exceed savings.

### Verdict

**Direction correct — Redukt strongly beats gzip-9 and brotli-11 on logs (-44% vs gzip, -31% vs brotli, mean of 3 log files, actual compression).**
But compute cost at current throughput (~80 B/s mean on logs) makes the case **economically negative**.

**Required for viable case**: ~~throughput ≥2 KB/s~~ — corrected 2026-11-10: `scripts/breakeven.py` requires 18-100 KB/s
**per vCPU** (S3 Standard, 7 years, spot); Fase 19 reaches ~1.2 KB/s per 12-vCPU machine (~0.1 KB/s per vCPU).

---

## Executive Summary (Corrected 2026-10-27)

| Case | Status | Key data | Blocker |
|---|---|---|---|
| E-invoicing mid | **BLOCKED** | XML: -48% vs brotli (actual compression) [MEASURED] | PDF never benchmarked (67% of mix) |
| E-invoicing large | **BLOCKED** | Same | Same |
| PACS hospital | **DEAD** | x-ray: +42% WORSE [MEASURED ce_bpb] | Redukt loses on medical imaging |
| Log archival | **DIRECTION CORRECT** | Logs: -31% vs brotli (mean 3 files, actual) [MEASURED] | Throughput (~80 B/s mean) makes compute cost prohibitive |

### What Changed from Previous Version

1. **Case 3 (PACS)**: BPB numbers were fabricated (claimed 3.87, actual 6.524 for x-ray).
   Entire case invalidated — Redukt LOSES on medical imaging.
2. **Cases 1-2 (E-Invoicing)**: Weighted BPB removed — 90% of mix (PDF) was never measured.
3. **Case 4 (Logs)**: gzip-9 BPB corrected from 3.00 to 0.750 [MEASURED]. Redukt BPB
   corrected from 1.15 to 0.361 (ce_bpb) [MEASURED]. Savings reduced from 1,247K to 263K.
   Compute cost constraint added.
4. **All projections now traceable** to `gzip9-brotli11-comparison.csv`.
5. **(2026-10-28)**: Redukt BPB corrected from cross-entropy to actual AZTH compression.
   Apache logs: 0.245 (ce) → 0.309 (actual). Facturae: 0.406 (ce) → 0.462 (actual).
   Advantages reduced: logs -48% → -34%, XML -54% → -48%. Coder efficiency 80-95%.
