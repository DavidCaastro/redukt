# Domain Coverage — BPB per Domain + Gap Analysis

**Date**: 2026-10-15
**Purpose**: Comprehensive per-domain BPB tracking and coverage gap identification.
**Config**: Redukt post-(14 CM + RWKV-7 0.1B Q8 + match + LSTM BPTT=64
+ 48 ctx + order-chain + neural-blend + emb surgery center0.3)

---

## Tier 1 — Quick Eval (5 files, 10KB each)

| Domain | Type | Best BPB | Date | Current  | Delta | Best config |
|--------|------|----------|------|---------------|-------|-------------|
| samba | C source code | **1.0437** | 2026-10-09  | 1.0437 | = | order-chain + neural-blend |
| enwik8 | Text EN (Wikipedia) | **1.1633** | 2026-10-09  | 1.1633 | = | order-chain + neural-blend |
| dickens | Text EN (literary) | **1.3573** | 2026-10-09  | 1.3573 | = | order-chain + neural-blend |
| mozilla | Executable x86 | **1.6404** | 2026-10-07 (S3) | 1.8184 | +0.1780 | S3 without order-chain |
| OEIS | Integer sequences | **1.8184** | 2026-10-09  | 1.8184 | = | order-chain + neural-blend |
| **Composite** | | **mean=1.4412** | | 1.4412 | = | sigma=0.2574, worst=1.8184 |

Note: mozilla regressed with order-chain/neural-blend (+0.18 at 10KB). Best for
mozilla was S3 without those flags. But composite improved because the other 4
files improved more.

---

## Tier 2b — Silesia Corpus (12 files, 100KB each)

| Domain | Type | Cluster | Best BPB | Date | Current  | Delta | Notes |
|--------|------|---------|----------|------|---------------|-------|-------|
| xml | XML markup | A | **0.2537** | 2026-10-09  | 0.2537 | = | Saturated |
| nci | Chemical data (SMILES) | A | **0.3124** | 2026-10-09  | 0.3124 | = | Saturated |
| samba | C source code | B | **1.0523** | 2026-10-09  | 1.0523 | = | |
| mozilla | Executable x86 | C | **1.1253** | 2026-10-09  | 1.1253 | = | |
| webster | English dictionary | B | **1.1969** | 2026-10-09  | 1.1969 | = | |
| dickens | English literary text | B | **1.3387** | 2026-10-09  | 1.3387 | = | |
| reymont | Polish literary text | C | **1.3151** | 2026-10-08  | 1.3557 | +0.0406 | regressed (UTF-8 diacritics) |
| mr | Medical MR image | C | **1.3964** | 2026-10-09  | 1.3964 | = | DICOM int16 |
| osdb | MySQL database dump | C | **2.3343** | 2026-10-09  | 2.3343 | = | 4-byte aligned records |
| ooffice | Office binary (OLE2) | D | **2.9265** | 2026-10-08  | 3.2881 | +0.3616 | regressed (heterogeneous) |
| x-ray | X-ray DICOM | D | **3.7753** | 2026-10-09  | 3.7753 | = | int16 pixel data |
| sao | Astronomy IEEE-754 | D | **4.9844** | 2026-10-09  | 4.9844 | = | Dense float32 arrays |
| **Composite** | | | **mean=1.8678** | | 1.8678 | = | sigma=1.3857, worst=4.9844 |

Note: ooffice (+0.36) and reymont (+0.04) are worse under than baseline.
The current config (order-chain + neural-blend) hurts those specific domains.

---

## enwik8 at Multiple Scales

| Scale | Best BPB | Date | Config |
|-------|----------|------|--------|
| 10KB | **1.1633** | 2026-10-09  | order-chain + neural-blend |
| 100KB | **1.1810** | 2026-10-08  | BPTT=64 + 40 ctx (pre-enrichment) |
| 200KB | **1.1255** | 2026-10-14  | Same config, larger scale |

BPB improves monotonically with scale. Convergence threshold ~350-700KB .

---

## Tier 3 — Modern Corpus (11 files, 100KB/10KB)

| Domain | Type | Cluster | BPB | Date | Eval bytes | Notes |
|--------|------|---------|-----|------|------------|-------|
| ml-weights | SafeTensors | A | **0.4842** | 2026-10-08  | 10KB | ML weights, highly structured |
| ait-B | Modern C source | A | **0.9730** | 2026-10-08  | 100KB | |
| structured-jsonl | JSON-Lines | C | **1.1195** | 2026-10-08  | 10KB | |
| ait-C | Wikipedia EN | B | **1.3319** | 2026-10-08  | 100KB | |
| ait-G | Astronomy image | C | **2.3078** | 2026-10-08  | 100KB | Likely int16 FITS |
| modern-x64-pe | x86-64 PE executable | D | **2.9698** | 2026-10-08  | 100KB | |
| ait-H | Executable | D | **3.0514** | 2026-10-08  | 100KB | |
| ait-A | Protein sequences | D | **3.9225** | 2026-10-08  | 100KB | ~20 amino acid alphabet |
| ait-F | Astronomy image | D | **6.2775** | 2026-10-08  | 100KB | FITS float probable |
| ait-E | CERN floats | D | **6.7401** | 2026-10-08  | 100KB | Dense IEEE-754 |
| ait-D | Pseudo-random | D | **7.9891** | 2026-10-08  | 100KB | Theoretical ceiling ~8.0 |
| **Composite** | | | **mean=3.3788** | | | sigma=2.5716 |

**T3 was never re-evaluated with config** (order-chain + neural-blend).
These are baseline numbers only.

---

## Regressions Under Current Config 

Two domains are worse under the current config vs their historical best:

| Domain | Best ever | Best config | Current  | Regression | Root cause |
|--------|-----------|-------------|---------------|------------|------------|
| ooffice | 2.9265 | baseline | 3.2881 | +0.3616 | OLE2 heterogeneous structure; order-chain creates spurious cross-sector hash contexts |
| reymont | 1.3151 | baseline | 1.3557 | +0.0406 | Polish UTF-8 diacritics; neural-blend may interfere with byte-pair patterns |
| mozilla (10KB) | 1.6404 | S3 | 1.8184 | +0.1780 | At 10KB, order-chain hasn't converged for variable-length x86 instructions |

---

## Domain Coverage Gaps

### Missing domains — High priority (would reveal architectural weaknesses)

| Absent domain | Why it matters | Est. BPB | Data source |
|---------------|---------------|----------|-------------|
| **Audio PCM (WAV)** | Array of int16 with strong temporal correlation. Direct test for RecordModel/LinearPredModel viability. If >4.0 BPB, confirms specialized models are urgent. | ~3-5 | Any uncompressed WAV |
| **Chinese/Japanese text** | UTF-8 3-4 bytes/char. RWKV "World" tokenizer is multilingual — never validated. If ~1.3 BPB, backbone isn't English-biased. If >2.0, tokenizer problem. | ~1.3-1.6 | Wikipedia CJK dumps |
| **Dense CSV/TSV** | Tabular with fixed delimiters. Should fall in cluster A (<1.0) by structural repetition. If not, RecordModel would help from another angle. | ~0.5-1.0 | Kaggle datasets |

### Missing domains — Medium priority

| Absent domain | Why it matters | Est. BPB | Data source |
|---------------|---------------|----------|-------------|
| **Python/JS/Rust code** | Modern code with different indentation than C. Only C tested (samba, ait-B). | ~1.0-1.2 | Any open source repo |
| **PDF** | Complex container: text + binary + compressed streams + fonts. Real-world mixed-domain stress test. | ~3-5 | Any PDF |
| **Genomic (FASTA)** | 4-letter alphabet (ACGT). Entropy ~2.0 bits/base but with biological patterns (codons, motifs). | ~1.5-2.5 | NCBI/GenBank |
| **Syslog/application logs** | Highly repetitive structured text with timestamps. Should be cluster A-B. | ~0.5-1.0 | /var/log |
| **Network capture (pcap)** | Repetitive IP/TCP headers + variable payload. Mixed structure. | ~2-4 | Wireshark samples |

### Missing domains — Low priority (sanity checks or niche)

| Absent domain | Why it matters | Est. BPB | Data source |
|---------------|---------------|----------|-------------|
| **Encrypted (AES output)** | Must give ~8.0. Sanity check (like ait-D). | ~8.0 | openssl rand |
| **Compressed (gzip output)** | Near-random. Indirectly covered by ait-D. | ~7.9-8.0 | gzip any file |
| **Raw video (YUV)** | Frames with temporal/spatial correlation. High entropy but patterned. | ~3-6 | ffmpeg extraction |
| **Sparse scientific (HDF5)** | Arrays with many zeros + floats. Mix of cluster A and E. | ~2-4 | CERN/NASA open data |
| **Social text (emojis/Unicode)** | UTF-8 4-byte sequences. Chat logs, social media. | ~1.5-2.0 | Twitter/Reddit dumps |

---

## Coverage Map

```
COVERED                               ABSENT (critical gaps marked *)
───────                                ──────
A (structured):  xml, nci, ml-wt       CSV*, TSV, logs
B (text):        EN, PL, code-C        ZH*, JA*, AR, code-Py/JS/Rust
C (binary):      exe, office, DB       PDF, pcap, audio*
D (hard):        x-ray, MR, astro      video raw, HDF5
E (float):       CERN, SAO             audio float, sparse scientific
∞ (random):      ait-D                 encrypted (sanity check)
```

Three critical gaps (audio, CJK text, CSV) would directly inform Phase 3d
(RecordModel, LinearPredModel) and validate RWKV backbone generalization.

---

## Per-Cluster Summary vs Targets 

| Cluster | Target | Best 100KB | Current 100KB | Files | Gap | Status |
|---------|--------|------------|---------------|-------|-----|--------|
| A (structured) | ≤0.25 | 0.29 avg | 0.29 avg | xml, nci | 0.04 | ~85% — closes with scale |
| B (text) | ≤1.05 | 1.05 avg | 1.05 avg | samba, webster, dickens | ~0 | ~70% — nearly met |
| C (binary) | ≤1.50 | 1.77 avg | 1.77 avg | mozilla, mr, osdb, reymont | 0.27 | ~50% — needs RecordModel |
| D (hard) | ≤3.50 | 3.69 avg | 3.69 avg | ooffice, x-ray, sao | 0.19 | ~35% — needs RecordModel + LinearPred |
| E (float) | ≤4.50 | 6.51 avg | 6.51 avg | ait-E, ait-F | 2.01 | ~20% — hardest gap, mantissa entropy |

---

## References

- T1/T2 baselines: `docs/results/INDEX.md`
- T2b/T3 detailed: research documentation
- T2b detailed: `docs/results/INDEX.md` §T2b R56
- ablation analysis: research documentation
- per-cluster targets: research documentation
- scale transition: research documentation
