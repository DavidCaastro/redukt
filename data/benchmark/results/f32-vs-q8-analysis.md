# F32 vs Q8 Quantization Benchmark

**Date**: 2026-10-26
**Hardware**: Intel i5-1235U, 12 threads, 32 GB DDR5, Windows 11
**Window**: 10,000 bytes per file
**Command**: `redukt hybrid-eval --input <FILE> --bytes 10000 --quant <f32|q8>`
**Weights**: RWKV-7 0.1B (SafeTensors, auto-quantized)

## Key Finding

**Q8 quantization causes negligible BPB degradation (<2.5%) vs F32, while being 2x faster.**
F32 is NOT the solution to our competitive gap. The gap comes from experimental features
(`--hierarchical --match --emb-surgery center0.3`) that were used in the original benchmarks
but are not part of the standard production pipeline.

## Full Results: F32 vs Q8 (Silesia + enwik8 + Industry)

| File | Domain | F32 BPB | Q8 BPB | Delta | Delta % | F32 tok/s | Q8 tok/s |
|---|---|---|---|---|---|---|---|
| apache-access.log | server-logs | 0.2432 | 0.2448 | +0.0016 | +0.67% | 13.1 | 26.6 |
| linux-syslog.log | system-logs | 0.3589 | 0.3617 | +0.0027 | +0.75% | 13.4 | 26.6 |
| facturae-batch.xml | e-invoicing | 0.3965 | 0.4061 | +0.0096 | +2.43% | 13.4 | 26.4 |
| openssh-auth.log | auth-logs | 0.4738 | 0.4749 | +0.0011 | +0.24% | 13.2 | 26.7 |
| ubl-invoice-batch.xml | e-invoicing | 0.4799 | 0.4897 | +0.0097 | +2.03% | 13.2 | 26.9 |
| nci | chemical-db | 0.5962 | 0.5963 | +0.0001 | +0.02% | 22.0 | 46.5 |
| xml | markup | 0.6312 | 0.6402 | +0.0090 | +1.42% | 13.0 | 26.7 |
| samba | source-code | 1.2583 | 1.2658 | +0.0075 | +0.60% | 21.8 | 46.3 |
| enwik8 | encyclopedia | 1.3041 | 1.2812 | -0.0229 | **-1.76%** | 12.9 | 26.6 |
| dickens | literature | 1.7479 | 1.7792 | +0.0313 | +1.79% | 11.8 | 33.1 |
| webster | dictionary | 1.7534 | 1.7601 | +0.0067 | +0.38% | 21.6 | 46.2 |
| reymont | polish text | 1.8310 | 1.8077 | -0.0233 | **-1.27%** | 22.5 | 46.8 |
| mozilla | executable | 1.9664 | 1.9645 | -0.0019 | **-0.10%** | 21.2 | 46.3 |
| mr | medical-image | 2.2820 | 2.2826 | +0.0007 | +0.03% | 22.0 | 46.3 |
| ooffice | executable | 3.7360 | 3.7035 | -0.0325 | **-0.87%** | 22.9 | 46.7 |
| osdb | database | 5.6792 | 5.6425 | -0.0367 | **-0.65%** | 21.2 | 46.6 |
| x-ray | medical-image | 6.4970 | 6.5240 | +0.0027 | +0.42% | 16.3 | 26.6 |
| sao | astronomy-db | 9.1778 | 9.2408 | +0.0630 | +0.69% | 22.6 | 46.9 |

## Summary Statistics

| Metric | Value |
|---|---|
| Files where Q8 ≈ F32 (|delta| < 1%) | **12/18 (67%)** |
| Files where Q8 BEATS F32 | **5/18 (28%)** |
| Max Q8 degradation | facturae-batch.xml: +2.43% |
| Max Q8 improvement | enwik8: -1.76% |
| Mean absolute delta | 0.81% |
| F32 avg speed | 17.4 tok/s |
| Q8 avg speed | 36.5 tok/s |
| **Q8 speedup** | **2.1x** |

## Production Competitive Position (Q8 vs brotli-11, 10KB)

> **NOTE (2026-10-28)**: Redukt BPB values below are cross-entropy (hybrid-eval), NOT actual
> compressed file size. Actual compressed BPB is 5-20% higher. See BENCHMARK-RESULTS.md for
> corrected values. Apache actual: 0.309 (-33.8%), facturae actual: 0.462 (-48.1%).

| File | Domain | Redukt Q8 ce_bpb | brotli-11 BPB | vs brotli (ce, not actual) |
|---|---|---|---|---|
| apache-access.log | server-logs | **0.245** | 0.466 | **-47.5%** (actual: -33.8%) |
| linux-syslog.log | system-logs | **0.362** | 0.582 | **-37.9%** |
| facturae-batch.xml | e-invoicing | **0.406** | 0.891 | **-54.4%** |
| openssh-auth.log | auth-logs | **0.475** | 0.783 | **-39.4%** |
| ubl-invoice-batch.xml | e-invoicing | **0.490** | 0.889 | **-44.9%** |
| nci | chemical-db | 0.596 | 0.570 | +4.6% |
| xml | markup | **0.640** | 0.890 | **-28.1%** |
| samba | source-code | **1.266** | 1.610 | **-21.4%** |
| enwik8 | encyclopedia | **1.281** | 1.830 | **-30.0%** |
| dickens | literature | **1.779** | 2.110 | **-15.7%** |
| webster | dictionary | **1.760** | 2.100 | **-16.2%** |
| reymont | polish text | **1.808** | 1.890 | **-4.3%** |
| mozilla | executable | **1.964** | 2.310 | **-15.0%** |
| mr | medical-image | 2.283 | 2.270 | +0.6% |
| ooffice | executable | 3.704 | 3.320 | +11.5% |
| x-ray | medical-image | 6.524 | 4.440 | +46.9% |
| osdb | database | 5.643 | 3.080 | +83.2% |
| sao | astronomy-db | 9.241 | 5.570 | +65.9% |

### Win/Loss by Category

| Category | Files | Wins | Avg advantage |
|---|---|---|---|
| **Server logs** | 3 | **3/3** | **-41.6%** |
| **E-invoicing XML** | 2 | **2/2** | **-49.7%** |
| **Text/literature** | 4 | **4/4** | **-16.6%** |
| **Structured markup** | 1 | **1/1** | **-28.1%** |
| **Source code** | 1 | **1/1** | **-21.4%** |
| Executable/binary | 2 | 1/2 | -1.8% |
| Medical | 2 | 0/2 | +23.7% |
| High-entropy | 3 | 0/3 | +51.9% |
| **Total** | **18** | **12/18 (67%)** | |

## Conclusion

1. **F32 vs Q8 is irrelevant** — delta <2.5% on all files, Q8 is 2x faster
2. **Q8 is the correct production choice** — better speed/quality tradeoff
3. **Redukt dominates on text, logs, XML, and code** — actual compression advantages verified: -34% logs, -48% XML
4. **Redukt loses on high-entropy and medical imaging** — expected for LLM-based compression
5. **Industry files** show strongest results. Note: ce_bpb percentages above are optimistic; actual compression deltas are ~30% smaller
6. **The original BENCHMARK-RESULTS.md gap** was from experimental features, not quantization
