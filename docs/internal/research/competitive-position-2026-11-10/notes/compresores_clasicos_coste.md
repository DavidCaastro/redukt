# Classical and domain-specific lossless compression in 2026: the "cheap baseline" (ratio + compute cost per GB)

Research date: 2026-11-10. All prices/benchmarks dated inline. Cost formula used throughout:

```
seconds_per_GB  = 1e9 bytes / (R MB/s * 1e6)  = 1000 / R
hours_per_GB    = 1000 / (3600 * R)           = 0.27778 / R
cost_per_GB ($) = P_vcpu_hour * 0.27778 / R        (R = single-thread MB/s on one vCPU)
```
Inputs used: P_spot = $0.0175/vCPU-h (c7a.large spot $0.035/h / 2 vCPU), P_on-demand = $0.0515/vCPU-h
(c7a.large $0.103/h / 2 vCPU), both from [instances.vantage.sh, fetched 2026-11-10](https://instances.vantage.sh/aws/ec2/c7a.large).
The requested bracket $0.01-0.04/vCPU-h is also shown where useful. Multithreading changes wall-clock, not cost per GB
(cost is per vCPU-second), so single-thread MB/s is the right input.

Repo-internal anchors used for comparison (local files, not web sources):
- `data/benchmark/results/dual-backbone-benchmark.csv` (2026-10-31): Redukt Mamba+CM 10 KB BPB and brotli-11 / gzip-9 BPB on the same 10 KB slices.
- `data/benchmark/results/competitors-fullsize-2026-11-01.csv`: gzip-9 / brotli-11 / xz-9 BPB and MB/s on full Silesia files and on the repo's log/XML samples (dev laptop i5-1235U).
- `CLAUDE.md` "Commercial Target": Redukt CPU lockstep ~1,180 B/s on 12 vCPU = $50-60/GB spot; GPU T4 N=128 ~4.2 KB/s = $9-17/GB.

---

## KQ1. General-purpose compressors (zstd, xz, brotli, bzip2, zpaq, paq8px, cmix, ppmd, lrzip, 7-zip): ratio and speed on enwik8 and Silesia

### Takeaway
On enwik8 the classical frontier runs from gzip 2.92 bpb (~10 MB/s) through xz/zstd/brotli ~2.0 bpb (0.5-3 MB/s), 7-zip PPMd 1.70 bpb (~2 MB/s), zpaq -m5 1.43 bpb (~0.15 MB/s), paq8px 1.27 bpb (~3.4 KB/s) to cmix 1.17 bpb (~1.6 KB/s). On a cloud vCPU that is $0.00002/GB (zstd -1) to ~$0.002-0.009/GB (xz -9 / zstd -22 / brotli -11), ~$0.03/GB (zpaq -m5) and $1.4-3/GB (paq8px / cmix) at spot prices; the neural 0.96 bpb point costs $50-60/GB on CPU today, i.e. 16-20x more than cmix for ~18% fewer bits.

### Cited Findings

**lzbench 2.4 (single thread, AMD EPYC 9555P 3.2 GHz, turbo off, Ubuntu 26.04, gcc 15.2, `silesia.tar` 211,947,520 B; README does not state the run date but lists zstd 1.5.7, xz 5.8.4, brotli 1.2.0, lzma 26.03, so 2026)** — [inikep/lzbench README](https://github.com/inikep/lzbench)

| Compressor / level | Comp MB/s | Decomp MB/s | Compressed bytes | % of original | bpb (x8) |
|---|---|---|---|---|---|
| zstd 1.5.7 -1 | 411 | 1351 | 73,193,704 | 34.53 | 2.76 |
| zstd 1.5.7 -5 | 126 | 1172 | 62,740,852 | 29.60 | 2.37 |
| zstd 1.5.7 -11 | 36.1 | 1320 | 57,963,230 | 27.35 | 2.19 |
| zstd 1.5.7 -15 | 9.39 | 1345 | 56,867,494 | 26.83 | 2.15 |
| zstd 1.5.7 -18 | 4.13 | 1175 | 53,288,948 | 25.14 | 2.01 |
| zstd 1.5.7 -22 (--ultra) | 1.99 | 1087 | 52,284,290 | 24.67 | 1.97 |
| xz 5.8.4 -5 | 3.84 | 121 | 50,228,752 | 23.70 | 1.90 |
| xz 5.8.4 -9 | 2.75 | 122 | 48,766,532 | 23.01 | 1.84 |
| lzma 26.03 -9 (7-zip SDK) | 2.55 | 78.2 | 48,682,482 | 22.97 | 1.84 |
| brotli 1.2.0 -5 | 53.3 | 419 | 59,555,449 | 28.10 | 2.25 |
| brotli 1.2.0 -11 | 0.56 | 383 | 50,407,795 | 23.78 | 1.90 |
| bzip2 1.0.8 -9 | 12.7 | 35.5 | 54,572,811 | 25.75 | 2.06 |
| zlib 1.3.2 -6 | 26.8 | 314 | 68,228,431 | 32.19 | 2.58 |
| zlib 1.3.2 -9 | 11.3 | 318 | 67,644,548 | 31.92 | 2.55 |
| ppmd8 26.03 -9 | 2.61 | 2.55 | 48,207,763 | 22.75 | 1.82 |
| lz4 1.10.0 default | 550 | 3832 | 100,880,800 | 47.60 | 3.81 |

- zstd -19 is not in the lzbench 2.4 table; -18 (4.13 MB/s) and -22 (1.99 MB/s) bracket it — [inikep/lzbench README](https://github.com/inikep/lzbench)
- facebook/zstd's own README (Core i7-9700K 4.9 GHz, Ubuntu 24.04, lzbench, Silesia) only lists the fast end: zstd 1.5.7 -1 ratio 2.896 at 510 / 1550 MB/s; brotli 1.1.0 -1 2.883 at 290 / 425 MB/s; zlib -1 2.743 at 105 / 390 MB/s; lz4 1.10.0 2.101 at 675 / 3850 MB/s — [facebook/zstd README](https://github.com/facebook/zstd)

**Large Text Compression Benchmark (mattmahoney.net, page header "Last updated Oct. 14, 2026"; times are ns/byte on enwik9 as reported by each submitter on heterogeneous machines; memory MB)** — [LTCB](https://mattmahoney.net/dc/text.html)

| Program | Options | enwik8 bytes | enwik8 bpb | enwik9 bytes | Comp ns/B -> MB/s | Decomp ns/B | Mem MB |
|---|---|---|---|---|---|---|---|
| cmix v21 (2024-09-17) | -t | 14,623,723 | 1.170 | 107,963,380 | 622,949 -> 0.0016 MB/s (1.6 KB/s) | 638,442 | 30,950 |
| nncp v3.2 (2023-10-23, Transformer, GPU) | — | 14,915,298 | 1.193 | 106,632,363 | 241,871 -> 0.0041 MB/s | 238,670 | 7,600 |
| paq8px_v206fix1 | -12L | 15,849,084 | 1.268 | 124,696,410 | 291,916 -> 0.0034 MB/s | 294,847 | 28,151 |
| paq8hp12any | -8 | 16,230,028 | 1.298 | 132,045,026 | 37,660 -> 0.027 MB/s | 37,584 | 1,850 |
| zpaq 6.42 | -m s10.0.5fmax6 (≈ -m5 class) | 17,855,729 | 1.428 | 142,252,605 | 6,699 -> 0.149 MB/s | 14,739 | 14,000 |
| ppmonstr J | -m1700 -o16 | 19,055,092 | 1.524 | 157,007,383 | 3,574 -> 0.28 MB/s | ~3,600 | 1,700 |
| lrzip 0.612 | -z -L 9 -p 1 (zpaq backend) | 19,847,690 | 1.588 | 169,318,794 | 2,987 -> 0.33 MB/s | 2,929 | 2,700 |
| 7zip 4.46a | -m0=ppmd:mem=1630m:o=10 | 21,197,559 | 1.696 | 178,965,454 | 503 -> 1.99 MB/s | 546 | 1,630 |
| xz 5.2.1 | --lzma2=preset=9e,dict=1GiB,lc=4,pb=0 | 24,703,772 | 1.976 | 197,331,816 | 5,876 -> 0.17 MB/s | 20 | 6,000 |
| zstd 0.6.0 | -22 --ultra | 25,405,601 | 2.032 | 215,674,670 | 701 -> 1.43 MB/s | 2.2 | 792 |
| brotli (2016-02-18) | -q 11 -w 24 | 25,764,698 | 2.061 | 223,597,884 | 3,400 -> 0.29 MB/s | 5.9 | 437 |
| bzip2 1.0.2 | -9 | 29,008,736 | 2.321 | 253,977,839 | 379 -> 2.64 MB/s | 129 | 8 |
| gzip 1.3.5 | -9 | 36,445,248 | 2.916 | 322,591,995 | 101 -> 9.9 MB/s | 17 | 1.6 |

- A newer zstd run on enwik8 (Broadcom KB, undated article but zstd with levels to 22): level 22 -> 25,340,452 bytes (ratio 3.946 = 2.027 bpb) at 0.46 MB/s compress / 221 MB/s decompress; level 1 ratio 2.455 at ~155 MB/s — [Broadcom "ZSTD Compression Metrics"](https://knowledge.broadcom.com/external/article/404122/zstd-compression-metrics.html)
- LTCB top of table (2026-10-14): ranks 4-8 are Hutter-style neural/CM entries at 91.8-96.1 MB on enwik9 (ex-girlfriend 1.0 91,778,237; cmix-neif 91,913,343; makimono 95,793,872; cmix-lex-transformer 95,836,613; zmix 96,096,261); ranks 1-3 were not in the fetched portion — [LTCB](https://mattmahoney.net/dc/text.html)
- LWN commenters (2024) on a 1.6 GB file: `zstd --ultra -22` took 21m43s for 41 MB vs `xz -9` 15.6 s for 43 MB (xz multi-threaded); another put zstd -22 at ~83.5x the time of xz -9 per core — [LWN](https://lwn.net/Articles/967398) (anecdotal, mixed thread counts)

**Silesia per-file ratios (x3-compressor README; lz4 -9, gzip --best, xz -9 -e, zstd --ultra -22, brotli -q 11; undated, repo active 2022)** — [xbarin02/x3-compressor](https://github.com/xbarin02/x3-compressor)

| File | gzip | xz -9e | zstd -22 | brotli -11 | best classical bpb |
|---|---|---|---|---|---|
| dickens | 2.6461 | 3.6000 | 3.5765 | 3.6044 | 2.22 (brotli) |
| mozilla | 2.6966 | 3.8292 | 3.3769 | 3.6922 | 2.09 (xz) |
| mr | 2.7138 | 3.6231 | 3.2132 | 3.5317 | 2.21 (xz) |
| nci | 11.2311 | 23.1519 | 20.7925 | 22.0780 | 0.346 (xz) |
| ooffice | 1.9907 | 2.5346 | 2.3587 | 2.4818 | 3.16 (xz) |
| osdb | 2.7138 | 3.5456 | 3.2855 | 3.5812 | 2.23 (brotli) |
| reymont | 3.6396 | 5.0374 | 4.9060 | 4.9747 | 1.59 (xz) |
| samba | 3.9950 | 5.7778 | 5.5267 | 5.7367 | 1.38 (xz) |
| sao | 1.3613 | 1.6386 | 1.4479 | 1.5812 | 4.88 (xz) |
| webster | 3.4372 | 4.9540 | 4.8970 | 4.9188 | 1.61 (xz) |
| xml | 8.0709 | 12.2910 | 11.8004 | 12.4145 | 0.644 (brotli) |
| x-ray | 1.4035 | 1.8868 | 1.6457 | 1.8096 | 4.24 (xz) |

- Repo-local full-file check (2026-11-01, i5-1235U laptop) agrees: dickens brotli-11 2.2195 bpb at 0.3 MB/s, xz-9 2.2217 at 0.5 MB/s, gzip-9 3.0233 at 7.9 MB/s; nci brotli 0.3624, xz 0.4146; xml brotli 0.6444, xz 0.6784; samba xz 1.3935 — `data/benchmark/results/competitors-fullsize-2026-11-01.csv` (local). Laptop MB/s are 3-5x below the EPYC lzbench figures (thermal/laptop), so use lzbench for cost math.
- Squash benchmark (quixdb) exists but the fetched index gave no table; it was not used — [Squash unstable](https://quixdb.github.io/squash-benchmark/unstable)

**Compute cost per raw GB (formula above; single-thread MB/s from lzbench 2.4 / LTCB)**

| Tool / level | R MB/s | hours per GB | $/GB spot ($0.0175) | $/GB on-demand ($0.0515) | $/GB at $0.01 / $0.04 |
|---|---|---|---|---|---|
| zstd -1 | 411 | 0.00068 | $0.000012 | $0.000035 | $0.000007 / $0.000027 |
| zstd -11 | 36.1 | 0.0077 | $0.00013 | $0.00040 | $0.00008 / $0.00031 |
| bzip2 -9 | 12.7 | 0.0219 | $0.00038 | $0.0011 | $0.00022 / $0.00087 |
| zstd -18 (≈ -19) | 4.13 | 0.0673 | $0.0012 | $0.0035 | $0.00067 / $0.0027 |
| xz -9 | 2.75 | 0.101 | $0.0018 | $0.0052 | $0.0010 / $0.0040 |
| ppmd8 -9 | 2.61 | 0.106 | $0.0019 | $0.0055 | $0.0011 / $0.0043 |
| zstd --ultra -22 | 1.99 | 0.140 | $0.0024 | $0.0072 | $0.0014 / $0.0056 |
| brotli -11 | 0.56 | 0.496 | $0.0087 | $0.026 | $0.0050 / $0.020 |
| zpaq -m5 class (LTCB 6,699 ns/B) | 0.149 | 1.86 | $0.033 | $0.096 | $0.019 / $0.075 |
| paq8px -12 (LTCB) | 0.0034 | 81 | $1.42 | $4.17 | $0.81 / $3.24 |
| cmix v21 (LTCB) | 0.0016 | 173 | $3.03 | $8.91 | $1.73 / $6.92 |
| Redukt CPU lockstep (repo, 1,180 B/s on 12 vCPU = 98 B/s per vCPU) | 0.000098 | 2,830 | $49.5 | $146 | $28 / $113 |

Example arithmetic: xz -9 at 2.75 MB/s -> 1000/2.75 = 364 s = 0.101 h per GB -> 0.101 x $0.0175 = $0.0018 (spot).

### Inferences
- The realistic buyer baseline for "max classical ratio at negligible cost" is xz -9 / zstd -22 (1.84-1.97 bpb on Silesia, ~2.0 bpb on enwik8) at $0.002-0.007/GB; brotli -11 is 4-5x more expensive to compress than xz -9 for the same ratio and only wins on decompression speed.
- Context-mixing (zpaq -m5, paq8px, cmix) is the true classical ceiling: cmix reaches 1.17 bpb on enwik8 for ~$3/GB spot; Redukt's 0.964 bpb is 17.6% fewer bits for ~16x the compute cost on CPU (or ~3-6x on T4 at $9-17/GB). Any sales argument "vs classical" must therefore be framed against zstd/xz (cheap, ~2x worse) rather than against cmix, where the gap is small and the cost ordering is unfavorable.
- LTCB timings come from submitters' own machines (2003-2026), so the ns/B for xz 5.2.1 (0.17 MB/s with a 1 GiB dictionary) and brotli 2016 (0.29 MB/s) are not comparable with lzbench 2026; lzbench numbers were preferred for cost math.

### Gaps
- No single 2026 source reports zstd -19 specifically on Silesia (lzbench 2.4 jumps from -18 to -22) or paq8pxd on LTCB (row not in the fetched segment).
- No current TurboBench or Squash tables could be fetched (Squash index returned no data).
- LTCB ranks 1-3 (2026) were not in the fetched portion of the page.

---

## KQ2. Log-specific compressors (CLP, LogShrink, LogReducer, LogZip, Loghub; Elastic/Loki/ClickHouse; Splunk/Datadog)

### Takeaway
On Loghub logs, parser-based compressors reach 2-5x the ratio of lzma: LogShrink 55.9x (0.143 bpb) and LogReducer 43.0x (0.186 bpb) on Apache vs gzip 21.3x (0.376 bpb) and lzma 25.2x; on HDFS LogShrink 27.3x (0.293 bpb) vs gzip 10.6x. LogReducer runs at 5-32 MB/s (C++), LogShrink 1.3-5.5 MB/s (Python), CLP ingests >13x faster than Elasticsearch/Splunk; Uber reports 169x aggregate on Spark logs with CLP (2.16x better than zstd). Columnar stores (ClickHouse) get 38x with plain ZSTD(3) on nginx logs and 56-178x with typed columns and sort keys; Elastic logsdb cuts index size 65-77%; Splunk stores ~50% of raw.

### Cited Findings
- CLP (OSDI '21, Rodrigues/Luo/Yuan): "significantly higher compression ratio than all commonly used compressors", search comparable or better than Elasticsearch/Splunk, ingestion "over 13x" faster than both, scales to petabytes; abstract gives no absolute ratios — [USENIX OSDI 21](https://www.usenix.org/conference/osdi21/presentation/rodrigues)
- CLP OSDI slides (as extracted via search index, not verified in the PDF text): average compression ratio across test datasets CLP 32.20, gzip 16.38, Splunk Enterprise 2.86, Elasticsearch 1.75 — [OSDI 21 slides PDF](https://usenix.org/system/files/osdi21_slides_rodrigues.pdf)
- Uber (blog 2022-09-29): 5.38 PB of uncompressed Spark INFO logs over 30 days compressed to 31.4 TB (169x reported, ~171x implied); CLP end-to-end 2.16x better ratio than Zstandard and 2.28x than gzip on a Spark subset; Phase-1 IR + zstd 5-8% better than raw zstd; storage cost for 1-month retention fell from $1.8M/yr (uncompressed) to $10K/yr; a 4 MB compressed buffer = 400 MB+ raw; no MB/s reported — [Uber Engineering](https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp)
- LogShrink (ICSE 2024, arXiv 2309.09479) Table 5 compression ratios (LogZip and LogReducer run with lzma backend): Apache (4.90 MB): gzip 21.308, lzma 25.186, bzip2 29.557, LogZip 30.375, LogReducer 43.028, LogShrink 55.940. HDFS (1.47 GB): gzip 10.636, lzma 13.559, bzip2 14.059, LogZip 16.666, LogReducer 22.634, LogShrink 27.319. OpenSSH (70 MB): gzip 16.828, lzma 18.918, LogReducer 86.699, LogShrink 103.175. Spark (2.75 GB): gzip 17.825, lzma 19.908, LogReducer 57.135, LogShrink 59.739. Windows (26 GB): gzip 17.798, lzma 202.568, LogShrink 456.301 — [ar5iv 2309.09479](https://ar5iv.labs.arxiv.org/html/2309.09479)
- LogShrink Table 6 compression speed (MB/s): LogReducer Apache 1.686, HDFS 20.570, Spark 18.871, Windows 31.938, Thunderbird 19.532; LogShrink Apache 1.880, HDFS 3.253, Spark 2.821, Windows 5.507; LogZip 0.009-1.357 — [ar5iv 2309.09479](https://ar5iv.labs.arxiv.org/html/2309.09479)
- LogShrink headline: 16%-356% average ratio gain over baselines; 1.16x-5.54x vs lzma, 1.14x-6.76x vs bzip2; 1.66x avg over LogZip; 2.95 MB/s average speed (1.31-5.51) — [arXiv 2309.09479](https://arxiv.org/abs/2309.09479)
- LogReducer is a FAST '21 paper (Wei et al.), not FAST '23: 1.19x-5.30x ratio vs LZMA, 0.56x-3.16x LZMA's speed on logs >100 MB; 1.03x-4.01x ratio and 2.05x-182x speed vs Logzip — [USENIX FAST 21](https://www.usenix.org/conference/fast21/presentation/wei)
- A 2023 follow-up measured LogReducer on JSON data at compressed fraction 0.219 (4.57x), 7.23 MB/s compress, 12.72 MB/s decompress — [arXiv 2311.13947 (PBC)](https://arxiv.org/pdf/2311.13947); the PBC abstract claims ~2x the ratio of state-of-the-art techniques on machine-generated data with per-record random access — [arXiv abs](https://arxiv.org/abs/2311.13947)
- A 2024 numeric-token-parsing add-on reports LogReducer's average ratio over 16 datasets as 60.928, raised to 74.240 with numeric parsing (different ratio convention) — [arXiv 2408.05760](https://arxiv.org/pdf/2408.05760)
- ClickHouse (blog 2025-10-23, 66.7M nginx rows, ~20 GB raw): LZ4 20x, GZIP 31x, ZSTD(3) 38x on the raw file; single String column (ZSTD(1)) ~35x; regex-structured columns ~56x; optimized types/codecs (IPv4, LowCardinality, Delta+ZSTD(1), ZSTD(6) on paths) ~92x; plus ordering key (referer, user_agent, remote_user, request_path) 178x (21,237,294,480 / 118,984,801); time-first ordering ~50x. The article has internal inconsistencies (structured table 56x in text vs 15.3x in summary) — [ClickHouse blog](https://clickhouse.com/blog/log-compression-170x)
- Elastic logsdb index mode: "up to 65%" storage reduction at launch (2024-12-13); later Elastic benchmarks 162.7 GB -> 39.4 GB (76%) and 161.9 GB -> 37.5 GB (77%); index sorting up to 30%, synthetic _source another 20-40% (Enterprise); more CPU at indexing; default for logs-* since 9.2 — [BusinessWire press release](https://www.businesswire.com/news/home/20241213828255/en/Elastic-Announces-Elasticsearch-Logsdb-Index-Mode-to-Reduce-Log-Data-Storage-Footprint-by-Up-to-65), [Elastic Observability Labs](https://www.elastic.co/observability-labs/blog/elasticsearch-logsdb-index-mode-storage-savings), [Elastic storage evolution](https://www.elastic.co/observability-labs/blog/elasticsearch-logsdb-storage-evolution)
- Splunk capacity planning (docs 9.3/10.2): rawdata file typically 15% of pre-indexed size, TSIDX ~35%; sizing guidance uses a 50% compression estimate — [Splunk docs](https://help.splunk.com/en/splunk-enterprise/get-started/deployment-capacity-manual/10.2/hardware-capacity-planning/estimate-your-storage-requirements)
- Grafana Loki: gzip is the default chunk encoding (best ratio, slow), snappy recommended for high throughput (worse ratio), lz4 in between; no official ratio figures — [Loki best practices](https://grafana.com/docs/loki/latest/configure/bp-configure/). Community test on Loki 3.3.1: snappy 14.4 GB vs zstd 9.5 GB in S3 (~34% smaller) — [Grafana community](https://community.grafana.com/t/size-chunks-and-index-difference-in-s3/90428)
- Repo-local anchors on real access logs (2026-11-01): apache-access.log 171 KB -> gzip-9 0.4298 bpb (18.6x), brotli-11 0.3132 (25.5x), xz-9 0.3384 (23.6x); linux-syslog 216 KB -> brotli 0.3789, xz 0.4321; openssh-auth 225 KB -> brotli 0.3592, xz 0.4209 — `data/benchmark/results/competitors-fullsize-2026-11-01.csv` (local). Redukt Mamba+CM on a 10 KB slice of apache: 0.2164 bpb vs brotli 0.4664 on the same slice — `dual-backbone-benchmark.csv` (local).

### Inferences
- Converting ratios to bpb: Loghub Apache LogShrink 55.9x = 0.143 bpb, LogReducer 43.0x = 0.186 bpb, lzma 25.2x = 0.318 bpb, gzip 21.3x = 0.376 bpb. Redukt's 0.216 bpb on a (different) 10 KB Apache access-log slice is therefore not better than parser-based log compressors on Loghub Apache, which run at 1.7-32 MB/s (10^3-10^4 x Redukt's CPU throughput). The honest neural-vs-classical comparison on logs is against LogReducer/CLP, not gzip.
- Cost of LogReducer at ~20 MB/s (HDFS/Spark): 0.0139 h/GB -> $0.00024/GB spot; LogShrink at 3 MB/s -> $0.0016/GB. Both negligible vs storage.
- Columnar/typed storage (ClickHouse 92-178x) changes the problem (reorders rows, needs a schema); only the 35-38x "single string column + zstd" figure is a like-for-like lossless file baseline.
- The repo's brotli baseline worsens sharply at 10 KB (0.466 bpb) vs 171 KB (0.313 bpb); the published 54% margin on apache is partly a small-window artifact and shrinks to ~32% at 100 KB (repo scale test, `CLAUDE.md`).

### Gaps
- Could not read the OSDI '21 paper/slides PDF text directly; the per-dataset CLP vs zstd/xz table and CLP MB/s are not captured (only the aggregate 32.20 vs gzip 16.38 via search extraction, and Uber's 2.16x vs zstd).
- No Datadog compression figure found in any search result.
- No Loki official ratio numbers; only community measurements.

---

## KQ3. XML/JSON-specific compression (EXI, XMill, xz on XML, zstd dictionaries, e-invoice batches, Parquet)

### Takeaway
No published compression ratios exist for e-invoice XML archives (Facturae, UBL/Peppol, FatturaPA, CFDI) in any source found; the only anchors are the repo's own measurements (brotli-11 0.247-0.395 bpb on 39-83 KB batches, xz 0.31-0.47) and Silesia `xml` (brotli 12.4x = 0.644 bpb). zstd trained dictionaries give ~4.7x better ratio on ~300 B JSON records (1.33-1.37x -> 5.86-6.83x) with no speed loss; W3C's EXI evaluation (2009) found EXI "consistently smaller than gzipped XML", in some cases >10x, with 9.2x faster decoding than gzipped XML; XMill improves gzip by up to 15% but worsens bzip2/PPM.

### Cited Findings
- zstd README (0.6.0-era dictionary benchmark on small JSON records, ~300 B): ratio without dictionary ~1.33-1.37x, with dictionary ~5.86-6.83x (~4.7x gain); large (~6 KB) docs ~2.8x gain; "no speed loss, even some faster decompression"; dictionaries help only for related data, one dictionary per data type — [zstd README mirror (Android)](https://android.googlesource.com/platform/external/zstd/+show/refs/heads/android14-release/README.md). Current README describes the test as "roughly 10K records weighing about 1KB each" with charts only — [facebook/zstd](https://github.com/facebook/zstd)
- Dictionary training workflow: `zstd --train FullPathToTrainingSet/* -o dictionaryName`, compress with `zstd -D dictionaryName FILE`; default max dictionary size 110 KB (third-party doc) — [Mintlify zstd dictionary guide](https://www.mintlify.com/facebook/zstd/guides/dictionary-compression)
- W3C EXI Evaluation (Working Draft 2009-04-07; 94 documents in 21 test groups): EXI "consistently smaller than gzipped XML regardless of document size, document structure or the availability of schema information", "in some cases, EXI is over 10 times smaller than gzip"; smaller than ASN.1 PER, sometimes 20x; decoding 14.5x (avg) / 6.7x (median) faster than XML and 9.2x / 4.4x faster than gzipped XML; encoding 6.0x / 2.4x vs XML and 5.4x / 2.7x vs gzipped XML (Efficient XML 4.0, Pentium 4 3.0 GHz). Numeric per-group compactness only in graphs — [W3C EXI Evaluation](https://www.w3.org/TR/exi-evaluation/)
- XMill/xmlppm era results: bzip2 beats gzip by 20-30% on raw XML; XMill "speeds up and improves compression using gzip and bounded-context ppmd+ by up to 15%, but worsens compression for unbounded-context compressors such as bzip2 and ppm*" — [xmlppm paper (Cheney)](https://xmlppm.sourceforge.net/paper/node8.html); XMill authors claim "factor of 2" over gzip on data-like XML — [codemia summary](https://codemia.io/knowledge-hub/path/best_compression_algorithm_for_xml) (secondary)
- 2008 arXiv survey of XML compressors: gzip-based tools worst, PPM-based/XML-aware best but much slower — [arXiv 0806.0075](https://arxiv.org/pdf/0806.0075)
- Silesia `xml` (5.3 MB): gzip 8.07x, xz -9e 12.29x, zstd -22 11.80x, brotli -11 12.41x (= 0.644 bpb) — [x3-compressor README](https://github.com/xbarin02/x3-compressor); repo-local full-file: brotli 0.6444 bpb at 0.6 MB/s, xz 0.6784 at 3.3 MB/s, gzip 0.9912 at 20.4 MB/s (laptop) — `competitors-fullsize-2026-11-01.csv` (local)
- Repo-local e-invoice samples (2026-11-01): facturae-batch.xml 39,384 B -> gzip-9 0.5556 bpb (14.4x), brotli-11 0.3951 (20.2x), xz-9 0.4737 (16.9x); ubl-invoice-batch.xml 83,260 B -> gzip 0.3886 (20.6x), brotli 0.2470 (32.4x), xz 0.3121 (25.6x). On 10 KB slices: brotli 0.8904 / 0.8880 bpb vs Redukt 0.3891 / 0.4461 — `competitors-fullsize-2026-11-01.csv`, `dual-backbone-benchmark.csv` (local)
- Parquet codec comparison (Vertica export, not invoices): Snappy 2.6 GB, GZIP 1.9 GB, ZSTD 1.7 GB on the same data — [OpenText/Vertica blog](https://blogs.opentext.com/quick-tip-export-to-parquet-compression-with-gzip-brotli-and-zstd); Uber: GZIP -> ZSTD(3) on large Hive tables gave ~7% storage drop — [Uber cost efficiency](https://www.uber.com/en-US/blog/cost-efficiency-big-data/)
- MinIO server-side compression defaults include `.xml`, `.json`, `.log`, `.txt`, `.csv`, `text/*`, `application/xml`, `application/json`; objects < 4 KiB are never compressed; throughput "500 MB/s or greater per CPU core" write, ~1 GB/s decompress; no ratios and no dedup stated — [MinIO AIStor docs](https://docs.min.io/enterprise/aistor-object-store/administration/objects-and-versioning/data-compression/)

### Inferences
- For small-invoice workloads (one XML per object, 5-80 KB) the cheap classical answer is zstd with a per-format trained dictionary (expected several-x gain on the first KBs per the zstd README), or batching invoices into larger archives so that xz/brotli reach the 20-32x seen on the repo's 39-83 KB batches. A buyer will treat ~0.25-0.40 bpb at ~$0.002-0.009/GB as the baseline on invoice XML, not the 0.89 bpb 10 KB-slice brotli figure.
- EXI is an interchange format with schema-aware parsing, not an archival compressor; its ">10x smaller than gzip" cases are tiny SOAP messages, so it is relevant to Peppol/UBL transport, not to cold archives.

### Gaps
- No published ratios for Facturae, UBL/Peppol, FatturaPA or SAT CFDI archives from any vendor, tax authority or paper (searches returned only generic XML/Parquet material).
- No measured zstd-dictionary ratio on invoice XML (only the JSON README example); the repo has not yet run `zstd --train` on its invoice batches.
- W3C EXI numeric compactness table is only available as images in the 2009 draft.

---

## KQ4. Typical total cost per GB to compress with zstd -19 or xz -9 on a cloud vCPU

### Takeaway
At 2.75 MB/s single-thread (xz -9) a raw GB takes 364 s = 0.101 vCPU-h, i.e. $0.0018 spot / $0.0052 on-demand; zstd -18/-22 (1.99-4.13 MB/s) costs $0.0012-0.0024 spot / $0.0035-0.0072 on-demand; brotli -11 (0.56 MB/s) $0.0087 / $0.026. These are 2-4 orders of magnitude below the ~$50/GB of CPU neural compression and below one month of S3 Standard storage for the bytes they save.

### Cited Findings
- c7a.large: 2 vCPU, on-demand $0.103/h ($0.10264), spot $0.035/h -> $0.0515 and $0.0175 per vCPU-hour (region/OS not shown on page; fetched 2026-11-10) — [instances.vantage.sh](https://instances.vantage.sh/aws/ec2/c7a.large)
- Single-thread throughputs: zstd -18 4.13 MB/s, zstd -22 1.99 MB/s, xz -9 2.75 MB/s, lzma -9 2.55 MB/s, brotli -11 0.56 MB/s, bzip2 -9 12.7 MB/s (EPYC 9555P, lzbench 2.4) — [inikep/lzbench](https://github.com/inikep/lzbench)
- Decompression is far cheaper for LZ-family: zstd 1,087-1,345 MB/s, brotli 383 MB/s, xz 122 MB/s, bzip2 35.5 MB/s, ppmd8 2.55 MB/s (symmetric) — [inikep/lzbench](https://github.com/inikep/lzbench)
- Repo's own break-even model uses the same formula (`C_comp = P_cpu * 1e9 / (3600 * R)`) with P_cpu spot $0.022, on-demand c7g $0.03625, m7i $0.0504, Hetzner dedicated ~EUR0.0068/thread-h — `scripts/breakeven.py` (local)

Worked arithmetic (P_spot = $0.0175/vCPU-h, P_od = $0.0515):

| Tool | R MB/s | s/GB | vCPU-h/GB | $/GB spot | $/GB on-demand | decomp $/GB spot |
|---|---|---|---|---|---|---|
| zstd -18 | 4.13 | 242 | 0.0673 | $0.00118 | $0.00346 | $0.000004 (1175 MB/s) |
| zstd -22 | 1.99 | 503 | 0.1396 | $0.00244 | $0.00719 | $0.000004 |
| xz -9 | 2.75 | 364 | 0.1010 | $0.00177 | $0.00520 | $0.00004 (122 MB/s) |
| brotli -11 | 0.56 | 1,786 | 0.496 | $0.00868 | $0.0255 | $0.00001 (383 MB/s) |
| bzip2 -9 | 12.7 | 79 | 0.0219 | $0.00038 | $0.00113 | $0.00014 (35.5 MB/s) |
| cmix v21 | 0.0016 | 623,000 | 173 | $3.03 | $8.91 | ≈ same (symmetric) |

### Inferences
- Even at the pessimistic $0.04/vCPU-h and the slow brotli -11, classical max-ratio compression costs ~$0.02/GB; zstd -19 or xz -9 at $0.002-0.005/GB are effectively free relative to any hot storage saving (see KQ6), so the buyer's baseline cost is ≈ $0.
- Neural CPU compression at $50-60/GB is 10,000-30,000x the cost of xz -9 per GB; GPU T4 at $9-17/GB is ~2,000-5,000x.

### Gaps
- Spot prices vary by region/hour; only a single c7a.large snapshot (2026-11-10) was captured. No Hetzner/OVH dedicated per-thread price was fetched (repo uses ~EUR0.0068).

---

## KQ5. Dedup + compression in object storage (S3 Intelligent-Tiering, MinIO, Ceph BlueStore)

### Takeaway
Public object stores do not compress or dedup customer bytes: S3 Intelligent-Tiering only moves objects between access tiers (40-95% price reduction by tier, not by size) and bills by stored bytes; MinIO compresses text/log/XML/JSON transparently (algorithm unnamed in current docs, >500 MB/s per core, no ratio published); Ceph BlueStore applies zstd/lz4/snappy/zlib per pool and only keeps the compressed copy when it is <= 70% of the original, with one measured 1.9x on enwik9. So the "free" reduction a buyer gets from the storage layer is 0x on S3/GCS/Azure/R2/B2/Wasabi and roughly gzip/zstd-level (2-3x on text) on self-hosted MinIO/Ceph.

### Cited Findings
- S3 Intelligent-Tiering page describes only access-based tier movement: 30 days no access -> Infrequent Access (saves 40%), 90 days -> Archive Instant Access (68%), opt-in Archive Access (71%) and Deep Archive Access (up to 95%); objects < 128 KB not auto-tiered; no mention of compression or deduplication — [AWS S3 Intelligent-Tiering](https://aws.amazon.com/s3/storage-classes/intelligent-tiering/)
- S3 pricing page: GET $0.0004 per 1,000 (Standard), first 100 GB/month internet egress free, $0.09/GB example from eu-west-1; storage table did not render in the fetch — [AWS S3 pricing](https://aws.amazon.com/s3/pricing/)
- MinIO AIStor: compression on PUT / decompression on GET for `.txt .log .csv .json .tar .xml .bin` and `text/*`, `application/json`, `application/xml`, `binary/octet-stream`; excludes audio/video/images/archives and objects < 4 KiB; disabled for encrypted objects by default; "500 MB/s or greater per available CPU core" write, "approximately 1 GB/s or greater per CPU core" decompress; no compression ratios or dedup mentioned — [MinIO docs](https://docs.min.io/enterprise/aistor-object-store/administration/objects-and-versioning/data-compression/)
- Ceph BlueStore (Red Hat Ceph Storage 9 docs): compression applied only when compressed output is 70% or smaller (required ratio 0.7); supports zstd among algorithms; modes none/passive/aggressive/force — [Red Hat docs](https://docs.redhat.com/en/documentation/red_hat_ceph_storage/9/html/edge_guide/bluestore-server-side-compression_edge)
- Ceph Days Raleigh slide (Bryan Stillwell): BlueStore zstd on enwik9 ~2.8 GiB uncompressed -> ~1.5 GiB (~1.9x); column headers garbled in extraction — [ceph.io slides](https://main.ceph.io/assets/pdfs/ceph_days_raleigh/bryan_stillwell.pdf)
- Third-party Ceph blogs claim 3-5x on logs/JSON/XML/CSV (unsourced; conflicts with the 1.9x measurement) — [oneuptime (2026-03-31)](https://oneuptime.com/blog/post/2026-03-31-rook-understand-when-compression-helps-vs-hurts-ceph/markdown) (low reliability)

### Inferences
- On the public clouds the buyer pays for every byte uploaded, so client-side compression is the only lever; the relevant competitor is therefore "zstd/xz before PUT" (cost ≈ $0.002-0.007/GB), not the storage service.
- Self-hosted MinIO/Ceph deliver zstd-class reduction (≈2x on text, likely 10-20x on logs by analogy with zstd results) at ~zero marginal cost, which lowers the incremental value of neural compression for on-prem buyers.

### Gaps
- No MinIO published compression ratios; the current docs do not even name the algorithm (older docs cited klauspost S2/snappy, not confirmed here).
- No vendor data on block-level dedup gains for logs/XML in Ceph or MinIO (neither offers dedup per the fetched docs).
- S3 per-tier storage prices and Intelligent-Tiering monitoring fee were not readable from the AWS page (taken from a secondary comparison in KQ6).

---

## KQ6. Dollar value of the marginal neural reduction at 2026 hot object storage prices

### Takeaway
Each raw GB of enwik8-like text saves 0.1735 GB (brotli 2.352 -> neural 0.964 bpb), worth $0.0040/month on S3 Standard ($0.023) and $0.0012/month on B2 ($0.00695): $0.048-0.34 per raw GB over 1-7 years on S3, $0.014-0.10 on B2. Apache logs (0.466 -> 0.216 bpb) save only 0.031 GB per raw GB = $0.00072/month on S3 ($0.060 over 7 years); invoice XML (0.890 -> 0.389) saves 0.063 GB = $0.0014/month ($0.121 over 7 years). All are far below the $50-60/GB CPU compute cost of the neural step and comparable to or below a single egress of the saved bytes ($0.09/GB).

### Cited Findings

**Storage prices (per GB-month, US single-region)**
- AWS S3 (us-east-1, table dated May 2026): Standard $0.023 (first 50 TB), $0.022 (50-500 TB), $0.021 (>500 TB); Intelligent-Tiering $0.023 + monitoring fee; Standard-IA $0.0125; Glacier Instant $0.004; Glacier Flexible $0.0036; Deep Archive $0.00099; egress $0.09/GB first 10 TB — [Finout comparison, published 2026-09-08](https://finout.io/blog/cloud-storage-pricing-comparison)
- Azure Blob LRS East US: Hot $0.018, Cool $0.010, Cold $0.0045, Archive $0.00099 — [Finout](https://finout.io/blog/cloud-storage-pricing-comparison); a 2026 cut of East US Hot LRS from $0.0184 to $0.0165 reported by a secondary source — [leanopstech](https://leanopstech-web.pages.dev/blog/cloud-storage-pricing-comparison-2026/)
- GCS regional US: Standard $0.020, Nearline $0.010, Coldline $0.004, Archive $0.0012 (multi-region $0.0024); egress ~$0.12/GB (secondary) — [Finout](https://finout.io/blog/cloud-storage-pricing-comparison); official GCS page fetch returned no table — [GCS pricing](https://cloud.google.com/storage/pricing)
- Cloudflare R2 (official, fetched 2026-11-10): Standard $0.015/GB-month, Infrequent Access $0.01 (30-day minimum, $0.01/GB retrieval); Class A $4.50/M, Class B $0.36/M; egress free; 10 GB-month free — [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)
- Backblaze B2 (official, fetched 2026-11-10): $6.95/TB-month (= $0.00695/GB), first 10 GB free, egress free up to 3x average monthly storage then $0.01/GB, no minimum duration — [Backblaze pricing](https://www.backblaze.com/cloud-storage/pricing)
- Wasabi (official, fetched 2026-11-10): $7.99/TB-month pay-as-you-go (≈ $0.0078/GB base-2), 1 TB minimum monthly charge, 90-day minimum retention, no egress or API fees — [Wasabi pricing](https://wasabi.com/pricing); secondary sources say the rise from $6.99 to $7.99 took effect 2026-07-01 — [leanopstech](https://leanopstech-web.pages.dev/blog/wasabi-pricing-2026/)

**Bits saved per raw GB (repo measurements, 10 KB slices, 2026-10-31 / 2026-11-10)** — `dual-backbone-benchmark.csv`, `CLAUDE.md` (local)
- enwik8: brotli-11 2.352 -> Redukt 0.964 bpb: Δ = 1.388 bpb -> 0.1735 GB saved per raw GB
- apache-access.log: 0.466 -> 0.216: Δ = 0.250 bpb -> 0.0313 GB saved per raw GB
- facturae-batch.xml: 0.890 -> 0.389: Δ = 0.501 bpb -> 0.0626 GB saved per raw GB
- Against the stronger classical baselines on full files (local csv): apache brotli 0.3132 / xz 0.3384 at 171 KB; facturae brotli 0.3951 at 39 KB; enwik8-class text cmix 1.170 (LTCB). Using these, Δ shrinks to: apache 0.313 - 0.224 (Redukt 100 KB scale test) = 0.089 bpb -> 0.011 GB; facturae 0.395 - 0.389 ≈ 0.006 bpb -> ~0.0008 GB (near zero, but Redukt was measured on a 10 KB slice, brotli on the 39 KB file); enwik8 vs cmix 1.170 - 0.964 = 0.206 bpb -> 0.026 GB.

**Worked table: monthly and cumulative saving per raw GB = (Δbpb / 8) × price**
Formula: `saving_month = (bpb_base - bpb_neural) / 8 * P_storage`; `saving_T_years = saving_month * 12 * T`.

| Storage class ($/GB-mo) | enwik8 Δ0.1735 GB: $/mo | 1 yr | 7 yr | apache Δ0.0313 GB: $/mo | 7 yr | facturae Δ0.0626 GB: $/mo | 7 yr |
|---|---|---|---|---|---|---|---|
| S3 Standard $0.023 | $0.00399 | $0.0479 | $0.335 | $0.000719 | $0.0604 | $0.00144 | $0.121 |
| GCS Standard $0.020 | $0.00347 | $0.0416 | $0.291 | $0.000625 | $0.0525 | $0.00125 | $0.105 |
| Azure Hot $0.018 | $0.00312 | $0.0375 | $0.262 | $0.000563 | $0.0473 | $0.00113 | $0.0946 |
| R2 Standard $0.015 | $0.00260 | $0.0312 | $0.219 | $0.000469 | $0.0394 | $0.000939 | $0.0789 |
| Wasabi $0.0078 | $0.00135 | $0.0162 | $0.114 | $0.000244 | $0.0205 | $0.000488 | $0.0410 |
| B2 $0.00695 | $0.00121 | $0.0145 | $0.101 | $0.000217 | $0.0183 | $0.000435 | $0.0365 |
| S3 Deep Archive $0.00099 | $0.000172 | $0.00206 | $0.0144 | $0.000031 | $0.0026 | $0.000062 | $0.0052 |

Example: enwik8 on S3 Standard: 0.1735 × $0.023 = $0.00399 per month; × 84 months = $0.335 per raw GB over 7 years. This matches the repo's own threshold (<= $0.33/GB for text in hot storage, `CLAUDE.md`).
- One-time egress value of the saved bytes at $0.09/GB: enwik8 0.1735 × 0.09 = $0.0156 per download of the raw GB; apache $0.0028; facturae $0.0056.

### Inferences
- At the current CPU cost ($50-60/GB) the neural step never pays back on storage alone: even the best case (enwik8-class text, S3 Standard, 7 years) returns $0.335 per raw GB, 150-180x less than the compute spent. On B2/Wasabi the 7-year return is $0.10 (text), $0.018 (logs), $0.037 (invoices). GPU at $9-17/GB is still 27-50x off on text and 150-900x off on logs.
- Against the right classical baseline (xz/brotli on whole files, or LogReducer/CLP on logs) the marginal GB saved shrinks further (apache Δ -> 0.011 GB, invoices Δ -> ~0), so the storage-only value proposition on logs and invoice XML is close to zero unless small-object (10 KB) storage without batching is a hard constraint for the buyer.
- Break-even compute cost for text on S3 Standard over 7 years is $0.33/GB -> required throughput R* = 0.0175 × 0.27778 / 0.33 = 0.0147 MB/s = 14.7 KB/s per vCPU (spot); on B2 it is 48 KB/s per vCPU. Compare: cmix 1.6 KB/s, paq8px 3.4 KB/s, Redukt 0.098 KB/s per vCPU today.

### Gaps
- Official AWS/GCS/Azure pricing tables could not be fetched directly (page tables did not render); S3/GCS/Azure figures come from a 2026-09-08 secondary comparison and should be rechecked against the official calculators before quoting to a client.
- Intelligent-Tiering monitoring fee per 1,000 objects (relevant for many small invoice objects) was not captured.
- Redukt's full-file (not 10 KB slice) BPB on facturae/ubl batches is not in the repo results, so the "vs whole-file brotli" invoice delta above is an approximation.
