# Market position after checks E2-E4 — 2026-11-15

> Asked by the owner after checks E2, E3, E3b and E4 (`strata-backbone/10-strong-cm-check.md`,
> `strata-backbone/11-short-message-cm-check.md`): compared with what exists on the market today, where is Redukt?
> Method: seven web searches on 2026-11-15 with generic queries (no project or owner identifiers), read through
> search summaries; no page was fetched in full. Own figures come from the repository documents named in each row.
> Labels: MEASURED (own measurement, document named), REPORTED (a third party's claim, source named, not
> reproduced here), INFERENCE. This document updates `competitive-position-2026-11-10/REPORT.md`; it does not
> replace it.

## 0. Result first — weak points before anything else

1. **Redukt is technically level with the best neural text compressors and has no product advantage outside
   natural-language text.** On text its ratio matches the published neural systems; on logs and structured data
   free tools already do the same or better (MEASURED, E2); for short machine messages the sample of traffic does
   the work, not the model (MEASURED, E3 / E3b).
2. **The category is still empty.** No commercial neural lossless compressor for text, logs or XML was found. The
   funded 2026 startups in "AI compression" do lossy Earth-observation imagery and video — a different business.
   Absence in seven searches is not proof of absence.
3. **The technical differential is packaging, not ratio.** ts_zip and Nacrith reach equal or better text ratios
   with models of similar size and are free. What is specific to Redukt: a single zero-dependency binary, a
   130M backbone that decodes on CPU, and the numeric contract declared in the header.
4. **The owner's openness to leave CPU / edge (2026-11-15) matches the state of the art.** The published neural
   text compressors run on a GPU; ts_zip reports up to 1 MB/s on an RTX 4090 at a similar ratio, 500-1000x the
   Redukt CPU path (INFERENCE from REPORTED and MEASURED figures on different hardware and files).
5. **The one gap without a direct competitor is free text over expensive links** (narrowband satellite). No
   product was found there — and no evidence of demand either. Telemetry on those links is already solved by
   hand-packed binary fields.
6. Limits of this document. (a) Third-party figures are self-reported: Nacrith is a single-author preprint with no
   independent reproduction; OpenZL numbers are Meta's on datasets it chose. (b) No ts_zip speed on enwik9 was
   found, only "up to 1 MB/s" on its page. (c) Current satellite prices and the effect of direct-to-cell services
   on the value of a narrowband byte were not researched; it is an open risk for path 3. (d) Search summaries can
   be wrong; one source was internally inconsistent (an OpenZL table with two rows apparently swapped; an SBD
   limit given as 320 and 340 bytes on the same site). (e) Nothing was run for this document.

## 1. By segment

### 1.1 Natural-language text (ratio)

| System | Figure | Hardware / speed | Status | Source |
|---|---|---|---|---|
| ts_zip (Bellard), RWKV 169M v4, 8-bit | enwik9 135,443,237 B = 1.084 bpb (xz 1.707); enwik8 1.11 bpb as cited by the Nacrith paper | GPU needed for a reasonable speed; up to 1 MB/s on an RTX 4090; 4 GB RAM | free binary, not a product | REPORTED, ts_zip page |
| Nacrith (Feb 2026 preprint) | enwik8 0.9389 bpb; alice29.txt 0.918 bpb (cmix v21 1.63 there) | ~500 MB GGUF weights, ~1.2 GB VRAM per worker, consumer GPU; no throughput comparison published | research code | REPORTED, self-reported, not reproduced |
| NNCP | ~1.19 bpb enwik8; per-file training inflates small files (~3.96 bpb) | GPU | research | REPORTED via the Nacrith paper |
| Redukt, Mamba-130M + CM | enwik8 0.9648 (10 KB), 0.9697 (500 KB), 0.9781 (1 MB), 0.9998 (2 MB) bpb | CPU ~1.2-2 KB/s on 12 vCPU; T4 through the CLI 1.8-2.1 KB/s | this repository | MEASURED, `CLAUDE.md` |
| paq8px v217 `-8` | 43-58 % larger than Mamba-130M on prose (171 KB - 1 MiB) | 11 KB/s on one CPU thread | free | MEASURED, E2 |

Reading: on text Redukt's ratio is in the same band as the best published neural systems (INFERENCE: different
files and lengths, not a head-to-head). Nobody in that band sells a product. The speed gap to ts_zip is the GPU.

### 1.2 Logs

| System | Figure | Status | Source |
|---|---|---|---|
| CLP (YScope; Apache-2.0; in production at Uber) | ratio 2.16x that of Zstandard and 2.28x that of gzip on a subset of Uber's Spark logs; search without decompressing; no managed offering confirmed | open source | REPORTED, Uber engineering blog, YScope docs |
| zpaq `-m5`, lpaq1 | 9-20 % smaller than Strata `cm` on unseen logs at 0.6-1.2 MB/s | free | MEASURED, E2 |
| paq8px v217 | 34-41 % smaller than Mamba-130M on logs (171 KB - 1 MiB) | free | MEASURED, E2 |
| Redukt, Mamba-130M | apache 10 KB window -11 % against paq8px; loses on long inputs | this repository | MEASURED, E4, E2 |

Reading: no advantage. A log customer pays for search over compressed data (CLP) at least as much as for ratio,
and the storage value of logs is capped near $0.02-0.05 per raw GB over seven years (DERIVED, doc 10).

### 1.3 Structured data

| System | Figure | Status | Source |
|---|---|---|---|
| OpenZL (Meta, Oct 2025; 0.3 in 2026) | `sao` 2.06x against xz 1.64x and zstd 1.31x at ~340 MB/s compression; numeric streams 7.76x against zstd -19 5.75x and xz -9 5.92x; falls back to zstd on plain text | open source, 0.x | REPORTED, Meta's own numbers |
| paq8px v217 | 27 % smaller than Mamba-130M on JSON / `nci`; `sao` 10 KB window: Redukt +87 % | free | MEASURED, E2, E4 |

Reading: no advantage. Format-aware compression wins this segment at five to six orders of magnitude more speed.

### 1.4 Short messages over narrowband satellite (path 3)

| Fact | Value | Source |
|---|---|---|
| Iridium SBD message size | 340 B mobile-originated, 270 B mobile-terminated; IMT up to 100 KB | REPORTED, reseller documentation |
| Current practice | hand-packed binary payloads (a tracker fits date, time, position, altitude, speed, battery, temperature in 17 bytes); "compact JSON of 50-60 bytes" in industrial guidance | REPORTED, reseller blog |
| Products with a language model for this channel | none found; one research proposal (speech sent as text at ~70 bps, predicted 20-30 bps with a language model) | REPORTED, arXiv |
| Redukt, text | Mamba-130M payload 25-48 % smaller than the best primed public CM compressor | MEASURED, E3 |
| Redukt, machine traffic with a sample | primed Mamba 13-17 % smaller than lpaq1 (9 MB) with a sample, 11-19 % larger than paq8px with a sample; priming is not a product feature | MEASURED, E3b |

Reading: the gap is real and narrow. Redukt wins clearly only on free text; telemetry is already packed by hand to
a density no general compressor reaches. Whether anyone sends enough free text over $0.3-3 per KB links to pay for
a 130 MB model on the terminal is R-116, still unverified.

### 1.5 Who raises money for "AI compression" in 2026

| Company | What | Relevance |
|---|---|---|
| The Compression Company (UK / US, pre-seed $3.4M, Feb 2026) | neural codecs for Earth-observation data on orbital GPUs; lossless mode "for high-value targets"; first on-orbit demo announced for 2026 | adjacent (satellite downlink, imagery), mostly lossy |
| Neurovia AI (subsidiary of a listed company) | "visually lossless" video (12.15 GB -> 421 MB demo) | lossy video, not comparable |
| Cloudflare Unweight | lossless compression of model weights (~22 %), internal system | not a product, different payload |

## 2. What follows (INFERENCE; decisions are the owner's)

1. Any external claim must compare against the context-mixing family and the neural systems above, not against
   brotli / gzip (already a rule in `CLAUDE.md` after E2-E4).
2. If the target is natural-language text, the honest reference is ts_zip on a GPU. A direction consistent with
   the owner's note would be: text only, GPU or accelerator allowed, measured head-to-head against ts_zip, paq8px
   and cmix on the same bytes — and it still needs a buyer for whom text bytes are expensive (the storage value of
   text is about $0.22 per raw GB over seven years; DERIVED, doc 10).
3. Path 3 needs R-116 before any engineering: is there free-text traffic on narrowband links, and do direct-to-cell
   services erase the price that makes the byte expensive?
4. Logs and structured data: nothing measured or found supports further work.

## 3. Sources (searched 2026-11-15)

- Nacrith: Neural Lossless Compression — https://arxiv.org/abs/2602.19626v2
- ts_zip: Text Compression using Large Language Models — https://bellard.org/ts_zip/
- OpenZL: A Graph-Based Model for Compression — https://arxiv.org/pdf/2510.03203v2
- InfoQ on OpenZL — https://www.infoq.com/news/2025/10/openzl-structured-compression
- Uber: Reducing logging cost by two orders of magnitude using CLP —
  https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp
- CLP documentation — https://docs.yscope.com/clp/main
- Ground Control: IP vs messaging for satellite IoT —
  https://www.groundcontrol.com/blog/ip-vs-messaging-for-satellite-iot-data-transmission
- Ground Control: Iridium SBD documentation — https://docs.groundcontrol.com/iot/rockblock/iridium
- SatNews on The Compression Company —
  https://satnews.com/2026/02/12/software-defined-ai-compression-startup-secures-3-4m-to-unblock-the-space-to-ground-bottleneck/
- EU-Startups on The Compression Company —
  https://www.eu-startups.com/2026/02/imperial-college-founders-raise-e2-8-million-for-the-compression-company-a-satellite-data-compression-platform/
- PR Newswire on Neurovia AI —
  https://www.prnewswire.com/news-releases/neurovia-ai-a-subsidiary-of-roboai-concludes-participation-at-isnr2026--demonstrating-neurostream-visual-data-infrastructure-and-96-37-visually-lossless-compression-302778921.html
- Cloudflare Unweight — https://blog.cloudflare.com/unweight-tensor-compression
- STCTS (speech as text over constrained links) — https://arxiv.org/html/2512.00451v2
