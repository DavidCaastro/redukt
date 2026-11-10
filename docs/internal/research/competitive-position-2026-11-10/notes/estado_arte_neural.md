# State of the Art in Neural / LLM-based Lossless Compression (status as of 2026-11-10)

Scope note for the report writer: all bpb figures below are `compressed_bytes x 8 / original_bytes`. Where the source only
gave bytes or a compression ratio (CR, %), I converted (CR% x 0.08 = bpb) and marked it "derived". "Actual file" means the
figure comes from a real compressed file; "estimate" means cross-entropy / code-length estimate without a real file.
Throughput conversions from LTCB "ns/byte" are mine (1e9 / ns-per-byte = bytes/s). Context (not researched, given by the
coordinator): Redukt = Mamba-130M + CM + arithmetic coding, CPU-only Rust single binary, 0.964 bpb on enwik8 10 KB (actual
file), ~1,180 B/s aggregate on 12 vCPU, ~4.2 KB/s on a T4 at batch 128.

Method caveat: an early WebFetch pass of the LTCB page mis-reported the table as "starting at rank 4". I downloaded the raw
HTML (705,280 bytes, page says last updated Oct 14, 2026) and read the table directly; the rows below are verbatim from it.

---

## Key Question 1 — Current leaders and their numbers (dense comparison table)

### Takeaway
As of Nov 2026 the best published lossless text results are ~0.70-0.74 bpb on enwik9 (cmix-neif / ex-girlfriend, CPU
single-core, ~5 KB/s, Oct 2026) and 0.9389 bpb on the full enwik8 for a pretrained-LLM compressor (Nacrith, SmolLM2-135M,
GTX 1050 Ti, Feb 2026). Every neural system that beats ~1.0 bpb runs at single-digit KB/s per core or per consumer GPU; the
only neural systems in the MB/s class (L3TC, ts_zip) sit at 1.1-1.4 bpb and need an A100/RTX 4090 or an Apple ANE.

### Cited Findings — master comparison table

| System (version, date) | Model / size | enwik8 bpb | enwik9 bpb | Measured on | Throughput | Hardware | Lang / deps | Binary files? | Public? / License | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **ex-girlfriend 1.0** (Oct 14, 2026) | cmix-neif + online auxiliary predictor on transformer MLP features; "Tr" | — | **0.734** (91,778,237 B total incl. decompressor, derived) | full 1 GB, actual file | 220,071 ns/B comp = ~4.5 KB/s; 241,636 ns/B decomp = ~4.1 KB/s; ~61 h for 1 GB | AMD Threadripper 3970X, Linux, 10.2 GB RAM | C/C++ (cmix lineage) | text-tuned (enwik9) | Yes, github.com/rlanday/ex-girlfriend; "not eligible" for Hutter | [LTCB](https://mattmahoney.net/dc/text.html) |
| **cmix-neif 1.0** (Oct 5, 2026) | cmix-lex + transformer from fx2-cmix-transformer + small LSTM from cmix-obias + zmix models/mixer | — | **0.735** total (91,913,343 B; archive 87,912,654 + 4,000,689 decompressor => 0.703 bpb data-only, derived) | full 1 GB, actual file | 175,392 ns/B = ~5.7 KB/s; 48.72 h comp / 48.73 h decomp | GCP c2d-highcpu-8 (AMD EPYC 7B13), Geekbench5 single-core 1,306; 9,750 MB | C/C++ | text-tuned | Yes (LTCB entry) | [LTCB](https://mattmahoney.net/dc/text.html) |
| makimono 1.0 (Oct 13, 2026) | zmix with retrained transformer weights | — | 0.766 (95,793,872) | full 1 GB, actual | 228,540 ns/B = ~4.4 KB/s | one core of AMD EPYC 9755 KVM VM, 16 GB | Zig (zmix lineage) | text-tuned | Yes | [LTCB](https://mattmahoney.net/dc/text.html) |
| cmix-lex-transformer lexth11c (Sep 27, 2026) | transformer weights externally retrained, quantization-optimized | — | 0.767 (95,836,613; archive + 3,477,137 decompressor = 99,313,750 Hutter-style) | full 1 GB | 199,440 ns/B = ~5.0 KB/s; 55.4 h comp / 59.4 h decomp | 2 vCPU Intel Xeon Platinum Ice Lake (Modal), 12 GiB | C/C++ | text-tuned | Yes; Hutter "pending" | [LTCB](https://mattmahoney.net/dc/text.html) |
| **zmix 1.0** (Sep 14, 2026) | fx2-cmix-transformer stack **reimplemented in Zig**, retrained transformer | — | 0.769 (96,096,261) | full 1 GB | 131,364 ns/B = ~7.6 KB/s; ~36.5 h | Ryzen 9 5900X, one core, Geekbench5 T=1,724 | **Zig** | text-tuned | Yes; Hutter "pending" | [LTCB](https://mattmahoney.net/dc/text.html) |
| rata-cmix 1.0.0 (Sep 12, 2026) | fx4-cmix-transformer lineage | — | 0.775 (96,849,689) | full 1 GB | 190,633 ns/B = ~5.2 KB/s | GCP n4d-highmem-2 (EPYC Turin) | C/C++ | text-tuned | Yes (open source) | [LTCB](https://mattmahoney.net/dc/text.html) |
| **fx2-cmix-transformer** (Jul 24, 2026; Hutter award Sep 30, 2026) | fx2-cmix with LSTM replaced by a **6M-param transformer pre-trained on enwik9 on 8x RTX 5090 for 26 h** (+2.9 MB to binary); "does not need a GPU to run" | — | 0.776 (96,996,198 on LTCB); **0.803** (100,424,672 B official Hutter total) | full 1 GB, actual | 133,766 ns/B = ~7.5 KB/s, ~37 h on Ryzen 5900X; "8 days" on i7-1165G7 under WSL2 (12 GB) | Ryzen 9 5900X one core; Hutter test machine i7-1165G7 | C/C++ | text-tuned | Yes (Hutter requires open source) | [LTCB](https://mattmahoney.net/dc/text.html); [Hutter Prize](http://hutter1.net/prize/index.htm) |
| fx2-cmix-t-ft8 (Oct 10, 2026) | same, transformer trained 8 more epochs "on a rented GPU for about $20" | — | 0.776 (96,945,005) | full 1 GB | 111,600 ns/B = ~9.0 KB/s; 31.0 h comp / 36.1 h decomp | Ryzen 9 5900X | C/C++ | text-tuned | Yes; "not eligible" | [LTCB](https://mattmahoney.net/dc/text.html) |
| fractal-zip (Sep 16, 2026) | cmix-lex-transformer derivative | **0.984** (12,303,845) | 0.781 (97,617,144) | full files | 151,632 ns/B = ~6.6 KB/s | i7-11850H, Geekbench5 T~1,603 | C/C++ | text-tuned | Yes; not eligible | [LTCB](https://mattmahoney.net/dc/text.html) |
| tufazip v2 (Sep 30, 2026) | transformer in C/Python/CUDA, weights trained offline, quantized | — | 0.808 archive (100,958,324) + 4.43 MB decompressor = 0.843 total | full 1 GB | 647,595 ns/B = **~1.5 KB/s** on 2x B200 | NVIDIA DGX B200 (2x B200, 2x Xeon 8570), 73.7 GB mem | C/Python/CUDA | text | Yes | [LTCB](https://mattmahoney.net/dc/text.html) |
| altxs 1.0.0 (Aug 25, 2026) | cmix-lex + transformer with external weights trained on enwik9; **requires CUDA GPU** | — | 0.747 archive (93,434,410) + 13.49 MB decompressor = 0.855 total | full 1 GB | 63 h compress, **330 h decompress**; ~80 GB GPU memory | 40-core EPYC 9V84 + GH100 94 GB | C/C++/CUDA | text | Yes | [LTCB](https://mattmahoney.net/dc/text.html) |
| **nncp v3.2** (Bellard, Oct 23, 2023; v3.3 binaries Jun 5, 2024) | Transformer, ~199M params (v3 description), online-trained (no pretrained weights) | **1.193** (14,915,298) | **0.853** (106,632,363) + 628,955 B program = 0.858 | full files, actual | 241,871 ns/B = ~4.1 KB/s comp, ~4.2 KB/s decomp; ~67 h per GB | GeForce RTX 3090 (v2 note; v3 "supports recent NVIDIA GPUs"), 7.6 GB | C + CUDA; Linux and Windows x64 builds | general bytes (text-oriented) | Yes, free/experimental; license not stated on page | [bellard.org/nncp](https://bellard.org/nncp/); [LTCB](https://mattmahoney.net/dc/text.html) |
| **ts_zip** (Bellard, 2024-03-02 release) | **RWKV-4 169M**, pretrained, 8-bit weights, BF16 eval | **1.106** (13,825,741) | **1.084** (135,443,237) | full files, actual | **up to 1 MB/s** comp and decomp on RTX 4090; third-party: 7-128 KB/s (older version) | RTX 4090; 4 GB RAM; "GPU necessary for reasonable speed"; CPU works (slow) | binary distribution (Linux/Windows); language not stated | **text only** ("binary files see little compression") | Yes (binaries); license not stated; "no backward compatibility between versions" | [bellard.org/ts_zip](https://bellard.org/ts_zip/); [Hackaday](https://hackaday.com/2023/08/27/text-compression-gets-weirdly-efficient-with-llms) |
| **cmix v21** (Knoll, Sep 17, 2024) | CM + LSTM, ~2,000 models | **1.170** (14,623,723) | 0.864 (107,963,380) + 281,387 program | full files, actual | 622,949 ns/B = **~1.6 KB/s**; ~173 h per GB; 30.9 GB RAM | i7-7700K, one core | C++ | general bytes | Yes, open source | [LTCB](https://mattmahoney.net/dc/text.html) |
| **paq8px v206fix1 -12L** (LTCB row); latest release **v217 (Sep 28, 2026)** | CM + LSTM mode; v217 adds 3-layer mixer, lower memory at level 10+ | **1.268** (15,849,084) | 0.998 (124,696,410) | full files | 291,916 ns/B = ~3.4 KB/s; 28.2 GB RAM | (LTCB tester machine, note 93) | C++ | **general bytes, strong file-type models (image, audio, exe)** | Yes, GPL | [LTCB](https://mattmahoney.net/dc/text.html); [paq8px releases](https://github.com/hxim/paq8px/releases) |
| **Nacrith** (Tacconelli, arXiv 2602.19626, Feb 23-24, 2026) | **SmolLM2-135M** (F32 GGUF ~500 MB) + online predictors (token n-gram, adaptive bias head) + 32-bit AC with 2^24 CDFs | **0.9389** (11.2 MB from 95.4 MiB; actual file) ; alice29 0.918 | not reported | full enwik8 | paper: 50-70 tok/s start -> 20-30 tok/s when 2,048-token KV cache full; 3 workers 60-90 tok/s aggregate; limitations section "~21 tok/s". README example: alice (148.5 KB) compressed in 42.3 s = ~3.5 KB/s (my calc) | **GTX 1050 Ti 4 GB** (CUDA 6.1); ~1.2 GB VRAM/worker, max 8 workers; CPU fallback exists | **Python** + llama-cpp-python (C/C++), PyTorch fallback, transformers | **Yes (NC06 hybrid): text runs >= 64 B to the LLM, binary runs to LZMA/gzip/raw** | Yes, github.com/robtacconelli/Nacrith-GPU, **Apache-2.0**, 23 stars, 46 commits | [arXiv abs](https://arxiv.org/abs/2602.19626); [arXiv HTML](https://arxiv.org/html/2602.19626); [GitHub](https://github.com/robtacconelli/Nacrith-GPU) |
| **StateSMix** (Tacconelli, arXiv 2605.02904, Apr 5, 2026) | online-trained **Mamba-style SSM** (DM=32, NL=2, ~120K active params) + 9 sparse n-gram tables (16M slots each) + AC; no pretrained weights | 2.123 (1 MB), 2.149 (3 MB), 2.162 (10 MB), **2.130 (100 MB)** | — | excerpts + full enwik8 | ~2,000 tokens/s; OpenMP 1.9x on 4 cores | **commodity x86-64 CPU with AVX2, no GPU** | **pure C** | not stated | code URL not given in abstract; third-party listing cites GitHub | [arXiv abs](https://arxiv.org/abs/2605.02904); [search summary](https://arxiv.deeppaper.ai/papers/2605.02904v1) |
| **L3TC** (Alipay, AAAI-25, arXiv 2412.16642) | RWKV-based, trained from scratch on enwik8; variants 200K / 800K / 3.2M / 12M params | CR 17.55% = **1.404** (200K); 15.68% = 1.254 (800K); 13.81% = 1.105 (3.2M); 11.24% = 0.899 (12M; **in-sample, ACR 35.24%**) (derived) | CR 18.48% = **1.478** (200K); 17.56% = 1.405 (800K); 16.23% = 1.298 (3.2M); 16.00% = 1.280 (12M) (derived) | full files | **L3TC-200K: 4.35 MB/s on A100 (batch 2048), 1.30 MB/s on iPhone 12 ANE (batch 256)**; iPhone 12 CPU batch 1: 11-27 KB/s; L3TC-3.2M 2.50 MB/s A100 | A100 80 GB; iPhone 12 | Python/PyTorch; weights on Google Drive | text | Yes, github.com/alipay/L3TC-...; LEGAL.md present, license not shown; 20 stars | [arXiv HTML](https://arxiv.org/html/2412.16642v2); [GitHub](https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33446) |
| **FineZip** (arXiv 2409.17141, Sep 2024) | **Llama-3 8B** (+ 4-bit variant), online memorization + dynamic context | ratio 0.12799 = **1.024** (10 MB subset); 4-bit 0.1445 = 1.156; "FineZip (AC)" 0.0797 = 0.638 (derived) | — | **first 10 MB of enwik8** | 10 MB in 250 min = **~0.67 KB/s**; 4-bit: 67 min = ~2.5 KB/s; (AC variant ~13,000+ min) | single **NVIDIA A6000 48 GB**, batch 70 (4-bit) | Python/PyTorch | text | Yes, github.com/fazalmittu/FineZip | [arXiv HTML](https://arxiv.org/html/2409.17141) |
| **LLMZip** (arXiv 2306.04050, Jun 2023) | LLaMA-7B | per FineZip table: ratio 0.1163 = **0.930** (10 MB subset); with AC 0.0795 = 0.636 (derived) | — | 10 MB subset (FineZip's rerun) | 10 MB in **13,651 min** (~9.5 days) = ~12 B/s | A6000 (FineZip rerun) | Python/PyTorch | text | research code | [arXiv abs](https://arxiv.org/abs/2306.04050); [FineZip](https://arxiv.org/html/2409.17141) |
| **DeepMind "Language Modeling Is Compression"** (ICLR 2024, arXiv 2309.10668) | Chinchilla 70B / 7B / 1B; Llama2 7B/13B (as re-tabulated by L3TC) | — | Chinchilla-70B raw CR 8.30% = **0.664**; Chinchilla-7B 10.20% = 0.816; Chinchilla-1B 11.30% = 0.904; Llama2-7B 8.62% = 0.690; Llama2-13B 8.04% = 0.643 (derived); **adjusted CR incl. model size: 1400% (70B), 140% (Llama2-7B)** | 1 GB in 2,048-byte chunks | Llama2-7B: **280 B/s**; Llama2-13B: 85 B/s on A100 (L3TC Table 5) | A100 (L3TC); DeepMind TPUs (not extracted) | research | general bytes (also ImageNet 43.4%, LibriSpeech 16.4%) | models not public (Chinchilla) | [L3TC Table 3/5](https://arxiv.org/html/2412.16642v2); [arXiv abs](https://arxiv.org/abs/2309.10668v2) |
| RWKV-1.5B / RWKV-7B as compressor (L3TC baseline) | pretrained RWKV | — | 10.89% = 0.871 (1.5B); 9.68% = 0.774 (7B) (derived) | full 1 GB | **20 KB/s** (1.5B), 4 KB/s (7B) on A100 | A100 | — | text | — | [L3TC](https://arxiv.org/html/2412.16642v2) |
| **LMCompress** (Li et al., Nature Machine Intelligence 7:794-799, 2025; arXiv 2407.07723) | large pretrained LMs (iGPT for images; text LLM) | not reported in bpb | not reported | various | **no speed figures published** (criticized for this) | not stated | research | images/audio/video/text via tokenization + AC | no public code found | [arXiv](https://arxiv.org/abs/2407.07723); [critique](https://buttondown.com/anatol/archive/lmcompress-lossless-data-compression/) |
| **TRACE** (Mao et al., 2022, arXiv 2203.16114) | single-layer transformer, byte-level, online | — | CR 5.29:1 = **1.512** (derived) | full 1 GB | 952.3 kb/min (~16 KB/s if KB) comp; "3x DZip" | GPU (V100-class; paper) | Python/PyTorch | general bytes | Yes | [arXiv](https://arxiv.org/pdf/2203.16114) |
| **DZip** (2019/2021, arXiv 1911.03572) | biGRU + bootstrap, online | — | CR 4.47:1 = **1.79** (derived) | full 1 GB | 399.4 kb/min (~6.7 KB/s); **34-40 KB/s on RTX 4060 Ti** (2026 remeasure) | GPU | Python/PyTorch | general bytes | Yes | [TRACE paper](https://arxiv.org/pdf/2203.16114); [Kim & Belyaev 2026](https://arxiv.org/abs/2604.15472) |
| Kim & Belyaev chained predictors (arXiv 2604.15472, Apr 16, 2026) | chained lightweight neural predictors with "information inheritance" | "close to PAC" (no numbers in abstract) | — | — | 1.2-6.3x PAC encode, 2.8-12.3x PAC decode | consumer GPU | — | — | under review | [arXiv](https://arxiv.org/abs/2604.15472) |
| **Diffuse to Compress** (Nardone & Ferragina, arXiv 2608.11249, Aug 4, 2026) | diffusion LMs instead of autoregressive | evaluated on enwik8, numbers not in abstract | — | — | claimed to attack the "one-symbol-per-step bottleneck"; no numbers in abstract | not stated | — | text | no repo linked | [arXiv](https://arxiv.org/abs/2608.11249) |
| **Hybrid-Neural Archival** (Armstrong et al., arXiv 2603.25526, Mar 26, 2026) | LLM + classical hybrid | 0.39 BPC memorized literature; 0.75 BPC unseen news | — | custom | "**~2,600x slower than Zstd**"; "limits deployment to ultra-cold storage" | not stated | — | text | — | [arXiv](https://arxiv.org/abs/2603.25526) |
| **BOA Constrictor** (Gupta, Doglioni, Elliott, arXiv 2511.11337, Nov 14, 2025) | **Mamba**, model ~4.5 MB | n/a (HEP data) | n/a | ATLAS/CMS structured physics data | not reported; "Storage-Saving Rate trails ZLIB" | not stated | — | **structured scientific data** | code not linked | [arXiv](https://arxiv.org/abs/2511.11337) |
| ZipNN (IBM, arXiv 2411.05239, Nov 2024) | **not a neural compressor**: Huffman coding of float exponents to compress model weights | n/a | n/a | BF16 weights (-33%) | up to 80 GB/s decompress, 13 GB/s compress | CPU multithread | open source | model weights only | Yes | [IBM blog](https://research.ibm.com/blog/Zip-NN-AI-compression); [arXiv](https://arxiv.org/html/2411.05239v2) |

Classical references from the same LTCB table (same page, for the Pareto question): paq8px_v206fix1 -12L 1.268 bpb;
zpaq 6.42 -m s10.0.5fmax6 1.428; xz 5.2.1 -9e 1.976; zstd 0.6.0 -22 --ultra 2.032; brotli -q11 -w24 2.061; gzip -9 2.916
(enwik8, derived from bytes) — [LTCB](https://mattmahoney.net/dc/text.html).

Additional cited points:
- Nacrith's own comparison table (enwik8 100 MB): gzip -9 2.916, zstd -19 2.156, xz -9 1.989, bzip2 2.321, brotli -q11
  2.059, PAQ8px -12L ~1.27, CMIX v21 1.17, NNCP v3 ~1.19, ts_zip ~1.11, FineZip 1.024, Nacrith 0.9389; OOD post-cutoff text
  (334 KB): CMIX 1.148, ts_zip 0.964, FineZip 0.977, Nacrith 0.723 — [arXiv HTML](https://arxiv.org/html/2602.19626).
- Nacrith claims "first LLM-based compressor to handle non-text files, to our knowledge" — [arXiv HTML](https://arxiv.org/html/2602.19626).
- Nacrith's abstract: beats CMIX v21 by 44% and ts_zip by 20% on alice29; model "60x smaller" than FineZip's with no
  fine-tuning — [arXiv abs](https://arxiv.org/abs/2602.19626).
- ts_zip page: "the output is deterministic and reproducible; results don't depend on GPU or CPU model or thread count" —
  [bellard.org/ts_zip](https://bellard.org/ts_zip/). Bellard's encode.su post: "evaluated in a deterministic and
  reproducible way so that the compressed data can be decompressed using a different GPU or CPU" —
  [encode.su thread 4109](https://encode.su/threads/4109-ts_zip-Text-Compression-using-Large-Language-Models).
- NNCP took "about 9 hours to compress 95 MB enwik8 on a V100" (as cited by TRACE) — [arXiv 2203.16114](https://arxiv.org/pdf/2203.16114);
  EDPC (2025) cites NNCP at 261-432.6 KB/min — [arXiv 2507.18969](https://arxiv.org/pdf/2507.18969).
- L3TC supplementary Table 10 (enwik9): CMIX v20 CR 10.99% decoding 4 KB/s; NNCP v3.2 10.66% at 1.6 KB/s; RWKV-1.5B
  10.89% at 20 KB/s (hardware for CMIX/NNCP rows unspecified) — [L3TC](https://arxiv.org/html/2412.16642v2).
- FineZip authors' conclusion: "LLMs are still not a viable solution for large-scale text compression" — [arXiv](https://arxiv.org/html/2409.17141).
- MambaByte itself: 0.93 bpb on PG19 (language-model perplexity, not a file compressor); no MambaByte-based file compressor
  found — [emergentmind summary](https://www.emergentmind.com/topics/mambabyte-model); the only Mamba-based compressors found
  are StateSMix and BOA Constrictor (above).

### Inferences
- The frontier split in two in 2026: (a) **CPU, online-learning CM + small (6M) pretrained transformer** at 0.70-0.78 bpb
  enwik9 and 5-9 KB/s single-core (Hutter lineage), and (b) **GPU, pretrained 135M-8B LLM + light online models** at
  0.94-1.02 bpb enwik8 with throughput from ~90 B/s to a few KB/s (Nacrith, FineZip). Redukt (Mamba-130M + CM) is
  architecturally in group (b) but runs on CPU like group (a).
- Nacrith's throughput is internally inconsistent: the paper's ~21-30 tok/s (~90-130 B/s at ~4.3 B/token) vs the README
  example implying ~3.5 KB/s (possibly 4 instances loaded). Treat Nacrith's single-worker speed as 0.1-3.5 KB/s on a GTX
  1050 Ti until independently measured.
- Nacrith's FineZip row says "(SmolLM2-135M) 1.024", but FineZip's own paper gets 1.024 with Llama-3 8B on a 10 MB subset;
  Nacrith likely re-quoted FineZip's number and mislabeled the model.
- Redukt's 0.964 bpb (10 KB) vs Nacrith 0.9389 (100 MB) are **not directly comparable**: online-adaptive components gain
  substantially with file length (StateSMix shows the opposite trend only because it is purely online). For a pretrained
  backbone + CM the 10 KB figure is probably pessimistic relative to a 100 MB run; this needs measurement, not inference.
- All neural systems that exceed ~0.9 bpb on text and also handle arbitrary binaries are hybrids that route binary regions to
  LZMA/CM (Nacrith NC06, paq8px, cmix); no pure-LLM compressor handles binaries well.

### Gaps
- No public per-GB $ figures exist for any system; only times/hardware (see Key Question 8 for my derived GPU-h and core-h
  per GB).
- nncp v3.3 (Jun 2024) results and license not published on the page; v3.2 figures remain the latest.
- LMCompress has no speed or bpb-on-enwik numbers in any source found; Nature MI paper paywalled.
- Diffuse-to-Compress and Kim & Belyaev abstracts contain no absolute numbers; full PDFs not read.
- OmniZip (arXiv 2602.22286, "unified lightweight lossless compressor for multimodal data") surfaced in search but was not
  read.
- Could not verify L3TC code license (LEGAL.md not rendered) or whether CPU inference is supported (CLI sample uses `--device cuda`).

---

## Key Question 2 — LTCB standings and Hutter Prize (status, cost, time)

### Takeaway
LTCB (page last updated Oct 14, 2026) is led by ex-girlfriend (91,778,237 B total on enwik9) and cmix-neif (91,913,343),
both CPU single-core transformer+CM hybrids taking ~49-61 hours per GB; nncp v3.2 (GPU) has dropped to ~12th. The Hutter
Prize, dormant at ~5% total improvement from 2019 to May 2026, jumped to 15% with three awards in June-July 2026; the current
record is fx2-cmix-T at 100,424,672 bytes (0.803 bpb incl. decompressor), awarded 37,300 EUR, under a ~49 h single-core /
10 GB RAM / no-GPU regime.

### Cited Findings
**LTCB top 12 by total size (enwik9 + decompressor), raw table rows, Oct 14, 2026** — [LTCB](https://mattmahoney.net/dc/text.html):

| # | Program | enwik9 total (B) | bpb (derived) | Comp ns/B | Decomp ns/B | ~comp speed | Hours/GB | Mem MB | Alg | Hardware (tester note) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ex-girlfriend 1.0 | 91,778,237 | 0.734 | 220,071 | 241,636 | 4.5 KB/s | 61 | 10,204 | Tr | Threadripper 3970X (124) |
| 2 | cmix-neif 1.0 | 91,913,343 | 0.735 | 175,392 | 175,428 | 5.7 KB/s | 48.7 | 9,750 | Tr | GCP c2d-highcpu-8 EPYC 7B13, GB5 1,306 (119) |
| 3 | makimono 1.0 | 95,793,872 | 0.766 | 228,540 | 220,740 | 4.4 KB/s | 63 | 9,970 | Tr | 1 core EPYC 9755 VM (123) |
| 4 | cmix-lex-transformer lexth11c | 95,836,613 | 0.767 | 199,440 | 213,840 | 5.0 KB/s | 55 | 9,765 | Tr | 2 vCPU Xeon Ice Lake, Modal (117) |
| 5 | zmix 1.0 | 96,096,261 | 0.769 | 131,364 | 131,184 | 7.6 KB/s | 36.5 | 9,370 | Tr | Ryzen 9 5900X one core, GB5 1,724 (115) |
| 6 | rata-cmix 1.0.0 | 96,849,689 | 0.775 | 190,633 | 187,200 | 5.2 KB/s | 53 | 9,537 | Tr | GCP n4d-highmem-2 EPYC Turin (113) |
| 7 | fx2-cmix-t-ft8 | 96,945,005 | 0.776 | 111,600 | 129,960 | 9.0 KB/s | 31 | 9,681 | Tr | Ryzen 9 5900X (121) |
| 8 | fx2-cmix-transformer | 96,996,198 | 0.776 | 133,766 | 131,561 | 7.5 KB/s | 37 | 9,547 | Tr | Ryzen 9 5900X (108) |
| 9 | fractal-zip | 97,617,144 | 0.781 | 151,632 | 159,840 | 6.6 KB/s | 42 | 9,999 | Tr | i7-11850H (118) |
| 10 | tufazip v2 | 105,388,353 | 0.843 | 647,595 | 630,136 | 1.5 KB/s | 180 | 73,748 | Tr | DGX B200, 2x B200 GPU (112) |
| 11 | altxs 1.0.0 | 106,924,811 | 0.855 | 227,372 | 1,189,976 | 4.4 KB/s (decomp 0.84 KB/s) | 63 (decomp 330) | 88,000 | Tr | 40-core EPYC 9V84 + GH100 (110) |
| 12 | nncp v3.2 | 107,261,318 | 0.858 | 241,871 | 238,670 | 4.1 KB/s | 67 | 7,600 | Tr | RTX 3090 (88) |
| 13 | fx-deepmix | 107,828,411 | 0.863 | — | 301,180 | — | — | 10,233 | LSTM | i7-1165G7 (106) |
| 14 | cmix-obias (Aug 18, 2026) | 108,009,834 | 0.864 | — | 119,920 | — | — | 9,836 | LSTM | Ryzen 9 5900X (108) |
| 15 | cmix v21 -t | 108,244,767 | 0.866 | 622,949 | 638,442 | 1.6 KB/s | 173 | 30,950 | CM | i7-7700K (83) |

- enwik8 column is blank for almost all 2026 entries (they are enwik9-tuned Hutter entries); only fractal-zip (12,303,845 =
  0.984 bpb), nncp v3.2 (14,915,298 = 1.193) and cmix v21 (14,623,723 = 1.170) report enwik8 among the top 15 — [LTCB](https://mattmahoney.net/dc/text.html).
- cmix-neif progression: pre (Sep 15, 2026) 99,125,735 total -> pre2 98,249,835 (118.55 h) -> pre3 97,261,675 (107.73 h) ->
  1.0 (Oct 5) 95,914,051 Hutter-style total in 48.72 h — [LTCB](https://mattmahoney.net/dc/text.html).
- fx2-cmix-transformer: "replacing the LSTM model with a transformer with 6M parameters pre-trained on enwik9 on 8 RTX 5090
  GPUs for 26 hours, adding 2.9 MB to the compressor and decompressor. It does not need a GPU to run"; a segfault on Intel
  i7-1165G7 was fixed Aug 21, 2026; Intel run took 8 days under WSL2 — [LTCB](https://mattmahoney.net/dc/text.html).
- Hutter Prize awards table (official site) — [hutter1.net](http://hutter1.net/prize/index.htm):

| Date | Winner | Program | Total size (B) | Improvement | Prize |
|---|---|---|---|---|---|
| 24 Jul 2026 (awarded Sep 30, 2026 per LTCB) | Vladimir Ivanov | fx2-cmix-T | 100,424,672 | 7.46% | 37,300 EUR |
| 19 Jul 2026 | David Freelan | cmix-obias | 108,521,870 | 1.05% | 5,240 EUR |
| 26 Jun 2026 | Ibrahim Marcouch & Kaido Orav | cmix-lex | 109,671,639 | 1.01% | 5,050 EUR |
| 3 Sep 2024 | Kaido Orav & Byron Knoll | fx2-cmix | 110,793,128 | 1.59% | 7,950 EUR |
| 2 Feb 2024 | Kaido Orav | fx-cmix | 112,578,322 | 1.38% | 6,911 EUR |
| 16 Jul 2023 | Saurabh Kumar | fast cmix | 114,156,155 | 1.04% | 5,187 EUR |
| 31 May 2021 | Artemiy Margaritov | starlit | 115,352,938 | 1.1% | 9,000 EUR |
| 4 Jul 2019 (baseline) | Alexander Rhatushnyak | phda9v1.8 | 116,673,681 | — | — |

- Hutter rules: S = compressor + archive < L = 100,424,672 B; prize = 500,000 EUR x (1 - S/L), min claim 5,000 EUR (1%);
  "<~50 hours" on a single CPU core, < 10 GB RAM, < 100 GB disk, Linux or Windows; total paid 85,485 EUR —
  [hutter1.net](http://hutter1.net/prize/index.htm). Time limit formula: < 70,000/T hours with T = Geekbench5 score;
  test machine Lenovo i7-1165G7 (T~1,427 single core -> ~49 h); **GPU usage is prohibited** —
  [hrules](https://hutter1.net/prize/hrules.htm).
- Epoch AI (updated Oct 6, 2026): "between the prize's inauguration in July 2019 and May 2026 total progress amounted to a
  5% improvement over the baseline; rapid submissions in June and July 2026 tripled that to 15%" —
  [epoch.ai](https://epoch.ai/data/hutter-prize).
- Gwern (essay Jan 22, 2026, rev. Mar 10, 2026): the Prize has had "almost no visible influence on mainstream AI" in ~20
  years; winning approaches resemble demoscene optimization; a model 1 TB larger that saves 0.01 bits/byte breaks even only
  after ~800 TB of text; "GPT-5 would end up stupider than a `cat`" under the literal objective; nncp v2 (56M-param
  Transformer) beat cmix on enwik9 from a single GPU in 2021 — [gwern.net](https://gwern.net/blog/2026/hutter-prize).
- Several 2026 LTCB entries are explicitly "not eligible" (ex-girlfriend, fx2-cmix-t-ft8, fractal-zip, fx-deepmix) or
  "pending" (zmix, cmix-lex-transformer) — [LTCB](https://mattmahoney.net/dc/text.html).

### Inferences
- The 2026 surge is one lineage: cmix -> fx2-cmix -> fx2-cmix-transformer (6M transformer pretrained on enwik9) -> zmix /
  cmix-lex-transformer / cmix-neif / ex-girlfriend. The key unlock was a tiny (6M) **pretrained** transformer small enough
  (2.9 MB) to amortize inside the decompressor size, running on CPU.
- LTCB and Hutter totals differ for the same program (fx2-cmix-transformer 96,996,198 on LTCB vs 100,424,672 official):
  the official figure likely corresponds to a different build/parameters submitted for eligibility; cite the official one for
  the prize, the LTCB one for the benchmark.
- Cost of a Hutter-class result: ~37-61 single-core-hours per GB. At any plausible cloud vCPU price this is tens of dollars
  per GB, i.e. the same order as Redukt's reported CPU cost (~$50-60/GB per the coordinator), but at 0.70-0.78 bpb vs 0.96.
- GPU-based entries (tufazip, altxs) are currently *slower* and *worse* than the CPU leaders on LTCB, a reversal of the
  2021-2023 nncp era.

### Gaps
- Whether any 2026 entries were produced with AI coding agents: no source states this (Epoch AI and LTCB are silent); a
  search hit for an "AI Village" dashboard was not followed. Unverified.
- Official Hutter page over HTTPS (prize.hutter1.net) failed with an expired certificate; data taken from the HTTP mirror.
- No LTCB entry reports both enwik8 and a 2026-era bpb under 0.98, so there is no 2026 enwik8 full-file SOTA from the
  Hutter lineage; fractal-zip's 0.984 is the only one.

---

## Key Question 3 — CPU-only / CPU-capable vs GPU-required, and CPU speeds

### Takeaway
The entire 2026 LTCB/Hutter top-9 is CPU single-core at 4.4-9 KB/s (0.73-0.78 bpb enwik9); among pretrained-LLM compressors
only ts_zip (CPU fallback, deterministic, slow) and Nacrith (llama.cpp CPU fallback, speed not reported) can run without a
GPU, and StateSMix is CPU-only pure C at ~2,000 tok/s but only xz-class ratios.

### Cited Findings
- CPU single-core, no GPU at runtime: ex-girlfriend, cmix-neif, makimono, cmix-lex-transformer, zmix, rata-cmix,
  fx2-cmix-t-ft8, fx2-cmix-transformer ("does not need a GPU to run"), fractal-zip, cmix v21, paq8px — all with ns/byte
  figures above (4.4-9.0 KB/s for the transformer hybrids; 1.6 KB/s cmix v21; 3.4 KB/s paq8px -12L) —
  [LTCB](https://mattmahoney.net/dc/text.html).
- GPU required: nncp (RTX 3090; forum: "GPU is mandatory", early versions "only NVIDIA Ampere supported") —
  [LTCB](https://mattmahoney.net/dc/text.html), [encode.su NNCP thread](https://encode.su/threads/3094-NNCP-Lossless-Data-Compression-with-Neural-Networks/page5);
  altxs ("require a CUDA compatible GPU"), tufazip (2x B200) — [LTCB](https://mattmahoney.net/dc/text.html);
  FineZip/LLMZip (A6000) — [FineZip](https://arxiv.org/html/2409.17141); L3TC speed measured on A100 and iPhone ANE —
  [L3TC](https://arxiv.org/html/2412.16642v2).
- GPU optional: ts_zip — "A GPU is necessary to get a reasonable speed"; decodable on a different GPU or CPU —
  [bellard.org/ts_zip](https://bellard.org/ts_zip/), [encode.su 4109](https://encode.su/threads/4109-ts_zip-Text-Compression-using-Large-Language-Models).
  Nacrith — "falls back to CPU if no GPU is available"; any CUDA GPU with >= 2 GB VRAM works —
  [GitHub](https://github.com/robtacconelli/Nacrith-GPU).
- CPU-only by design: StateSMix, pure C, AVX2, ~2,000 tokens/s, OpenMP 1.9x on 4 cores — [arXiv](https://arxiv.org/abs/2605.02904).
- Mobile CPU: L3TC-200K on iPhone 12 CPU at batch 1: 11-27 KB/s (Table 12); on the ANE 1.30 MB/s — [L3TC](https://arxiv.org/html/2412.16642v2).
- Hutter Prize machine: i7-1165G7, Geekbench5 ~1,427 single-core, ~49 h budget; GPU prohibited — [hrules](https://hutter1.net/prize/hrules.htm).

### Inferences
- Per-core CPU throughput of the CM+6M-transformer hybrids (5-9 KB/s at 0.73-0.78 bpb) is roughly 50-90x higher than
  Redukt's per-vCPU figure (~98 B/s per vCPU at 1,180 B/s on 12 vCPU) while also compressing ~20-25% better on enwik9 — but
  these are enwik9-tuned text models with 9.5-10 GB RAM, 1-GB files and 30-60 h runs; none reports behavior on 10 KB files
  or binaries.
- No pretrained-LLM compressor reports a CPU throughput number at all; Redukt's CPU figures appear to be the only published
  CPU numbers for a >=100M-param backbone compressor.

### Gaps
- No measured CPU speed for ts_zip or Nacrith anywhere in the sources.
- L3TC server-CPU speed is not reported (only A100 and iPhone).

---

## Key Question 4 — Structured data (XML, JSON, logs)

### Takeaway
No neural compressor in the literature reports results on XML/JSON/e-invoice corpora; the closest are a Mamba compressor for
structured HEP data (BOA Constrictor, Nov 2025), a GPT-2 pre-processor for logs/HTML with sub-6% gains (Aug 2025), and
Nacrith's out-of-distribution text test.

### Cited Findings
- BOA Constrictor (arXiv 2511.11337, Nov 14, 2025): Mamba-based, model ~4.5 MB, structured High Energy Physics datasets;
  vs LZMA-9: ATLAS 2.21x vs 1.69x, CMS 44.14x vs 27.14x; Storage-Saving Rate "trails ZLIB"; no throughput, hardware or
  code given — [arXiv](https://arxiv.org/abs/2511.11337).
- "GPT-2 as a Compression Preprocessor: Improving Gzip for Structured Text Domains" (arXiv 2508.14061, Aug 2025): rewrites
  logs/HTML/config into predictable form before gzip; gains 0.34% on defence logs, 5.8% on HTML — [arXiv](https://arxiv.org/html/2508.14061v1).
- LogLite (arXiv 2507.10337, Jul 2025): lightweight streaming log compressor for TEXT and JSON logs — **not neural** —
  [deeppaper summary](https://arxiv.deeppaper.ai/papers/2507.10337v1?locale=en).
- "LLM-generated data compression" (arXiv 2505.06297, 2025): 14 LLMs on 8 LLM-generated datasets, covers structured text
  such as logs — [arXiv](https://arxiv.org/pdf/2505.06297.pdf) (abstract-level only).
- Lossless prompt compression via in-context dictionary encoding for log data sent to LLMs (arXiv 2604.13066, Mar 2026) —
  [alphaxiv](https://www.alphaxiv.org/abs/2604.13066.md).
- Nacrith OOD post-cutoff document (334 KB): 0.723 bpb vs CMIX 1.148, ts_zip 0.964 — [arXiv HTML](https://arxiv.org/html/2602.19626).
- PETRA III synchrotron study (arXiv 2608.00168, Jul 31, 2026): 212 TiB, nine classical tools; ratios vary >2 orders of
  magnitude across datasets; ZPAQ best ratio, Zstandard/LZ4 occupy the high-throughput Pareto region; heterogeneous
  per-category strategies beat uniform policies — [arXiv](https://arxiv.org/abs/2608.00168).

### Inferences
- Redukt's facturae/UBL XML (0.389/0.446 bpb) and apache-log (0.216 bpb) results have no published neural counterpart to be
  compared against; the competitive comparison for those segments is only against classical tools (brotli/zstd/xz/zpaq).

### Gaps
- No paper found reporting bpb on XML or JSON corpora with an LLM compressor (searched 2025-2026 arXiv terms).

---

## Key Question 5 — Decompression-speed vs ratio Pareto frontier on enwik8 (neural vs classical)

### Takeaway
On enwik8 the frontier spans ~8 orders of magnitude in decode speed: zstd -22 at 2.03 bpb / ~455 MB/s, brotli-11 2.06 /
~169 MB/s, xz -9e 1.98 / ~50 MB/s, zpaq -m5-class 1.43 / ~68 KB/s, paq8px -12L 1.27 / ~3.4 KB/s, cmix v21 1.17 / ~1.6 KB/s,
nncp v3.2 1.19 / ~4.2 KB/s (GPU), ts_zip 1.106 / up to 1 MB/s (RTX 4090), Nacrith 0.939 / ~0.1-3.5 KB/s (GTX 1050 Ti); no
2025-2026 paper or forum thread charting neural vs classical on one machine was found.

### Cited Findings (all enwik8, full 100 MB, actual files; speeds derived from LTCB ns/byte unless noted; machines differ)

| Compressor | bpb | Decompress speed | Compress speed | Hardware | Source |
|---|---|---|---|---|---|
| gzip 1.3.5 -9 | 2.916 | 17 ns/B = ~59 MB/s | 101 ns/B = ~9.9 MB/s | LTCB tester | [LTCB](https://mattmahoney.net/dc/text.html) |
| zstd 0.6.0 -22 --ultra | 2.032 | 2.2 ns/B = ~455 MB/s | 701 ns/B = ~1.4 MB/s | LTCB tester (76) | [LTCB](https://mattmahoney.net/dc/text.html) |
| brotli -q 11 -w 24 (2016 build) | 2.061 | 5.9 ns/B = ~169 MB/s | 3,400 ns/B = ~294 KB/s | LTCB tester (48) | [LTCB](https://mattmahoney.net/dc/text.html) |
| xz 5.2.1 -9e (1 GiB dict) | 1.976 | 20 ns/B = ~50 MB/s | 5,876 ns/B = ~170 KB/s | LTCB tester (73) | [LTCB](https://mattmahoney.net/dc/text.html) |
| 7zip PPMd o=10 | 1.696 | 546 ns/B = ~1.8 MB/s | 503 ns/B = ~2.0 MB/s | LTCB (23) | [LTCB](https://mattmahoney.net/dc/text.html) |
| bsc-m03 0.4.0 (BWT) | 1.623 | 135 ns/B = ~7.4 MB/s | 160 ns/B = ~6.3 MB/s | i9-9700K (96) | [LTCB](https://mattmahoney.net/dc/text.html) |
| zpaq 6.42 -m s10.0.5fmax6 | 1.428 | 14,739 ns/B = ~68 KB/s | 6,699 ns/B = ~149 KB/s | LTCB (61) | [LTCB](https://mattmahoney.net/dc/text.html) |
| paq8px_v206fix1 -12L | 1.268 | 294,847 ns/B = ~3.4 KB/s | 291,916 ns/B = ~3.4 KB/s | LTCB (93), 28 GB | [LTCB](https://mattmahoney.net/dc/text.html) |
| nncp v3.2 (GPU) | 1.193 | 238,670 ns/B = ~4.2 KB/s | 241,871 ns/B = ~4.1 KB/s | RTX 3090 | [LTCB](https://mattmahoney.net/dc/text.html) |
| cmix v21 -t | 1.170 | 638,442 ns/B = ~1.6 KB/s | 622,949 ns/B = ~1.6 KB/s | i7-7700K, 31 GB | [LTCB](https://mattmahoney.net/dc/text.html) |
| ts_zip (RWKV-4 169M) | 1.106 | up to 1 MB/s | up to 1 MB/s | RTX 4090 | [bellard.org/ts_zip](https://bellard.org/ts_zip/) |
| L3TC-200K | 1.404 | 4.35 MB/s (A100 batch 2048); 1.30 MB/s (iPhone 12 ANE) | — | A100 / iPhone 12 | [L3TC](https://arxiv.org/html/2412.16642v2) |
| L3TC-3.2M | 1.105 | 2.50 MB/s (A100) | — | A100 | [L3TC](https://arxiv.org/html/2412.16642v2) |
| fractal-zip (CPU, 2026) | 0.984 | 159,840 ns/B = ~6.3 KB/s | 151,632 ns/B = ~6.6 KB/s | i7-11850H | [LTCB](https://mattmahoney.net/dc/text.html) |
| FineZip (Llama-3 8B, 10 MB subset) | 1.024 | not reported (symmetric design) | ~0.67 KB/s (250 min / 10 MB) | A6000 | [FineZip](https://arxiv.org/html/2409.17141) |
| Nacrith (SmolLM2-135M) | 0.9389 | alice: 148.5 KB in 39.1 s = ~3.8 KB/s (README) vs paper ~21-30 tok/s | ~3.5 KB/s (README) | GTX 1050 Ti | [GitHub](https://github.com/robtacconelli/Nacrith-GPU); [arXiv HTML](https://arxiv.org/html/2602.19626) |
| StateSMix (CPU, pure C) | 2.130 | ~2,000 tok/s | ~2,000 tok/s | x86-64 AVX2 | [arXiv](https://arxiv.org/abs/2605.02904) |

- Asymmetric neural decode: altxs decompresses 5.2x slower than it compresses (330 h vs 63 h, GPU) —
  [LTCB](https://mattmahoney.net/dc/text.html). All other neural entries are near-symmetric (encoder and decoder run the
  same model).
- Hybrid-neural archival paper: "~2,600x slower than Zstd" — [arXiv 2603.25526](https://arxiv.org/abs/2603.25526).
- The only 2026 Pareto-front benchmark paper found (PETRA III) covers classical tools only — [arXiv 2608.00168](https://arxiv.org/abs/2608.00168).

### Inferences
- Between zpaq (1.43 bpb, 68 KB/s decode) and paq8px/cmix (1.17-1.27 bpb, 1.6-3.4 KB/s) there is a ~20-40x decode-speed
  cliff for ~12-18% fewer bits; neural CPU compressors (Redukt included) live on the slow side of that cliff. ts_zip and
  L3TC are the only systems that break the cliff (MB/s at 1.1-1.4 bpb) and both need an accelerator.
- Redukt at 0.964 bpb (10 KB) / ~1.2 KB/s (12 vCPU) sits near cmix v21 in speed and near fractal-zip/Nacrith in ratio, but
  on a 10 KB file; a like-for-like 100 MB enwik8 run is needed to place it on this chart.

### Gaps
- LTCB speeds come from different machines and years (zstd 0.6.0 from 2016, brotli 2016 build); no single-machine 2025-2026
  chart including neural compressors was found on encode.su, HN or arXiv. lzbench/Squash results were not retrieved.

---

## Key Question 6 — Community view on practical viability (speed, determinism, GPU dependency)

### Takeaway
Authors and practitioners converge on the same verdict: LLM compressors win on ratio but are 100-10,000x too slow for
general use (FineZip: "not a viable solution"; hybrid-archival: "ultra-cold storage only"), and floating-point
non-determinism across GPUs ("GPU Butterfly Effect", A100-encoded data failing to decode on H100) is a recognized
correctness hazard that only a few systems (ts_zip, logit-quantization protocols) explicitly solve.

### Cited Findings
- FineZip authors: "LLMs are still not a viable solution for large-scale text compression" despite a 54x speedup over
  LLMZip — [arXiv](https://arxiv.org/html/2409.17141).
- Hybrid-neural archival (Mar 2026): latency "limits immediate deployment to ultra-cold storage"; identifies the "GPU
  Butterfly Effect" where hardware non-determinism prevents data recovery; solved with a logit-quantization protocol —
  [arXiv 2603.25526](https://arxiv.org/abs/2603.25526).
- A lossless point-cloud decoder on an H100 failed to decode data encoded on an A100; CUDA/cuDNN docs do not guarantee
  determinism; a 2024 plug-and-play mitigation adds ~1% bitstream overhead — [Towards Reproducible Learning-based
  Compression, arXiv 2410.09872](https://arxiv.org/pdf/2410.09872) and [arXiv 2601.10678](https://arxiv.org/pdf/2601.10678).
- "A Model-Driven Lossless Compression Algorithm Resistant to Mismatch" (Hu & Tang, arXiv 2601.17684, Jan 25, 2026):
  "even small mismatches can cause the decoding to fail"; proposes an algorithm tolerant to structured mismatches with
  proven correctness — [arXiv](https://arxiv.org/abs/2601.17684).
- ts_zip: deterministic across GPU/CPU and thread count by design — [bellard.org](https://bellard.org/ts_zip/);
  [encode.su 4109](https://encode.su/threads/4109-ts_zip-Text-Compression-using-Large-Language-Models).
- Nacrith: FP32 model, float64 bias head, integer CDFs; "bit-exact reproducibility" claimed **only under identical GPU
  architecture, BLAS library and driver**; cross-hardware determinism not tested — [arXiv HTML](https://arxiv.org/html/2602.19626).
- NNCP forum: GPU "mandatory because of the processing load"; early versions Ampere-only —
  [encode.su 3094](https://encode.su/threads/3094-NNCP-Lossless-Data-Compression-with-Neural-Networks/page5).
- Third-party write-up on ts_zip: speed is "the weak point" and gains "rarely justify the cost" —
  [lavx.hu](https://news.lavx.hu/article/ts-zip-pushing-text-compression-boundaries-with-language-model-priors).
- Gwern: Hutter-style constraints exclude GPU-scale models; "almost no visible influence on mainstream AI" — [gwern.net](https://gwern.net/blog/2026/hutter-prize).
- HN threads exist: "Ts_zip: Text Compression Using Large Language Models" (item 42549083, Dec 2024), "Compressed
  filesystems a la language models" (item 46058033, Nov 2025), "Compression is prediction" (item 49263497, 2026);
  commenters note ts_zip's 8-bit/BF16 model (~171 MB download) — [HN 42549083](https://news.ycombinator.com/item?id=42549083);
  [HN 46058033](https://news.ycombinator.com/item?id=46058033); [HN 49263497](https://news.ycombinator.com/item?id=49263497).
- 2026 LTCB entries show a reproducibility incident even on CPU: fx2-cmix-transformer segfaulted on Intel i7-1165G7
  (fixed Aug 21, 2026) and cmix-obias needed separate AMD and Intel builds (Jul 19 / Aug 11, 2026) —
  [LTCB](https://mattmahoney.net/dc/text.html).

### Inferences
- Determinism is treated in the literature as a GPU problem; the fact that cmix-lineage CPU entries needed AMD/Intel-specific
  builds in 2026 indicates CPU-side numeric portability (FMA/SIMD paths) is also live, which is exactly what Redukt's AZTH v5
  numeric-contract header addresses. This is a differentiator worth stating explicitly in the competitive analysis.
- Community consensus on "viability" is driven by the general-purpose use case; no source evaluates the "bytes are expensive"
  niche (metered links, regulated retention) that Redukt targets.

### Gaps
- Could not read HN or encode.su thread bodies (only titles/snippets); specific comment-level opinions beyond those quoted are
  unverified. No r/compression thread retrieved.

---

## Key Question 7 — Rust or single-binary zero-dependency neural compressors besides Redukt

### Takeaway
No Rust neural lossless compressor was found. The closest single-binary, zero-runtime-dependency neural compressors are
zmix (Zig reimplementation of the cmix+transformer stack, CPU, Sep 2026), StateSMix (pure C, AVX2, CPU), and Bellard's
nncp/ts_zip (C/CUDA, shipped as Linux/Windows binaries); everything LLM-based (Nacrith, FineZip, L3TC) is Python + PyTorch or
llama.cpp.

### Cited Findings
- zmix: "the model stack is reimplemented in Zig, the transformer weights are retrained, and the decompressor is ..."
  (self-extracting Linux archive); one core, 9.4 GB — [LTCB](https://mattmahoney.net/dc/text.html).
- StateSMix: "pure C", commodity x86-64 with AVX2, no GPU — [arXiv](https://arxiv.org/abs/2605.02904).
- nncp v3: "coded in C and supports recent NVIDIA GPUs"; v3.3 Linux (incl. CUDA) and Windows x64 builds —
  [bellard.org/nncp](https://bellard.org/nncp/); [LTCB](https://mattmahoney.net/dc/text.html).
- ts_zip: binary downloads for Linux and Windows, 4 GB RAM — [bellard.org/ts_zip](https://bellard.org/ts_zip/).
- Nacrith: Python 3.10+, llama-cpp-python, PyTorch fallback, transformers, accelerate, numpy, nvidia-smi —
  [GitHub](https://github.com/robtacconelli/Nacrith-GPU).
- L3TC: "Official PyTorch implementation", `pip install -r requirements.txt` — [GitHub](https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression).
- FineZip: Python, github.com/fazalmittu/FineZip — [arXiv](https://arxiv.org/html/2409.17141).
- Rust search only surfaced `arcode-rs`, a Rust arithmetic coder library (2019, ~23 stars), not a compressor —
  [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fcgbur%2Farcode-rs).

### Inferences
- Redukt appears to be the only pretrained-backbone (>=100M params) neural compressor delivered as a zero-dependency single
  binary; the Zig/C systems that match that packaging either use tiny (6M / 120K) models or online-only learning.

### Gaps
- GitHub was not searched directly for "neural compressor rust" (web search only); a small private/unindexed project could
  exist.

---

## Key Question 8 — Derived cost per GB (hours of hardware per GB; prices NOT sourced)

### Takeaway
Measured throughputs translate to roughly 30-180 single-core-hours per GB for the CPU Hutter-lineage leaders, ~67 RTX-3090-h
per GB for nncp v3.2, ~0.3 RTX-4090-h per GB for ts_zip, ~0.06 A100-h per GB for L3TC-200K, and 110-23,000 A6000-h per GB
for FineZip/LLMZip; the report writer should multiply by their own cloud prices (none were sourced here).

### Cited Findings (throughput sources) and derived hours/GB (my arithmetic: 1e9 B / bytes-per-s / 3600)
- ex-girlfriend 4.5 KB/s -> 61 core-h/GB; cmix-neif 5.7 KB/s -> 48.7 h; zmix 7.6 KB/s -> 36.5 h; fx2-cmix-transformer
  7.5 KB/s -> 37 h (Ryzen 5900X) or ~192 h (i7-1165G7, "8 days"); cmix v21 1.6 KB/s -> 173 h; paq8px -12L 3.4 KB/s -> 81 h —
  [LTCB](https://mattmahoney.net/dc/text.html).
- nncp v3.2 4.1 KB/s -> ~67 GPU-h/GB on RTX 3090 — [LTCB](https://mattmahoney.net/dc/text.html).
- ts_zip up to 1 MB/s -> ~0.28 GPU-h/GB on RTX 4090 — [bellard.org/ts_zip](https://bellard.org/ts_zip/).
- L3TC-200K 4.35 MB/s -> ~0.064 A100-h/GB; L3TC-3.2M 2.5 MB/s -> 0.11 A100-h/GB — [L3TC](https://arxiv.org/html/2412.16642v2).
- RWKV-1.5B 20 KB/s -> ~14 A100-h/GB; Llama2-7B 280 B/s -> ~990 A100-h/GB — [L3TC Table 5](https://arxiv.org/html/2412.16642v2).
- FineZip 0.67 KB/s -> ~417 A6000-h/GB; FineZip-4bit 2.5 KB/s -> ~112 h; LLMZip ~12 B/s -> ~22,700 h — [FineZip](https://arxiv.org/html/2409.17141).
- Nacrith 0.09-3.5 KB/s (inconsistent, see Q1) -> ~80 to ~3,000 GTX-1050Ti-h/GB — [GitHub](https://github.com/robtacconelli/Nacrith-GPU); [arXiv HTML](https://arxiv.org/html/2602.19626).
- tufazip 1.5 KB/s on 2x B200 -> ~180 DGX-h/GB — [LTCB](https://mattmahoney.net/dc/text.html).
- Hutter test budget: ~49 h single core (i7-1165G7) for 1 GB — [hrules](https://hutter1.net/prize/hrules.htm).
- Coordinator-provided Redukt: 1,180 B/s on 12 vCPU -> ~235 wall-h = ~2,800 vCPU-h/GB; T4 4.2 KB/s -> ~66 T4-h/GB (not
  researched, given).

### Inferences
- In hardware-hours per GB, Redukt on a T4 (~66 h) is in the same band as nncp v3.2 on an RTX 3090 (~67 h) but at a worse
  ratio on short files; on CPU, Redukt's ~2,800 vCPU-h/GB is ~50-75x the cmix-neif/zmix core-hours, which is the gap the
  coordinator already describes as 30-500x to viability.
- The only neural systems with sub-1-GPU-hour-per-GB economics (ts_zip, L3TC) sit at 1.1-1.4 bpb, i.e. they trade ~15-45%
  of the ratio advantage for 100-1,000x speed; this is the "fast tier" Redukt's roadmap R-105 references.

### Gaps
- No source gives $/GB for any neural compressor; cloud spot prices were deliberately not researched (out of scope), so the
  report writer must supply them (e.g., via `scripts/breakeven.py`).
- Nacrith's true single-GPU throughput is unresolved (paper vs README differ by ~30x).
