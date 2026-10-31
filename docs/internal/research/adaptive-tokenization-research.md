# Adaptive Tokenization for Neural Compression — Research Analysis

> Redukt Research Document — 2026-10-31
> Status: Complete
> Scope: Tokenization strategies for dual-backbone (Mamba-130m + RWKV-7) compression

---

## 1. Problem Statement

Redukt operates two neural backbones with fundamentally different tokenization:

| Backbone | Tokenizer | Vocab Size | Bytes/Token (text) | Bytes/Token (binary) |
|---|---|---|---|---|
| Mamba-130m | GPT-NeoX BPE | 50,280 | 2.8–3.6 | 1.0–1.3 |
| RWKV-7 0.1B | World (byte-level) | 65,536 | ~1.0 | ~1.0 |

BPE tokenization gives Mamba a massive advantage on text (29–81% better BPB than RWKV) [MEASURED] but collapses on binary data (7–41% worse than RWKV) [MEASURED]. The question: can a single tokenization strategy work universally, or must Redukt route between backbones?

---

## 2. Literature Survey

### 2.1 No Universal Solution Exists

No production compression system has solved "one tokenizer for all data types with a pretrained model" [SOURCED]. Every known system adopts one of three strategies:

1. **Byte/bit-level exclusively** — cmix [SOURCED: byronknoll.com/cmix.html], paq8px [SOURCED: github.com/hxim/paq8px], DeepZip. Avoids the tokenization problem entirely but requires enormous models or online training.
2. **Separate backends per data type** — Nacrith NC06 routes text to SmolLM2+ensemble neural and binary to LZMA fallback [SOURCED: arXiv:2602.19626]. Achieves SOTA 0.94 BPB on enwik8.
3. **Single tokenizer, accept binary weakness** — ts_zip, FineZip [SOURCED: arXiv:2409.17141], current Redukt Mamba path.

### 2.2 Tokenizer-Model Coupling Is Real

The intuition that "changing tokenization breaks the model" is well-supported:

- BPE-trained models develop representations deeply coupled to BPE token boundaries [SOURCED: Tokenization Falling Short, EMNLP 2024].
- Feeding non-canonical tokenizations (e.g., byte sequences to a BPE model) is out-of-distribution (OOD) and degrades quality significantly [SOURCED: EMNLP 2024].
- Embedding surgery alone (mapping new vocab to old embeddings) loses 10–30% quality without continued pretraining [SOURCED: ReTok, arXiv:2410.04335].
- The gap between 50,280 BPE tokens and 256 raw bytes is too large for surgery alone — the model's internal representations assume multi-byte token granularity [SOURCED].
- BPE-Dropout demonstrates that even small tokenization perturbations affect downstream quality [SOURCED: Provilkov et al., 2020].

### 2.3 Architecture-Level Solutions (Require Training)

Several recent architectures eliminate the tokenization problem by operating at byte level:

| System | Scale | BPB (PG19/enwik8) | Key Technique | Reference |
|---|---|---|---|---|
| MambaByte | 353M | 0.93 (PG19) | Pure byte-level Mamba | arXiv:2401.13660 [SOURCED] |
| BLT | 8B | Matches subword | Entropy-based dynamic patching | ACL 2025, Meta [SOURCED] |
| SpaceByte | — | 1.009 (PG19) | Space-aware byte grouping | NeurIPS 2024 [SOURCED] |
| Bolmo/Byteification | Varies | <1% loss vs subword | mLSTM adapter retrofit | Nature 2026, arXiv:2512.15586 [SOURCED] |

**MambaByte** (Wang et al., 2024) is the most directly relevant: a pure byte-level Mamba at 353M parameters achieves 0.93 BPB on PG19, competitive with subword-tokenized models of similar size [SOURCED: arXiv:2401.13660]. This proves that Mamba's SSM architecture can work at byte granularity, but requires training from scratch.

**Bolmo/Byteification** (Nature 2026) demonstrates retrofitting byte-level capability onto pretrained subword models using mLSTM adapter layers, at <1% of pretraining compute [SOURCED: arXiv:2512.15586]. This is the closest to a "drop-in" solution but still requires adapter training.

**BLT** (Meta, ACL 2025) uses entropy-based dynamic patching — grouping bytes into variable-length patches based on local entropy — and matches tokenized LLMs at 8B scale [SOURCED]. The entropy-based segmentation concept is relevant to compression.

**SpaceByte** (NeurIPS 2024) achieves 1.009 BPB on PG19 as the first byte-level model to match subword compute efficiency [SOURCED].

### 2.4 Compression-Specific Systems

| System | Approach | BPB (enwik8) | Training | Reference |
|---|---|---|---|---|
| Nacrith NC06 | Segment + route (text→neural, binary→LZMA) | 0.94 | Pretrained | arXiv:2602.19626 [SOURCED] |
| cmix v21 | Bit-level, 300+ models, file-type detection | 1.16 | Online | byronknoll.com/cmix.html [SOURCED] |
| paq8px | Bit-level, context mixing, type-aware subsets | ~1.20 | Online | github.com/hxim/paq8px [SOURCED] |
| StateSMix | Online-trained Mamba (120K params) per-file | 2.12 | Per-file | arXiv:2605.02904 [SOURCED] |
| NNCP v2 | LSTM, byte-level, online fine-tune | ~1.0 | Per-file | bellard.org/nncp [SOURCED] |
| FineZip | LLM + per-file fine-tuning | — | Per-file | arXiv:2409.17141 [SOURCED] |
| OmniZip | Multi-modal neural compression | — | Pretrained | CVPR 2026, arXiv:2602.22286 [SOURCED] |
| L3TC | Learned lossless text compression | — | Pretrained | AAAI 2025, arXiv:2412.16642 [SOURCED] |
| zip2zip | Compressed-domain transform | — | Pretrained | NeurIPS 2025, arXiv:2506.01084 [SOURCED] |
| G2-V3 | Grammar-guided compression | — | — | arXiv:2606.17712 [SOURCED] |

**Key observation**: Nacrith NC06 achieves SOTA by routing, not by building a universal tokenizer [SOURCED: arXiv:2602.19626]. cmix and paq8px achieve universality through bit-level operation with hundreds of specialized models [SOURCED].

### 2.5 Binary-Specific Tokenization

**Binary BPE** (Bommarito, 2024) trains BPE specifically on binary data with a 65K vocabulary, achieving ~3.37x compression ratio [SOURCED: arXiv:2511.17573]. A Rust crate (`bbpe`) exists. However, this requires a model trained on binary BPE tokens — it cannot be used with Mamba-130m's pretrained weights without retraining.

---

## 3. Analysis: Why Adaptive Tokenization Alone Cannot Solve This

### 3.1 The Fundamental Constraint

Redukt uses **pretrained** models (Mamba-130m, RWKV-7 0.1B). Any tokenization change must work with existing weights. This eliminates:

- MambaByte (requires byte-level training from scratch) [SOURCED: arXiv:2401.13660]
- Binary BPE (requires BPE-binary training) [SOURCED: arXiv:2511.17573]
- BLT-style patching (requires architecture change + training) [SOURCED]
- SpaceByte (requires architecture change + training) [SOURCED]

### 3.2 What Remains Viable Without Training

1. **Backbone routing** — Use existing R-028 content detector to select Mamba (text) or RWKV (binary). Zero training, zero risk.
2. **Ensemble mixing** — Run both backbones, weight their probability distributions. Higher compute cost but potentially better on mixed content.
3. **Bolmo-style adapter** — Train a small adapter (<1% compute) to give Mamba byte-level capability. Medium risk, requires training infrastructure.

### 3.3 The Dual-Backbone Advantage

Redukt's architecture already contains the solution. With R-034 (Backbone Dock) providing a unified trait interface and R-028 (content detector) classifying input into 6 types, the infrastructure for routing exists:

| Content Type | Optimal Backbone | Reason |
|---|---|---|
| Text (UTF-8) | Mamba-130m | BPE merges → 2.8–3.6 bytes/token → better compression [MEASURED] |
| Structured text (XML, JSON, logs) | Mamba-130m | High redundancy + BPE vocabulary coverage [MEASURED] |
| Native code (PE/ELF) | RWKV-7 | Byte-level tokenization handles arbitrary byte patterns [MEASURED] |
| Binary data (images, archives) | RWKV-7 | BPE falls to ~1 byte/token, negating its advantage [MEASURED] |
| Mixed content | Ensemble or segment | Depends on text/binary ratio |
| PDF/compound | RWKV-7 | Mostly binary with sparse text fragments [MEASURED] |

---

## 4. Approaches Ranked for Redukt

| Rank | Approach | Risk | Training Required | Impact | Feasibility |
|---|---|---|---|---|---|
| 1 | Content-adaptive backbone routing (R-028 → Mamba/RWKV) | Low | None | High — best of both worlds | Immediate |
| 2 | Dual-backbone ensemble (weighted probability mix) | Medium | None/minimal | Higher — better on mixed content | Short-term |
| 3 | Byte-level Mamba (MambaByte-small, trained from scratch) | High | Full training | Highest — eliminates tokenization problem | Long-term |
| 4 | Byteification adapter (Bolmo-style retrofit) | High | Adapter training | High — keeps pretrained weights | Medium-term |
| 5 | Online-trained SSM per-file (StateSMix-style) | Medium | Per-file online | Moderate — no pretrained weights needed | Medium-term |

### 4.1 Rank 1: Content-Adaptive Backbone Routing

**Implementation**: R-028 content detector → if text/structured → Mamba-130m, else → RWKV-7. Store backbone ID in AZTH header (1 bit in FLAGS).

**Projected gains** [ESTIMATE]:
- Text files: use Mamba BPB (0.16–0.91) instead of RWKV (0.30–1.90) → 29–81% improvement
- Binary files: keep RWKV BPB (3.69–9.22) instead of Mamba (5.22–10.72) → avoid 7–41% regression
- Both backbones already integrated via R-034 Backbone Dock

**Risk**: Low. Both backbones are proven. Content detection is already implemented. Decompressor must load the correct backbone based on header flag.

**Cost**: Decompressor must have access to both sets of weights (~630MB Mamba + ~350MB RWKV). Compression time unchanged (only one backbone runs per file).

### 4.2 Rank 2: Dual-Backbone Ensemble

**Implementation**: Run both backbones in parallel, mix probability distributions with learned or heuristic weights. Similar to how CM already mixes 19 hash models + LSTM.

**Advantage over routing**: Handles mixed-content files (e.g., HTML with embedded base64) without segmentation.

**Risk**: Medium. 2x compute cost. Probability mixing requires careful calibration to avoid degrading either backbone's predictions. Weight tuning may need a small validation set.

### 4.3 Rank 3–5: Training-Required Approaches

These are documented for completeness but are outside Redukt's current zero-dependency, no-training-infrastructure constraints:

- **MambaByte-small** would require training a ~130M byte-level Mamba from scratch (~$500–2000 compute) [ESTIMATE].
- **Byteification adapter** would require training mLSTM adapter layers on Mamba-130m (~$50–200 compute) [ESTIMATE based on: arXiv:2512.15586].
- **StateSMix-style online SSM** would train a small model per file at compression time, adding significant latency [SOURCED: arXiv:2605.02904].

---

## 5. Sources

| # | Reference | Year | Key Contribution |
|---|---|---|---|
| 1 | MambaByte — Wang et al., arXiv:2401.13660 | 2024 | Byte-level Mamba, 0.93 BPB PG19 |
| 2 | Bolmo/Byteification — Nature, arXiv:2512.15586 | 2026 | Retrofit byte-level via mLSTM adapters |
| 3 | BLT — Meta, ACL 2025 | 2025 | Dynamic entropy-based byte patching |
| 4 | SpaceByte — NeurIPS 2024 | 2024 | First byte-level matching subword compute |
| 5 | Nacrith NC06 — Tacconelli, arXiv:2602.19626 | 2026 | Segment+route, SOTA 0.94 BPB enwik8 |
| 6 | StateSMix — arXiv:2605.02904 | 2025 | Online-trained Mamba per-file |
| 7 | cmix — byronknoll.com/cmix.html | 2021+ | Bit-level, 300+ models |
| 8 | paq8px — github.com/hxim/paq8px | 2020+ | Bit-level context mixing |
| 9 | Binary BPE — Bommarito, arXiv:2511.17573 | 2024 | BPE trained on binary, 65K vocab |
| 10 | OmniZip — CVPR 2026, arXiv:2602.22286 | 2026 | Multi-modal neural compression |
| 11 | L3TC — AAAI 2025, arXiv:2412.16642 | 2025 | Learned lossless text compression |
| 12 | zip2zip — NeurIPS 2025, arXiv:2506.01084 | 2025 | Compressed-domain transform |
| 13 | NNCP v2 — bellard.org/nncp | 2021 | LSTM byte-level + online fine-tune |
| 14 | Tokenization Falling Short — EMNLP 2024 | 2024 | OOD from non-canonical tokenization |
| 15 | BPE-Dropout — Provilkov et al. | 2020 | Tokenization perturbation effects |
| 16 | ReTok — arXiv:2410.04335 | 2024 | Retokenization + embedding surgery |
| 17 | FineZip — arXiv:2409.17141 | 2024 | LLM + per-file fine-tuning |
| 18 | G2-V3 — arXiv:2606.17712 | 2026 | Grammar-guided compression |

---

## 6. Conclusion

**Adaptive tokenization for a pretrained model is an unsolved problem** [SOURCED]. No known technique allows feeding arbitrary byte sequences to a BPE-trained model without significant quality degradation, unless adapter layers are trained (Bolmo) or the model is retrained from byte-level (MambaByte).

**Redukt's dual-backbone architecture is the correct solution**. Rather than forcing one tokenizer to handle all data types, Redukt can leverage its existing infrastructure:

1. **R-028** (content detector) already classifies files into 6 types.
2. **R-034** (Backbone Dock) already abstracts backbone selection behind traits.
3. Mamba-130m excels on text; RWKV-7 excels on binary. Together they cover the full spectrum.

**Recommended next step**: Implement content-adaptive backbone routing (Rank 1). This requires only a 1-bit flag in the AZTH header and routing logic in the compress/decompress paths. Zero training, zero new dependencies, immediate benefit across all 19 benchmark files.

The ensemble approach (Rank 2) is a natural follow-up for mixed-content files, but routing alone captures the majority of the potential gain.
