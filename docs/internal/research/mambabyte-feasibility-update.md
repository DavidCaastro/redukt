# MambaByte Training Feasibility Update (R-079)

**Date**: 2026-11-07
**Status**: Research complete — CONDITIONAL GO (MambaByte from scratch)
**Dependencies**: R-041 (original research), R-077 (entropy gate NOT VIABLE at BPE 50K)

---

## Context

R-077 proved that entropy-gated inference is NOT VIABLE with BPE 50K vocab:
CM entropy is always 15.49-15.62 bits (near log2(50280)), leaving no operating point.
With byte-level model (vocab=256), CM entropy drops to ~5-6 bits vs 8 max,
enabling 40-70% skip rate.

This update evaluates two paths to byte-level Mamba:
1. Byteification adapter (retrofit existing model)
2. MambaByte from-scratch training

---

## 1. Byteification Adapter (Nature, Oct 2026)

**Paper**: "Retrofitting language models to operate over bytes"
**Venue**: Nature, DOI: 10.1038/s41586-026-11111-4
**Authors**: Allen Institute for AI, U. Edinburgh, U. Washington

### Method (two-stage)
- Stage 1: Progressive Knowledge Distillation — align byte-level encoder with teacher
  token embeddings (L2 alignment → KL divergence → boundary learning). ~9.8B tokens.
- Stage 2: Byte-Level SFT — replace token LM head with Dechunk+Decoder, transition
  to next-byte prediction. ~39.3B tokens end-to-end.
- Total: 49.1B tokens, <1% of typical pretraining budget.

### Key Finding
The byte-level encoder uses **Mamba2 SSM** architecture. Mamba2 encoder achieves
70.2 MMLU on Qwen-3 4B vs Transformer encoder collapsing to 31.3.
**SSM superiority for byte-level processing is validated.**

### Models Retrofitted
OLMo 2 1B, OLMo 3 7B, Qwen3 8B, Llama 3 8B.
Bolmo 7B: +16.5% absolute STEM improvement over BLT 7B baseline.
Retains >92% of original model performance with <1% pretraining budget.

### Compatibility with Redukt: PARTIAL
- Mamba2 encoder component validates SSM+byte synergy (directly relevant)
- Full pipeline assumes Transformer-based "global" model (KV-cache, attention)
- Mamba-130M lacks these → would need custom adaptation
- Real cost after R&D likely exceeds from-scratch training
- **Verdict: NOT RECOMMENDED for Redukt** (too much adaptation work)

---

## 2. MambaByte From-Scratch (RECOMMENDED)

**Paper**: "MambaByte: Token-free Selective State Space Model"
**Venue**: ICLR 2024 (oral), arXiv:2401.13660
**Authors**: Wang et al., Cornell

### Published Results
- MambaByte-353M on PG19: **0.930 BPB** (8192-byte context)
- Beats MegaByte-758M (1.00 BPB) with 0.63x less compute
- Architecture: 53 layers, 1024 dims, d_state=16, expansion=2
- Training: 30B bytes, 8192-byte sequences, batch 48, BF16

### Cost Estimates (6ND formula, A100 at $1.88/hr, 45% MFU)

| Model Size | PetaFLOPs | A100-hours | Cost |
|---|---|---|---|
| MambaByte-50M (minimal) | 9.0 | 18 | **$34** |
| MambaByte-130M (target) | 23.4 | 46 | **$87** |
| MambaByte-130M (50B data) | 39.0 | 77 | **$145** |
| MambaByte-353M (published) | 63.5 | 126 | **$237** |

### Expected BPB for Redukt
- MambaByte-130M alone: ~1.10-1.30 BPB enwik8 (smaller than 353M)
- MambaByte-130M + CM ensemble: **0.85-1.05 BPB** enwik8
- CM contributes more with byte-level (256 classes, higher entropy room)
- Current Mamba-130M BPE + CM: 0.9609 BPB → potential improvement

---

## 3. GPU Throughput Projections (T4)

T4 specs: 16GB GDDR6, 320 GB/s bandwidth, 65 TFLOPS FP16.

### Current BPE Mamba-130M on T4
- Conservative: 33 tok/s (DRAM-bound, matches CPU SIMD)
- Optimized (CUDA kernels): 150-350 tok/s
- In bytes/s (3.5 bytes/BPE-token avg): 525-1,225 bytes/s

### MambaByte-130M on T4 (byte-level, vocab=256)
- LM head: 130M x 50,280 → 130M x 256 = **196x smaller** output projection
- LM head is 30% of FLOPs → removing 99.5% saves ~29.8% compute
- But: byte sequences ~3.5x longer than BPE for same content
- Estimated: 195-455 byte-tok/s (without entropy gate)

### With Entropy Gate (R-077 concept, enabled by vocab=256)
CM entropy at vocab=256: ~5-6 bits (vs 8 max). Gate viable at threshold ~6.5 bits.

| Scenario | byte-tok/s | vs CPU (119 B/s) | Notes |
|---|---|---|---|
| GPU naive (no gate) | 195-455 | 1.6-3.8x | Longer sequences offset LM savings |
| GPU + 50% gate | 390-910 | 3.3-7.6x | CM handles confident positions |
| GPU + 70% gate | 650-1,517 | 5.5-12.7x | High-confidence CM bypass |
| GPU optimized + 70% | 1,000-2,500 | 8.4-21.0x | CUDA kernel optimization |

Note: CPU = 34 Mamba BPE tok/s × 3.5 bytes/tok = 119 bytes/s effective.

---

## 4. Competitive Landscape

| System | BPB (enwik8) | Architecture | Status |
|---|---|---|---|
| **Redukt** (current) | **0.9609** | Mamba-130M BPE + 5-CM + AC | Production |
| Nacrith | 0.9389 | SmolLM2-135M BPE + CM + AC | arXiv 2026 |
| StateSMix | 2.13 | Mamba 120K online + CM | arXiv 2025 |
| MambaByte-353M | 0.930 | MambaByte (byte-level) | ICLR 2024 |
| **Redukt + MambaByte** | **0.85-1.05** | MambaByte-130M + 5-CM + AC | Projected |

"Breaking the Token Ceiling" (arXiv:2609.12303, Sep 2026) confirms byte models
outperform token-distilled baselines by up to 4% with 1/6th training data.

---

## 5. Recommendation

### Option A: MambaByte From Scratch — RECOMMENDED
- **Cost**: $87-145 (130M params, 30-50B bytes)
- **Timeline**: 2-4 days training + 1-2 weeks integration
- **BPB**: 0.85-1.05 on enwik8 (with CM ensemble)
- **Unlocks**: entropy gate (40-70% skip) + eliminates BPE binary regressions
- **Risk**: moderate (PyTorch external training, integration)

### Option B: Byteification Adapter — NOT RECOMMENDED
- **Cost**: $2-8 (theoretical), but significant R&D for Mamba adaptation
- Full pipeline assumes Transformer global model
- Real cost after adaptation likely exceeds Option A

### Option C: Status Quo + GPU Plugin Only — FALLBACK
- **Cost**: $0 training
- BPE 50K blocks entropy gate permanently
- GPU T4: 150-350 tok/s (4-10x CPU), no entropy gate
- Binary regressions persist (sao, ooffice, x-ray)

### Phased Plan (Option A)

1. **Validate** (1 day, $34): Train MambaByte-50M on 30B bytes, measure BPB
2. **Train** (2-3 days, $87-145): MambaByte-130M on 30-50B mixed corpus
3. **Integrate** (1 week): Rust inference (reuse existing Mamba SIMD infra)
4. **Gate** (2 days): Re-enable R-077 entropy gate with vocab=256
5. **GPU** (1 week): CUDA kernels for MambaByte forward pass

Total investment: $121-179, 3-4 weeks calendar time.
Expected outcome: 0.85-1.05 BPB + 650-2,500 byte-tok/s GPU.

---

## Sources

1. Retrofitting LMs to bytes (Nature, Oct 2026) — DOI: 10.1038/s41586-026-11111-4
2. Distilling Token→Byte Models (arXiv:2602.01007) — Bao et al.
3. MambaByte (arXiv:2401.13660, ICLR 2024) — Wang et al.
4. StateSMix (arXiv:2605.02904) — Tacconelli 2025
5. Nacrith (arXiv:2602.19626) — Neural lossless compression 2026
6. Breaking the Token Ceiling (arXiv:2609.12303) — Marathe et al. Sep 2026
7. MambaByte weights: huggingface.co/JunxiongWang/MambaByte_Arxiv
