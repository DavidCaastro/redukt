# MambaByte Training Infrastructure Audit

**Date**: 2026-11-07
**Purpose**: Complete inventory of existing vs required components for MambaByte training
**Target**: Kaggle TPU v5e-8 (20h/week free) + A100 fallback ($87-145)

## 1. What Exists (Inference Only)

### Rust Mamba-130M Inference Engine
- `src/infrastructure/mamba/model.rs` (522 lines): Complete forward pass, AVX2 SIMD
- Config: d_model=768, d_inner=1536, d_state=16, d_conv=4, n_layer=24, vocab=50280
- MambaLayerWeights: norm, in_proj, conv1d, x_proj, dt_proj, a_log, d_vec, out_proj
- MambaState + MambaScratch for RNN state management
- Throughput: 34 tok/s SIMD (CPU DDR4), 72.4 tok/s (Kaggle Xeon)

### Tokenizer
- `src/infrastructure/mamba/tokenizer.rs` (447 lines): GPT-NeoX BPE, 50,280 vocab
- 13 synthetic byte fallback tokens for binary safety (BUG-001 fix)
- Format: HuggingFace tokenizer.json

### Weights
- `weights/mamba-130m/model.safetensors` (493 MB): Apache-2.0
- Source: state-spaces/mamba-130m-hf (HuggingFace)
- Key format: `backbone.layers.N.mixer.xxx`, `backbone.embeddings.weight`

### GPU Plugin (Skeleton)
- `plugins/redukt-gpu/src/lib.rs`: FFI interface, 4 exported functions
- BackendVTable contract (R-072), placeholder uniform logits
- No actual compute kernels

### Kaggle Infrastructure
- `scripts/kaggle-setup.sh`, `kaggle-push.sh`, `kaggle-status.sh`
- `notebooks/gpu-validation.py`: Validated on T4, determinism PASS, roundtrip PASS
- `scripts/download-mamba-weights.sh`: HuggingFace download

### SafeTensors Reader
- `src/infrastructure/rwkv7/safetensors.rs`: Zero-dep Rust SafeTensors parser
- Supports F32, F16, BF16 with automatic conversion to F32

## 2. What Needs To Be Built

### R-080: MambaByte Training Notebook
**Purpose**: Complete PyTorch training script for MambaByte on Kaggle TPU/GPU
**Deliverable**: `notebooks/mambabyte-train.py`

Components:
1. **MambaByte model definition** (PyTorch)
   - Identical architecture to Mamba-130M but vocab_size=256
   - Layers: Embedding(256, 768) -> 24x MambaBlock -> RMSNorm -> Linear(768, 256)
   - SSM params: d_state=16, d_conv=4, dt_rank=48
   - Total params: ~96M (vs 130M -- LM head shrinks from 768x50280 to 768x256)
   - Tied embeddings (embedding.weight == lm_head.weight)

2. **Training loop**
   - Optimizer: AdamW (lr=6e-4, beta1=0.9, beta2=0.95, wd=0.1)
   - Schedule: linear warmup 2000 steps -> cosine decay to 6e-5
   - Loss: cross-entropy on next-byte prediction
   - Gradient clipping: max_norm=1.0
   - Mixed precision: BF16 (native on TPU), FP16+AMP on GPU
   - Batch size: 48 sequences x 8192 bytes = 393KB per step
   - Total: 30B bytes / 393KB = ~76K steps

3. **Dataset pipeline**
   - Byte-level tokenization (identity: byte value = token ID)
   - Sources: PG19 (books, ~11GB), SlimPajama subset (~20GB), Wikipedia (~6GB)
   - Streaming DataLoader (no full corpus in memory)
   - Random 8192-byte chunks from shuffled files

4. **TPU adaptation** (torch_xla)
   - `import torch_xla.core.xla_model as xm`
   - 8-way data parallelism via `xm.xla_device()`
   - `xm.optimizer_step()` for gradient sync
   - `xm.mark_step()` for XLA graph compilation
   - Batch size per core: 6 (total 48 across 8 cores)

5. **Checkpointing**
   - Save every 5000 steps to `/kaggle/working/checkpoints/`
   - Resume from latest checkpoint on session restart
   - Export final model to SafeTensors format
   - Critical: Kaggle TPU sessions max 9 hours

6. **Validation**
   - Every 1000 steps: BPB on enwik8 10KB (byte-level, no BPE)
   - Every 5000 steps: BPB on 4 files (enwik8, dickens, apache, samba)
   - Log: step, loss, lr, BPB, tok/s, GPU/TPU memory

### R-081: Weight Conversion Pipeline
**Purpose**: Convert PyTorch checkpoint -> SafeTensors with Rust-compatible key names
**Deliverable**: `scripts/convert-mambabyte-weights.py`

Key mapping (PyTorch -> SafeTensors):
- `layers.{i}.norm.weight` -> `backbone.layers.{i}.norm.weight`
- `layers.{i}.mixer.in_proj.weight` -> `backbone.layers.{i}.mixer.in_proj.weight`
- `layers.{i}.mixer.conv1d.weight` -> `backbone.layers.{i}.mixer.conv1d.weight`
- `layers.{i}.mixer.conv1d.bias` -> `backbone.layers.{i}.mixer.conv1d.bias`
- `layers.{i}.mixer.x_proj.weight` -> `backbone.layers.{i}.mixer.x_proj.weight`
- `layers.{i}.mixer.dt_proj.weight` -> `backbone.layers.{i}.mixer.dt_proj.weight`
- `layers.{i}.mixer.dt_proj.bias` -> `backbone.layers.{i}.mixer.dt_proj.bias`
- `layers.{i}.mixer.A_log` -> `backbone.layers.{i}.mixer.A_log`
- `layers.{i}.mixer.D` -> `backbone.layers.{i}.mixer.D`
- `layers.{i}.mixer.out_proj.weight` -> `backbone.layers.{i}.mixer.out_proj.weight`
- `embedding.weight` -> `backbone.embeddings.weight`
- `norm_f.weight` -> `backbone.norm_f.weight`
- `lm_head.weight` -> `lm_head.weight` (or tied to embedding)

Steps:
1. Load PyTorch checkpoint
2. Remap keys to HuggingFace naming convention
3. Verify all tensor shapes match expected config
4. Save as SafeTensors (F32)
5. Print SHA-256 for provenance

### R-082: Rust Byte Tokenizer + Model Loader
**Purpose**: Integrate MambaByte into Redukt inference pipeline
**Deliverables**:
- `src/infrastructure/mamba/byte_tokenizer.rs`
- Updates to `model_loader.rs`, `backbone.rs`

Components:
1. **ByteTokenizer** implementing `BackendTokenizer` trait
   - `encode(&self, data: &[u8]) -> Vec<u32>`: identity (each byte = token ID)
   - `decode(&self, tokens: &[u32]) -> Vec<u8>`: identity (token ID = byte)
   - `vocab_size(&self) -> usize`: 256
   - No merges, no tokenizer.json, no BPE logic

2. **BackboneChoice::MambaByte** variant
   - New enum variant in `src/domain/backbone.rs`
   - AZTH v4 FLAG 0x40 backbone byte: 0=RWKV, 1=Mamba(BPE), 2=MambaByte
   - CLI: `--backbone mambabyte`

3. **model_loader.rs** updates
   - Detect MambaByte weights (vocab_size=256 in config or by directory name)
   - Load with MambaConfig { vocab_size: 256, ... }
   - Return DynModel::Mamba with ByteTokenizer instead of BpeTokenizer
   - Weight file: `weights/mambabyte-96m/model.safetensors`

4. **MambaModel changes**: NONE needed
   - MambaModel already parameterized by MambaConfig.vocab_size
   - Embedding and LM head sizes are config-driven
   - SSM layers are vocab-independent
   - Only difference: vocab_size (256 vs 50280)

### R-083: Cross-Framework Validation Notebook
**Purpose**: Verify PyTorch training output matches Rust inference bit-level
**Deliverable**: `notebooks/mambabyte-validate.py`

Steps:
1. Load same weights in PyTorch and export logits for 100 bytes of enwik8
2. Load same weights in Rust and compute logits for same 100 bytes
3. Compare: max absolute difference should be <1e-5 (F32 precision)
4. Compress enwik8 10KB with MambaByte, verify BPB matches expected range
5. Roundtrip test: compress -> decompress -> byte-identical

## 3. Training Timeline

### On Kaggle TPU v5e-8 (free)
| Phase | Duration | Quota Used |
|---|---|---|
| MambaByte-50M validation | 5 weeks | 100h TPU |
| MambaByte-96M production | 10-12 weeks | 200-240h TPU |
| **Total** | 15-17 weeks | 300-340h TPU |

### On Paid A100 ($87-145)
| Phase | Duration | Cost |
|---|---|---|
| MambaByte-50M validation | 18h | $34 |
| MambaByte-96M production | 46h | $87 |
| **Total** | 64h (2.7 days) | $121 |

### Hybrid (recommended)
- Validate 50M on Kaggle TPU: 5 weeks, $0
- Train 96M on A100: 2 days, $87
- Total: 5 weeks + 2 days, $87

## 4. MambaByte-96M Architecture (vs Mamba-130M)

| Parameter | Mamba-130M (BPE) | MambaByte-96M (Byte) |
|---|---|---|
| vocab_size | 50,280 | 256 |
| d_model | 768 | 768 |
| d_inner | 1536 | 1536 |
| d_state | 16 | 16 |
| d_conv | 4 | 4 |
| n_layer | 24 | 24 |
| Embedding params | 38.6M | 197K |
| LM head params | 38.6M | 197K |
| Layer params | 53.1M | 53.1M |
| **Total params** | **130.3M** | **~96M** |
| Weight file size | 493 MB | ~365 MB |
| Tokens per byte | ~0.29 (BPE) | 1.0 (identity) |
| Sequence length | ~2,900 tok/10KB | 10,240 tok/10KB |

Key insight: MambaByte processes 3.5x more tokens per byte, but each forward pass is cheaper (smaller LM head). The SSM recurrence cost is identical.

## 5. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| BPB worse than Mamba-130M BPE | High | Validate 50M first ($0 on Kaggle) |
| TPU training instability | Medium | Checkpointing every 5K steps |
| Rust inference numerical mismatch | Medium | R-083 cross-validation |
| Dataset quality | Low | Use established corpora (PG19, Wikipedia) |
| Kaggle session interruption | Low | Resume from checkpoint |
| Training divergence | Medium | LR schedule, gradient clipping, monitor loss |

## 6. Expected Outcomes

### BPB Projections (enwik8 10KB, with CM ensemble)
- MambaByte-50M (validation): 1.10-1.40 BPB
- MambaByte-96M (production): 0.85-1.05 BPB
- Current Mamba-130M BPE: 0.9609 BPB

### Entropy Gate Unlock
- Byte-level vocab=256: CM entropy ~5-6 bits (vs 15.5 at BPE 50K)
- Viable gate threshold: ~6.5 bits
- Expected skip rate: 40-70%
- Effective throughput: 2-5x multiplier

### Throughput (GPU T4, with entropy gate 70%)
- MambaByte-96M: 650-2,500 byte-tok/s
- vs Current CPU: 119 bytes/s (34 tok/s x 3.5 bytes/tok)
- Speedup: 5.5-21x

## 7. Sources

1. MambaByte (arXiv:2401.13660, ICLR 2024) -- Wang et al.
2. Mamba (arXiv:2312.00752) -- Gu & Dao, 2023
3. Retrofitting LMs to bytes (Nature, Oct 2026) -- Allen Institute
4. Breaking the Token Ceiling (arXiv:2609.12303) -- Marathe et al.
5. Kaggle TPU v5e-8 announcement -- kaggle.com/product-announcements/607202
6. PyTorch XLA documentation -- pytorch.org/xla
7. MambaByte weights -- huggingface.co/JunxiongWang/MambaByte_Arxiv
