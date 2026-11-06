# Kaggle Validation Protocol — Redukt GPU Backend

> Protocol for validating GPU plugin on Kaggle's free GPU tier.
> Date: 2026-11-06.

---

## 1. Kaggle Resources

| Resource | Value |
|---|---|
| GPU options | T4 (16GB GDDR6), P100 (16GB HBM2), T4x2 |
| Weekly quota | 30 hours GPU, 20 hours TPU |
| Max session | 12 hours (GPU), 9 hours (TPU) |
| Idle timeout | 20 minutes |
| System RAM | ~29 GB |
| Disk (scratch) | ~20 GB (non-persistent) |
| Datasets | Up to 100 GB (read-only mount) |
| Internet | Yes (pip install, HuggingFace) |
| CUDA | 12.8+ |
| PyTorch | 2.10+ |

## 2. Setup

### API Access
1. Create Kaggle account (kaggle.com)
2. Settings → API → Create New Token → download `kaggle.json`
3. Place at `~/.kaggle/kaggle.json`
4. Install CLI: `pip install kaggle`

### Dataset Upload
Upload Mamba-130M weights as Kaggle Dataset:
```bash
kaggle datasets create -p weights/mamba-130m/ --dir-mode zip
```

### Plugin Build
Build plugin in Kaggle notebook (first cell):
```python
!git clone https://github.com/DavidCaastro/redukt.git
!cd redukt/plugins/redukt-gpu && cargo build --release
!cp redukt/plugins/redukt-gpu/target/release/libredukt_gpu.so .
```

## 3. Validation Protocol

### Phase 1: Plugin loads correctly
```bash
# Verify the plugin .so exists and exports symbols
nm -D libredukt_gpu.so | grep redukt_backend
# Expected: 4 symbols (init, forward, reset_state, free)
```

### Phase 2: Throughput benchmark (12 files)
Standard benchmark suite:
```
apache-access.log, facturae-batch.xml, ubl-invoice-batch.xml, nci,
dickens, enwik8, samba, mozilla, mr, ooffice, x-ray, sao
```

Each file at 10KB, measure:
1. **tok/s** (tokens per second)
2. **BPB** (bits per byte of compressed AZTH output)
3. **Roundtrip**: decompress → diff → byte-identical?

### Phase 3: Determinism verification
```bash
# Compress twice, outputs must be identical
redukt compress input.bin out1.azth --backend plugin
redukt compress input.bin out2.azth --backend plugin
diff out1.azth out2.azth  # must be empty
```

### Phase 4: CPU vs GPU BPB comparison
BPB from plugin MUST match CPU Mamba BPB exactly (same model, same weights).
Any difference indicates a determinism bug in GPU kernels.

## 4. Benchmark Template

### Kaggle notebook structure:
```
Cell 1: Setup (clone repo, build plugin, download weights)
Cell 2: Prepare test files (download/create 12 benchmark files)
Cell 3: CPU baseline (redukt compress each file, record BPB + tok/s)
Cell 4: GPU plugin (same files, --backend plugin, record BPB + tok/s)
Cell 5: Determinism check (compress twice, diff)
Cell 6: Results table + analysis
Cell 7: Save results to /kaggle/working/
```

## 5. MambaByte Training (R-041) on Kaggle

### Feasibility on free tier:
| Phase | A100-hours | T4-hours (x6) | Weeks @30h/wk |
|---|---|---|---|
| 0: Validate 353M | 1h | 6h | <1 |
| 1: Train 50M | 100h | 600h | 20 |
| 2: Train 130M | 250h | 1,500h | 50 |

**Recommendation**: Fase 0 viable on Kaggle. Fase 1 marginal (5 months).
For 130M training, use paid GPU (Lambda/RunPod, ~$500).

### Training notebook structure:
```
Cell 1: Install mamba-ssm, causal-conv1d
Cell 2: Load/prepare corpus (from Kaggle Dataset)
Cell 3: Define MambaByte config (vocab=256, d_model=512/768)
Cell 4: Training loop (AdamW, cosine LR, gradient clipping)
Cell 5: Eval BPB on enwik8 every 1000 steps
Cell 6: Save checkpoint to /kaggle/working/
Cell 7: Export to SafeTensors
```

## 6. Monitoring & Automation

### Via Kaggle API (from Claude Code):
```bash
kaggle kernels push -p notebook/          # Submit notebook
kaggle kernels status user/notebook-name  # Check status
kaggle kernels output user/notebook-name  # Download results
```

### Checkpointing strategy:
- Save every 1000 steps to `/kaggle/working/`
- Download after each session
- Resume from last checkpoint in next session
- Keep `best_bpb` checkpoint separately

---

## Sources

- Kaggle Docs: https://www.kaggle.com/docs/notebooks
- Kaggle GPU Guide (2026): https://aicreditmart.com/ai-credits-providers/kaggle-free-gpu-tpu-30-hours-week-access-guide-2026/
- mamba-ssm: https://github.com/state-spaces/mamba
- MambaByte: arXiv:2401.13660
