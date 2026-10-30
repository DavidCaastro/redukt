# Weights Manifest — Redukt

> CRITICAL: This file documents the exact provenance of every weights directory.
> Without this metadata, weights cannot be reliably restored after deletion.
> NEVER delete a weights directory without first verifying this manifest.

## Active Weights

### rwkv7-0.1b (World v2.8) — DEFAULT

- **Status**: ACTIVE (default `--weights` path)
- **Source**: BlinkDL/rwkv-7-world on HuggingFace
- **File**: `RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth`
- **URL**: https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
- **Format**: Originally `.pth` (PyTorch), converted to `.safetensors` locally
- **License**: Apache 2.0
- **Training**: ~1T tokens, World v2.8 text data, ctx 4096
- **Architecture**: D=768, L=12, H=12, head_size=64, vocab=65536
- **Key format**: BlinkDL (`blocks.N.att.xxx`, `emb.weight`, `head.weight`)
- **PTH size**: 382,195,690 bytes
- **SafeTensors size**: 382,105,536 bytes
- **Tokenizer**: `rwkv_vocab_v20230424.txt` (shared across all RWKV-7 models)
- **Baseline BPB**: 1.1680 (enwik8 10KB, `hybrid-eval --hierarchical --match --emb-surgery center0.3`)

#### Restoration procedure

```bash
# 1. Download .pth from HuggingFace
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='BlinkDL/rwkv-7-world',
    filename='RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth')
print(path)
"

# 2. Convert to safetensors
python -c "
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
pth = hf_hub_download(repo_id='BlinkDL/rwkv-7-world',
    filename='RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth')
sd = torch.load(pth, map_location='cpu', weights_only=True)
save_file(sd, 'weights/rwkv7-0.1b/model.safetensors')
"

# 3. Copy tokenizer from any other RWKV-7 directory
cp weights/rwkv7-0.1b-g1d/rwkv_vocab_v20230424.txt weights/rwkv7-0.1b/

# 4. Verify baseline
cargo run --release -- hybrid-eval --hierarchical --match --emb-surgery center0.3 --bytes 10000 data/enwik8
# Expected: 1.1680 BPB
```

### mamba-130m (GPT-NeoX BPE) — R-035 EXPERIMENTAL

- **Status**: ACTIVE (experimental, `--weights weights/mamba-130m`)
- **Source**: state-spaces/mamba-130m-hf on HuggingFace
- **Files**: `model.safetensors` (493 MB), `tokenizer.json` (2.1 MB)
- **License**: Apache 2.0
- **Architecture**: Mamba SSM, d_model=768, d_inner=1536, d_state=16, d_conv=4, n_layer=24, vocab=50280
- **Tokenizer**: GPT-NeoX BPE (tokenizer.json, 50,254 entries, 50,009 merges)
- **Key format**: HuggingFace (`backbone.layers.N.mixer.xxx`, `backbone.embeddings.weight`)
- **Baseline BPB**: 0.9074 (enwik8 10KB, `mamba-eval`)
- **Throughput**: ~15.5 tok/s (scalar, no SIMD optimization)

#### Restoration procedure

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='state-spaces/mamba-130m-hf', filename='model.safetensors')
shutil.copy(path, 'weights/mamba-130m/model.safetensors')
path = hf_hub_download(repo_id='state-spaces/mamba-130m-hf', filename='tokenizer.json')
shutil.copy(path, 'weights/mamba-130m/tokenizer.json')
"
```

---

## Rules

1. **NEVER delete a weights directory without checking this manifest first**
2. **NEVER delete the DEFAULT weights directory** (`rwkv7-0.1b`) — all baselines depend on it
3. **Every new weights directory MUST have an entry here** before use in experiments
4. **Every experiment result MUST specify** which `--weights` path was used
5. **Restoration procedures MUST be tested** before considering a deletion safe
