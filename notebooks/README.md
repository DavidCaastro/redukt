# Redukt Notebooks

## gpu-validation.py

Kaggle kernel script for validating the GPU backend plugin.

### Usage

```bash
# 1. Setup (one-time)
scripts/kaggle-setup.sh

# 2. Push and run on Kaggle GPU
scripts/kaggle-push.sh

# 3. Check status / download results
scripts/kaggle-status.sh --wait
```

### What it does

1. Installs Rust toolchain on Kaggle
2. Clones and builds Redukt + GPU plugin
3. Downloads Mamba-130M weights
4. Runs CPU baseline benchmark (9 files x 10KB)
5. Tests GPU plugin loading
6. Verifies determinism (2 identical runs)
7. Verifies roundtrip (compress -> decompress -> diff)
8. Saves results to `/kaggle/working/`
