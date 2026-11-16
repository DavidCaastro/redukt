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

## Kaggle account clean-up (2026-11-16)

The owner asked to remove the kernels from Kaggle. Checked before deleting: the code of every kernel against this
directory, and every result file (CSV, logs, reports) against `data/benchmark/results/`.

| Kernel | Code on Kaggle vs this repo | Kept from its output | Kaggle |
|---|---|---|---|
| `redukt-gpu-fullsize-bench` | identical | results in `kaggle-fullsize/` (tracked); two Linux archives in `data/test-fixtures/linux-x86_64/` | deleted |
| `redukt-gpu-fullsize-bench-b` | identical | results in `kaggle-fullsize/` (tracked) | deleted |
| `short-message-bench` | identical | results in `short-message-bench/run-1` | deleted |
| `reset-period-sweep` | identical | results in `reset-period-sweep/run-1` | deleted |
| `log-specialist-train` | one line older than the repo (`.contiguous()` fix) | results in `log-specialist/run-1`; the four trained models (`lstm-xs/s/m.pt`, `tf-s.pt`, 22 MB) downloaded to `log-specialist/run-1/models/` (gitignored, laptop only) | deleted |
| `small-backbone-probe` | one line differs (last run defaulted `PROBE_ONLY` to `l3tc`) | results in `small-backbone-probe/` | deleted |
| `redukt-gpu-cuda-test` | one line older (reference bpb before R-108) | results in `kaggle-cuda/` | deleted |
| `mambabyte-50m-validation` | two leading `# %% [code]` markers | log in `mambabyte-50m-v4/`; no output files | deleted |
| `strata-ladder` | identical | integer weights tracked in `weights/strata/` (see `weights/MANIFEST.md`) | **kept**: its output is the only copy of the training checkpoints `ckpt.pt` outside the laptop |

Not kept anywhere else (regenerable from public corpora with these scripts): the working directories of the runs
(repository clones, downloaded corpora, cut messages, the 128 batch archives of `-bench-b`, third-party L3TC
checkpoints).

The raw kernel logs (`*.log`) are gitignored: they exist only on the laptop, next to the tracked CSV and report
files of each run.
