# Redukt GPU CUDA Plugin (R-090)

CUDA implementation of the Mamba SSM forward pass for NVIDIA GPUs.
Exports the BackendVTable FFI v1 contract as `libredukt_gpu.so`.

## Architecture

```
SafeTensors → convert-weights-gpu.py → model.rbin (flat binary)
                                           ↓
mamba_cuda.cu → nvcc → libredukt_gpu.so → BackendVTable FFI
                                           ↓
                                    Redukt (--backbone plugin)
```

## CUDA Kernels

| Kernel | Purpose | Launch config |
|---|---|---|
| `k_gemv` | Matrix-vector multiply y = Ax | M blocks × 256 threads |
| `k_gemv_bias` | GEMV + bias: y = Ax + b | M blocks × 256 threads |
| `k_rms_norm` | RMSNorm (in-place) | 1 block × 256 threads |
| `k_layer_norm` | LayerNorm with bias | 1 block × 256 threads |
| `k_silu` | SiLU activation (in-place) | ceil(N/256) blocks |
| `k_conv1d_shift` | Conv1d shift register + SiLU | ceil(d_inner/256) blocks |
| `k_softplus_bias` | Softplus(x + bias) for dt | ceil(d_inner/256) blocks |
| `k_ssm_recurrence` | Selective scan (d_state generic) | ceil(d_inner/256) blocks |
| `k_gated_silu` | out = y * silu(z) | ceil(d_inner/256) blocks |
| `k_emb_lookup` | Embedding row copy | ceil(d_model/256) blocks |
| `k_residual_add` | x += residual | ceil(d_model/256) blocks |

## .rbin Weight Format

Flat binary for trivial GPU loading (no JSON, no parsing):

```
Header (44 bytes, all LE uint32):
  [0:4]   magic    = 0x55504752 ("RGPU")
  [4:8]   version  = 1
  [8:12]  vocab_size
  [12:16] d_model
  [16:20] d_inner
  [20:24] d_state
  [24:28] d_conv
  [28:32] dt_rank
  [32:36] n_layer
  [36:40] flags    (bit 0: norm_bias, bit 1: untied_head)
  [40:44] reserved = 0

Body (sequential f32 arrays):
  embeddings[vocab_size * d_model]
  per layer i:
    norm_w[d_model]
    [norm_b[d_model]]          if flag bit 0
    in_proj_w[2*d_inner * d_model]
    conv1d_w[d_inner * d_conv]
    conv1d_b[d_inner]
    x_proj_w[(dt_rank+2*d_state) * d_inner]
    dt_proj_w[d_inner * dt_rank]
    dt_proj_b[d_inner]
    a_log[d_inner * d_state]
    d_vec[d_inner]
    out_proj_w[d_model * d_inner]
  norm_f_w[d_model]
  [norm_f_b[d_model]]          if flag bit 0
  [lm_head_w[vocab_size * d_model]]  if flag bit 1
```

## Build (Linux / Kaggle T4)

```bash
cd plugins/redukt-gpu-cuda
make              # → libredukt_gpu.so
make info         # show CUDA version and GPU
```

Requirements: `nvcc` (CUDA Toolkit), NVIDIA GPU with sm_75+ (T4, A100, etc.)

## Convert Weights

```bash
python scripts/convert-weights-gpu.py \
    --input weights/mamba-130m/model.safetensors \
    --output weights/mamba-130m/model.rbin
```

## Test

```bash
python notebooks/redukt-gpu-cuda-test.py \
    --weights weights/mamba-130m \
    --benchmark
```

## Use with Redukt

```bash
# Place libredukt_gpu.so next to redukt binary
cp plugins/redukt-gpu-cuda/libredukt_gpu.so ./

# Compress with GPU backend
./redukt compress --input file.txt --output file.azth --backbone plugin
```

## Performance Projections (Mamba-130M, T4)

| Operation | CPU (i5-1235U) | GPU (T4) | Speedup |
|---|---|---|---|
| GEMV bandwidth | 12.8 GB/s | ~190 GB/s | 15x |
| Per-token latency | ~30 ms | ~2 ms | 15x |
| Throughput | ~34 tok/s | ~500 tok/s | 15x |

## License

Apache-2.0. See repository LICENSE.
