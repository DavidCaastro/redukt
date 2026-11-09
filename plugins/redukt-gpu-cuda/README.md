# Redukt GPU CUDA Plugin (R-090, R-096)

CUDA implementation of the Mamba-1 SSM forward pass for NVIDIA GPUs, built as
`libredukt_gpu.so`. It exports:

- **BackendVTable FFI v1**: `redukt_backend_{init,forward,reset_state,free}`, unchanged.
- **FFI v2 batch API** (added alongside v1): `redukt_backend_{abi_version,batch_init,batch_reset,forward_batch}`.
  The full contract is in [docs/api/PLUGIN-FFI-V2.md](../../docs/api/PLUGIN-FFI-V2.md).

**Status:** the plugin compiles with zero warnings, but it has not yet been run on a GPU
(validation is pending on Kaggle T4).

## Architecture

```
SafeTensors -> scripts/convert-weights-gpu.py -> model.rbin (v2) [+ tokenizer.json for BPE]
mamba_cuda.cu -> nvcc -> libredukt_gpu.so -> BackendVTable v1 (+ v2 batch)
                                          -> redukt --backbone plugin --weights <dir>
```

v1 runs as the batched pipeline with `n_streams = 1` on its own state pool. As a result, a
stream's logits are bit-identical whether it runs alone in v1 or inside any v2 batch.

## Kernels (all batched; state indexed by slot)

| Kernel | Purpose | Launch |
|---|---|---|
| `k_gemm_rows` | `Y[b,r] = W[r,:]·X[b,:]`. The weight row goes to dynamic shared memory once and is reused for all streams. Fixed reduction tree, no atomics. | M blocks × 256 |
| `k_rms_norm` / `k_layer_norm` | Norm, out of place | n blocks × 256 |
| `k_embed` | Embedding gather per stream | (⌈d/256⌉, n) |
| `k_conv1d` | Conv1d shift register + bias + SiLU | (⌈di/256⌉, n) |
| `k_softplus_bias` | `dt = softplus(dt + b)` | (⌈di/256⌉, n) |
| `k_ssm_step` | Selective scan step (sequential over d_state) | (⌈di/256⌉, n) |
| `k_gated_silu` | `y · silu(z)` | (⌈di/256⌉, n) |
| `k_residual_add` | `x += out` | ⌈n·d/256⌉ |
| `k_zero_slots` | Reset the listed slots | (⌈stride/256⌉, n) |

## .rbin Weight Format

The header is 44 bytes, all little-endian `uint32`. The plugin and the Rust host both accept v1 and v2.

```
  [0:4]   magic    = 0x55504752 ("RGPU")
  [4:8]   version  = 2 (1 = legacy)
  [8:12]  vocab_size
  [12:16] d_model     [16:20] d_inner   [20:24] d_state
  [24:28] d_conv      [28:32] dt_rank   [32:36] n_layer
  [36:40] flags    (bit 0: norm_bias/LayerNorm, bit 1: untied_head)
  [40:44] v2: tokenizer (0 = byte identity, vocab 256; 1 = GPT-NeoX BPE, mamba-130m)
          v1: reserved = 0 -> tokenizer implied by vocab (256 = byte, else BPE)
```

The body is a sequence of f32 arrays, unchanged from v1. In order:

- `embeddings`
- then, for each layer:
  - `norm_w`
  - `[norm_b]`
  - `in_proj_w`
  - `conv1d_w`
  - `conv1d_b`
  - `x_proj_w`
  - `dt_proj_w`
  - `dt_proj_b`
  - `A_log`
  - `D`
  - `out_proj_w`
- `norm_f_w`
- `[norm_f_b]`
- `[lm_head_w]`

## Build (no GPU needed)

```bash
plugins/redukt-gpu-cuda/build.sh     # local nvcc, or nvidia/cuda:12.4.1-devel-ubuntu22.04 via Docker
# -> libredukt_gpu.so with sm_75 (T4) and sm_89 (L4) SASS, plus compute_89 PTX
#    build.log: ptxas registers and smem per kernel; the build fails on any warning
```

## Convert Weights

```bash
python scripts/convert-weights-gpu.py \
    --input weights/mamba-130m/model.safetensors \
    --output weights/mamba-130m/model.rbin          # v2, tokenizer=auto (bpe for vocab 50280)
```

## Use with Redukt

```bash
cp plugins/redukt-gpu-cuda/libredukt_gpu.so target/release/   # next to the real binary
./target/release/redukt compress   --backbone plugin --weights weights/mamba-130m \
    --input f --output f.azth
./target/release/redukt decompress --weights weights/mamba-130m --input f.azth --output f.out
```

How the tokenizer is chosen:

- **Vocab 256:** byte-level.
- **Vocab 50280:** the BPE tokenizer, loaded from the `tokenizer.json` next to `model.rbin`
  (falling back to `weights/mamba-130m`).

## Test (Kaggle T4)

See the header of `notebooks/redukt-gpu-cuda-test.py`. The notebook checks the following:

- ABI version
- v1 against a NumPy reference
- v1 against batch, bit-exact
- CLI roundtrip and BPB against the 0.9609 CPU reference
- tok/s for v1 and for batch sizes N = 1, 8, 32, 128

## License

Apache-2.0. See repository LICENSE.
