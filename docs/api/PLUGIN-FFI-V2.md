# Plugin FFI v2 — Batched multi-stream forward (R-096)

Status: implemented in `src/domain/backend_ffi.rs`, `src/infrastructure/plugin_backend/`,
`plugins/redukt-gpu-cuda/mamba_cuda.cu`. GPU execution has **not yet been validated** (it's
pending on Kaggle T4 via `notebooks/redukt-gpu-cuda-test.py`).

## 1. Compatibility model

v2 only **adds** to v1. Nothing in v1 changes:

| Item | v1 | v2 |
|---|---|---|
| `redukt_backend_{init,forward,reset_state,free}` | required | unchanged, still required |
| `BackendInfo` layout, `BackendInfo.ffi_version` | `1` | still `1` |
| `redukt_backend_abi_version()` | absent | returns `>= 2` |
| `redukt_backend_batch_{init,reset}`, `redukt_backend_forward_batch` | absent | required if abi >= 2 |

How the host detects v2 (`loader.rs::resolve_v2`): v2 is enabled only if all four v2 symbols
resolve **and** `abi_version() >= 2`. Otherwise `PluginLibrary::vtable_v2 = None`, and the plugin is
used through v1 exactly as before.

- **Old host + new plugin:** the old host never looks up the new symbols, and `ffi_version` is still 1, so it works.
- **New host + old plugin:** the symbols are missing, so the host uses v1 only.

`BACKEND_FFI_VERSION` is deliberately **not** bumped to 2, even though ROADMAP R-096 proposes
it. Bumping it would make every existing host reject every new plugin
(`info.ffi_version != BACKEND_FFI_VERSION`).

## 2. C signatures

```c
/* Result codes: same values as BackendResult; v2 adds 5. v2 functions return int32_t. */
enum { OK = 0, INIT_FAILED = 1, FORWARD_FAILED = 2, VERSION_MISMATCH = 3,
       WEIGHTS_NOT_FOUND = 4, INVALID_ARGUMENT = 5 };

uint32_t redukt_backend_abi_version(void);                       /* >= 2 */
int32_t  redukt_backend_batch_init(uint32_t max_streams);        /* 0 = release pool */
int32_t  redukt_backend_batch_reset(const uint32_t* slots,       /* NULL = all slots */
                                    uint32_t n_slots);
int32_t  redukt_backend_forward_batch(uint32_t n_streams,
                                      const uint32_t* slots,     /* [n_streams] */
                                      const uint32_t* tokens,    /* [n_streams] */
                                      float* out_logits,         /* [n_streams * vocab_size] */
                                      uint32_t vocab_size);
```

Rust (`backend_ffi.rs`): `BackendAbiVersionFn`, `BackendBatchInitFn`, `BackendBatchResetFn`,
`BackendForwardBatchFn`, `BackendVTableV2`, `SYM_ABI_VERSION`, `SYM_BATCH_INIT`,
`SYM_BATCH_RESET`, `SYM_FORWARD_BATCH`, `BackendResult::from_code`, `validate_batch_args`.

v2 functions return a plain `int32_t`, not the Rust `#[repr(C)] enum`. If a foreign plugin
returned an unexpected value as an enum, that would be an invalid discriminant (undefined behaviour).
`from_code` maps any unknown code to `ForwardFailed`.

### Design choice: slots instead of `states[]` pointers or a `ctx`

- **No `ctx` parameter.** The v1 plugin is already a process-global singleton (one model per process),
  so the batch pool belongs to that singleton.
- **States are `u32` slot indices.** Each slot is a set of buffers in a device-resident pool that
  `batch_init` allocates. The host never holds device pointers. The plugin can lay the states out
  contiguously, as `[layer][slot][...]`, and gather them by index inside the kernels.
  An index is easy to validate (`< max_streams`, no duplicates). A raw pointer is not.
- **This fits the lockstep multi-stream codec.** Each file gets a slot. At every step the active
  streams pass `(slot, token)` pairs. A stream that has finished is simply left out:
  `n_streams` can shrink and slots don't need to be contiguous.

## 3. Semantics

| Call | Contract |
|---|---|
| `batch_init(N)` | Requires a successful `backend_init`. Frees any previous pool, then allocates `N` slots (`N <= 4096` in the CUDA plugin) with every state set to zero. Device scratch is sized for `N` rows. `N = 0` only frees. Errors: `INIT_FAILED` (no model or out of memory), `INVALID_ARGUMENT` (N too large). |
| `batch_reset(slots, n)` | Sets the conv and SSM state of the listed slots to zero (`slots == NULL` means every slot). Other slots are left untouched. |
| `forward_batch(n, slots, tokens, out, vocab)` | Advances each listed slot by exactly one token and writes the logits of batch position `b` to `out[b*vocab .. (b+1)*vocab)`. Slots not listed are untouched. `n = 0` is a no-op. Errors: `INIT_FAILED` (no pool), `INVALID_ARGUMENT` (`vocab` mismatch, `n > N`, slot `>= N`, duplicate slot, token `>= vocab`, null pointer). On `INVALID_ARGUMENT` nothing is launched and no state changes. |

- **Synchronous.** All logits are in `out` when the call returns.
- **Memory ownership.** The caller owns `slots`, `tokens` and `out`, and needs them only for the
  duration of the call. The plugin owns all device memory (weights, the pool of `N` states and the
  scratch). `backend_free` releases everything, including the batch pool.
- **Threading.** Like v1, calls are single-threaded with no reentrancy. v1 and v2 use separate state
  pools: `backend_forward` and `reset_state` never touch batch slots, and the batch calls never touch the v1 state.
- **Validation in two places.** `PluginBackend::forward_batch` validates the arguments on the host
  (`validate_batch_args`) before crossing the FFI boundary. The plugin validates them again.

## 4. Determinism (expected; not yet measured on GPU)

1. **Same build, same GPU architecture, same input → bit-identical logits run after run.**
   Every reduction uses a fixed tree:
   - Each thread accumulates over a fixed `k` sequence using explicit `fmaf`.
   - Inside each warp, a `__shfl_down_sync` tree (offsets 16, 8, 4, 2, 1).
   - Across warps, a sequential sum in shared memory.

   There are **no atomics**. The CUDA C++ Programming Guide explains the risk under
   "Deterministic Results": the ordering of arithmetic atomic operations depends on hardware
   scheduling, which can vary between runs. Floating-point addition is also not associative.
   (<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#associativity>)
2. **v1 ≡ v2, independent of batch size and slot.** v1 is the same batched pipeline run with
   `n = 1` on a private pool of capacity 1. The reduction order for a given `(stream, row)` does
   not depend on `n`, on the stream's position in the batch, or on its slot. So the logits for a
   stream are bit-identical whether it runs alone in v1 or inside any batch.
   `notebooks/redukt-gpu-cuda-test.py` checks this (`v1_vs_batch8_bit_exact`).
3. **Not guaranteed:**
   - **GPU vs CPU.** The kernels and the Rust CPU path use different reduction orders.
   - **Different SASS targets (sm_75 vs sm_89).** The code generated by `expf`, `logf` and `rsqrtf`
     and by contraction may differ.

   Archives compressed with `--backbone plugin` are tagged backbone 3 in the AZTH header. Decode
   them with the same plugin build on the same GPU architecture (see ROADMAP R-104).

## 5. CUDA implementation (Mamba-1)

The same kernels serve v1 and v2. `blockIdx.y` (or the grid row) selects the stream.

| Step | Kernel | Shape |
|---|---|---|
| embedding gather | `k_embed` | (ceil(d/256), n) |
| RMSNorm / LayerNorm | `k_rms_norm` / `k_layer_norm` | n blocks |
| in_proj, x_proj, dt_proj, out_proj, LM head | `k_gemm_rows` | M blocks. The weight row is loaded **once** into dynamic shared memory and reused for all n streams in tiles of 8 accumulators. |
| conv1d shift + bias + SiLU | `k_conv1d` | (ceil(di/256), n); state indexed by slot |
| softplus(dt + b) | `k_softplus_bias` | (ceil(di/256), n) |
| selective scan step | `k_ssm_step` | (ceil(di/256), n); sequential loop over d_state, state indexed by slot |
| y·SiLU(z), residual | `k_gated_silu`, `k_residual_add` | per element |
| slot reset | `k_zero_slots` | (ceil(stride/256), n) |

- **Execution and synchronization.** Every launch goes to one `cudaStreamNonBlocking` stream, so
  the kernels run in the order they were enqueued. Slots and tokens are copied H2D with
  `cudaMemcpyAsync` from pinned buffers allocated with `cudaHostAlloc`. The logits are copied D2H
  into the caller's buffer, followed by `cudaStreamSynchronize`. `cudaGetLastError` is checked after
  the launches.
- **Shared memory limit.** The dynamic shared memory per block is `K * 4` bytes, with
  `K = max(d_model, d_inner)`. It must fit in 48 KB, the limit without opt-in, so the plugin
  rejects models with `K > ~12000` at init.

### The `.rbin` header (v2) and the tokenizer

The header is still 44 bytes. In v2, word 10 (previously `reserved`) holds the tokenizer:
`0` = byte-level identity (vocab must be 256), `1` = GPT-NeoX BPE (mamba-130m, vocab 50280).
In v1 files the tokenizer is inferred from the vocab size: 256 means byte-level, anything else
means BPE. Both readers accept v1 and v2: the plugin (`validate_header`) and the host
(`plugin_backend::model::parse_rbin_header`).

For BPE, the host loads `tokenizer.json`. It looks first next to the `.rbin`, then in
`weights/mamba-130m`. `convert-weights-gpu.py` copies the tokenizer file to the output directory.

## 6. Host API (Rust)

The lockstep codec consumes this API, but the application layer does not use it yet:

```rust
impl PluginBackend {
    fn batch_abi(&self) -> Option<u32>;
    fn batch_init(&mut self, max_streams: u32) -> Result<(), BackendResult>;
    fn batch_reset(&self, slots: Option<&[u32]>) -> Result<(), BackendResult>;
    fn forward_batch(&mut self, slots: &[u32], tokens: &[u32], out: &mut [f32])
        -> Result<(), BackendResult>;
}
```

All four return `Err(VersionMismatch)` when the plugin is v1-only.

## 7. References (NVIDIA, consulted 2026-11-10)

- **Shared memory and dynamic shared memory**, `__syncthreads`:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory>
- **Warp shuffle functions**:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions>
- **Floating-point computation** (associativity, FMA, deterministic results):
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html>
- **Asynchronous execution** (stream ordering, non-blocking streams, pinned memory):
  <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html>
- **Compute capabilities** (shared memory per block, warp size):
  <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- **Runtime API**:
  - Memory: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>
  - Errors: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html>
- **nvcc** (`--generate-code`, `--ptxas-options=-v`, `--compiler-options`):
  <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>
