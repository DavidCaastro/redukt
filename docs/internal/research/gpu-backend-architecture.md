# GPU Backend Architecture — Redukt

> Research document: portable GPU acceleration without breaking zero-dep core.
> Date: 2026-11-06. Status: Architecture defined (R-072 to R-075).

---

## 1. Problem Statement

Fases 9-12 exhaustively proved that CPU throughput is DRAM-bandwidth-limited:
- Mamba-130M: 27-34 tok/s (70-85% of DDR4-2667 theoretical max)
- RWKV-7 0.1B: 94.8 tok/s (60% DRAM saturation)
- PGO, operator fusion, SIMD, quantization, sparsity — ALL exhausted
- Hard ceiling: DDR4 single-channel 21.3 GB/s

GPU HBM bandwidth breaks this ceiling:
- T4: 300 GB/s (14x DDR4)
- P100: 732 GB/s (34x)
- A100: 2,039 GB/s (96x)
- Expected throughput: 600-5,000 tok/s (20-150x CPU)

## 2. Architecture Decision: Plugin Model

### Why NOT embedded GPU support:
- CUDA = NVIDIA-only lock-in
- wgpu/Vulkan crates = breaks zero-dep constraint
- GPU not available everywhere (edge, servers, CI)

### Why plugin model:
- Core binary stays zero-dep, compiles everywhere
- Plugin is optional .dll/.so loaded at runtime
- No GPU? Silent fallback to CPU. Zero user friction.
- Multiple plugins for different hardware (CUDA, Vulkan, Metal, OpenCL)
- Future accelerators (RISC-V, FPGA, NPU) = new plugin, zero core changes

### Alternative approaches considered:
| Approach | Pros | Cons | Decision |
|---|---|---|---|
| Embedded CUDA | Best perf on NVIDIA | NVIDIA-only, breaks zero-dep | REJECTED |
| Embedded wgpu | Portable GPU | Adds ~50 crate deps | REJECTED |
| Child process IPC | Zero-dep core | Latency per token (~1ms IPC overhead) | REJECTED |
| **Runtime plugin (dlopen)** | **Zero-dep core, any GPU** | Unsafe FFI boundary | **SELECTED** |
| WebAssembly WASI | Portable, sandboxed | No GPU access from WASI yet | FUTURE |

## 3. FFI Interface Design (R-072)

### Contract: 4 functions, version-checked

```c
// redukt_backend_init: load model + init GPU
BackendResult redukt_backend_init(
    const char* weights_path,  // null-terminated UTF-8
    int device_id,             // GPU index (0 = default)
    BackendInfo* info_out      // filled on success
);

// redukt_backend_forward: single-token RNN step
BackendResult redukt_backend_forward(
    uint32_t token,            // input token ID
    float* out_logits,         // pre-allocated output buffer
    uint32_t vocab_size        // buffer length
);

// redukt_backend_reset_state: clear RNN state (between files)
void redukt_backend_reset_state(void);

// redukt_backend_free: release all resources
void redukt_backend_free(void);
```

### Version protocol:
- Plugin reports `ffi_version` in BackendInfo
- Host checks `ffi_version == BACKEND_FFI_VERSION` (currently 1)
- Mismatch → BackendResult::VersionMismatch → fallback to CPU

### Symbol names:
All prefixed with `redukt_` to avoid collisions with system libraries.

## 4. Plugin Loader (R-073)

### Search strategy:
1. Find directory of current executable (`std::env::current_exe()`)
2. Look for platform-specific filename next to it
3. Load via OS API (LoadLibraryW / dlopen)
4. Resolve 4 symbols by name
5. If any step fails → return None (silent)

### Platform support:
| OS | Library name | Load API | Symbol API |
|---|---|---|---|
| Windows | `redukt-gpu.dll` | `LoadLibraryW` (kernel32) | `GetProcAddress` |
| Linux | `libredukt_gpu.so` | `dlopen` (libc) | `dlsym` |
| macOS | `libredukt_gpu.dylib` | `dlopen` (libc) | `dlsym` |

### Safety:
- `Send + Sync` impl (process-global handle, single-threaded forward calls)
- `Drop` calls `backend_free()` then `FreeLibrary`/`dlclose`
- All function pointers validated non-null before use

## 5. Backbone Integration (R-074)

### Type hierarchy:
```
Backbone trait
├── Rwkv7Model     (CPU, byte-level, 65K vocab)
├── MambaModel     (CPU, BPE, 50K vocab)
└── PluginBackend  (GPU, byte-level, 256 vocab)
    ├── PluginState  (holds reset_fn pointer)
    ├── PluginScratch (empty — GPU manages buffers)
    └── PluginTokenizer (identity: byte → token ID)
```

### DynModel dispatch:
```
DynModel::Rwkv(Rwkv7Model)
DynModel::Mamba(MambaModel)
DynModel::Plugin(PluginBackend)
```

### Auto-fallback:
- `--backend auto` (default): Mamba CPU
- `--backend plugin`: try GPU, panic if not found
- `--backend rwkv|mamba`: force CPU backbone

### AZTH header:
FLAG 0x40 + backbone byte: 0=RWKV, 1=Mamba, **2=Plugin**
Decompression requires matching backbone (same model).

## 6. wgpu Plugin (R-075)

### Status: SKELETON (placeholder logits)

### Planned architecture:
```
redukt-gpu.dll
├── wgpu Device + Queue initialization
├── SafeTensors weight loader (→ GPU buffers)
├── Compute shaders (WGSL):
│   ├── embedding_lookup.wgsl
│   ├── rms_norm.wgsl
│   ├── mat_vec_mul.wgsl  (GEMV, main bottleneck)
│   ├── conv1d_silu.wgsl
│   ├── ssm_recurrence.wgsl (selective scan, d_state=16)
│   ├── gated_silu.wgsl
│   └── softmax_logits.wgsl
├── RNN state management (GPU-side buffers)
└── Synchronization (GPU→CPU logits transfer per token)
```

### Performance model:
Single-token inference is memory-bound (GEMV dominates).
GPU advantage = HBM bandwidth:

| Hardware | BW | tok/s est. | vs CPU | Latency/tok |
|---|---|---|---|---|
| CPU DDR4 | 21 GB/s | 27-34 | 1x | ~30ms |
| T4 | 300 GB/s | 600-800 | ~20x | ~1.5ms |
| P100 | 732 GB/s | 1500-2000 | ~50x | ~0.6ms |
| RTX 4090 | 1008 GB/s | 2000-3000 | ~80x | ~0.4ms |
| A100 | 2039 GB/s | 4000-5000 | ~150x | ~0.2ms |

### Key challenge: GPU→CPU sync
Each token requires logits back on CPU for arithmetic coding.
PCIe latency: ~5-10us per transfer. At 5000 tok/s = 5M us budget → 1us/tok.
Sync overhead: <1% of total. NOT a bottleneck.

## 7. Determinism Requirements

For lossless compression, `compress → decompress` MUST produce identical bytes.
This means the plugin must produce BIT-IDENTICAL logits on the same hardware.

### GPU determinism challenges:
- Floating-point non-associativity (parallel reductions)
- wgpu doesn't guarantee bit-exact across runs by default
- Solution: deterministic reduction algorithms in WGSL shaders
- Verification: roundtrip test on every build

### Cross-platform:
- Same GPU, same driver → deterministic (required)
- Different GPU/driver → NOT guaranteed (acceptable)
- Decompress uses same backbone byte → loads same plugin → same GPU → deterministic

## 8. Throughput Projections

### Single file (10KB):
| Hardware | tok/s | Time | vs CPU |
|---|---|---|---|
| CPU (Mamba) | 30 | 120s | 1x |
| T4 | 700 | 5s | 24x |
| A100 | 4500 | 0.8s | 150x |

### 1 TB cold storage (batch, 8 GPU streams):
| Hardware | Throughput | Time | Cost |
|---|---|---|---|
| CPU x50 | ~1.5 MB/s | ~8 days | $0 (hardware) |
| T4 x1 | ~3 KB/s | ~4 years | N/A |
| A100 x8 | ~1 MB/s | ~12 days | ~$600 |
| A100 x32 | ~4 MB/s | ~3 days | ~$600 |

## 9. Future Extensions

### Plugin variants:
- `redukt-gpu-cuda`: NVIDIA-optimized CUDA kernels (Tri Dao's selective scan)
- `redukt-gpu-metal`: Apple Metal compute shaders
- `redukt-gpu-opencl`: AMD/Intel/FPGA via OpenCL
- `redukt-gpu-npu`: Neural Processing Units (Qualcomm Hexagon, Intel NPU)

### MambaByte integration (R-041):
When MambaByte is trained, the GPU plugin loads byte-level weights.
vocab_size=256, no tokenizer complexity. Identity tokenizer already in place.

### Speculative decoding on GPU (R-053):
CM-draft on CPU, batch-verify on GPU → higher acceptance rate with GPU speed.

---

## Sources

- wgpu: https://wgpu.rs/ (Rust WebGPU implementation)
- WebGPU spec: https://www.w3.org/TR/webgpu/
- Mamba CUDA kernels: https://github.com/state-spaces/mamba (Tri Dao)
- mamba-rs: https://github.com/silvermpx/mamba-rs (Rust inference)
- StateSMix: arXiv:2605.02904 (SSM compression, 1100 tok/s CPU)
- High-Throughput LLM Compression: arXiv:2610.17586 (181 KB/s A100)
