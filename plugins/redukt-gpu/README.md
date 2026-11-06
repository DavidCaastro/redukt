# redukt-gpu -- Portable GPU Backend for Redukt

GPU compute backend plugin for the [Redukt](../../README.md) compression engine.
Uses wgpu for portable GPU acceleration (Vulkan/Metal/DX12/OpenGL).

## Status

**SKELETON** -- compiles and exports the 4 FFI functions with placeholder logic.
Actual GPU compute kernels are pending.

## Supported Platforms

| Backend | OS | GPUs |
|---|---|---|
| Vulkan | Linux, Windows, Android | NVIDIA, AMD, Intel, ARM Mali |
| Metal | macOS, iOS | Apple Silicon, Intel |
| DX12 | Windows | NVIDIA, AMD, Intel |
| OpenGL | All (fallback) | Any OpenGL 4.6+ |

## Build

```bash
cd plugins/redukt-gpu
cargo build --release
```

Output: `target/release/redukt_gpu.dll` (Windows) / `libredukt_gpu.so` (Linux) / `libredukt_gpu.dylib` (macOS).

## Install

Copy the shared library next to the `redukt` executable:

```bash
# Windows
cp target/release/redukt_gpu.dll /path/to/redukt/redukt-gpu.dll

# Linux
cp target/release/libredukt_gpu.so /path/to/redukt/libredukt_gpu.so

# macOS
cp target/release/libredukt_gpu.dylib /path/to/redukt/libredukt_gpu.dylib
```

## Usage

Redukt auto-detects the plugin at startup:

```bash
redukt compress --backend plugin input.bin output.azth
```

If no plugin is found, Redukt falls back to CPU automatically.

## FFI Interface (v1)

The plugin exports 4 C functions:

| Symbol | Signature | Purpose |
|---|---|---|
| `redukt_backend_init` | `(path, device, info) -> Result` | Load model, init GPU |
| `redukt_backend_forward` | `(token, logits, vocab) -> Result` | Single-step forward |
| `redukt_backend_reset_state` | `() -> void` | Reset RNN state |
| `redukt_backend_free` | `() -> void` | Release all resources |

## Architecture

```
Redukt (zero-dep Rust) --dlopen--> redukt-gpu.dll (wgpu + safetensors)
     |                                  |
     +-- BackendVTable (4 fn ptrs)      +-- wgpu Device + Queue
     +-- PluginBackend: Backbone        +-- Compute Shaders (WGSL)
     +-- Auto-fallback to CPU           +-- SafeTensors weight loader
```

## License

Apache-2.0 (same as Redukt)
