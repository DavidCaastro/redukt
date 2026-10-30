# Backbone Scalability Analysis — Redukt

> Investigación estratégica: adaptación RWKV vs reemplazo de backbone.
> Fecha: 2026-10-30. Metodología: 4 agentes de investigación paralelos + web recursiva.

---

## 1. El Problema

### Throughput actual
- ~1.1 KB/s agregado (3 hilos, i5-1235U) [MEASURED 2026-10-29]
- 94.8 tok/s single-thread (Q8, AVX2/VNNI) [MEASURED]

### Escala de producción
| Volumen | Tiempo 1 core | Cores para 24h | Cores para 1 semana |
|---|---|---|---|
| 1 TB | ~30 años | ~10,800 | ~1,540 |
| 10 TB | ~300 años | ~108,000 | ~15,400 |

### Root cause: recurrencia matricial RWKV-7

```
S_t = diag(w_t) · S_{t-1} + outer(S_{t-1} @ (-kk_t), kk_t · a_t) + outer(v_t, k_mod_t)
```

- Estado: matriz (64x64) por head → O(D²) por token
- State update = 62% del forward pass [MEASURED]
- DRAM saturada al 60% de bandwidth a 94.8 tok/s [MEASURED]
- El término rank-1 `outer(S @ (-kk), kk·a)` (delta rule) es lo que da poder expresivo
  Y lo que destruye la paralelizabilidad eficiente

### Clase de complejidad
- Comprimir N bytes: O(N · D²) — **polinomial**, NO es NP
- Recurrencia no lineal → probablemente **P-completa** (no paralelizable intra-archivo)
- Inter-archivo: embarazosamente paralelo
- Constante multiplicativa: ~3,000x vs brotli (~590K FLOPs/byte vs ~200)

---

## 2. Opción 1 — Adaptar RWKV

### 2.1 Parallel scan para RWKV-7

La recurrencia ES lineal en S: `S_t = M_t · S_{t-1} + B_t` donde
`M_t = diag(w_t) + outer(-kk_t, kk_t · a_t)` es diagonal+rank-1 data-dependent.

Parallel scan teóricamente aplicable (asociatividad de multiplicación matricial).
**Pero la composición requiere O(n³) por par** vs O(n²) secuencial por paso:
- Parallel scan total work: O(T · n³ · log T)
- Secuencial total work: O(T · n²)
- Para n=64, T=10000: scan es ~640x MÁS trabajo total

**Veredicto**: parallelizable en teoría, SLOWER en práctica. [SOURCED: Blelloch 1990]

### 2.2 Chunkwise parallel inference

Establecido para training/prefill (FLA, chunkRWKV6). NO aplica a decode autoregresivo
porque cada token depende del estado del anterior.
[SOURCED: arXiv:2406.06484 (ICLR 2025), github.com/00ffcc/chunkRWKV6]

### 2.3 ParaRNN (Apple, ICLR 2026)

Newton iteration + parallel scan para RNNs no lineales. 100-1000x speedup para TRAINING.
**Paper dice explícitamente**: "decode one step at a time" — no ayuda a inference.
[SOURCED: arXiv:2510.21450]

### 2.4 Speculative state branching

Pre-computar forward passes para top-K tokens más probables.
Cuando el token real coincide, se ahorra el forward.
- Speedup teórico: 2-3x (limitado por acceptance rate)
- **Sin implementación existente** para compresión con RWKV
- Engineering complexity: muy alta
- [SOURCED: concepto derivado de speculative decoding + EDPC (arXiv:2507.18969)]

### 2.5 Técnicas universales (aplican a cualquier backbone)

| Técnica | Ganancia | Coste | Riesgo BPB |
|---|---|---|---|
| BPE tokenizer | 3-5x throughput | Alto (entrenar vocab) | Desconocido |
| L3TC outlier-aware tokenizer | 2.2x throughput | Medio | Bajo |
| Expanded CM (19→50+ modelos) | -3% a -8% BPB | Medio | Positivo |
| Pipeline model↔coder (EDPC) | 1.1-1.5x | Medio | Cero |

[SOURCED: L3TC — AAAI 2026; Nacrith — arXiv:2602.19626; EDPC — ACM MM 2025]

### 2.6 Techo de Opción 1

Con TODAS las técnicas combinadas: ~5-15 KB/s en CPU.
Mejora 5-15x pero NO resuelve el problema estructural O(D²).

---

## 3. Opción 2 — Nuevo Backbone

### 3.1 Comparativa de arquitecturas

| Arquitectura | Tipo | Coste/token | Estado | BPB conocido | CPU impl | Relevancia |
|---|---|---|---|---|---|---|
| **RWKV-7** (actual) | Matrix RNN | O(D²) | Matriz (D²) | 1.28 CE [MEASURED] | Rust (Redukt) | Baseline |
| **Mamba** | Selective SSM | O(D·N), N=16 | Vector (D·N) | 0.93 PG19 (353M) | C, Rust | **ALTA** |
| **Griffin/Hawk** | Diagonal RNN | **O(D)** | Vector (D) | No publicado | Parcial Rust | **ALTA** |
| **xLSTM (mLSTM)** | Matrix LSTM | O(D²) | Matriz (D²) | No publicado | Tesis AVX2 | BAJA |
| **BASED** | Linear attn+window | O(D·F+W·D) | O(D·F) | No publicado | Ninguna | BAJA |
| **Hyena** | Long conv+gate | O(D) recurrent | Vector (D) | No publicado | Ninguna | BAJA |

### 3.2 Mamba — candidato principal

**Evidencia cuantitativa**:
- StateSMix (Mamba 120K params, CPU, C+AVX2): **1,100 tok/s** — 11.6x Redukt
  [SOURCED: arXiv:2605.02904]
- MambaByte (353M, byte-level, 8K context): 0.93 BPB en PG19
  [SOURCED: arXiv:2401.13660]
- `mamba-rs`: Rust crate, SafeTensors, f32/bf16/f16, deterministic GEMMs
  [SOURCED: github.com/silvermpx/mamba-rs]
- Differential Mamba 127M en enwik8: 2.321 BPB (byte-level, sin CM)
  [SOURCED: arXiv:2507.06204]

**Por qué Mamba resuelve el bottleneck**:
- Estado diagonal O(D·N) con N=16 vs RWKV's O(D²=4096) → ~256x menos FLOPs/step
- Composición scan: O(D·N) elementwise vs O(N³) matricial
- SIMD trivial: operaciones elementwise (multiply, add) vs matmul
- Selective scan mantiene poder expresivo: parámetros input-dependent

**Zero-dep feasibility**:
- `mamba-rs` es crate con dependencias (safetensors, serde, etc.)
- Redukt NO puede usar crates externos → portar lógica de inference manualmente
- Reusar parser SafeTensors propio de Redukt (ya existe, zero-dep)
- Las operaciones matemáticas de Mamba son elementwise + matmul → ya las tenemos en tensor.rs

### 3.3 Griffin/Hawk — candidato teórico ideal

- RG-LRU: O(D) estrictamente diagonal, elementwise, máximo SIMD
- Pero: no hay modelos pretrained small (<1B) disponibles públicamente
- RecurrentGemma: 2B y 9B (demasiado grandes para CPU)
- Requeriría entrenar desde cero → alto coste, alto riesgo
- [SOURCED: arXiv:2402.19427]

### 3.4 xLSTM — descartado

- mLSTM tiene O(D²) por outer product update — MISMO bottleneck que RWKV-7
- No resolvería el problema de throughput
- [SOURCED: arXiv:2405.04517]

---

## 4. Dock Architecture (abstracción de backbone)

### 4.1 Coupling actual Redukt → RWKV

| Tipo | Referencias directas | Archivos |
|---|---|---|
| `Rwkv7Model` | 11 | codec(3), api(3), benchmark(2), loader(3) |
| `Rwkv7State` | 11 | codec(7), api(2), benchmark(2) |
| `Scratch` | 2 | codec(2) |
| `WorldTokenizer` | 9 | codec(2), api(3), benchmark(2), loader(2) |
| Traits existentes | **0** | — |

El codec NUNCA inspecciona internals de State o Scratch (opaque pass-through).
`embedding_surgery` es RWKV-específico.

### 4.2 Esfuerzo del dock

| Tarea | Líneas | Riesgo |
|---|---|---|
| Definir traits (`Backbone`, `BackboneState`, `Tokenizer`) | ~50 nuevas | Ninguno |
| `impl Backbone for Rwkv7Model` | ~30 nuevas | Ninguno |
| `impl Tokenizer for WorldTokenizer` | ~10 nuevas | Ninguno |
| Refactorizar `codec.rs` a generics | ~80 cambiadas | Medio |
| Refactorizar `api.rs` a generics | ~30 cambiadas | Bajo |
| Refactorizar `benchmark.rs` a generics | ~20 cambiadas | Bajo |
| Refactorizar `model_loader.rs` | ~20 cambiadas | Bajo |
| **Total** | **~240 líneas** | **Bajo-medio** |

### 4.3 Decisiones clave

- **Generics vs dyn Trait**: generics. `forward_into` se llama ~10K veces/archivo.
  Con un solo implementador, monomorphización es zero-cost.
- **embedding_surgery**: pre-processing concreto ANTES de pasar al trait genérico.
- **perf_summary**: método opcional en trait con default no-op.
- **Feature flags**: `--features mamba` para incluir segundo backend en el binario.

---

## 5. Seguridad — Mamba y Pesos

### 5.1 Estado del ecosistema ML (2025-2026)

- **Pickle RCE**: CVE-2025-1716 a CVE-2025-10157 — múltiples bypass de PickleScan
  [SOURCED: Sonatype, CVSS 9.3-9.8]
- **ShadowPickle (2026)**: evasión stealth de scanners
  [SOURCED: arXiv:2607.17503]
- **SafeTensors Won**: formato sin ejecución de código, default en HuggingFace
  [SOURCED: themlsecopshacker.com/p/safetensors-won]

### 5.2 Redukt ya está blindado

- **Zero pickle**: solo SafeTensors (.safetensors) y GGUF (.gguf)
- **Zero unsafe** en weight loading
- **Parser propio** zero-dep Rust: JSON header + raw tensors
- Bounds checking: `chunks_exact()`, `checked_add()`, file cap 2GB
- GGUF anti-DoS: 6 constantes límite (MAX_TENSOR_COUNT, MAX_NUMEL, etc.)
- Tokenizer: parse manual de syntax literals, no `eval()`

### 5.3 Mamba añade CERO riesgo nuevo

- `mamba-rs` usa SafeTensors — mismo formato
- Redukt portará la lógica de inference, NO el crate (zero-dep constraint)
- Reutiliza el parser SafeTensors existente de Redukt
- Solo cambian los nombres de tensores (mixer.A_log, mixer.D, etc.)
- SHA-256 checksums en `weights/MANIFEST.md` para integridad

---

## 6. Contexto Competitivo (SOTA 2025-2026)

| Sistema | BPB enwik8 | Throughput | Hardware | Backbone |
|---|---|---|---|---|
| **Redukt** | 1.33 [MEASURED] | 1.1 KB/s [MEASURED] | CPU i5 | RWKV-7 |
| StateSMix | 2.13 [SOURCED] | ~1.1 MB/s [SOURCED] | CPU (AVX2) | Mamba 120K |
| Nacrith | 0.94 [SOURCED] | ~2-3 KB/s [SOURCED] | GTX 1050 Ti | Transformer 135M |
| L3TC | — | 842 KB/s decode [SOURCED] | GPU | RWKV + tokenizer |
| ts_zip | 1.11 [SOURCED] | ~1 MB/s [SOURCED] | RTX 4090 | RWKV-169M |
| cmix v21 | 1.17 [SOURCED] | 0.5-5 KB/s [SOURCED] | CPU+GPU | 2000 CM + LSTM |
| NNCP v3 | — | ~0.1 KB/s [SOURCED] | GPU | Transformer-XL |
| Best 2026 | — | 181 KB/s [SOURCED] | A100 | Transformer |

[SOURCED: arXiv:2602.19626 (Nacrith), arXiv:2605.02904 (StateSMix), AAAI 2026 (L3TC),
bellard.org (ts_zip, NNCP), mattmahoney.net (cmix), arXiv:2610.17586 (High-Throughput)]

### Hallazgos clave de la industria

1. **Nadie ha resuelto el throughput** — el mejor reportado es 181 KB/s en A100
2. **Neural compression es ~2600x más lento que Zstd** [SOURCED: arXiv:2603.25526]
3. **Determinismo es ventaja competitiva** — paper de feasibility lo identifica como blocker
4. **Cold storage es el beachhead** — 60% de datos empresariales nacen "cold"
5. **CPU-only es nicho legítimo** — deployment sin GPU, data sovereignty, edge

---

## 7. Estrategia Recomendada

### Orden: Fase A → B → C → D

**Fase A — Dock (R-034)**: ~240 líneas, 1-2 días.
RWKV funciona idéntico, pero el sistema acepta cualquier backbone.

**Fase B — Mamba (R-035)**: 1-2 semanas.
Portar inference Mamba a zero-dep Rust. Benchmark comparativo 19 archivos.

**Fase C — Técnicas universales (R-036, R-037)**: 1 semana c/u, paralelas con B.
BPE tokenizer + expanded CM. Benefician a AMBOS backbones.

**Fase D — Decisión y producción**:
- Mamba BPB ±5% de RWKV → Mamba default (throughput domina)
- Mamba BPB >5% peor → flag `--backend rwkv|mamba` (usuario elige calidad vs velocidad)

### Target

| Métrica | Actual | Target Fase 4 | Factor |
|---|---|---|---|
| Single-thread | 1.1 KB/s | >10 KB/s | 10x |
| Aggregate (3T) | 3.3 KB/s | >50 KB/s | 15x |
| 1 TB @ 50 cores | 30 años | ~6 días | ~1800x |
| BPB enwik8 | 1.33 | ≤1.40 (±5%) | — |

---

## Fuentes

### Papers
- [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456) — Peng et al., 2025
- [Parallelizing Linear Transformers with Delta Rule](https://arxiv.org/abs/2406.06484) — ICLR 2025
- [ParaRNN](https://arxiv.org/abs/2510.21450) — Apple, ICLR 2026
- [GateLoop](https://arxiv.org/abs/2311.01927) — Katsch, 2024
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023
- [MambaByte](https://arxiv.org/abs/2401.13660) — Wang et al., CoLM 2024
- [StateSMix](https://arxiv.org/abs/2605.02904) — SSM for lossless compression, 2026
- [xLSTM](https://arxiv.org/abs/2405.04517) — Hochreiter et al., NeurIPS 2024
- [Griffin/Hawk](https://arxiv.org/abs/2402.19427) — Google DeepMind, 2024
- [Nacrith](https://arxiv.org/abs/2602.19626) — Tacconelli, 2026
- [L3TC](https://ojs.aaai.org/index.php/AAAI/article/view/33446/35601) — AAAI 2026
- [EDPC](https://arxiv.org/abs/2507.18969) — ACM Multimedia 2025
- [High-Throughput LLM Compression](https://arxiv.org/abs/2610.17586) — October 2026
- [Hybrid-Neural Archival Feasibility](https://arxiv.org/abs/2603.25526) — March 2026
- [ShadowPickle](https://arxiv.org/abs/2607.17503) — 2026
- [Differential Mamba](https://arxiv.org/abs/2507.06204) — 2025

### Implementaciones
- [mamba-rs](https://github.com/silvermpx/mamba-rs) — Mamba SSM in Rust
- [chunkRWKV6](https://github.com/00ffcc/chunkRWKV6) — Chunkwise parallel RWKV-6
- [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) — FLA kernels
- [Nacrith-GPU](https://github.com/robtacconelli/Nacrith-GPU) — Neural lossless compression
- [PickleScan CVEs](https://ai-alert.org/posts/hugging-face-security-incidents/) — 2025 security incidents
