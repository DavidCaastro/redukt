# Strata & Competitive Landscape Research

> R-087 — Investigación de proyectos "Strata" y ecosistema de compresión neural.
> Fecha: 2026-11-09. Método: investigación web recursiva + análisis de código.
>
> **ERRATA 2026-11-10**: superado en la parte competitiva por `docs/internal/research/competitive-position-2026-11-10/REPORT.md`.
> Correcciones: Nacrith tiene fallback CPU y maneja binarios (vía LZMA), no es "solo texto, requiere GPU"; su throughput
> publicado es inconsistente (paper ~20-30 tok/s por worker vs README ~3.5 KB/s); cmix v21 = 1.170 bpb enwik8 a 1.6 KB/s
> (no ~1.25); la estirpe Hutter 2026 (fx2-cmix-transformer, zmix, cmix-neif, ex-girlfriend) 0.73-0.78 bpb enwik9 en CPU
> no estaba cubierta; L3TC 4.35 MB/s corresponde al modelo de 200K (1.404 bpb), el de 3.2M da 1.105 bpb a 2.5 MB/s (A100).

---

## 1. Proyectos "Strata" Evaluados

Se identificaron 8 proyectos distintos con el nombre "Strata". Solo 1 tiene relevancia
directa para Redukt.

### 1.1 stratalab/strata-inference — RELEVANCIA ALTA

- **URL**: https://github.com/stratalab/strata-inference
- **Qué es**: Motor de inferencia Rust puro (zero C/C++ deps) para modelos Transformer GGUF.
- **Licencia**: MIT
- **Lenguaje**: Rust
- **Arquitectura**: Trait `ComputeBackend` con 25+ operaciones, backends CPU/Metal/CUDA.

**Técnicas adoptables para Redukt:**

| Técnica | Descripción | Aplicabilidad |
|---|---|---|
| **PTX-as-string** | Kernels CUDA embebidos como PTX inline, cargados via `cuModuleLoadData` | Directa — elimina dependencia nvcc |
| **Fused dequant** | Dequant Q8/Q4 dentro del kernel GEMV sin materializar f32 | Directa — reduce bandwidth GPU |
| **M=1 fast path** | Kernel GEMV optimizado para decode single-token | Directa — nuestro caso exacto |
| **Dequant cache** | Primera dequant a f32 en VRAM, cache para reuso | Directa — pesos reusados N veces |
| **Command batching** | Múltiples dispatches Metal en un solo command buffer | Relevante si portamos a Metal |

**Limitaciones para Redukt:**
- NO soporta SSM (Mamba/RWKV) — solo Transformers
- CPU backend sin SIMD (0.1-6.9 tok/s vs Redukt 34 tok/s)
- Trait ComputeBackend orientado a attention/KV-cache, no recurrence/scan

**Rendimiento reportado:**
- GPU Metal: 33-365 tok/s decode (competitivo con llama.cpp)
- CPU: 0.1-6.9 tok/s (15-250x más lento que GPU, sin SIMD)

### 1.2 Niko1221/Strata (31K stars) — RELEVANCIA BAJA

- **URL**: https://github.com/Niko1221/Strata
- **Qué es**: Motor MoE 125B en gaming GPUs (C++, basado en llama.cpp/ggml).
- **Técnicas**: Dynamic expert caching, jerarquía VRAM/RAM/SSD, CPU co-processing.
- **Por qué no aplica**: MoE sparse (24K expertos, 10 activos/token) vs modelos densos 130M.
  La jerarquía de memoria no aplica: nuestro modelo cabe completamente en RAM.

### 1.3 RustingStrata — RELEVANCIA MEDIA

- **URL**: https://github.com/RustingStudio/RustingStrata
- **Qué es**: Reimplementación Rust de Niko1221/Strata. Kernels CUDA via NVRTC.
- **Por qué no aplica**: Mismo enfoque MoE/Qwen, no SSM.

### 1.4 Otros (irrelevantes)

| Proyecto | Tipo | Razón de descarte |
|---|---|---|
| Strata OSDI 2026 | KV-cache jerárquico cloud | Cloud-scale serving, no aplica |
| Strata Hybrid Memory | Placement HBM/DRAM proactivo | Requiere hardware HBM |
| StrataFlow | Expert streaming desde disco | Específico para MoE |
| Strata File System (SOSP 2017) | Cross-media FS | File system research |
| NVlabs/STRATA | Simulación atmosférica | Sin relación |

---

## 2. Ecosistema de Compresión Neural (HuggingFace/Kaggle)

### 2.1 MambaByte Pre-entrenado — HALLAZGO CRÍTICO

**Modelos disponibles en HuggingFace (JunxiongWang):**

| Modelo | Params | Datos | BPB enwik8 (CE) |
|---|---|---|---|
| MambaByte_Arxiv | 353M | 30B bytes (arXiv) | 0.396 |
| MambaByte_Books | 353M | 30B bytes (PG19) | ~0.40 |
| MambaByte_Code | 353M | 30B bytes (code) | ~0.40 |
| MambaByte_PG19_972M | 972M | 150B bytes (PG19) | ~0.35 |

- **Paper**: "MambaByte: Token-free Selective State Space Model" (CoLM 2024)
- **GitHub**: https://github.com/jxiw/MambaByte
- **Arquitectura**: Mamba-1 (Selective SSM), vocab=256, byte-level puro.

**Implicación para Redukt:**
- Si los pesos 353M son compatibles con nuestro runtime Mamba, podemos **saltarnos
  el entrenamiento** ($87-145, 3-4 semanas) e ir directo a conversión + integración.
- BPB proyectado con coder efficiency 95%: ~0.42 BPB enwik8 (2.3x mejor que 0.961 actual).
- **Caveat**: son Mamba-1, no Mamba-2. Hay que verificar compatibilidad arquitectónica.
- **Caveat**: 353M params → ~353MB en DRAM → throughput CPU ~11 tok/s (vs 34 actual).

### 2.2 Nacrith — Competidor Directo

- **URL**: https://github.com/robtacconelli/Nacrith-GPU
- **Paper**: arXiv:2602.19626
- **Qué hace**: SmolLM2-135M + context mixing ensemble + arithmetic coder 32-bit.
- **BPB enwik8**: **0.9389** (Redukt: 0.9609, diferencia: 2.3%)
- **Throughput**: 21-70 tok/s en GTX 1050 Ti (con llama.cpp).
- **Limitaciones**: Python + llama.cpp; GPU recomendada (fallback CPU sin velocidad publicada); binarios delegados a
  LZMA/gzip (variante NC06), no modelados. (Corregido 2026-11-10; antes: "solo texto, requiere GPU".)

**Redukt vs Nacrith:**

| Aspecto | Redukt | Nacrith |
|---|---|---|
| BPB enwik8 | 0.9609 | **0.9389** |
| GPU requerida | No | Recomendada (fallback CPU) |
| Tipos de archivo | Cualquiera (modelados) | Texto modelado; binarios vía LZMA (NC06) |
| Dependencias | Zero | PyTorch + llama.cpp |
| Archivos estructurados | **0.216-0.446** | No evaluado |
| Lenguaje | Rust (single binary) | Python |

### 2.3 L3TC (Alipay, AAAI 2025) — RWKV para Compresión

- **URL**: https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression
- **Qué hace**: RWKV 3.2-12M params + outlier-aware tokenizer + arithmetic coding.
- **Throughput**: **4.35 MB/s** decompression, 1.30 MB/s en iPhone 12 ANE.
- **Trade-off**: Prioriza velocidad sobre ratio (modelos 50x más pequeños que Redukt).
- **Valida**: Arquitectura RWKV+AC es viable para compresión.

### 2.4 Otros Compresores Neurales

| Proyecto | BPB enwik8 | Modelo | GPU | Notas |
|---|---|---|---|---|
| ts_zip (Bellard) | ~1.11 | RWKV-4 169M | Sí | Solo texto, predecesor conceptual |
| NNCP v2 (Bellard) | 1.19 | Transformer | Sí | Binario cerrado |
| cmix v21 | 1.170 | CM 2000+ modelos + LSTM | No | C++, 1.6 KB/s, 31 GB RAM (LTCB) |
| FineZip | 1.024 | LLaMA-7B | Sí | 4h por corpus |
| LLMZip | ~0.709* | LLaMA-7B | Sí | *Bound asintótico, 9.5 días |
| DZip | >gzip | Dual RNN | Sí | Entrena online |
| NeuralZip | — | GRU | No | PoC, geometric mixing |

### 2.5 Recursos Adicionales

- **NNLCB Benchmark**: 17 compresores × 28 datasets × 4600h CPU/GPU
  (https://github.com/fahaihi/NNLCB) — útil para posicionar Redukt formalmente.
- **RWKV-Lite**: Compresión del propio modelo RWKV, 3.4-5x reducción. Potencial
  para reducir DRAM traffic de nuestros pesos.

---

## 3. Mapeo: Limitaciones Redukt → Soluciones Encontradas

| Limitación | Solución | Fuente | Impacto |
|---|---|---|---|
| DRAM bandwidth ceiling | GPU offload | strata-inference PTX pattern | 4-10x throughput |
| BPE 50K → CM ineficaz | MambaByte 353M pre-entrenado | HuggingFace JunxiongWang | vocab=256, CM viable |
| LM head 30% FLOPs | MambaByte vocab=256 | MambaByte paper | 196x reducción head |
| Entropy gate bloqueado | MambaByte byte-level | R-077 + MambaByte | 40-70% token skip |
| GPU kernels pendientes | PTX-as-string pattern | strata-inference | Zero nvcc dependency |
| Competitive gap 2.3% | MambaByte → 0.42 BPB | Pre-trained weights | Supera Nacrith 2x |

---

## 4. Recomendaciones (Priorizadas)

### P0: Evaluar pesos MambaByte 353M pre-entrenados (R-088)
- Descargar `JunxiongWang/MambaByte_Arxiv` desde HuggingFace
- Verificar compatibilidad Mamba-1 vs runtime Mamba de Redukt
- Si compatible: convertir con script existente → benchmark inmediato
- **Potencial: 0.42 BPB** sin entrenar nada ($0 vs $87-145)

### P1: Adoptar PTX-as-string para plugin GPU (R-090)
- Escribir kernels SSM GEMV como PTX inline en Rust
- Elimina dependencia de compilador CUDA externo
- Compatible con BackendVTable FFI existente

### P2: Benchmark formal contra Nacrith (R-089)
- Expandir benchmark con datasets compartidos (enwik8 completo, alice29.txt)
- Posicionar Redukt formalmente en SOTA

---

## 5. Fuentes

1. stratalab/strata-inference — https://github.com/stratalab/strata-inference
2. Niko1221/Strata — https://github.com/niko1221/strata
3. RustingStrata — https://github.com/RustingStudio/RustingStrata
4. MambaByte (JunxiongWang) — https://huggingface.co/collections/JunxiongWang/mambabyte
5. MambaByte Paper — https://huggingface.co/papers/2401.13660
6. Nacrith — https://arxiv.org/abs/2602.19626
7. L3TC (AAAI 2025) — https://arxiv.org/abs/2412.16642
8. ts_zip — https://bellard.org/ts_zip/
9. NNCP v2 — https://bellard.org/nncp/
10. cmix-neif-2.0 — https://github.com/marwoelk/cmix-neif-2.0
11. FineZip — https://arxiv.org/abs/2409.17141
12. LLMZip — https://arxiv.org/abs/2306.04050
13. NNLCB — https://github.com/fahaihi/NNLCB
14. RWKV-Lite — https://huggingface.co/papers/2412.10856
15. Strata OSDI 2026 — https://arxiv.org/html/2508.18572v1
