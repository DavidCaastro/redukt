# CPU Throughput Reformulation: Hardware-Aware Execution Pipeline

> R-091 to R-097 — Reasignación de responsabilidades por componente hardware.
> Fecha: 2026-11-09. Método: auditoría del código fuente + investigación PIM/NDP previa.

---

## 1. Diagnóstico: Por qué estamos DRAM-bound

### El loop actual (codec.rs, per-token RNN)

```
Para CADA token (×10,000 en archivo 10KB):
  cm.compute_biases()           ~0.2 ms  (L1/L2 cache, KB-scale)
  cm_only_entropy() check       ~0.01 ms (entropy gate R-077, ineficaz a BPE 50K)
  model.forward_into(tok)       ~3.0 ms  (LEE 127 MB de DRAM)   ← 75-80% del tiempo
  cm.combine_into(logits)       ~0.05 ms
  enc.encode_symbol(cdf)        ~0.01 ms
  cm.update(probs, tok)         ~0.03 ms

DRAM total por archivo 10KB: 10,000 × 127 MB = 1.27 TERABYTES
```

### Desglose de DRAM por token (Mamba-130M, Q8)

| Operación | Dimensiones | DRAM (Q8) | % total |
|---|---|---|---|
| in_proj (×24 capas) | 3072 × 768 | 56.6 MB | 44.5% |
| out_proj (×24 capas) | 768 × 1536 | 28.3 MB | 22.3% |
| x_proj + dt_proj (×24) | ~(80+1536) × dims | 4.4 MB | 3.5% |
| conv1d + bias (×24) | d_inner × d_conv | 0.4 MB | 0.3% |
| **LM head** | **50,280 × 768** | **38.6 MB** | **30.4%** |
| Otros (norm, etc.) | — | ~0.2 MB | 0.2% |
| **Total** | | **~127 MB** | **100%** |

### Lo que ya se intentó y fracasó (Fases 10-12)

| Técnica | Resultado | Por qué fracasó |
|---|---|---|
| R-059: Activation sparsity | 1.7-1.9% sparse | SiLU raramente produce zeros |
| R-061: Wanda column pruning | +20.4% BPB (1 canal) | 130M demasiado pequeño |
| R-062: Highway layer skip | +713% BPB (5.5% skip) | Capas no son highways |
| R-064: Mamba Q8 PTQ | +16.6% BPB | 130M demasiado pequeño para PTQ |
| R-065: NTA prefetch | ±0% | 493MB >> 12MB L3, pollution irrelevante |
| R-066: Mixed Q4/Q8 | Bloqueado | Q8 ya inaceptable |
| R-067: L2-tile layout | ±0% | GEMV streaming, tiling solo ayuda GEMM |
| R-068 to R-071: Compiler opts | All ❌ | DRAM-bound, no compute-bound |

**Resultado neto Fases 10-12**: 12 técnicas, 0 viables. DRAM bandwidth = hard ceiling.

---

## 2. La reformulación: qué cambia con MambaByte

### Cambio 1: LM Head 256 vs 50,280

```
Mamba-130M BPE:    50,280 × 768 × 1 byte (Q8) = 38.6 MB/token (30.4% del total)
MambaByte vocab=256: 256 × 768 × 1 byte (Q8) =  0.19 MB/token (0.2% del total)

DRAM por token: 127 MB → ~89 MB (-30%)
```

Este cambio es **automático** al usar MambaByte — no requiere optimización alguna.

### Cambio 2: Entropy Gate funcional

El código de entropy gate (R-077) ya está en codec.rs:

```rust
// codec.rs, líneas ~809-828
let cm_h = cm.cm_only_entropy(&mut ensemble_buf);
if adaptive_skip > 0 && consec_skips < 3 && cm_h < entropy_thr_f {
    gated = true;  // salta model.forward_into()
}
```

Con BPE 50K: entropy CM ~15.5 bits → **nunca dispara** (threshold inalcanzable).
Con MambaByte vocab=256: entropy CM ~3-6 bits → **gate rate ~70%**.

Impacto: solo el 30% de tokens ejecuta backbone. Throughput efectivo: ×3.3.

### Cambio 3: Tamaño de modelo

MambaByte-353M (pre-entrenado): ~353 MB → ~11 tok/s base.
MambaByte-96M (entrenado from-scratch): ~96 MB → ~40 tok/s base.

Con gate + LM head reducido, la velocidad efectiva cambia dramáticamente.

---

## 3. Stack CPU-only: optimizaciones viables

### Tier 1: Zero BPB risk, quirúrgico (< 100 líneas cada uno)

#### R-091: NTA Prefetch para MambaByte weight streaming

**Por qué R-065 fracasó y esto es diferente**: R-065 falló en Mamba-130M con F32 (493 MB).
Con MambaByte Q8 (~89 MB por token) y entropy gate (solo 30% de tokens ejecutan backbone),
el L3 (12 MB) importa de nuevo — el SSM state (48 KB) debe mantenerse caliente.

```
Actual (tensor.rs): _mm_prefetch(ptr, _MM_HINT_T0)  // L1 exclusive
Propuesto:          _mm_prefetch(ptr, _MM_HINT_NTA)  // L2 only, L3 libre para state
```

- **Dependencia**: MambaByte integrado (R-088 o R-086)
- **Riesgo BPB**: 0%
- **Medición**: Controlada A/B, 3+ rounds, REDUKT_NO_PREFETCH=1

#### R-092: L2-tile weight layout para MambaByte

**Por qué R-067 fracasó y esto es diferente**: R-067 falló porque GEMV streaming no se
beneficia de tiling cuando TODO el modelo es DRAM-bound sin pausa. Con entropy gate, los
tokens que SÍ ejecutan backbone están espaciados en el tiempo — los pesos tienen oportunidad
de ser re-leídos desde L2/L3 si el tile cabe. Además, el LM head ahora es 0.19 MB (cabe
completo en L2 1.25 MB).

```
Propuesto: reordenar pesos durante carga a tiles 256×512 Q8 (~128 KB)
Archivos: model_loader.rs (carga), mamba/model.rs (forward loop order)
```

- **Dependencia**: MambaByte integrado
- **Riesgo BPB**: 0% (mismos pesos, diferente layout en memoria)
- **Validación**: A/B throughput + roundtrip byte-identical

### Tier 2: Cambio arquitectónico CPU, zero GPU requerido

#### R-093: Parallel scan para compresión batch (CPU)

**Concepto**: Para COMPRESIÓN (todos los bytes conocidos), procesar N tokens por capa en
vez de 1 token por todas las capas. Pesos de una capa leídos UNA VEZ para N tokens.

```
HOY (RNN per-token):                    PROPUESTO (layer-by-layer batch):
Para cada token (×N):                   Para cada capa (×24):
  Para cada capa (×24):                   Cargar pesos capa (~3.7 MB, UNA VEZ)
    Leer pesos capa de DRAM               Parallel scan N tokens a través de capa

DRAM: N × 89 MB = 890 GB (10KB)        DRAM: 89 MB (UNA VEZ, 10KB)
```

**Constraint del código actual**: El forward pass es RNN puro (mamba/model.rs):
```rust
fn ssm_recurrence_d16(...) {
    for j in 0..d_inner {
        for n in 0..16 {
            ss[j][n] = da * ss[j][n] + dt[j] * conv_out[j] * b[n]; // sequential
        }
    }
}
```

Requiere reformular como parallel associative scan:
- Cada step produce tuple (A_t, B_t⊕x_t) con operador asociativo
- Prefix scan binario en Blelloch (ya existe R-052: `parallel_scan.rs`) (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte)
- El módulo R-052 es genérico y model-agnostic

**Constraint de roundtrip**: Compresión y descompresión DEBEN producir distribuciones
idénticas. Dos opciones:

- **Opción A (recomendada)**: Backbone procesa TODOS los tokens (sin gate a nivel de
  estado). Gate se convierte en gate de MIXING (cuánto pesar CM vs backbone en la
  distribución final). Descompresión en RNN produce resultados matemáticamente idénticos
  al parallel scan. El gate ya no ahorra compute — pero con parallel scan el compute es
  tan barato que no importa.

- **Opción B**: Mantener gate actual (salta forward_into). El parallel scan solo procesa
  tokens marcados como "inciertos". Pero el mixer SGD acopla CM y backbone:
  ```rust
  // mixer.rs — update usa probs MEZCLADAS
  grads[0] = -ngram_bias[correct_token];
  for i in 0..probs.len() { grads[0] += probs[i] * ngram_bias[i]; }
  self.w_ngram -= self.eta * grads[0]; // feedback del backbone
  ```
  El pre-scan CM sería aproximado, no exacto. Requiere validación empírica.

**Estimación de throughput (CPU, compresión, Opción A)**:
- 89 MB / 21.3 GB/s = 4.2 ms para leer todos los pesos una vez
- Compute: 24 capas × N tokens × ~3.7M multiplies
- A ~50 GOPS (AVX2 Q8, 2 P-cores): N=10K → ~18s
- Plus CM+encode secuencial: ~3s
- **Total: ~21s para 10KB → ~476 tok/s** (vs 34 tok/s actual)

**Para descompresión**: Sigue RNN per-token (no conoce bytes futuros).
Con entropy gate: 30% de tokens × 89 MB = 26.7 MB DRAM per "backbone token".
**~64 tok/s** con gate activo.

- **Implementación**: Nuevo método `forward_batch()` en MambaModel + adaptación de codec.rs
  compress path para batch mode
- **Riesgo BPB**: 0% si Opción A (matemáticamente equivalente a RNN)
- **Complejidad**: ALTA — parallel scan del SSM, buffers intermedios, refactor codec

#### R-094: CM pre-scan optimizado

Con el modelo batch (R-093 Opción A), el loop de compresión se reestructura:

```
FASE 1 (batch, bandwidth-efficient):
  Backbone: parallel scan todos los N tokens → logits[0..N]

FASE 2 (secuencial, cache-local):
  Para cada token t:
    cm.compute_biases()
    entropy = cm_only_entropy()
    if entropy < threshold:
      usar distribución CM-only (logits[t] ignorados en mixing)
    else:
      cm.combine_into(logits[t])  // backbone pre-computado
    enc.encode_symbol(cdf)
    cm.update(probs, tok)
```

Esta separación es natural: Fase 1 es bandwidth-intensive (pero O(1) lecturas),
Fase 2 es compute-trivial y 100% cache-local.

- **Dependencia**: R-093
- **Riesgo BPB**: 0% (mismo resultado, diferente orden de ejecución)

### Tier 3: Extensiones para edge/embedded (sin GPU)

#### R-095: HugePages para Linux deployment

```rust
// Después de mmap de pesos:
libc::madvise(ptr, size, libc::MADV_HUGEPAGE);
```

- 89 MB = ~22,000 páginas 4KB → 45 páginas 2MB
- Elimina TLB miss overhead (24% medido en nemotron-30b-cpu)
- **Solo Linux** — Windows requiere SeLockMemoryPrivilege (inaceptable para CLI)
- **Riesgo BPB**: 0%
- **Dependencia**: ninguna

---

## 4. Proyecciones cuantitativas por escenario

### Escenario 1: MambaByte-353M pre-entrenado (R-088, $0)

```
Base (353M, RNN, sin gate):           ~11 tok/s
+ LM head 256 (-30% DRAM):           ~14 tok/s   (×1.3)
+ Entropy gate 70% (descompresión):  ~48 tok/s   (×3.3)
+ NTA prefetch (R-091, +15%):        ~55 tok/s   (×1.15)
+ L2 tiling (R-092, +20%):           ~66 tok/s   (×1.20)
COMPRESIÓN + parallel scan (R-093):  ~200-400 tok/s (weight amortization)
```

### Escenario 2: MambaByte-96M entrenado (R-085, $87)

```
Base (96M, RNN, sin gate):            ~40 tok/s
+ LM head 256 (-30% DRAM):           ~52 tok/s   (×1.3)
+ Entropy gate 70% (descompresión):  ~173 tok/s  (×3.3)
+ NTA + L2 tiling (R-091+R-092):     ~237 tok/s  (×1.36)
COMPRESIÓN + parallel scan (R-093):  ~800-1500 tok/s
```

### Escenario 3: Hardware edge/embedded (sin GPU)

| Plataforma | DRAM BW | vs i5-1235U | Proyección con stack |
|---|---|---|---|
| Apple M2 (unified) | 100 GB/s | 4.7x | ~310 tok/s (96M+gate) |
| Apple M4 Pro | 150 GB/s | 7.0x | ~470 tok/s (96M+gate) |
| RPi 5 (LPDDR4x) | 34 GB/s | 1.6x | ~105 tok/s (96M+gate) |
| Snapdragon 8 Gen 3 | 51 GB/s | 2.4x | ~155 tok/s (96M+gate) |
| Intel N100 (edge) | 16 GB/s | 0.75x | ~50 tok/s (96M+gate) |
| AWS Graviton 3 | 51 GB/s | 2.4x | ~155 tok/s (96M+gate) |

### Escenario 4: CPU + GPU (referencia, Fase 17+ existente)

```
GPU T4 (320 GB/s): ~1,500-3,000 tok/s con entropy gate
GPU A100 (2 TB/s): ~5,000-10,000 tok/s teórico
```

---

## 5. Constraint de roundtrip en parallel scan (análisis detallado)

### El problema

El parallel scan para compresión produce resultados matemáticamente idénticos al RNN
**si y solo si** ambos procesan exactamente los mismos tokens en el mismo orden.

El entropy gate actual (R-077) SALTA el `forward_into()`, lo que significa que el backbone
NO actualiza su estado para tokens gateados. Esto crea una divergencia:

```
RNN con gate:     backbone ve tokens [0, 1, _, _, 4, _, 6, 7, ...]  (gaps)
Parallel scan:    backbone ve tokens [0, 1, 2, 3, 4, 5, 6, 7, ...]  (todos)
```

Las distribuciones serán diferentes → arithmetic decoder falla.

### Solución: Opción A (backbone always-on, gate solo en mixing)

1. Backbone procesa TODOS los tokens (parallel scan o RNN, ambos ven todos)
2. CM computa biases independientemente
3. Gate decide mixing weight: `w_backbone = 0.0` (CM-only) o `w_backbone = 1.0` (mixed)
4. Arithmetic encoder usa la distribución resultante
5. Descompresión: RNN procesa TODOS los tokens (idéntico estado), gate en mixing

**Ventaja**: Backbone state es DETERMINISTA independientemente del gate.
**Desventaja**: Descompresión ejecuta backbone para 100% de tokens (no 30%).
**Mitigación**: Con GPU, el 100% backbone es aceptable (320 GB/s). En CPU, la
descompresión será ~14-40 tok/s (sin gate savings) a menos que se use GPU.

### Alternativa: Gate bidireccional estricto

Mantener gate actual (skip forward). Requiere que compress y decompress ejecuten
exactamente la misma secuencia de gate decisions. Esto ya funciona hoy — el gate es
determinista (misma CM, misma historia de tokens). Pero el parallel scan no es compatible
con este modo porque el scan procesa todos los tokens.

**Recomendación**: Opción A para compress (parallel scan, always-on backbone).
Gate actual (skip forward) para decompress (más eficiente en CPU per-token).
**Nuevo flag AZTH v4** para indicar modo de compresión (batch vs streaming).

---

## 6. Relación con optimizaciones PIM/NDP previas

De `docs/internal/research/pim-ndp-bandwidth-research.md`:

| Técnica PIM | Resultado original | Resultado con MambaByte |
|---|---|---|
| R-065 NTA prefetch (±0%) | Fracasó: 493MB >> 12MB L3 | **Revisitable**: 89MB + gate 30% = 26.7 MB effective |
| R-067 L2-tile (±0%) | Fracasó: GEMV streaming continuo | **Revisitable**: gate espacía tokens, LM head 0.19 MB en L2 |
| HugePages | No implementado | **Viable en Linux**: 89MB = 45 hugepages |
| Q4 selective | +12.7% BPB, rechazado | **Revisitable con MambaByte**: diferente modelo, diferente sensibilidad |
| AVX-VNNI repacking | Depende de Q4 | **Depende de Q4 en MambaByte** |

**Clave**: Las técnicas que fracasaron en Fases 10-12 operaban bajo condiciones diferentes
(Mamba-130M F32/Q8, 493 MB, sin gate, LM head 38.6 MB). Con MambaByte, tres condiciones
cambian simultáneamente: modelo más pequeño en DRAM effective, gate activo, LM head
insignificante. Esto justifica re-evaluar NTA y tiling.

---

## 7. Impacto en FFI plugin (GPU path futuro)

El `BackendVTable` actual (backend_ffi.rs) es single-token:

```rust
pub type BackendForwardFn = unsafe extern "C" fn(
    token: u32, out_logits: *mut f32, vocab_size: u32
) -> BackendResult;
```

Para habilitar parallel scan en GPU, se necesitaría FFI v2:

```rust
// Propuesto (R-096):
pub type BackendForwardBatchFn = unsafe extern "C" fn(
    tokens: *const u32,      // array de N tokens
    n_tokens: u32,           // longitud
    out_logits: *mut f32,    // N × vocab_size
    vocab_size: u32,
) -> BackendResult;
```

Esto es **backward-compatible**: el plugin puede exportar solo `forward` (v1) o ambos
`forward` + `forward_batch` (v2). El host detecta disponibilidad por symbol resolution.

- **Dependencia**: R-093 (parallel scan en CPU primero, luego portar a GPU)
- **No bloquea nada CPU**: las optimizaciones CPU son independientes del plugin

---

## 8. Fuentes

1. Código: `src/application/codec.rs` — loop de compresión/descompresión per-token
2. Código: `src/domain/tensor.rs` — matmul AVX2, prefetch `_MM_HINT_T0`
3. Código: `src/infrastructure/mamba/model.rs` — forward pass RNN, SSM recurrence
4. Código: `src/domain/backend_ffi.rs` — FFI vtable single-token
5. Código: `src/domain/mixer.rs` — SGD update acoplado a probs mezcladas
6. Código: `src/domain/cm_ensemble.rs` — CM predict + entropy gate
7. Research: `docs/internal/research/pim-ndp-bandwidth-research.md` — Fases 10-12
8. Research: `docs/internal/research/strata-competitive-landscape.md` — R-087
9. R-052: `src/domain/parallel_scan.rs` — Blelloch scan genérico existente (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte)
10. R-077: entropy gate implementation (codec.rs ~L809-828)
