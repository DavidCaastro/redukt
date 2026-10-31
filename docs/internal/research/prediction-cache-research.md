# PredCache — Neural Prediction Cache

> Research: R-048/R-049/R-050
> Date: 2026-10-31
> Status: DISEÑO VALIDADO, pendiente implementación
> Origin: User insight — "almacenar recurrencias en tabla hash para trasladar
>         conocimiento del backbone a los CM, iteración tras iteración"

---

## 1. Problema

El backbone neural (Mamba-130M) ejecuta ~140M FLOPs por token (~29ms):
24 layers × (in_proj + conv1d + SSM recurrence + gate + out_proj) + LM head.
Esto es el **99.5% del tiempo de compresión**. CM + coder cuestan ~0.1ms.

El throughput está limitado a ~34 tok/s (SIMD) = ~97 bytes/s.
Brotli-11 opera a ~50 MB/s — una diferencia de **500,000x**.

No existe optimización de software que cierre esta brecha. El cuello de
botella es **inherente**: inferencia neural secuencial por token.

## 2. Insight Core

El usuario identificó que el problema no es QUÉ (necesitar backbone) sino
CUÁNDO (ejecutarlo en cada token). Si almacenamos las predicciones del
backbone indexadas por contexto, los tokens con contextos repetidos pueden
reusar predicciones anteriores sin ejecutar el forward pass.

**Formalización**: El backbone es una función `f(state, token) → logits`.
El state codifica todo el historial previo, pero para datos estructurados
(XML, logs), muchas secuencias de contexto producen distribuciones similares.
Un cache de contexto captura esta redundancia.

Con el tiempo, el cache absorbe el conocimiento del backbone. El sistema
se vuelve progresivamente más rápido — exactamente lo contrario del
comportamiento actual donde cada token cuesta lo mismo.

## 3. Modelo Matemático

### 3.1 Costo actual

```
T_compress = N × T_backbone + N × T_cm + N × T_coder
           ≈ N × 29ms
```

### 3.2 Costo con PredCache

```
T_compress = N_miss × T_backbone + N × T_lookup + N × T_cm + N × T_coder
           = N × (hit_rate × T_lookup + (1 - hit_rate) × T_backbone + T_cm + T_coder)

Speedup = T_backbone / (hit_rate × T_lookup + (1 - hit_rate) × T_backbone)
```

Para T_backbone = 29ms, T_lookup = 0.01ms:

| Hit rate | Speedup | Costo/token |
|----------|---------|-------------|
| 0% | 1.0x | 29ms |
| 20% | 1.25x | 23.2ms |
| 50% | 2.0x | 14.5ms |
| 70% | 3.3x | 8.7ms |
| 90% | 9.7x | 3.0ms |

### 3.3 Impacto en BPB

Cuando se usa cache en vez de backbone, la distribución puede diferir
de la que produciría el backbone con estado actualizado. El costo en BPB
está acotado por la divergencia KL:

```
delta_BPB ≈ Σ_hits KL(P_real || P_cached) / (N × ln(2))
```

Para contextos vistos múltiples veces con distribuciones estables,
KL → 0. La pérdida se concentra en:
1. State divergence (backbone state no actualizado durante skips)
2. Cuantización top-K (tokens fuera del top-K pierden precisión)

### 3.4 Hit rate estimado por tipo de datos

| Tipo de datos | Hit rate (order=8) | Hit rate (order=4) |
|---------------|-------------------|-------------------|
| Server logs (apache, syslog) | 50-70% | 70-90% |
| XML estructurado (facturas) | 30-50% | 50-70% |
| Texto natural (enwik8) | 5-15% | 15-30% |
| Binario (ejecutables) | 1-5% | 5-10% |

**Breakeven**: Hit rate > 2% ya justifica el overhead del lookup.

## 4. Diferencia vs Adaptive Skip (R-033)

Ya existe `V4_FLAG_ADAPTIVE_SKIP` que salta backbone cuando el modelo
fue muy confiado en el token actual (`probs_buf[tok] > threshold`).

| Aspecto | Adaptive Skip (R-033) | PredCache (R-048) |
|---------|----------------------|-------------------|
| Trigger | Confianza alta en token actual | Contexto visto antes |
| Cache | Ninguno — reusa logits previos | Hash table persistente |
| Alcance | Solo tokens consecutivos confiados | Cualquier contexto repetido |
| Aprendizaje | No aprende | Acumula conocimiento del backbone |
| Escala | Limitado por MAX_CONSEC_SKIP | Crece con el archivo |
| Speedup típico | 5-15% (pocos tokens seguros) | 25-300%+ (datos estructurados) |

**PredCache subsume adaptive skip**: si el cache tiene una entrada para
el contexto actual Y la confianza del entry es alta, es estrictamente
mejor que reussar los logits del token anterior.

## 5. Arquitectura

### 5.1 Estructura de datos

```rust
const TOP_K: usize = 32;
const TABLE_BITS: usize = 16;  // 64K entries
const TABLE_SIZE: usize = 1 << TABLE_BITS;

pub struct PredCache {
    table: Vec<Option<CacheEntry>>,
    order: usize,          // context window (default: 8 tokens)
    min_obs: u16,          // min observations before trusting (default: 2)
    max_skips: u16,        // max consecutive skips (default: 8)
    enabled: bool,
}

struct CacheEntry {
    tokens: [u16; TOP_K],  // top-K token IDs (by logit)
    logits: [f32; TOP_K],  // corresponding logits
    base_logit: f32,       // default for non-top-K tokens
    hash: u64,             // full hash for collision check
    observations: u16,     // times seen
}
```

**Memoria**:
- Entry: 32×2 + 32×4 + 4 + 8 + 2 = 206 bytes
- Tabla completa: 64K × 206 = **13.2 MB** (worst case)
- Típico (10KB file, ~3500 tokens): <1MB usado
- Backbone model: 493MB → cache es <3% adicional

### 5.2 Context hash

```rust
fn context_hash(tokens: &[u32], pos: usize, order: usize) -> u64 {
    let start = pos.saturating_sub(order);
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for &t in &tokens[start..pos] {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}
```

Costo: ~1ns (vs 29ms backbone). Negligible.

### 5.3 Flujo en el codec

```
PARA CADA TOKEN t (compress y decompress):

  1. Calcular ctx_hash = FNV-1a(tokens[t-order..t])

  2. cache_hit = pred_cache.lookup(ctx_hash)

  3. DECISIÓN (determinista en ambos lados):
     use_cache = cache_hit.is_some()
                 && cache_hit.observations >= MIN_OBS
                 && consecutive_skips < MAX_SKIPS

  4a. SI use_cache:
      → Reconstruir logits desde cache entry (top-K + base)
      → consecutive_skips += 1
      → backbone NO se ejecuta (state congelado)

  4b. SI NO use_cache:
      → backbone.forward_into(tok, state, scratch, logits)  // 29ms
      → pred_cache.insert(ctx_hash, logits)                 // capturar
      → consecutive_skips = 0

  5. CM siempre ejecuta (ngram, match, run, word, segment)    // ~0.01ms
  6. Mixer combina logits + CM biases                         // ~0.001ms
  7. Coder codifica/decodifica con CDF resultante             // ~0.001ms
  8. CM observa token decodificado                            // ~0.01ms
```

### 5.4 Reconstrucción de logits desde top-K

```rust
fn reconstruct_logits(entry: &CacheEntry, out: &mut [f32]) {
    // Llenar todo con base_logit (tokens raros)
    for v in out.iter_mut() { *v = entry.base_logit; }
    // Sobrescribir top-K con logits reales
    for i in 0..TOP_K {
        let tok = entry.tokens[i] as usize;
        if tok < out.len() {
            out[tok] = entry.logits[i];
        }
    }
}
```

Top-K=32 captura >99% de la masa de probabilidad en la mayoría de
distribuciones. Los tokens fuera del top-K reciben un logit base
(promedio de los logits fuera del top-K del momento de inserción).

## 6. Prueba de Determinismo

**Teorema**: El protocolo PredCache es determinista — compressor y
decompressor toman decisiones idénticas de skip.

**Prueba**:
1. PredCache comienza vacío en ambos lados.
2. En cada paso t, ambos lados conocen tokens[0..t] (idénticos).
3. context_hash es función pura de tokens[t-order..t] → mismo hash.
4. Cache lookup es función pura del hash + tabla → mismo resultado.
5. insert() se ejecuta con los mismos logits (mismo backbone, same-tier).
6. observations se incrementa idénticamente.
7. consecutive_skips sigue la misma trayectoria.
8. ∴ use_cache es idéntico en ambos lados para todo t. QED.

**Requisitos** (ya cumplidos por R-013):
- Mismo SIMD tier (SIMD↔SIMD o scalar↔scalar)
- Mismos hiperparámetros (almacenados en header)
- Mismo backbone (almacenado en header via FLAG_BACKBONE)
- PredCache params en header (order, min_obs, max_skips)

## 7. State Divergence Analysis

Cuando el backbone se salta (cache hit), su estado SSM no se actualiza.
El estado queda "congelado" en el último forward pass ejecutado.

### 7.1 Impacto en Mamba

Estado SSM: `h[layer][j][n]` donde j ∈ [0, 1536), n ∈ [0, 16)
Recurrencia: `h_new = da * h_old + dt * x * b`
Donde da = exp(dt * a) < 1 (decay constante).

Si saltamos K tokens, el estado pierde las K contribuciones más recientes.
El error es:
```
||h_real - h_stale|| ≈ Σ_{i=0}^{K-1} da^i × ||dt × x_i × b||
```

Para K pequeño (≤ MAX_SKIPS = 8) y datos estructurados donde los tokens
saltados son predecibles (alta confianza del cache), el error es pequeño.

### 7.2 Mitigación: MAX_SKIPS

MAX_SKIPS limita los skips consecutivos. Después de MAX_SKIPS cache hits,
se fuerza un forward pass del backbone, que:
1. Actualiza el estado con el token actual
2. Re-sincroniza parcialmente el estado (el decay exponencial
   hace que las contribuciones antiguas desaparezcan)

El estado se "autocorrige" gracias al decay. Tras un backbone sync,
el error acumulado decae exponencialmente en los siguientes pasos.

### 7.3 Impacto BPB estimado

| MAX_SKIPS | Divergencia estado | BPB overhead |
|-----------|-------------------|--------------|
| 1 | Mínima | <0.1% |
| 4 | Baja | 0.1-0.5% |
| 8 | Moderada | 0.5-2% |
| 16 | Alta | 2-5% |
| ∞ | Ilimitada | 5-20%+ |

Recomendación: MAX_SKIPS = 8 como default, tunable por content type.

## 8. Escenario de Uso: Facturas XML en Batch

Caso: 100 facturas XML de ~50KB cada una, procesadas en batch.

**Sin PredCache** (hoy):
- 100 × 50KB × 1 token/2.84 bytes × 29ms = 100 × 17,600 × 29ms = 51,040 s ≈ **14.2 horas**

**Con PredCache** (estimado 60% hit rate en facturas):
- Primera factura: ~0% hits (cache frío) → 17,600 × 29ms = 510s
- Facturas 2-10: ~40% hits (cache calentando) → 17,600 × 17.4ms = 306s
- Facturas 11-100: ~70% hits (cache caliente) → 17,600 × 8.7ms = 153s
- Total: 510 + 9×306 + 90×153 = 510 + 2,754 + 13,770 = **17,034s ≈ 4.7 horas**

**Speedup: 3.0x** en escenario batch.

**Nota**: el cache persiste entre archivos en batch mode (model loaded once).
Para archivos del mismo dominio, el cache se calienta progresivamente.

## 9. Integración con Adaptive Skip (R-033)

PredCache y adaptive skip son complementarios:

```
if pred_cache.has_hit(ctx_hash) && entry.obs >= MIN_OBS && consec < MAX_SKIPS:
    // PredCache hit: usa distribución aprendida del backbone
    use_cached_logits()
elif adaptive_skip.active && probs[tok] > skip_threshold && consec < 3:
    // Adaptive skip: reusa logits del paso anterior
    reuse_previous_logits()
else:
    // Full backbone forward
    backbone.forward_into(...)
```

PredCache tiene prioridad porque sus predicciones son contexto-específicas
(no solo "repite lo último"). Adaptive skip es fallback para tokens de
alta confianza donde el cache no tiene entrada.

## 10. Formato AZTH

### Nueva flag: V4_FLAG_PRED_CACHE

No necesaria si usamos los parámetros existentes de adaptive skip.
Alternativamente, extender el header con:

```
FLAG: 0x80 ya ocupado (ENHANCED_CM). Siguiente disponible: necesita
extensión de flags a 16-bit, o reuso de bits existentes.

Opción pragmática: PredCache usa V4_FLAG_ADAPTIVE_SKIP (0x20) con
un campo extendido que indica el tipo de skip:
  - skip_type = 0: classic adaptive skip (R-033)
  - skip_type = 1: pred_cache (R-048)
  - skip_type = 2: both

O mejor: PredCache es siempre activo (determinista, sin flag).
Si el file tiene contextos repetidos → cache hits → speedup.
Si no → 0% hits → sin impacto → sin necesidad de flag.
```

**Decisión recomendada**: PredCache activo siempre. Los parámetros
(order, min_obs, max_skips) hardcoded como constantes. Sin flag adicional.
Si el cache no tiene hits, el overhead es ~1ns/token (hash + failed lookup).

Justificación: el cache se construye de forma idéntica en compress y
decompress sin información adicional en el header.

## 11. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| BPB regression por state divergence | Alta | Bajo (1-3%) | MAX_SKIPS acotado, tunable |
| Hit rate bajo en texto natural | Alta | Ninguno | Cache inactivo = 0 overhead |
| Colisiones hash | Baja | Medio | Full hash verification en CacheEntry |
| Memoria excesiva (files grandes) | Media | Bajo | Table size fijo (64K entries), LRU eviction |
| Cross-tier mismatch | Baja | Alto | Ya mitigado por R-013 (same-tier requirement) |

## 12. Tareas de Implementación

### R-048: PredCache — Estructura de datos
- `src/domain/pred_cache.rs` (nuevo)
- CacheEntry: top-K tokens + logits + base + hash + observations
- context_hash: FNV-1a sobre últimos N tokens
- lookup / insert / reconstruct_logits
- Unit tests: insert/lookup, collision, reconstruct accuracy, eviction
- Sin integración al codec aún (solo data structure)

### R-049: PredCache Integration — Skip Protocol
- Integrar PredCache en compress loop (codec.rs, api.rs, benchmark.rs)
- Integrar en decompress loop (todas las versiones v4)
- Decisión determinista: ambos lados construyen cache idénticamente
- Convivencia con adaptive skip existente
- Roundtrip verification
- A/B benchmark: con/sin PredCache en 12 archivos

### R-050: PredCache Tuning & Scale
- Benchmark: speedup vs BPB tradeoff por tipo de archivo
- Escala: 10KB → 50KB → 100KB → 500KB
- Batch mode: cache persistence entre archivos
- Tuning: order, min_obs, max_skips óptimos por content type
- Memory profiling
- Integración con content_detect.rs para parámetros adaptativos

## 13. Referencias

- PAQ/cmix: context mixing con hash tables (Matt Mahoney, public domain)
- Nacrith (2602.19626): CDF-24, architecture similar a Redukt
- Proxy Compression (2602.04289): proxy distributions para segmentación
- Speculative Decoding (Leviathan et al., 2023): draft/verify con modelo pequeño
- BLT (2412.09871): entropy-based segmentation
- Redukt R-033: adaptive skip existente (probs_buf threshold)
