# Throughput Paradigm Shift: Beyond the Autoregressive Bottleneck

> R-098 — Investigacion formal: enfoques no explorados para viabilidad comercial.
> Fecha: 2026-11-09. Metodo: investigacion web recursiva multi-agente (5 lineas paralelas).
> Estado: INVESTIGACION. No requiere cambios de codigo.
> Nota (2026-11-09): "R-098" es aquí la etiqueta de esta investigación; en el ROADMAP, R-098 es Q8 weight-only Mamba (Fase 19).

---

## Abstract

Redukt alcanza compresion de clase mundial (0.961 BPB enwik8, -50.5% vs brotli-11 en
datos estructurados) pero opera a 43 B/s — cuatro ordenes de magnitud por debajo del
minimo comercial viable (~100 KB/s). Las Fases 10-12 agotaron 12 tecnicas de optimizacion
bajo el regimen actual (DRAM-bound, Mamba-130M, BPE 50K). El roadmap pendiente (Fases
16-18) proyecta ~1.5 KB/s con TODAS las optimizaciones CPU ejecutadas — insuficiente.

Esta investigacion explora **enfoques no considerados** que reformulan el ROL del modelo
neuronal en el pipeline de compresion, en lugar de optimizar su velocidad de ejecucion.

**Hallazgo central**: el campo esta convergiendo hacia arquitecturas donde el modelo
neuronal actua como "consultor" — proporcionando guia de alto nivel a codificadores
clasicos rapidos — en vez de computar distribuciones de probabilidad por cada byte.

---

## Tabla de contenidos

1. [Diagnostico: por que la optimizacion directa fracasa](#1-diagnostico)
2. [Paradigma 1: Diffusion Language Models](#2-paradigma-1-diffusion-language-models)
3. [Paradigma 2: Modelo como consultor (Amortized Neural Guidance)](#3-paradigma-2-modelo-como-consultor)
4. [Paradigma 3: Speculative Decoding para compresion](#4-paradigma-3-speculative-decoding)
5. [Paradigma 4: Two-stage commercial architecture](#5-paradigma-4-two-stage-architecture)
6. [Tecnicas complementarias](#6-tecnicas-complementarias)
7. [Analisis matematico: limites fundamentales](#7-analisis-matematico)
8. [Competidores y validacion externa](#8-competidores-y-validacion-externa)
9. [Evaluacion comparativa](#9-evaluacion-comparativa)
10. [Recomendacion estrategica](#10-recomendacion-estrategica)
11. [Fuentes](#11-fuentes)

---

## 1. Diagnostico

### 1.1 El bottleneck autoregresivo

El loop de compresion actual ejecuta un forward pass neuronal por CADA token:

```
Para CADA token (x10,000 en archivo 10KB):
  cm.compute_biases()           ~0.2 ms   (L1/L2 cache, KB-scale)
  model.forward_into(tok)       ~3.0 ms   (LEE 127 MB de DRAM)   <- 75-80% del tiempo
  cm.combine_into(logits)       ~0.05 ms
  enc.encode_symbol(cdf)        ~0.01 ms

DRAM total por archivo 10KB: 10,000 x 127 MB = 1.27 TERABYTES
```

Este patron es inherente a la factorizacion autoregresiva:
P(x) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ... * P(x_N|x_1...x_{N-1})

Cada factor requiere un forward pass completo del modelo.

### 1.2 Lo que ya se intento y fracaso (Fases 10-12)

| Fase | Tecnicas | Resultado | Causa raiz |
|------|----------|-----------|------------|
| Fase 10: Sparsity | R-059 activation, R-061 pruning, R-062 highway, R-064 Q8 PTQ | 4/5 NOT VIABLE | Modelo 130M demasiado pequeno |
| Fase 11: DRAM BW | R-065 NTA, R-066 mixed Q, R-067 L2-tile | 4/4 NOT VIABLE | 493 MB >> 12 MB L3 |
| Fase 12: Compiler | R-068 to R-071 | 4/4 NOT VIABLE | DRAM-bound, no compute-bound |

**12 tecnicas, 0 viables. DRAM bandwidth = hard ceiling bajo el regimen actual.**

### 1.3 Proyeccion del roadmap pendiente (Fases 16-18)

| Escenario | Throughput proyectado | vs minimo comercial (100 KB/s) |
|-----------|----------------------|-------------------------------|
| MambaByte-96M + gate + NTA + tiling (CPU) | ~237 tok/s (~237 B/s) | 420x insuficiente |
| + Parallel scan compresion (R-093) | ~800-1,500 tok/s (~1.5 KB/s) | 67x insuficiente |
| + GPU T4 | ~1,500-3,000 tok/s (~3 KB/s) | 33x insuficiente |
| + GPU A100 | ~5,000-10,000 tok/s (~10 KB/s) | 10x insuficiente |

**Conclusion**: las optimizaciones del roadmap actual no cierran el gap de 4 ordenes de
magnitud. Se requiere un cambio de paradigma, no una optimizacion incremental.

### 1.4 Confirmacion teorica

"The Invocation Bottleneck in LLM-Based Lossless Compression" (arXiv:2611.05871,
noviembre 2026) formaliza el problema. El tiempo de decodificacion se descompone en:

- **Execution redundancy** (eliminable): overheads de scheduling, kernel launch, etc.
  Eliminable con batch-invariant pipelines. Hasta 5.8x speedup.
- **Sequential dependency** (irreducible): P(x_t) depende de x_{1..t-1}. No se puede
  eliminar — solo **intercambiar** por mas bits o mas llamadas al modelo.

Esta es una limitacion teorica, no solo ingenieril. Las unicas salidas son:

1. Cambiar la clase de modelo (eliminar autoregresividad)
2. Cambiar el rol del modelo (amortizar su costo)
3. Cambiar la arquitectura comercial (esquivar el problema)

---

## 2. Paradigma 1: Diffusion Language Models

### 2.1 Mecanismo

"Diffuse to Compress" (arXiv:2608.11249, agosto 2026) introduce el uso de Diffusion
Language Models (DLMs) para compresion lossless de texto.

A diferencia de modelos autoregresivos (que generan texto izquierda-a-derecha), los DLMs
generan texto mediante denoising iterativo. Un bloque completo de bytes se "desruida"
simultaneamente en cada paso. El numero y las posiciones de simbolos codificados por paso
de inferencia se convierten en una **decision de diseno**, no en una restriccion del modelo.

```
AR (actual):          DLM (propuesto):
  Para cada byte:       Para cada paso de denoising (~50-100 pasos):
    forward(1 byte)       forward(TODOS los bytes del bloque)
    encode(1 symbol)      encode(batch de symbolos)
  Total: N passes       Total: ~50-100 passes (independiente de N)
```

### 2.2 Resultados reportados

| Metrica | AR-LLM baseline | DLM ("Diffuse to Compress") |
|---------|-----------------|---------------------------|
| Throughput compresion | ~1x | Hasta 10^4x mayor |
| Throughput descompresion | ~1x | Hasta 10^6x mayor |
| Simetria encode/decode | Asimetrico (decode mas lento) | Casi simetricos |
| Paralelizacion | Secuencial obligatorio | Ventanas no solapadas, multi-GPU |
| Ratio de compresion | Referencia | Competitivo con AR-LLM |
| Evaluado en | — | enwik8 |

### 2.3 Implicaciones para Redukt

- En vez de 10,000 forward passes para 10KB, un DLM haria ~50-100 pasos de denoising
  sobre el bloque completo. Cada paso toca todos los bytes simultaneamente.
- La secuencialidad desaparece — el cuello de botella pasa de O(N) a O(pasos_denoising).
- El modelo de difusion es entrenable con la misma infraestructura de MambaByte.
- Coste estimado de entrenamiento: $100-500 en GPU cloud para un modelo de 100M params.

### 2.4 Limitaciones

- Paper reciente (agosto 2026), sin implementacion publica disponible.
- Requiere entrenar un modelo de difusion byte-level desde cero.
- La integracion con arithmetic coding difiere fundamentalmente del approach actual.
- El pipeline de denoising no es trivial de implementar en Rust zero-dep.
- Los resultados de ratio son "competitivos" pero no se reportan cifras exactas vs SOTA.

### 2.5 Viabilidad para Redukt

**Alta a largo plazo.** Este es el cambio de paradigma mas radical — elimina la
autoregresividad por completo. Requiere investigacion de 3-6 meses y entrenamiento de
modelo. Es la unica via conocida a ordenes de magnitud de mejora en throughput sin
sacrificar calidad de compresion.

---

## 3. Paradigma 2: Modelo como consultor

### 3.1 Concepto central

En vez de que el modelo neuronal compute P(x_t | context) para CADA byte, el modelo
analiza el archivo (o un bloque) UNA VEZ y produce **guia de alto nivel** que un
codificador clasico rapido usa para la compresion byte-a-byte.

```
ACTUAL (modelo = ejecutor):
  Para cada byte: modelo.forward() -> P(byte) -> arithmetic code
  Costo: O(N) forward passes

PROPUESTO (modelo = consultor):
  FASE 0: modelo.analyze(archivo) -> {CM_weights, transform_chain, routing_map}
  FASE 1: CM_ensemble(weights=optimized) + transforms -> arithmetic code
  Costo: O(1) forward passes + O(N) CM inference (cache-local, rapido)
```

La distincion clave: el modelo NO desaparece — su conocimiento se **destila** en
parametros que un sistema rapido puede ejecutar.

### 3.2 Validacion externa

#### 3.2.1 OpenZL (Meta, 2025-2026)

Descompone la compresion en un DAG (grafo dirigido aciclico) de codecs clasicos
componibles. Un "trainer" neuronal construye la frontera de Pareto de compresores LZ
optimizados para los datos. El grafo selecciona que codec maneja cada region del archivo.

- Resultado: supera a xz en ratio de compresion a 10x mas velocidad.
- Mecanismo: la inteligencia esta en la SELECCION, no en la ejecucion.
- Fuente: arXiv:2605.09928

#### 3.2.2 G1-V22 (ganador AIT Data Compression Challenge 2026)

Dos clasificadores XGBoost con vector de 38 caracteristicas por bloque:
- Entropia (Shannon, condicional)
- Estadisticas de runs (longitud, frecuencia)
- Distribuciones de bytes (uniformidad, picos)
- Entropias condicionales de orden 1-3
- Caracteristicas LZ77 (distancia media de match, cobertura)

El primer clasificador selecciona la cadena optima de transforms:
BWT, MTF, RLE, delta, x86 BCJ, weighted frequency coding.

El segundo selecciona el modelo de compresion post-transform.

- Resultado: competitivo entre 117 submissions en el AIT Challenge 2026.
- Clave: **no usa redes neuronales pesadas** — un arbol de decision ligero es el consultor.
- Soporta modos Fast/Normal/Accurate con trade-off velocidad/ratio.
- Fuente: arXiv:2606.17712

#### 3.2.3 OmniZip / DualComp (CVPR 2026)

Mixture-of-Experts con 3 expertos (cada uno mitad del tamano del MLP original), routing
top-2 por token. Modality-routing automatico.

| Metrica | Valor |
|---------|-------|
| Throughput CPU | 200 KB/s |
| Throughput NPU | 1.1 MB/s |
| BPB texto | 1.107 |
| vs gzip | -42% a -62% segun modalidad |

- Fuente: arXiv:2602.22286

#### 3.2.4 Meta-Compression (ICLR 2024)

Meta-aprende tanto la etapa de encoding como de decoding. El modelo de meta-prediccion
ejecuta en ~0.002 segundos dado un constraint de compresion, luego produce parametros
para un decoder rapido.

- Fuente: OpenReview (ICLR 2024)

### 3.3 Diseno propuesto para Redukt

```
FASE 0 — Profiling (una vez por archivo, ~500ms):
  (a) Extraer vector de 30-40 features por bloque de 4KB:
      - Shannon entropy (order-0, order-1, order-2)
      - Byte distribution statistics (peaks, uniformity, kurtosis)
      - Run-length statistics (max, mean, coverage)
      - LZ match coverage (within block)
      - Repetition structure (autocorrelation)

  (b) Clasificador ligero selecciona por bloque:
      - Transform chain: {none, E8E9, delta, shuffle, BWT+MTF}
      - CM configuration: {text-optimized, binary-passthrough, structured-XML}
      - Backbone policy: {full, sampled-1-in-K, skip}

FASE 1 — Compresion (rapida, KB/s a MB/s):
  Para cada bloque:
    Aplicar transform chain seleccionada
    CM ensemble con configuracion optimizada
    Si backbone_policy != skip:
      Mamba forward cada K bytes (amortizado)
    Arithmetic encode con CDF de CM (+ backbone si activo)
```

### 3.4 Throughput proyectado

| Componente | Costo por bloque 4KB | vs actual |
|------------|---------------------|-----------|
| Feature extraction | ~0.1 ms | Nuevo |
| Clasificador XGBoost | ~0.01 ms | Nuevo |
| Transform aplicacion | ~0.05 ms | Similar |
| CM-only encoding | ~2 ms (4096 bytes, cache-local) | ~10x mas rapido |
| Mamba sampled (1-in-8) | ~1.5 ms (512 forward passes) | 8x menos passes |
| Total por bloque | ~3.7 ms | ~8x speedup |
| **Throughput estimado** | **~1.1 MB/s** | **~25,000x vs actual** |

Nota: la estimacion CM-only asume vocab=256 (MambaByte). Con BPE 50K, CM no es viable
como predictor standalone (entropia ~15.5 bits, R-077 medido).

### 3.5 Impacto en BPB

El costo de la amortizacion es perdida de precision en regiones donde el backbone
aportaria informacion que CM no captura:

- **Texto/logs/XML** (CM fuerte): estimado +2-5% BPB
- **Binario/high-entropy** (CM debil): estimado +10-20% BPB
- **High-entropy puro** (sao, ooffice): sin impacto — ya perdemos vs brotli

Para los dominios donde Redukt gana (-50% vs brotli), un +5% BPB sigue dejando una
ventaja aplastante (-47.5% vs brotli).

---

## 4. Paradigma 3: Speculative Decoding

### 4.1 Mecanismo

Un modelo "draft" pequeno y rapido predice K tokens adelante. El modelo "target" grande
verifica los K tokens en un unico forward pass (verificacion paralela). Los tokens
aceptados se codifican con la distribucion del target; los rechazados disparan fallback.

Para compresion, el arithmetic coder DEBE usar las distribuciones del target (no del
draft), asi que el draft solo ahorra llamadas de inferencia — no degrada el ratio.

```
ACTUAL:                         SPECULATIVE:
  Para cada byte:                 Draft predice K bytes
    target.forward(1 byte)        Target verifica K bytes en 1 pass
    encode(1 symbol)              Si accept: encode K symbols con target dist
  N forward passes                Si reject: encode hasta rejection point
                                  ~N/K forward passes del target
```

### 4.2 Resultados publicados

#### 4.2.1 SDMC — Speculative Dual-Model Coding

Del paper "The Invocation Bottleneck" (arXiv:2611.05871, noviembre 2026):

| Metrica | Valor |
|---------|-------|
| Speedup encoding | 5.8x |
| Speedup decoding | 3.0x |
| Impacto en BPB | 0% (usa distribuciones del target) |
| Mecanismo | GPU-resident batch-invariant pipelines |

#### 4.2.2 MSMC — Multi-token Single-Model Coding

Del mismo paper:
- Un solo modelo predice multiples tokens desde un prefijo compartido.
- Menos llamadas pero cada llamada cuesta mas bits (degradacion de prediccion).
- Trade-off configurable entre throughput y ratio.

#### 4.2.3 Speculative KV Coding (Doubleword, mayo 2026)

Aplica el paradigma especulativo a compresion de KV cache de LLMs:
- Un predictor barato maneja el arithmetic coder.
- Hasta 4x compresion lossless del KV cache del target.
- 8x combinado con FP8 lossy quantization.
- Demuestra que speculative coding funciona con arithmetic coding en practica.

#### 4.2.4 PMATIC (ICLR 2026)

"Synchronizing Probabilities for Multidevice Arithmetic Coding" (arXiv:2601.10678):
- Resuelve el problema de sincronizacion encoder-decoder en approaches paralelos.
- Usa "helper bits" para tolerar divergencias de probabilidad cuantizada.
- Drop-in replacement para arithmetic encoder estandar.
- Mantiene losslessness con robustez ante no-determinismo de hardware.

### 4.3 Aplicacion directa a Redukt

El CM ensemble de Redukt ES el draft model natural:

```
PROPUESTO:
  1. CM predice distribucion para K bytes adelante (K=4..16)
     Costo: ~0.2 ms * K (cache-local, sin DRAM)
  2. Mamba verifica los K bytes en batch
     Costo: ~3 ms (1 forward pass que procesa K tokens)
  3. Para bytes aceptados: encode con distribucion Mamba
     Para bytes rechazados: encode hasta punto de rechazo, retry
```

| Metrica | Valor estimado |
|---------|---------------|
| Acceptance rate (texto, CM vocab=256) | ~40-70% |
| Acceptance rate (texto, CM BPE 50K) | ~5-15% |
| Speedup efectivo (vocab 256) | 2-5x |
| Speedup efectivo (BPE 50K) | 1.2-1.5x |
| Impacto BPB | 0% (target distributions) |

**Nota critica**: con BPE 50K, CM acceptance rate es bajo (CM entropy ~15.5 bits,
predicciones near-uniform). Con MambaByte vocab=256, CM se vuelve un draft model
efectivo. Este paradigma se complementa con el roadmap existente (R-088 MambaByte).

### 4.4 Compatibilidad con pipeline actual

La infraestructura de speculative decoding (R-053, `speculative.rs`) ya existe en el
codebase. El modulo R-053 implementa CM-draft para K tokens adelante. No esta integrado
en el loop de compresion principal pero el scaffolding esta listo.

> **Nota (2026-11-09)**: `speculative.rs` y `parallel_scan.rs` ya no existen (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte).
> El párrafo anterior describe el estado previo al borrado.

PMATIC podria integrarse como drop-in replacement del arithmetic coder actual para
habilitar verificacion paralela sin riesgo de desincronizacion.

---

## 5. Paradigma 4: Two-stage architecture

### 5.1 Concepto: esquivar el throughput en vez de resolverlo

En vez de hacer a Redukt mas rapido, **separar la experiencia de usuario de la velocidad
de compresion**:

```
STAGE 1 — Ingest (instantaneo, visible al usuario):
  datos originales -> zstd/brotli (10+ GB/s) -> hot storage
  Reduccion inmediata: ~60-70%
  Latencia: imperceptible

STAGE 2 — Recompresion (background, invisible):
  hot storage -> decompress zstd -> compress Redukt -> cold storage
  Reduccion adicional: 30-60% sobre zstd (en dominios fuertes)
  Latencia: horas/dias — irrelevante para UX

Ahorro neto: Stage 1 da 60% inmediato, Stage 2 lleva a 92-97%
```

### 5.2 Precedentes industriales

#### 5.2.1 Google Colossus

Datos llegan al cluster con replicacion 3x en flash (rapido, hot). Procesos background
llamados "custodians" escanean continuamente el cluster. Cuando un chunk se "enfria"
(sin lecturas recientes), los custodians **re-codifican transparentemente** con erasure
coding Reed-Solomon y migran de flash a HDD.

- Resultado: overhead de almacenamiento baja de 3x a ~1.5x.
- La re-codificacion ocurre en background, fuera del path de lectura/escritura.
- Fuente: documentacion interna Google (via analisis publico).

#### 5.2.2 Facebook f4 / ROMP

Fotos migran de Haystack (hot, 3.6x replicacion) a f4 (warm, 2.1x via erasure coding)
despues de 1-3 meses. Latencia sube de 14ms a 17ms.

Separadamente, el codec ROMP recomprime losslessly 260+ mil millones de JPEGs para -15%
adicional. L-ROMP logra -28% con perdida imperceptible. Todo en background.

- Fuente: OSDI'14 (f4), arXiv:1912.11145 (ROMP).

#### 5.2.3 AWS S3 Intelligent-Tiering

Monitorea patrones de acceso automaticamente:
- 30 dias sin acceso: Infrequent Access (-40% costo)
- 90 dias: Archive Instant Access (-68%)
- 180 dias: Deep Archive (-95%)

S3 no recomprime con mejores algoritmos — solo mueve a medios mas baratos. Pero el patron
de migracion temporal es identico al que Redukt usaria.

### 5.3 Modelo PetaGene (validacion comercial directa)

PetaGene vende compresion genomica lenta (BAM/FASTQ, 60-90% savings) comercialmente:

| Aspecto | PetaGene | Aplicable a Redukt |
|---------|----------|-------------------|
| Arquitectura | Docker containers, AWS Batch, auto-scaling | Si |
| Throughput | 200K+ archivos/24h (batch paralelo) | Adaptable |
| Pricing | **Per-TB-saved** (fraccion del ahorro) | Si |
| Descompresion | **Gratuita** (PetaLink, open-access) | Critico |
| Break-even | Clientes recuperan licencia en ~2 meses | Depende de volumen |
| Modelo de revenue | Subscripcion multi-anual | Si |

**Insight clave**: la descompresion gratuita elimina la barrera de adopcion mas grande
(lock-in fear). El cliente puede siempre descomprimir sin pagar.

### 5.4 Economia del modelo two-stage para Redukt

Escenario: empresa con 2 TB/dia de logs, retencion 5 anos.

```
STAGE 1 (inmediato):
  2 TB/dia -> zstd-3 (5 GB/s) -> 800 GB/dia almacenado
  Costo S3 Standard: $18.40/dia = $6,716/ano

STAGE 2 (background Redukt, 43 B/s actual, 100 cores):
  800 GB zstd -> ~320 GB Redukt (BPB 0.424 logs vs zstd ~1.2)
  Recompresion: 800 GB / (43 B/s * 100 cores) = ~2.2 dias (just-in-time)
  Costo S3 despues de recompresion: $7.36/dia = $2,686/ano

AHORRO ANUAL: $4,030
AHORRO 5 ANOS (acumulativo): $50,375
```

Con throughput mejorado (1 KB/s con roadmap parcial, 10 cores):
- Recompresion de 800 GB: ~9.3 dias
- Cola de batch: viable con buffer de 2 semanas

**El throughput deja de ser blocker** — solo determina el tamano de la cola batch.

### 5.5 Modelo de pricing propuesto

Basado en PetaGene + precedentes de storage optimization:

| Tier | Volumen | Precio/TB ahorrado | Descompresion |
|------|---------|-------------------|---------------|
| Starter | < 10 TB/mes | EUR 12/TB | Gratuita |
| Growth | 10-100 TB/mes | EUR 8/TB | Gratuita |
| Enterprise | > 100 TB/mes | EUR 5/TB | Gratuita |

Fuente: analisis previo en `docs/internal/business-case/GO-TO-MARKET-STRATEGY.md`.

---

## 6. Tecnicas complementarias

### 6.1 RNN State Checkpointing para archivos similares

Si se comprimen N archivos del mismo tipo (ej: 10,000 logs Apache del mismo servidor):
- Todos comparten formato y estructura
- Mamba aprende el formato en los primeros ~500 bytes
- **Cachear el estado RNN** despues de procesar un header representativo
- Reutilizar como warm-start para los N-1 archivos restantes

| Metrica | Sin checkpoint | Con checkpoint |
|---------|---------------|---------------|
| Forward passes (header) | 500 x 10,000 = 5M | 500 x 1 = 500 |
| Ahorro | — | 4,999,500 forward passes |
| Speedup amortizado | 1x | ~1.05x por archivo individual, significativo en batch |

**Limitacion**: los archivos deben ser suficientemente similares para que el estado
cacheado sea valido. Requiere mecanismo de deteccion de divergencia (ej: si BPB del
primer bloque post-checkpoint supera un threshold, descartar y recalcular).

**Implementacion**: bajo costo. Serializar MambaState a disco (~48 KB para Mamba-130M
SSM state) y cargar en warm-start.

### 6.2 Chunked Parallel Compression (BOA Constrictor)

BOA Constrictor (arXiv:2511.11337, noviembre 2025) demuestra Mamba + parallel range
coder para datos lossless:

| Metrica | Valor |
|---------|-------|
| Throughput compresion | 3.5 - 45 MB/s |
| Throughput descompresion | 1.5 - 25 MB/s |
| Modelo | Mamba next-byte prediction |
| Dominio | High Energy Physics (byte streams) |
| Paralelismo | Streaming range coder parallelizado |

**Implicacion**: prueba que Mamba + range coder paralelo es viable y alcanza MB/s.
La tecnica es directamente portable a Redukt.

**Mecanismo**: particionar el archivo en chunks independientes, comprimir cada chunk
con su propio stream de arithmetic/range coding, compartiendo el modelo pero con estados
independientes. Los chunks pueden procesarse en paralelo.

**Trade-off**: perdida de contexto inter-chunk en los limites. Chunks mas grandes
preservan mas contexto pero reducen paralelismo. Chunk optimo: 4KB-64KB.

### 6.3 Chained Lightweight Neural Predictors

arXiv:2604.15472, abril 2026.

Cadena de redes neuronales minimas, cada una manejando fuentes de Markov de orden
creciente. "Information inheritance": predicciones de orden bajo alimentan predictores
de orden superior.

| Metrica | Valor |
|---------|-------|
| Speedup encoding vs PAC | 1.2x - 6.3x |
| Speedup decoding vs PAC | 2.8x - 12.3x |
| Aplicable a byte streams | Si, directamente |
| Hardware | Consumer GPU |

**Para Redukt**: podria reemplazar o aumentar el CM ensemble con una cadena de micro-redes
que imiten al backbone para patrones frecuentes, delegando al backbone completo solo
para contextos raros.

### 6.4 Distilacion por dominio

Entrenar un modelo microscopico (500K-2M params) que imite a Mamba-130M especificamente
para un tipo de archivo (logs Apache, XML Facturae, etc.).

| Aspecto | Modelo original | Modelo destilado |
|---------|----------------|-----------------|
| Parametros | 130M (127 MB Q8) | 2M (~2 MB Q8) |
| Ubicacion en cache | DRAM (21.3 GB/s) | L2/L3 cache (~200 GB/s) |
| Throughput estimado | 34 tok/s | 2,000-5,000 tok/s |
| BPB relativo | Referencia | +5-15% (domain-specific) |
| Entrenamiento | Pre-existente | $50-200 por dominio |

**Referencia**: Knowledge Distillation for Learned Image Compression (ICCV 2025)
demuestra 40% menos parametros y 57% menos FLOPs con solo 2.5% degradacion BD-rate.

**Referencia**: StateSMix (arXiv:2605.02904) logra 1,300 tok/s con modelo de solo
120K params (DM=32, NL=2). BPB 2.12 en enwik8 — peor que Redukt 0.961 porque no tiene
pesos pre-entrenados, pero demuestra que modelos tiny alcanzan throughput de ordenes
de magnitud superiores.

### 6.5 Interpolacion de distribuciones (especulativo)

El estado SSM evoluciona suavemente — la distribucion P(x_t) cambia poco de un byte
al siguiente en regiones predecibles. Propuesta no publicada:

```
t = 0:    computa P(x_0) exacto (forward pass completo)
t = 1..K: INTERPOLA P(x_t) ~ P(x_0) + gradiente_estimado
t = K:    computa P(x_K) exacto (corrige drift)
```

**Base matematica**: en el manifold de Fisher de distribuciones de probabilidad, la
evolucion suave del estado SSM implica que la distribucion se mueve a lo largo de una
geodesica local. La interpolacion lineal en log-space es optima para geodesicas cortas
en geometria informacional.

**Riesgo**: la interpolacion introduce error que degrada BPB. El error crece con K.
No hay publicaciones que validen esto para compresion.

**Potencial**: K-fold speedup en regiones suaves. Alto riesgo, alto reward.

### 6.6 Bits-Back Coding con SSM (IconoCLaSM)

IconoCLaSM (arXiv:2103.10150) generaliza bits-back ANS a State Space Models:
- Entrelaza pasos de bits-back con timesteps del SSM.
- Explota la estructura de Markov latente del SSM.
- O(1) overhead inicial independiente de la longitud de secuencia.

**Limitacion para Redukt**: bits-back coding requiere codificar/decodificar datasets
completos (o chunks grandes) para amortizar el overhead inicial. Archivos pequenos
(10KB) sufririan penalizacion significativa. Requiere acceso a la distribucion posterior
aproximada del modelo — no trivial para Mamba's selective state space.

### 6.7 EDPC — Pipeline Decoupling

EDPC (arXiv:2507.18969, julio 2025) — Decoupled Pipeline Compression Architecture:
- Separa prediccion de probabilidades (GPU) de encoding (CPU).
- Ambos corren en paralelo: GPU computa distribuciones del bloque siguiente mientras
  CPU codifica el bloque actual.
- Throughput: **170.23 KB/s** (2.7x vs PAC baseline).

**Para Redukt**: alinea naturalmente con el plugin GPU (Fase 13). GPU computa logits
en batch, CPU ejecuta CM + arithmetic coding. La latencia se oculta por pipelining.

---

## 7. Analisis matematico

### 7.1 Limites fundamentales de la compresion lossless

**Teorema de Shannon (Source Coding Theorem)**: para alcanzar el entropy rate H(X) de
una fuente, se requiere la distribucion condicional P(x_t | x_1...x_{t-1}) en cada
posicion. Todo approach que evita este computo (LZ, sketches, arboles de contexto MDL)
alcanza peor compresion que modelos neuronales.

**Implicacion**: no existe un atajo matematico que evite computar distribuciones
per-symbol y alcance el mismo ratio. Las unicas opciones son:
1. Computar distribuciones APROXIMADAS mas rapido (distilacion, interpolacion).
2. Computar distribuciones EXACTAS menos veces (speculative, amortizado, consultor).
3. Usar una CLASE DIFERENTE de modelo que computa distribuciones conjuntas en menos
   pasos (difusion).

### 7.2 Sufficient statistics

El estado oculto h_t del SSM ES una estadistica suficiente — resume x_0...x_{t-1} para
prediccion. Pero su dimension es fija (ej: 130K floats para Mamba-130M) y debe
actualizarse secuencialmente. No existe una estadistica suficiente de dimension fija para
fuentes byte arbitrarias (Pitman-Koopman-Darmois: solo familias exponenciales admiten
estadisticas suficientes de dimension finita).

**Conclusion**: el estado SSM ya es la mejor "compresion del contexto" posible. No hay
shortcut que lo evite.

### 7.3 MDL como criterio de gating

Minimum Description Length: seleccionar el modelo M que minimiza L(M) + L(D|M).

Aplicado como criterio de gate para Redukt:
- Si L(backbone) + L(bloque|backbone) > L(CM_only) + L(bloque|CM_only), usar CM-only.
- Esto es una formalizacion principiada del entropy gate (R-077).
- PTP-MDL (Kieffer & Yang, 2014) demuestra compresion parallelizable con este enfoque.

**Potencial**: 2-5x throughput en regiones high-entropy (donde backbone no aporta valor),
con coste BPB minimo (CM-only es optimo para esas regiones por definicion MDL).

### 7.4 Propiedades algebraicas del SSM

La recurrencia SSM h_t = A_t * h_{t-1} + B_t * x_t es un primer-orden lineal con
matrices dependientes del input. Propiedades explotables:

- **Asociatividad**: la composicion de mapas lineales es asociativa, habilitando parallel
  prefix scan (Blelloch). Computa TODOS los h_t en O(log N) profundidad paralela.
- **Chunked computation** (Mamba-2 SSD): split en chunks de tamano C. Intra-chunk via
  matmul paralelo, inter-chunk via scan de N/C elementos. 2-8x speedup sobre Mamba-1.
- **Dualidad state-space**: para A diagonales, la recurrencia es equivalente a
  convolucion causal, habilitando FFT en entrenamiento.

**Limitacion**: para COMPRESION (no entrenamiento), el parallel scan es aplicable pero
con restricciones. Cada h_t depende del byte real x_t (no una prediccion), por lo que
la compresion sigue siendo secuencial en su dependencia de datos. El parallel scan
solo ayuda si TODOS los bytes son conocidos (compresion, no descompresion).

R-093 del roadmap ya contempla parallel scan para compresion. Lo que NO se habia
considerado es combinarlo con chunked inference (Mamba-2 SSD style) para reducir el
overhead per-byte de propagacion de estado.

---

## 8. Competidores y validacion externa

### 8.1 Landscape de compresion neuronal lossless (2025-2026)

| Sistema | enwik8 BPB | Throughput | Hardware | Modelo | Ano |
|---------|-----------|------------|----------|--------|-----|
| **Redukt** | **0.961** | 43 B/s | CPU i5-1235U | Mamba-130M + CM | 2026 |
| Nacrith | 0.939 | 50-90 tok/s | GTX 1050 Ti | SmolLM2-135M + CM | 2026 |
| High-throughput LLM | N/A (7.17x ratio) | **181 KB/s** | A100 | TensorRT-LLM | 2026 |
| EDPC | N/A | 170.23 KB/s | GPU+CPU | Pipeline decoupled | 2025 |
| BOA Constrictor | N/A (HEP data) | 3.5-45 MB/s | GPU | Mamba + parallel RC | 2025 |
| StateSMix | 2.123 | **1,300 tok/s** | CPU AVX2 | Mamba-120K + CM | 2025 |
| OmniZip | 1.107 | 200 KB/s | CPU | MoE 3-expert | 2026 |
| cmix v21 | ~1.63 | 0.67 KB/s | CPU | 2077 models + LSTM | 2024 |
| NNCP v3 | 1.19 | ~KB/s | CPU multi-core | Transformer-XL | 2023 |

### 8.2 Productos comerciales de compresion neuronal (2026)

| Empresa | Producto | Funding | Foco | Tipo |
|---------|----------|---------|------|------|
| The Compression Company (UK) | Satellite codecs | EUR 2.8M pre-seed | Satelite EO, 10-50x | Lossy |
| Neurovia AI / Robo.ai (UAE) | NeuroStream | Publica (AIIO) | Video 4K, 96.37% | Visually lossless |
| Multiverse Computing (ES) | CompactifAI | EUR 189M Series B | Compresion de LLMs | Model compression |
| ZipNN (IBM+community) | ZipNN | Open source | Pesos de modelos AI | Lossless, narrow |
| PetaGene (UK) | PetaSuite | Private | Genomica BAM/FASTQ | Lossless, domain |

**Hallazgo critico**: NO existe ningun producto comercial de compresion neuronal lossless
de proposito general. El espacio esta vacio. Esto es tanto oportunidad (first mover)
como advertencia (el mercado puede no estar listo por limitaciones de throughput).

### 8.3 GPU-accelerated lossless (no-neural, referencia)

| Sistema | Throughput | Hardware |
|---------|-----------|----------|
| nvCOMP gANS | ~530 GB/s | RTX 4090 |
| nvCOMP Cascaded | hasta 500 GB/s | NVIDIA GPU |
| nvCOMP LZ4 | hasta 100 GB/s | NVIDIA GPU |
| Falcon (FP data) | 10.82 GB/s compress | GPU |
| L3 (learned) | 1.08-1.90 TB/s decompress | GPU |

La compresion clasica GPU opera 6 ordenes de magnitud mas rapido que la neuronal.
Ratios mucho peores (2-5x vs 8-10x neuronal), pero el gap de throughput es enorme.

### 8.4 Economia de la compresion neuronal

"Waiting to Decompress: The Economics of LLM-Based Compression" (VLDB/CIDR 2026, Kipf
et al.) establece:

- La compresion LLM tardaria **10 anos** en amortizar costos de compute vs almacenar sin
  comprimir.
- Comparado con compresion tradicional, el break-even seria **120 anos**.
- Los autores proyectan viabilidad "dentro de la proxima decada" con mejoras de hardware.

**Implicacion**: la via directa (hacer la compresion neuronal mas rapida hasta que sea
economica) NO es viable a corto/medio plazo. Se requiere un modelo comercial que no
dependa de throughput (two-stage, per-TB-saved pricing, archival batch).

---

## 9. Evaluacion comparativa

### 9.1 Matriz de paradigmas

| Paradigma | Throughput esperado | BPB risk | Costo impl. | Depende MambaByte | Madurez |
|-----------|-------------------|----------|-------------|-------------------|---------|
| **1. DLM (Diffuse to Compress)** | 10,000x+ | ~comparable AR | $100-500 training + alto impl. | No (reemplaza backbone) | Baja (paper reciente) |
| **2. Modelo-consultor** | 100-25,000x | +2-5% texto, +10-20% binario | Medio | Idealmente si (vocab 256) | Media (OpenZL, G1-V22) |
| **3. Speculative decoding** | 2-6x | 0% (target dist) | Bajo (infra existe) | Mejora con vocab 256 | Alta (SDMC, PMATIC) |
| **4. Two-stage commercial** | Infinito (UX) | 0% | Bajo (arquitectura) | No | Alta (Google, PetaGene) |

### 9.2 Tecnicas complementarias

| Tecnica | Throughput | BPB risk | Costo | Independiente |
|---------|-----------|----------|-------|---------------|
| RNN state checkpointing | 1.05-2x (batch) | 0% | Bajo | Si |
| Chunked parallel (BOA) | Linear en #chunks | +1-3% | Medio | Si |
| Chained lightweight predictors | 3-12x | Variable | Medio | Si |
| Distilacion por dominio | 100-1,000x | +5-15% | $50-200/dominio | Si |
| Interpolacion distribuciones | K-fold | Alto riesgo | Bajo | Si |
| Bits-back (IconoCLaSM) | N/A (diferente modelo) | N/A | Alto | Requiere posterior |
| EDPC pipeline decoupling | 2-3x | 0% | Medio | Requiere GPU |
| MDL gating | 2-5x (high-entropy) | Minimo | Bajo | Si |

### 9.3 Combinabilidad

Los paradigmas NO son mutuamente excluyentes. Combinaciones viables:

```
Combinacion A (inmediata, $0):
  Two-stage (zstd + background Redukt)
  + Speculative decoding (CM draft)
  + RNN state checkpointing (batch similar files)
  Throughput UX: infinito. Throughput real: ~2-6x actual (~86-258 B/s)

Combinacion B (medio plazo, $50-200):
  Two-stage + Speculative + MambaByte (vocab 256)
  + Modelo-consultor (CM tuning per block)
  + Distilacion por dominio (logs, XML)
  Throughput real: ~2-10 KB/s (depende de gate rate y distilacion)

Combinacion C (largo plazo, $200-500):
  DLM (Diffuse to Compress) reemplaza backbone
  + CM como refinamiento
  + Chunked parallel + GPU
  Throughput real: ~100 KB/s - 1 MB/s (estimado, no validado)
```

---

## 10. Recomendacion estrategica

### 10.1 Accion inmediata (0 costo, 0 riesgo BPB)

**Implementar two-stage architecture como modelo de deployment.**

No requiere cambios de codigo en el compresor. Requiere:
- Documentacion del workflow zstd-ingest + Redukt-batch
- Script de orquestacion (cola batch, prioridad por antiguedad)
- Metricas de ahorro (dashboard: GB ahorrados, costo evitado)

Esto **desbloquea la via comercial inmediatamente** — el throughput de Redukt deja
de ser un blocker de UX y se convierte en un parametro de sizing de infraestructura.

### 10.2 Corto plazo (semanas, bajo costo)

**Speculative decoding con CM como draft.**

La infraestructura existe (R-053 `speculative.rs`) (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte). Integrar en el loop de compresion
principal. Speedup esperado: 1.2-1.5x con BPE 50K, 2-5x con MambaByte vocab 256.

**RNN state checkpointing para batch processing.**

Serializar/deserializar MambaState. Implementable en <100 lineas. Beneficio en
workflows batch con archivos homogeneos (logs, invoices).

### 10.3 Medio plazo (meses, $50-200)

**Modelo-consultor: profiling + CM parameter tuning por bloque.**

Requiere:
- Feature extractor (38-dim vector per 4KB block)
- Clasificador ligero (XGBoost o decision tree, entrenado offline)
- CM configurations pre-definidas (text-optimized, binary-passthrough, etc.)
- Backbone sampling policy (full, 1-in-K, skip)

Throughput proyectado: 1-25 KB/s dependiendo de agresividad del sampling.

**Distilacion por dominio (logs Apache, XML Facturae).**

Entrenar micro-modelos (500K-2M params) que imiten al backbone en dominios especificos.
Throughput: miles de tok/s. BPB: +5-15% vs backbone completo, pero sigue ganando a
brotli por amplio margen en dominios fuertes.

### 10.4 Largo plazo (trimestres, $200-500)

**Evaluar Diffusion Language Model para compresion.**

Es la unica via conocida a >100 KB/s sin hardware especializado. Requiere:
- Entrenar DLM byte-level (~100M params) en corpus diverso
- Implementar denoising schedule + coding scheme en Rust
- Validar ratio vs autoregresivo en benchmark de 12 archivos

Si el ratio es competitivo (<=+10% vs AR), esto transforma a Redukt de "proof of concept
con throughput prohibitivo" a "producto comercial viable en multiples segmentos".

### 10.5 Lo que NO hacer

- **No invertir en R-091/R-092 (NTA, L2-tiling) antes de MambaByte**: son optimizaciones
  condicionales sobre proyecciones no validadas empiricamente. +35% sobre 34 tok/s sigue
  siendo comercialmente irrelevante.

- **No perseguir GPU como unica via**: GPU lleva el throughput a ~10 KB/s (A100), que
  sigue sin cerrar el gap para la mayoria de segmentos comerciales. GPU es complementaria,
  no solucion.

- **No ignorar two-stage architecture**: es la unica via que desbloquea revenue a
  throughput actual. Todas las demas opciones requieren meses de implementacion.

---

## 11. Fuentes

### Papers fundamentales

1. "Diffuse to Compress: Lossless Compression with Diffusion Language Models"
   arXiv:2608.11249, agosto 2026.

2. "The Invocation Bottleneck in LLM-Based Lossless Compression"
   arXiv:2611.05871, noviembre 2026. (SDMC, MSMC formalization)

3. "Synchronizing Probabilities for Multidevice Arithmetic Coding (PMATIC)"
   arXiv:2601.10678, ICLR 2026.

4. "Waiting to Decompress: The Economics of LLM-Based Compression"
   VLDB/CIDR 2026, Kipf et al.

5. "High-Throughput LLM-Based Compression and What It Takes"
   arXiv:2610.17586, VLDB/ADMS 2026.

### Arquitecturas de compresion

6. "Nacrith: Neural Lossless Compression via Ensemble Context Modeling"
   arXiv:2602.19626, febrero 2026.

7. "StateSMix: Online Lossless Compression via Mamba SSM and Sparse N-gram CM"
   arXiv:2605.02904, mayo 2025.

8. "BOA Constrictor: A Mamba-based Lossless Compressor for HEP Data"
   arXiv:2511.11337, noviembre 2025.

9. "OmniZip: Unified Lightweight Lossless Compressor"
   arXiv:2602.22286, CVPR 2026.

10. "DualComp: End-to-End Dual-Modality Lossless Compressor"
    arXiv:2505.16256, 2026.

11. "Chained Lightweight Neural Predictors for Lossless Compression"
    arXiv:2604.15472, abril 2026.

12. "EDPC: Accelerating Lossless Compression via Decoupled Parallel Dataflow"
    arXiv:2507.18969, julio 2025.

13. "SEP: Semantics Enhancement Framework for Byte-Stream Compression"
    IJCAI 2025.

### Tecnicas de aceleracion

14. "HPAC: Hierarchical Parallel Autoregressive ConvNet"
    arXiv:2511.10991, noviembre 2025.

15. "Parallelizing Stream-Based Lossless Compression on GPU/CPU"
    ACM 2026.

16. "Speculative KV Coding"
    Doubleword Blog, mayo 2026.

17. "Range-Coder with Fast Adaptation"
    arXiv:2601.06120, 2026.

### Latent variable models

18. "BB-ANS: Practical Lossless Compression with Latent Variables"
    arXiv:1901.04866, Townsend et al.

19. "Bit-Swap: Recursive Bits-Back Coding for Deep Generative Models"
    arXiv:1905.06845, ICML 2019.

20. "IconoCLaSM: Bits-Back for State Space Models"
    arXiv:2103.10150.

### Modelo-consultor y meta-learning

21. "OpenZL: Using Graphs to Compress Smaller and Faster"
    arXiv:2605.09928, Meta.

22. "The 2026 AIT Data Compression Challenge" (G1-V22)
    arXiv:2606.17712.

23. "Meta Compression"
    ICLR 2024 submission.

24. "WaLLoC: Wavelet Learned Lossy Compression"
    IEEE DCC 2025, arXiv:2501.13751.

25. "Brevis: Lossless Tensor Compression as Program Synthesis"
    arXiv:2608.02162.

### Matematica y teoria

26. "A Universal Parallel Two-Pass MDL Context Tree Algorithm"
    arXiv:1407.1514, Kieffer & Yang.

27. "Mamba-2: Algorithms and Systems (SSD)"
    Dao & Gu, Princeton.

28. "Compiler-First State Space Duality"
    arXiv:2603.09555.

29. "Sliding Window Recurrences for Sequence Models"
    arXiv:2512.13921.

### Precedentes comerciales

30. Facebook f4 Warm BLOB Storage, OSDI'14.

31. Facebook ROMP Photo Recompression, arXiv:1912.11145.

32. PetaGene on AWS Reference Architecture, AWS Blog.

33. Google Colossus architecture analysis (public documentation).

34. "Knowledge Distillation for Learned Image Compression"
    ICCV 2025.

35. "Fitted Neural Lossless Image Compression (FNLIC)"
    CVPR 2025.

36. "Test-Time Steering / Weighted Product of Experts"
    EMNLP 2025.

---

*Documento generado como investigacion formal. No se realizaron cambios de codigo.*
*Metodo: investigacion web recursiva multi-agente (5 lineas paralelas, ~30 queries).*
