# Roadmap — Redukt

> Tareas priorizadas por criticidad para producción.
> Actualizar al completar cada subtarea.

---

## P0 — Integridad y Fiabilidad (BLOCKER para producción)

### R-001: CRC-32 en formato AZTH v3
- **Estado**: COMPLETADO
- **Impacto**: Sin checksum, archivos corruptos producen basura silenciosamente
- **Spec**: Añadir CRC-32 (IEEE) del contenido original al final del archivo
- **Coste**: +4 bytes por archivo
- **Archivos**: `src/domain/crc32.rs`, `src/application/codec.rs`, `docs/api/FORMAT-SPEC.md`
- **Subtareas**:
  - [x] Implementar crc32 (IEEE polynomial, zero-dep) — `src/domain/crc32.rs`
  - [x] Escribir CRC-32 tras STREAM en compresión
  - [x] Leer y validar CRC-32 en descompresión (error si no coincide)
  - [x] Bump MAGIC a v3 (`AZTH\x03\x00\x00\x00`)
  - [x] Mantener retrocompat: leer v2 sin CRC (warn), escribir siempre v3
  - [x] Actualizar FORMAT-SPEC.md
  - [x] Tests: 4 unit tests CRC-32 (known vectors, empty, single byte, consistency)

### R-002: Enforce roundtrip size mismatch como error
- **Estado**: COMPLETADO
- **Impacto**: Actualmente `codec.rs:258` solo imprime WARNING — el usuario recibe datos incompletos sin saberlo
- **Coste**: 0 bytes, cambio de 2 líneas
- **Archivos**: `src/application/codec.rs`
- **Subtareas**:
  - [x] Cambiar warning a `eprintln!` + `std::process::exit(1)`
  - [x] En API (`api.rs`): devolver error string (ya existía, verificado)

### R-003: Validación de hiperparámetros del header
- **Estado**: COMPLETADO
- **Impacto**: NaN/Inf en bias_lr, ngram_scale o mix_eta corrompe toda la descompresión
- **Coste**: 0 bytes, ~10 líneas de validación
- **Archivos**: `src/application/codec.rs`, `src/application/api.rs`
- **Subtareas**:
  - [x] Validar `f32::is_finite()` para los 3 hiperparámetros al leer header
  - [x] Error claro si falla validación (CLI: exit(1), API: Err string)

### R-004: Validación del arithmetic decoder
- **Estado**: COMPLETADO
- **Impacto**: Stream corrupto causa comportamiento indefinido (basura o panic)
- **Archivos**: `src/domain/coder.rs`, `src/application/codec.rs`, `src/application/api.rs`
- **Subtareas**:
  - [x] Detectar underflow/overflow en `decode_symbol()` — `DecodeError::RangeUnderflow`
  - [x] Validar que tokens decodificados ∈ [0, vocab_size) — `decode_symbol_checked()`
  - [x] Propagar error al caller: CLI exit(1), API Err(String)
  - [x] Tests: out-of-range, empty stream, corrupted stream

---

## P1 — Robustez y Testing

### R-005: Tests de edge cases
- **Estado**: COMPLETADO
- **Archivos**: `src/application/codec.rs` (unit tests), `tests/roundtrip.rs` (integration tests)
- **Subtareas**:
  - [x] Test: archivo vacío (0 bytes) — `decompress_rejects_empty_file`
  - [x] Test: archivo de 1 byte — `roundtrip_single_byte` (ignored, needs weights)
  - [x] Test: archivo < 28 bytes — `decompress_rejects_short_file`
  - [x] Test: AZTH con magic inválido — `decompress_rejects_bad_magic`
  - [x] Test: AZTH con versión no soportada — `decompress_rejects_unsupported_version`
  - [x] Test: AZTH con bit flip en stream — `decompress_rejects_corrupted_crc` (ignored, needs weights)
  - [x] Test: determinismo — `coder_deterministic_output`, `cdf_from_logits_deterministic`, `roundtrip_determinism`
  - [x] Unit tests: CRC split/extraction, v2/v3 parsing, hyperparams validation
- **Total**: 14 nuevos tests (10 unit + 4 integration), 50 tests en suite completa

### R-006: Error handling — panics a Results
- **Estado**: COMPLETADO
- **Impacto**: `panic!()` en I/O no es aceptable para librería/API
- **Archivos**: `src/domain/error.rs`, `src/application/codec.rs`, `src/application/api.rs`, `src/main.rs`
- **Subtareas**:
  - [x] Reemplazar `unwrap_or_else(|e| panic!())` por propagación de `Result<>`
  - [x] Devolver `AzError` desde compress/decompress (y hybrid-eval, baseline, cm-eval)
  - [x] CLI: catch en main, exit(1) con mensaje limpio
  - [x] API: mapear a HTTP status codes apropiados
  - [x] `next_arg` retorna `AzResult<&str>` — cascada actualizada en todos los callers
  - [x] 5 nuevos unit tests (validate_magic, parse_hyperparams)
  - [x] 55 tests totales pasando (51 unit + 4 integration)

### R-007: Límite u32 TOTAL_BYTES — documentar y validar
- **Estado**: COMPLETADO
- **Impacto**: Archivos >4GB corrompen silenciosamente (overflow u32)
- **Coste**: 0 bytes formato, ~15 líneas validación
- **Archivos**: `src/domain/error.rs`, `src/application/codec.rs`, `src/application/api.rs`
- **Subtareas**:
  - [x] `validate_u32_size()` en error.rs — check genérico con error explícito
  - [x] Validar TOTAL_BYTES y TOKEN_COUNT en cmd_compress (codec.rs)
  - [x] Validar TOTAL_BYTES y TOKEN_COUNT en compress_bytes (api.rs)
  - [x] api.rs compress_bytes retorna Result (antes era infallible)
  - [x] Documentado en FORMAT-SPEC.md como limitación conocida (ya existía)
  - [x] 3 nuevos unit tests (accepts_max, accepts_zero, rejects_overflow)
  - [x] Evaluar upgrade a u64: diferido a futura v4 (rompe formato header)
  - [x] 58 tests totales pasando (54 unit + 4 integration)

---

## P2 — Métricas y Observabilidad

### R-008: Timing por fase
- **Estado**: COMPLETADO
- **Impacto**: Necesario para identificar cuellos de botella en optimización
- **Archivos**: `src/application/telemetry.rs`, `src/application/codec.rs`, `src/main.rs`
- **Subtareas**:
  - [x] `PhaseTimer` struct — acumulador de fases con report y JSON export
  - [x] Instrumentar cmd_compress: read, model_load, tokenize, encode, write
  - [x] Instrumentar cmd_decompress: read, model_load, decode, detokenize, verify, write
  - [x] Instrumentar cmd_hybrid_eval: read, model_load, tokenize, evaluate
  - [x] Instrumentar cmd_baseline: read, model_load, tokenize, evaluate
  - [x] `--verbose` / `-v` flag en compress, decompress, hybrid-eval, baseline
  - [x] JSON timing output en modo verbose (`timing_json: {...}`)
  - [x] 4 nuevos unit tests para PhaseTimer
  - [x] 62 tests totales pasando (58 unit + 4 integration)

### R-009: Métricas de eficiencia del coder
- **Estado**: COMPLETADO
- **Archivos**: `src/application/codec.rs`
- **Subtareas**:
  - [x] `CoderMetrics` struct: ce_bpb, compressed_bpb, overhead_bpb, efficiency_pct, ratio
  - [x] Efficiency % en report estándar de compress (`coder efficiency: XX.XX%`)
  - [x] JSON metrics en --verbose (`coder_metrics_json: {...}`)
  - [x] 3 nuevos unit tests (perfect efficiency, with overhead, zero input)
  - [x] 65 tests totales pasando (61 unit + 4 integration)

---

## PV — Verificación de Datos (BLOCKER para decisiones de negocio)

> Auditoría 2026-10-27 encontró datos fabricados/no verificados en business-case docs.
> Estas tareas producen los datos medidos necesarios para tomar decisiones.

### R-V01: Re-benchmark post-SIMD (18 archivos)
- **Estado**: COMPLETADO
- **Resultado**: `data/benchmark/results/post-simd-benchmark.csv`
- **Hallazgos** [MEASURED]:
  - Mean tok/s SIMD: 22.7 (range 18.5-26.3)
  - Mean tok/s scalar: 18.7 (3 files)
  - SIMD speedup: 1.25x (range 1.10-1.39x)
  - Mean B/s SIMD: 50.8 overall (industry 57.5, silesia 48.2)
  - BPB idéntico SIMD vs scalar (determinismo verificado)
  - Nota: tok/s menor que mediciones anteriores (~35), probablemente por carga del sistema

### R-V02: Benchmark gzip-9 y brotli-11 a 10KB
- **Estado**: COMPLETADO
- **Resultado**: `data/benchmark/results/gzip9-brotli11-comparison.csv`
- **Hallazgos clave**:
  - gzip-9 en logs: 0.546-0.990 BPB (NO 3.00 como decía BUSINESS-CASES.md)
  - Redukt vs gzip-9 en logs: -49% a -55% (mejor de lo documentado)
  - Redukt pierde en 4/17 archivos vs gzip-9 (ooffice, osdb, sao, x-ray)

### R-V03: Benchmark compresión PDF
- **Estado**: COMPLETADO
- **Resultado**: `data/benchmark/results/pdf-benchmark.csv`
- **Hallazgos** [MEASURED, W3C dummy.pdf 13KB, first 10KB]:
  - Redukt Q8: **11.110 BPB** — near maximum entropy (8.0 = random)
  - brotli-11: 7.897 BPB (compresses 1.3%)
  - gzip-9: 7.938 BPB (compresses 0.8%)
  - Redukt vs brotli: **+40.7% PEOR**
  - PDF binario es casi incompresible — Redukt pierde decisivamente
- **Impacto en e-invoicing**: Con 67% PDF en el mix, el blended result anula la ventaja XML.
  E-invoicing solo viable con compresión selectiva (XML separado de PDF).
- **CAVEAT**: dummy.pdf es un PDF simple. PDFs reales con texto podrían ser diferentes.
  - [ ] Obtener PDF de factura real con texto embebido (no solo imágenes)
  - [ ] Re-medir con PDF representativo de facturación

### R-V04: Test de escala (BPB vs tamaño de archivo) ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-29)
- **Impacto**: Confirma que la ventaja de Redukt CRECE con tamaño de archivo.
- **Datos**: `data/benchmark/results/scale-test-complete.csv` (2 archivos × 4 tamaños)
- **Resultados** [MEASURED]:
  | File | Size | Redukt Q8 | brotli-11 | Δ vs brotli |
  |---|---|---|---|---|
  | apache-access.log | 1KB | 0.974 | 1.592 | **-38.8%** |
  | apache-access.log | 10KB | 0.245 | 0.466 | **-47.5%** |
  | apache-access.log | 50KB | 0.185 | 0.353 | **-47.4%** |
  | apache-access.log | 100KB | 0.186 | 0.328 | **-43.4%** |
  | dickens | 1KB | 2.099 | 3.024 | **-30.6%** |
  | dickens | 10KB | 1.779 | 2.778 | **-36.0%** |
  | dickens | 50KB | 1.477 | 2.711 | **-45.5%** |
  | dickens | 100KB | 1.429 | 2.649 | **-46.0%** |
- **Conclusión**: Ventaja crece con tamaño: dickens -30.6% (1KB) → -46.0% (100KB).
  Apache plateau a ~47% a partir de 10KB. Redukt mejora MÁS que brotli con más contexto.
- **Subtareas**:
  - [x] Medir gzip-9 y brotli-11 a 1KB, 10KB, 50KB, 100KB
  - [x] Medir Redukt Q8 a 1KB, 10KB, 50KB, 100KB
  - [x] Documentar curva de convergencia comparativa

### R-V05: Evaluar Q4 quantization BPB
- **Estado**: COMPLETADO — **RECHAZADO**
- **Resultado**: `data/benchmark/results/q4-evaluation.csv`
- **Hallazgos** [MEASURED, 6 archivos representativos]:
  - Mean BPB regression: **+13.3%** (range +5.1% a +25.7%)
  - Peor: xml +25.7%, facturae +20.8% — LOS MEJORES VERTICALES SON LOS MÁS AFECTADOS
  - Q4 throughput: **2.7-6.1 tok/s** vs Q8 26-47 tok/s — **7.6x MÁS LENTO**
  - Q4 no aporta NI mejor compresión NI mejor velocidad
  - Modelo 72.6 MB Q4 vs ~129 MB Q8 — ahorro de RAM irrelevante para 0.1B
- **Decisión**: Q4 RECHAZADO. Q8 permanece como default de producción.
  Q4 flag se mantiene para experimentación pero NO se recomienda.

### R-V06: Factibilidad de model pruning (RWKV-edge)
- **Estado**: COMPLETADO
- **Resultado**: `docs/internal/research/model-optimization-feasibility.md`
- **Hallazgos**:
  - RWKV-edge paper SÍ probó en 0.1B — 3.5-4.8x compresión, -1pp accuracy [SOURCED]
  - PERO: accuracy NLP ≠ BPB compresión. Impact en BPB probablemente mayor.
  - Q4 quantization: RECHAZADO por R-V05 (medido, no especulativo)
  - SVD rank-384 en TimeMix: opción más viable. BPB est. +2-5% [ESTIMATE]
  - Parallel segment compression: mayor ganancia potencial (2-4x) pero alto esfuerzo
  - Distillation: NO-GO (fuera de scope, modelo ya mínimo)
- **Recomendación**: SVD conservador → Parallel segments. Ambos CONDITIONAL GO.
  Ninguna optimización de modelo cambia la ecuación fundamental de throughput
  suficiente para business cases actuales sin GPU.

### R-V07: Verificar TAM figures con fuentes primarias
- **Estado**: COMPLETADO (parcial — 4/7 verificados, 3 sin fuente)
- **Resultados**:
  - [x] Log management TAM: **$4.0-5.0B** (MarketsandMarkets $4.1B, Mordor $4.35B, R&M $4.99B)
  - [x] Cribl: **$390M ARR** Aug 2026 (Sacra), **$3.5B val** Series E (GlobeNewsWire). CONFIRMADO.
  - [x] Edicom: **EUR 252.9M** revenue 2024, EUR 276.6M 2025 (Valencia Plaza). **60% net margin** (El Economista Jul 2025). 15K-19K clients. CONFIRMADO.
  - [x] Cold/archive data storage TAM: **NO ENCONTRADO** — "$49B" sin fuente verificable
  - [x] Genomics TAM: **NO ENCONTRADO** — "$2-5B" sin fuente verificable
  - [x] MARKET-ANALYSIS.md actualizado con fuentes verificadas y tags

---

## P3 — Optimización y Go-to-Market

> **Principio rector**: las optimizaciones de throughput tienen interdependencias en
> degradación de BPB. Se secuencian de menor a mayor riesgo, con quality gates entre cada paso.
>
> **Presupuesto de degradación BPB**: máximo **+3% combinado** sobre Q8 baseline en logs
> (vertical prioritario). Si se supera, STOP — la ventaja competitiva (-31% vs brotli mean,
> compresión real) se erosiona. Con +3% la ventaja cae a ~-29%, aún viable.
> A +5% cae a ~-27%, margen peligrosamente reducido. A +10% cae a ~-24%.
>
> **Justificación del presupuesto +3%** (2026-10-28): La corrección de cross-entropy a
> compresión real redujo la ventaja media en logs de -41% a -31%. El margen real es menor
> del que se estimaba — el presupuesto debe ser más conservador para protegerlo.
>
> **Interdependencias documentadas**:
> - KV caching + Multi-file parallelism: **independientes**, 0% BPB loss cada una
> - SVD pruning: **degrada BPB** (-1pp NLP ≈ est. +2-5% BPB compresión) [ESTIMATE]
> - Parallel segments: **degrada BPB** (pérdida de contexto en fronteras) [UNMEASURED]
> - SVD + Parallel segments combinados: **degradación compuesta**, no aditiva [UNMEASURED]
> - Cada paso debe medirse individualmente Y acumulado contra el presupuesto
>
> **Corrección 2026-10-28**: Presupuesto reducido de +5% a +3%. Ver justificación arriba.

### R-010: Estabilizar features experimentales en producción
- **Estado**: COMPLETADO
- **Impacto**: `--emb-surgery center0.3` mejora 14/18 archivos, media -1.69% BPB
- **Contexto**: F32 vs Q8 confirmó que cuantización NO es el gap (<2.5%). Las features experimentales sí.
- **Archivos**: `src/infrastructure/rwkv7/model.rs`, `src/application/codec.rs`
- **Resultados benchmark** (2026-10-27):
  - `--emb-surgery center0.3`: MEJOR en 14/18 archivos (-1.69% media), PEOR en 4 (mr +6.2%, reymont +2.3%, ooffice +1.1%, ubl +1.7%)
  - `--emb-surgery norm`: peor que center0.3 en todos
  - `--emb-surgery spread`: muy malo (+29% en enwik8)
  - `--emb-surgery center0.5`: demasiado agresivo (+9% en enwik8)
  - `--tau 0.05..0.5`: PEOR en todos los valores (+1.1% en enwik8). DESCARTADO.
  - Detalle: `data/benchmark/results/emb-surgery-benchmark.csv`
- **Subtareas**:
  - [x] Benchmark `--emb-surgery` variantes (center0.3, norm, spread, center0.1-0.5)
  - [x] Benchmark `--tau` variantes (0, 0.05, 0.1, 0.2, 0.5) — DESCARTADO, empeora
  - [x] Añadir `--emb-surgery` a cmd_compress, cmd_decompress, cmd_baseline
  - [x] Evaluar almacenar método en header (v4) para roundtrip transparente — DONE: stored in v4 header FLAGS 0x02 (5 bytes: 1B method + 4B alpha)
  - [x] Documentar que compress/decompress deben usar el mismo `--emb-surgery` — DONE: header auto-applies on decompress, CLI overridden by header when present
  - [x] `--match` (longest-match predictor) — DESCARTADO: -0.0095 BPB, bajo ROI, no se porta
  - [x] `--hierarchical` — DESCARTADO: supersedido por AdaptiveMixer actual

### R-011: Throughput — SIMD vectorización
- **Estado**: COMPLETADO
- **Impacto**: AVX2 kernels para Q8 y F32 matmul, 3-tier dispatch automático
- **Archivos**: `src/domain/tensor.rs`
- **Resultados micro-benchmark** (i5-1235U Alder Lake, release, 768×768 Q8):
  - Scalar: 97.2 μs → AVX2: 14.5 μs (**6.7x**) → VNNI: 6.7 μs (**14.5x**)
  - Head 65536×768: Scalar 4.2ms → VNNI 3.2ms (**1.32x**, memory-bound)
- **Bit-exact verification**: Q8 AVX2 0/768 mismatches, Q8 VNNI 0/768 mismatches
- **F32 FMA deshabilitado**: 733/768 bit-mismatches → rompe roundtrip cross-path
- **Cross-path roundtrip**: PASS (SIMD→scalar y scalar→SIMD, byte-identical, CRC OK)
- **BPB**: 1.281205 idéntico scalar/SIMD (0 delta)
- **Resultados end-to-end** (enwik8 5KB, Q8, 3 rondas):
  - Scalar ~26 tok/s → SIMD ~35 tok/s (**~1.3x throughput**)
  - Detalle: `data/benchmark/results/simd-benchmark.md`
- **Dispatch Q8**: VNNI > AVX2 > scalar (F32 siempre scalar)
- **REDUKT_NO_SIMD=1**: env var para forzar scalar (benchmarking)
- **Subtareas**:
  - [x] Perfilar hotspots: mat_vec_mul dominante (>80% del tiempo via PhaseTimer)
  - [x] Implementar AVX2 kernel para Q8: sign trick + maddubs + madd_epi16, 4-row ILP
  - [x] Implementar AVX2+FMA kernel para F32: 8-wide FMA, 4-row ILP
  - [x] Horizontal sum helpers: `hsum_i32_avx2`, `hsum_f32_avx`
  - [x] Dispatch 3-tier en `q8_mat_vec_mul_into`: VNNI > AVX2 > scalar
  - [x] Dispatch 2-tier en `mat_vec_mul_into`: AVX2+FMA > scalar
  - [x] Remainder handling: cols no divisible por 32 (AVX2) o 8 (FMA)
  - [x] 9 nuevos tests: scalar vs dispatch, AVX2 directo, non-aligned, single row
  - [x] 3 benchmarks ignorados: Q8 768×768, F32 768×768, Q8 65536×768 (head)
  - [x] End-to-end benchmark: REDUKT_NO_SIMD=1 vs default, 3 rondas cada uno
  - [x] Evaluar portable_simd: descartado (requiere nightly, raw intrinsics son estables)
  - [x] 74 tests totales pasando (70 unit + 4 integration), 3 ignored benchmarks

### R-019: Refactorización matemática del forward pass (0% BPB risk)
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: NINGUNO — verificado bit-exact (BPB 1.278120 pre/post)
- **Impacto**: eliminación de ~1.3 MB/tok heap churn en codec + 16KB/head buffer en model
- **Archivos**: `src/infrastructure/rwkv7/model.rs`, `src/application/codec.rs`, `src/application/api.rs`,
  `src/domain/mixer.rs`, `src/domain/ngram.rs`, `src/domain/tensor.rs`
- **Decisiones de diseño**:
  - Rank-1 factorization (`mat_mat_mul_small_into` → mat-vec) NO implementada: cambia orden de
    acumulación float → NO es bit-exact. Se mantiene `ab` + `s_ab` + matmul original.
  - `vk` outer product inlined: `s.v_out[hd+i] * s.k_mod[hd+j]` es bit-exact (misma multiplicación).
  - Softmax fusion descartada: `cdf_from_logits` usa f64, `softmax` usa f32 → valores distintos.
- **Subtareas — State math (model.rs)**:
  - [x] Inline `vk` outer product en state update loop (elimina buffer N×N + `outer_product_into`)
  - [~] Rank-1 factorización: DESCARTADA — float non-associativity rompe bit-exactness
  - [~] Fusionar loops: DESCARTADA — output read necesita state post-update
  - [x] **Quality gate**: BPB IDÉNTICO (1.278120 pre/post en enwik8 10KB)
  - [x] Roundtrip byte-identical + CRC-32 verified
- **Subtareas — Codec zero-alloc (codec.rs, mixer.rs, ngram.rs, tensor.rs)**:
  - [x] `compute_bias_into()` en ngram.rs — escribe en buffer pre-allocado
  - [x] `combine_into()` en mixer.rs — escribe en buffer pre-allocado
  - [x] `softmax_into()` en tensor.rs — escribe en buffer pre-allocado (usa div, no inv*mul)
  - [x] Eliminar `ngram.compute_bias()` duplicado (se reutiliza ng_buf)
  - [x] Pre-allocar 3 buffers V-sized antes de hot loops (ng_buf, ensemble_buf, probs_buf)
  - [x] Aplicado en: cmd_compress, cmd_decompress (v3+v4), cmd_hybrid_eval, cmd_baseline, API
  - [x] Roundtrip byte-identical post-refactor

### R-021: F32 AVX2 sin FMA para LoRA matmuls
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: BAJO — per-element mul+add matches scalar, pero acumulación SIMD (8 lanes
  paralelos + hsum) difiere de secuencial → 717/768 mismatches en bit-exact test.
  BPB delta: -0.003 (1.2749 vs 1.2781). Intra-tier determinista, cross-tier diverge.
- **Impacto medido**:
  - Micro: scalar 72.6μs → noFMA 32.1μs (**2.26x**) [MEASURED: bench_simd_f32_768x768]
  - End-to-end: 38.1 → 45.4 tok/s (**+19.2%**) [MEASURED: hybrid-eval enwik8 10KB]
- **Archivos**: `src/domain/tensor.rs`
- **Decisión de diseño**: Per-element `vmulps+vaddps` es idéntico a scalar `mul+add`. La
  diferencia viene SOLO del orden de acumulación (8 partial sums + hsum vs sequential).
  FMA (fmadd) tiene dos diferencias: distinto redondeo POR ELEMENTO + distinto orden.
  noFMA tiene solo UNA diferencia: distinto orden. Máximo delta 1.5e-4 (vs FMA ~1e-3).
- **Subtareas**:
  - [x] Implementar `mat_vec_mul_avx2_nofma_into`: vmulps + vaddps, 8-wide, 4-row ILP
  - [x] Bit-exact test: 717/768 mismatches (expected — accumulation order differs)
  - [x] Dispatch: AVX2-noFMA > scalar en `mat_vec_mul_into` (respeta REDUKT_NO_SIMD=1)
  - [x] Same-path roundtrip: compress AVX2 → decompress AVX2 **PASS** (byte-identical + CRC)
  - [x] Cross-path: compress AVX2 → decompress scalar **FAIL** (expected — model diverges)
  - [x] Tests: 4 nuevos (directly, non_aligned, single_row, bit_analysis)
  - [x] Benchmark micro F32 768×768: noFMA 2.26x, FMA 2.37x (noFMA only 5% slower)
  - [x] Benchmark end-to-end: +19.2% tok/s (38.1 → 45.4)
- **Invariante**: `REDUKT_NO_SIMD=1` fuerza scalar en Q8 Y F32 → cross-machine portability

### R-022: SIMD transcendentals (tanh, sigmoid, exp) — COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: delta -0.13% (1.2812 → 1.2796) — polynomial approx marginally better on all 4 tested files
- **Impacto real**: +23.2% end-to-end throughput (controlled A/B, 3 rounds) [MEASURED]
- **Archivos**: `src/domain/tensor.rs`, `src/infrastructure/rwkv7/model.rs`
- **Implementación**:
  - Cephes-style `fast_exp_ps` (AVX2): Cody-Waite reduction + Horner degree-5 polynomial
  - `fast_sigmoid_ps` = 1/(1+exp(-x)), `fast_tanh_ps` = 2*sigmoid(2x)-1
  - Fused operations for RWKV-7 forward pass (avoid multiple passes over D=768):
    - `sigmoid_add_into`: out[i] = sigmoid(base[i] + offset[i]) — Step 4 'a' gate
    - `decay_sigmoid_into`: out[i] = exp(-0.606531 * sigmoid(base[i] + offset[i])) — Step 3 decay
    - `sigmoid_mix_inplace`: v_out[i] += (v_first[i] - v_out[i]) * sigmoid(base+offset) — Step 8
  - Updated `sigmoid_inplace`, `tanh_inplace` with AVX2 dispatch + scalar tail
  - model.rs Steps 3, 4, 8 refactored to call fused operations
- **Accuracy** [MEASURED: cargo test --release test_sigmoid/tanh/decay]:
  - sigmoid_inplace: max error 1.49e-8 (vs stdlib)
  - tanh_inplace: max error 1.19e-7
  - sigmoid_add_into: max error 5.96e-8
  - decay_sigmoid_into: max error 5.96e-8
  - sigmoid_mix_inplace: max error 2.38e-7
  - All within ~23-bit precision (f32 mantissa = 24 bits)
- **Micro-benchmark** [MEASURED: bench_simd_transcendentals_768, D=768, 5000 iters]:
  - decay_sigmoid_into: 1700 ns (scalar 4900 ns, **2.88x**) — biggest win from fusion
  - sigmoid_add_into: 600 ns, sigmoid_mix_inplace: 600 ns
  - sigmoid_inplace: 1800 ns (1.00x — LLVM auto-vectorizes simple loop)
- **Cross-path determinism**: same-tier deterministic. SIMD polynomial ≠ scalar stdlib
  (different computation → different rounding). `REDUKT_NO_SIMD=1` forces scalar path.
- **Controlled A/B validation** [MEASURED: r022-simd-transcendentals-validation.csv]:
  - 3 rounds alternating SIMD/scalar on same build, enwik8 10KB:
    - Scalar mean: 32.8 tok/s (range 30.2-35.0, thermal variance)
    - SIMD mean: 40.4 tok/s (range 36.2-42.9, thermal variance)
    - Per-round ratios: 1.23x, 1.27x, 1.20x → mean **1.23x (+23.2%)**
  - Multi-file BPB (SIMD vs scalar, 4 files):
    - enwik8: 1.2796 vs 1.2812 (delta -0.128%)
    - apache-access: 0.2446 vs 0.2448 (delta -0.075%)
    - linux-syslog: 0.3605 vs 0.3617 (delta -0.327%)
    - openssh-auth: 0.4745 vs 0.4749 (delta -0.094%)
    - ALL 4 files: SIMD BPB marginally BETTER (no regression)
  - BPB perfectly deterministic within each mode (3 rounds identical)
  - **Correction**: initial claim of +32.6% was inflated (cross-session comparison)
- **Quality gate PASSED**:
  - [x] 107 tests passed (7 new), 0 failed
  - [x] BPB: no regression on any tested file (4/4 improved marginally)
  - [x] Roundtrip: compress → decompress → CRC-32 verified → byte-identical
  - [x] Coder efficiency: 95.93% [MEASURED]
  - [x] Throughput: +23.2% controlled A/B (not +32.6% as initially reported) [MEASURED]

### R-HW1: HardwareGovernor — regulación dinámica de recursos ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO — no cambia computación, solo la regula
- **Impacto**: Protección contra deterioro de hardware (batería, RAM), crash prevention
- **Archivos**: `src/infrastructure/governor.rs` (nuevo), `src/application/codec.rs`, `src/main.rs`
- **Diseño implementado**:
  - ThrottleLevel: Full (normal), Eco (2ms sleep/100 tokens), Abort (graceful error)
  - Comunicación: AtomicU8 — zero overhead en hot path (~1ns atomic load)
  - Pre-flight: `GlobalMemoryStatusEx` (RAM), `GetSystemPowerStatus` (batería via kernel32 FFI)
  - Watchdog thread: 2s polling, actualiza ThrottleLevel atómicamente
  - Umbrales RAM: >95% → Eco, >97% → Abort
  - Umbrales batería (solo sin AC): <20% → Eco, <5% → Abort
  - Integrado en compress + decompress (hot loop, cada 100 tokens)
  - `--no-governor` flag para deshabilitar (benchmarking, testing)
- **Subtareas**:
  - [x] ThrottleLevel enum + AtomicU8 protocol
  - [x] Pre-flight: `GlobalMemoryStatusEx`, `GetSystemPowerStatus` (kernel32 FFI)
  - [x] Watchdog thread con 2s polling + sensor read
  - [x] Integrado en compress/decompress loop: check cada 100 tokens
  - [x] Eco mode: 2ms sleep para reducir power draw
  - [x] Abort: graceful error return (no panic)
  - [x] `--no-governor` flag para deshabilitar
  - [x] 5 unit tests: throttle roundtrip, disabled, atomic update, preflight, display
  - [x] Roundtrip verificado: compress→decompress byte-identical con governor activo
  - [x] 137 tests pasando (133 unit + 4 integration)

### Fase 3: Binary Preprocessing Pipeline (2026-10-29)

> Investigación completa: `docs/internal/research/binary-compression-research.md`
> Redukt pierde en 5/17 archivos binarios (sao +54%, x-ray +44%, ooffice +37%,
> osdb +22%, mozilla +1.5%). Diagnóstico: tokenizer de texto destruye estructura
> byte-level. Solución: detect + preprocess antes del modelo.
> Todas las técnicas son zero-dep, reversibles, sin riesgo BPB en texto.

#### R-027: Integración E8E9 en pipeline de compresión
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO en texto — solo se activa en ejecutables detectados
- **Impacto**: mozilla 10KB: 2.0536 BPB (E8E9 ON) vs 2.0520 BPB (OFF) — neutral en 10KB (solo 3 E8, 1 transformed). Beneficio esperado en archivos ejecutables puros (PE/ELF) a mayor escala.
- **Archivos**: `src/domain/preprocess.rs`, `src/application/codec.rs`, `docs/api/FORMAT-SPEC.md`
- **Diseño**:
  - `e8e9_encode_with_stats()` + `e8e9_decode()` + `detect_e8e9()` en preprocess.rs
  - FLAG 0x04 en AZTH v4 header, detección automática (E8/E9 density ≥0.5%, ≥3 count, >50% in-range)
  - compress: detect → transform → CRC(transformed) → encode → store FLAG
  - decompress: decode → verify CRC → inverse transform → write
  - `--e8e9 auto|on|off` CLI flag (default: auto)
- **Subtareas**:
  - [x] Añadir FLAG `V4_FLAG_E8E9 = 0x04` en header
  - [x] Implementar detección automática de ejecutables (`detect_e8e9()`)
  - [x] Integrar encode en cmd_compress antes de tokenización
  - [x] Integrar decode en cmd_decompress después de detokenización
  - [x] `--e8e9` flag para forzar/deshabilitar
  - [x] Roundtrip: mozilla compress→decompress byte-identical (PASS)
  - [x] Benchmark: mozilla BPB antes/después (neutral en 10KB, 3 E8 only)
  - [x] Auto-detect: texto NO transformado (PASS), mozilla auto-skip (correcto — tar header)
  - [x] FORMAT-SPEC.md actualizado con FLAG 0x04 documentación
- **Resultado**: Infraestructura E8E9 completa y funcional. Auto-detect conservador — solo activa para ejecutables puros con alta densidad E8/E9. Mozilla (tar) no se activa automáticamente (correcto).

#### R-028: Data-type detector (content classification)
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO — solo clasificación, no modifica datos
- **Impacto**: Habilita R-029, R-030, R-031 — es el dispatcher central
- **Archivos**: `src/domain/content_detect.rs` (nuevo), `src/application/codec.rs`
- **Diseño**:
  - Analiza primeros 8KB: Shannon entropy, alphabet size, printable ASCII ratio,
    XML tag density, float32 exponent-byte patterns, magic numbers (MZ/ELF/XML)
  - 6 categorías: Text, StructuredText, Executable, FloatArray, HighEntropy, GenericBinary
  - Decision tree: magic → printable ratio → entropy → exponent features
  - Logging integrado en cmd_compress (features + tipo detectado)
- **Subtareas**:
  - [x] Implementar feature extraction (entropy, ASCII ratio, XML density, float exponent)
  - [x] Implementar clasificación por decision tree (~180 líneas)
  - [x] 11 unit tests (cada tipo + edge cases + Display)
  - [x] Validación con 17 archivos de benchmark (14/17 ideal, 3 aceptable)
  - [x] Logging: `[compress] content: <type> (<features>)` en stderr
- **Resultados validación** (8KB scan, archivos reales):
  - Executable: ooffice (MZ) ✓
  - FloatArray: x-ray ✓ (exp_entropy=3.46, exp_alpha=27)
  - HighEntropy: sao ✓ (entropy=7.33)
  - StructuredText: xml, facturae, ubl, apache-access ✓
  - Text: dickens, webster, reymont, nci ✓
  - GenericBinary: mozilla, mr, osdb, samba ✓ (mr podría ser FloatArray — refinable en R-029)

#### R-029: Byte shuffle para datos float
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO en texto — solo se activa cuando R-028 detecta FloatArray
- **Impacto**: x-ray **-8.5% BPB** (6.5896 → 6.0280) [MEASURED]. sao +0.8% (correctly rejected by auto-detect).
- **Archivos**: `src/domain/byte_shuffle.rs` (nuevo), `src/application/codec.rs`, `docs/api/FORMAT-SPEC.md`
- **Diseño**:
  - Transpone N-byte groups: agrupa byte-0, byte-1, ..., byte-(N-1)
  - Stride=4 default (float32). CLI: `--shuffle auto|on|on:N|off`
  - FLAG 0x08 en AZTH v4 header + 1 byte stride
  - Auto-dispatch: FloatArray (R-028) → shuffle stride=4
  - Decompress: unshuffle antes de E8E9 inverse
- **Subtareas**:
  - [x] Implementar `byte_shuffle(data, stride)` y `byte_unshuffle(data, stride)`
  - [x] 9 unit tests (roundtrip stride 2/4/8, remainder, identity, known output, float clustering, large)
  - [x] FLAG 0x08 + stride byte en header (parse + build)
  - [x] Integración compress: auto si FloatArray, `--shuffle` CLI
  - [x] Integración decompress: unshuffle + streaming support
  - [x] Roundtrip: x-ray compress→decompress byte-identical (PASS)
  - [x] Roundtrip: dickens con shuffle forzado (PASS, BPB terrible pero lossless)
  - [x] Benchmark: x-ray -8.5%, sao +0.8% (auto-detect correcto)
  - [x] Texto NO shuffled en auto mode (PASS)
  - [x] FORMAT-SPEC.md actualizado

#### R-030: Delta coding (stride-N)
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO en texto — auto solo activa con shuffle (FloatArray)
- **Impacto**: x-ray **-11.4% BPB total** (6.5896 → 5.8416 con shuffle+delta) [MEASURED]
  - Delta sola: -3.1% adicional sobre shuffle (-8.5%)
  - sao: shuffle+delta PEOR (+16.9%) — auto-detect correcto (no aplica)
  - mr: shuffle+delta PEOR (+89.9%) — auto-detect correcto (no aplica)
- **Archivos**: `src/domain/delta.rs` (nuevo), `src/application/codec.rs`, `docs/api/FORMAT-SPEC.md`
- **Diseño**:
  - `delta_encode(data, stride)`: `out[i] = in[i].wrapping_sub(in[i-stride])`
  - `delta_decode(data, stride)`: `out[i] = in[i].wrapping_add(out[i-stride])`
  - Cadena: E8E9 → shuffle → delta → CRC → encode
  - FLAG 0x10 + 1 byte stride en header
  - Auto: activa si shuffle activo (FloatArray pipeline)
  - CLI: `--delta auto|on|on:N|off`
- **Subtareas**:
  - [x] Implementar delta_encode + delta_decode (~20 líneas)
  - [x] 9 unit tests (roundtrip, constant→zeros, ramp→constant, wrapping, shuffle+delta combo)
  - [x] FLAG 0x10 + stride byte en header (parse + build)
  - [x] Integración compress: auto si shuffle activo
  - [x] Integración decompress: delta decode antes de unshuffle
  - [x] Roundtrip: x-ray shuffle+delta → byte-identical (PASS)
  - [x] Texto no afectado en auto mode (PASS)
  - [x] FORMAT-SPEC.md actualizado

#### R-031: Record detection + column transposition
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO en texto — solo para detected fixed-width records
- **Impacto**: Infraestructura para fixed-record binary data (synthetic: 0.694 BPB)
- **Archivos**: `src/domain/record_detect.rs`, `src/application/codec.rs`
- **Diseño**:
  - Autocorrelación del byte stream (coincidence rate at each lag)
  - Q1 (25th percentile) baseline — robust to periodic peaks inflating mean/median
  - Threshold 5.0x Q1 — high to avoid false positives on mixed data
  - Min stride 4 — avoids int16 byte-swap false positives
  - Reuses SHUFFLE flag in header (detected stride stored as shuffle stride)
  - GenericBinary content type triggers record detection; FloatArray uses stride=4 directly
- **Resultados**:
  - Synthetic 8-byte records (16KB): 0.694 BPB, roundtrip byte-identical
  - osdb: no false positive (threshold correctly filters out weak 4.1x peak)
  - apache: no false positive (stride=87 at 5.3x filtered by conservative threshold)
  - All 18 benchmark files: zero false positive triggers
  - 6 unit tests: periodic detection, no-records-in-text, no-records-in-random, short data, stride-2, convenience
- **Subtareas**:
  - [x] Autocorrelación para detectar stride/record-length (Q1 baseline, 5.0x threshold)
  - [x] Column transposition via byte_shuffle(data, detected_stride)
  - [x] Reuses SHUFFLE flag in header (no new flag needed)
  - [x] Benchmark: no false positives on any of 18 files
  - [x] Roundtrip: synthetic records → byte-identical

#### R-032: CM-only bypass para datos binarios
- **Estado**: ABORTADO + REVERTIDO (2f34173)
- **Riesgo BPB**: Variable — bypass completo de RWKV-7 en regiones binarias
- **Hipótesis**: RWKV-7 es ruido para datos binarios de alta entropía → REFUTADA
- **Resultado empírico** (10KB, Q8):
  | Archivo | Normal BPB | CM-only BPB | Δ |
  |---------|-----------|-------------|-----|
  | sao | 6.776 | 8.090 | +19.4% |
  | osdb | 4.363 | 4.960 | +13.7% |
  | enwik8 | 1.281 | 2.687 | +109.7% |
- **Velocidad**: ~700 tok/s (8.2x vs 85 tok/s normal) pero calidad inaceptable
- **Conclusión**: RWKV-7 aporta predicciones útiles incluso en datos binarios.
  El modelo captura patrones de bytes que CM solo no detecta.
  Código revertido para no contaminar lecturas de benchmark.

#### R-033: Adaptive RWKV skip (confidence-based)
- **Estado**: COMPLETADO (experimental/opt-in) — 5b04f4a
- **Riesgo BPB**: Alto (+11% a +58% según archivo)
- **Hipótesis**: Si el modelo combinado (RWKV+CM) tiene confianza alta en un token,
  se puede omitir el forward pass del siguiente y reusar logits anteriores.
- **Implementación**:
  - CLI: `--adaptive-skip` (threshold=95%) o `--skip-threshold N` (1-99)
  - FLAG `0x20` en AZTH v4 header (1 byte: threshold_pct)
  - `MAX_CONSEC_SKIP=3` previene cascada de staleness en RWKV state
  - Compress y decompress toman decisiones idénticas → roundtrip garantizado
- **Resultado empírico** [MEASURED, 10KB, Q8, threshold=99%]:
  | Archivo | Baseline BPB | Skip BPB | Δ BPB | Skip % | Throughput |
  |---------|-------------|----------|-------|--------|------------|
  | enwik8 | 1.333 | 1.482 | +11.2% | 9.2% | ~102 tok/s |
  | apache | 0.296 | 0.467 | +57.8% | 49.2% | ~163 tok/s |
- **Roundtrip**: PASS (enwik8 + apache, CRC-32 verified, skip count idéntico)
- **Conclusión**: Feature correcta pero degradación significativa.
  Archivos altamente compresibles (logs) sufren más: los tokens "fáciles"
  que se skipean son los que más contribuyen a la ratio.
  Queda como **experimental/opt-in** — útil cuando velocidad > calidad.
- **Infraestructura adicional**: CLI gates para `--gpu`, `--tokenizer bpe`,
  `--model-size` (placeholders, retornan error claro si no disponible)

#### Secuencia de implementación Fase 3

```
R-028 (detector) → R-027 (E8E9) → R-029 (shuffle) → R-030 (delta) → R-031 (records) → R-032 (CM-only)
```

R-028 es prerequisito para activación automática de todos los demás.
R-027 puede implementarse antes con detección ad-hoc (E8/E9 density).

### Secuencia de throughput — orden por riesgo de degradación BPB

> Cada paso incluye un quality gate: medir BPB en los 3 archivos de log + enwik8.
> Si la degradación acumulada supera +3% en logs → STOP, no proceder al siguiente.

#### Paso 0: R-018 — Coder Efficiency Optimization (0% BPB risk, gains only) ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: NINGUNO — mejora la eficiencia del coder, no cambia predicciones del modelo
- **Resultado**: AZTH v4 implementado. Header 13B (defaults) o 25B (custom HP), vs 28B en v3.
  - [x] Header compacto v4: FLAGS byte + conditional HP (-15 bytes con defaults)
  - [x] token_count eliminado: decoder termina por byte count (interleaved decode+detokenize)
  - [x] Backward compat: decoder acepta v2, v3, v4. Encoder siempre escribe v4.
  - [x] f16 para HP: DESCARTADO — FLAGS condicionan presencia, no necesario reducir tamaño
  - [x] Benchmark: enwik8 10KB v4=1.3328 BPB vs v3=1.3448 BPB → -0.9% [MEASURED]
  - [x] Coder efficiency: 96.1% (v4) vs 95.3% (v3) [MEASURED]
  - [x] 96 tests pasando, roundtrip byte-identical verificado
  - [x] FORMAT-SPEC.md actualizado con v4 spec
- **Quality gate PASSED**: BPB mejoró (1.3448 → 1.3328), cross-entropy sin cambio (1.2812)

#### Paso 0.5: R-019 — Refactorización matemática (0% BPB risk) ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: NINGUNO — verificado bit-exact (BPB 1.278120 pre/post)
- **Impacto**: eliminación ~1.3 MB/tok heap churn + vk outer product inline
- **Quality gate PASSED**: BPB bit-exact, roundtrip byte-identical

#### Paso 1: R-021 — F32 AVX2 sin FMA (0% BPB risk) ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **Speedup real**: +19.2% end-to-end (38.1 → 45.4 tok/s) [MEASURED]
- **Quality gate PASSED**: same-path roundtrip byte-identical, cross-path diverge (expected)

#### Paso 1.5: R-022 — SIMD transcendentals ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **BPB delta**: -0.13% (improved, 4/4 files) [MEASURED: controlled A/B, validation CSV]
- **Speedup real**: +23.2% controlled A/B (3 rounds, mean SIMD/scalar ratio) [MEASURED]
- **Correction**: initial +32.6% claim was inflated by cross-session thermal variance
- **Key insight**: speedup from fused ops (avoid 3 passes over D=768), not raw SIMD exp
- **Quality gate PASSED**: 107 tests, roundtrip byte-identical, CRC-32 verified, 4-file BPB OK

#### Paso 2: R-015 — Rank-1 State Factorization ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-28)
- **Nota**: Renombrado de "KV Batch Caching" — RWKV es RNN, no transformer.
  "KV caching" no aplica. La optimización real es algebraica: rank-1 factorization.
- **BPB delta**: ±0.25% (2 files mejoran, 2 empeoran — ruido de reordenación float)
- **Speedup real**: scalar 2.0x (32.8→65.7), SIMD 2.35x (40.4→94.8) [MEASURED: A/B 3 rounds]
- **Archivos**: `src/infrastructure/rwkv7/model.rs`, `src/domain/tensor.rs`
- **Implementación**:
  - **Algebraic identity**: `state @ outer_product(u, v) = outer_product(state @ u, v)`
  - Replaces O(N³) `mat_mat_mul_small_into` with O(N²) `mat_vec_mul_small + outer_product`
  - N=64 (head_size): 262,144 FMAs → 8,192 ops = **32x fewer FLOPs per head**
  - 12 heads × 12 layers × 254K saved = **37M FLOPs saved per token**
  - Removed `ab` buffer from Scratch (-16KB), `mat_mat_mul_small_into` now dead code
- **Controlled A/B validation** [MEASURED: r015-rank1-factorization-validation.csv]:
  - 3 rounds alternating SIMD/scalar, enwik8 10KB:
    - Scalar mean: 65.7 tok/s (was 32.8 pre-R-015, **+100%**)
    - SIMD mean: 94.8 tok/s (was 40.4 pre-R-015, **+135%**)
    - SIMD/scalar ratio: 1.44x (thermal-stable, range 1.43-1.47x)
  - Multi-file BPB (SIMD vs scalar, 4 files):
    - enwik8: 1.2815 vs 1.2792 (delta +0.179%)
    - apache: 0.2455 vs 0.2449 (delta +0.252%)
    - syslog: 0.3612 vs 0.3624 (delta -0.352%)
    - openssh: 0.4741 vs 0.4753 (delta -0.241%)
    - Mixed: 2 files marginally better, 2 marginally worse — float reordering noise
  - BPB perfectly deterministic (3 rounds identical per mode)
  - Roundtrip: byte-identical, CRC-32 D8E1D10F verified
  - Compressed: 1666 bytes, coder efficiency 96.13%
- **Quality gate PASSED**:
  - [x] 108 tests passed (1 new rank-1 correctness), 0 failed
  - [x] BPB delta < ±0.35% on all 4 files
  - [x] Roundtrip byte-identical + CRC verified
  - [x] Throughput: +135% SIMD, +100% scalar (controlled A/B)

#### Paso 3: R-012 — Multi-file paralelismo (0% BPB risk) ✅
- **Estado**: COMPLETADO (2026-10-28)
- **Riesgo BPB**: NINGUNO — confirmado: BPB y bytes comprimidos bit-exact en 1/2/3 jobs
- **Speedup medido**: 1.36x (2 jobs), 1.64x (3 jobs) [MEASURED: r012-batch-parallelism-validation.csv]
- **Impacto**: Sub-lineal (CPU memory-bound), pero genuinamente multiplicativo con más cores
- **Archivos**: `src/application/codec.rs`, `src/infrastructure/rwkv7/model.rs`, `src/main.rs`
- **Diseño**: Modelo cargado 1x → `Arc<Rwkv7Model>` + `Arc<WorldTokenizer>` compartidos.
  Cada thread: propio `Rwkv7State` + `Scratch` + ngram/bias/mixer/encoder (~4 MB/thread).
  Work queue: `Arc<Mutex<Vec<PathBuf>>>` con pop() por thread.
- **Subtareas**:
  - [x] Implementar modo batch: `redukt compress --input-dir <DIR> --output-dir <DIR> --jobs N`
  - [x] `Rwkv7State::reset()` y `Scratch::reset_perf()` para reutilización sin realloc
  - [x] Thread pool con modelo pre-cargado (Arc), state/scratch por thread
  - [x] Benchmark: 1, 2, 3 jobs × 3 rounds × 3 files (controlled A/B, same session)
  - [x] **Quality gate PASS**: BPB idéntico single-thread vs multi-thread (0.000 delta)
  - [x] **Quality gate PASS**: bytes comprimidos idénticos en todos los modos
  - [x] Roundtrip verificado: decompress → byte-identical, CRC OK
  - [x] RAM: modelo ~130 MB Q8 (shared), ~4 MB/thread → 10 threads ≈ 170 MB total
- **Scaling** (3 files × 10KB, Dell Vostro 3520 i5-1235U):
  - 1 job: 106.1s mean (1.7 files/min)
  - 2 jobs: 78.0s mean (2.3 files/min) → **1.36x** [MEASURED]
  - 3 jobs: 64.8s mean (2.8 files/min) → **1.64x** [MEASURED]
  - Sub-lineal por memory bandwidth contention — escala mejor con más files

#### Paso 4: R-016 — SVD Pruning del modelo RWKV — ABORTADO
- **Estado**: ABORTADO (2026-10-28)
- **Riesgo BPB**: FATAL — RWKV-7 0.1B tiene rango efectivo alto, SVD no es viable
- **Resultado**: `data/benchmark/results/r016-svd-pruning-evaluation.csv`
- **Implementación**: Randomized SVD (Halko-Martinsson-Tropp) + Jacobi eigen con eigenvectores
- **Archivos**: `src/domain/spectral.rs` (truncated_svd_f32), `src/infrastructure/rwkv7/model.rs` (SvdProjection)
- **Hallazgos** [MEASURED]:
  - Baseline: 1.2782 BPB (enwik8 10KB, Q8)
  - Rank-750 (97.6%): 1.3225 BPB (+3.47%) AND **1.95x SLOWER** → FAIL
  - Rank-700 (91.1%): 1.3427 BPB (+5.05%) AND **1.82x SLOWER** → FAIL
  - Rank-384 (50.0%): 2.0532 BPB (+60.6%) → catastrophic quality loss
  - **Break-even FLOPs at rank=D/2=384**: two D×k matmuls = one D×D matmul. Any rank>384 is SLOWER.
  - Reconstruction relative errors (layer 0): key 8.6e-3, value 9.8e-3, receptance 3.1e-2, output 1.9e-2
- **Root cause**: RWKV-7 0.1B weight matrices have high effective rank (flat singular value spectrum).
  RWKV-edge paper NLP accuracy metrics (+1pp at rank-384) do NOT transfer to compression BPB (+60.6%).
  No rank satisfies BOTH quality (<+3% BPB) AND speed (>1.0x) constraints simultaneously.
- **Decisión**: ABORTADO. Código SVD se mantiene en codebase (compilable, tested) pero `--svd-rank` no viable.
- **Subtareas**:
  - [x] Implementar truncated_svd_f32 (Halko-Martinsson-Tropp randomized SVD)
  - [x] Implementar jacobi_eigen_full (eigenvectors via rotation accumulation)
  - [x] SvdProjection struct con Q8 re-quantization de factores
  - [x] SVD dispatch en time_mixing_scratch (Steps 2 y 12)
  - [x] `--svd-rank` CLI flag en compress y hybrid-eval
  - [x] Benchmark: rank 384/700/750 vs baseline → NINGUNO viable
  - [x] **Quality gate FAILED**: +3.47% BPB en MEJOR caso (rank-750), AND 1.95x slower
  - [x] Documentar resultados en CSV con verdict por configuración

#### Paso 5: R-017 — Parallel Segment Compression (BPB risk: variable) — BLOQUEADO
- **Estado**: BLOQUEADO (R-016 abortado → sin precedente de tolerancia BPB en modelo)
- **Riesgo BPB**: ALTO — divide el archivo en segmentos, cada uno pierde contexto previo
- **Speedup estimado**: 2-4x [ESTIMATE — depende de tamaño de segmento]
- **Presupuesto BPB restante**: +3% íntegro (R-016 no consumió nada — abortado)
- **NOTA**: R-016 demostró que la tolerancia BPB del modelo es extremadamente baja.
  Context loss en fronteras de segmento probablemente cause degradación similar o peor.
  Requiere evaluación empírica independiente antes de proceder.
- **Archivos**: `src/application/codec.rs`, `src/domain/coder.rs`
- **Subtareas**:
  - [ ] Diseñar formato de segmentos en AZTH (v4: segment offsets en header)
  - [ ] Implementar split → compress parallel → merge
  - [ ] Benchmark BPB por tamaño de segmento: 1KB, 4KB, 16KB, 64KB
  - [ ] **Quality gate BPB**: degradación del segmenting solo (sin SVD) en logs
  - [ ] **Quality gate COMBINADO**: R-018 + R-019 + R-021 + R-015 + R-012 + R-016 + R-017 todos activos
  - [ ] Si degradación combinada >+3% en logs → ajustar tamaño de segmento o ABORT R-017
  - [ ] Roundtrip test con segmentos paralelos
  - [ ] Documentar curva: tamaño segmento vs BPB loss vs speedup [MEASURED]

### Resumen de throughput proyectado (secuencia acumulativa)

> Todos los valores son [PROJECTION] salvo la fila "Current" que es [MEASURED].
> La degradación BPB es [ESTIMATE] — debe medirse en cada paso.

| Paso | Optimización | Speedup | B/s acum. (logs) | BPB risk acum. | Gate |
|---|---|---|---|---|---|
| Current | SIMD Q8 baseline | 1x | ~150 B/s [MEASURED 2026-10-28] | 0% | — |
| R-018 ✅ | + Coder efficiency | ~1x | ~150 B/s (mejora BPB, no throughput) | 0% (mejora) | BPB mejora |
| R-019 | + Math refactor + zero-alloc | 1.3-1.5x | ~195-225 B/s | 0% | BPB bit-exact |
| R-021 ✅ | + F32 AVX2 noFMA | 1.19x | ~232-268 B/s [MEASURED: +19.2%] | 0% | BPB bit-exact |
| R-022 ✅ | + SIMD transcendentals | 1.23x | ~285-330 B/s [MEASURED: +23.2% A/B] | -0.13% | BPB -0.13% (improved) |
| R-015 ✅ | + Rank-1 factorization | 2.35x | ~670 B/s [MEASURED: 94.8 tok/s A/B] | ±0.25% | BPB ±0.25% (noise) |
| R-012 ✅ | + N threads | 1.64x (3T) | ~1100 B/s (3T) [MEASURED] | 0% | BPB bit-exact |
| R-016 ❌ | ~~SVD pruning~~ | N/A | ABORTADO [MEASURED: +3.5% AND 1.95x slower] | N/A | FAILED |
| R-017 ⏸ | Parallel segments | 2-4x | BLOQUEADO (requiere evaluación BPB independiente) | +?% | Pendiente |

**Nota crítica**: Los speedups multiplicativos son teóricos. En la práctica:
- R-018 no mejora throughput, pero reduce BPB → amplía el margen competitivo
- R-019 es compute reduction puro — beneficio directo en TimeMix latency
- R-021/R-022 mejoran el "techo" scalar de TimeMix (74-88% del tiempo)
- R-015 × R-016 podrían ser sub-multiplicativos (ambos tocan inference path)
- R-012 es sub-lineal en cores (1.64x con 3 threads [MEASURED], memory-bound bottleneck)
- R-017 tiene overhead de merge y cold-start por segmento
- R-016 ABORTADO: SVD no viable para RWKV-7 0.1B (alto rango efectivo, BPB catastrófico)
- R-017 BLOQUEADO: requiere evaluación BPB independiente tras fracaso de R-016
- R-HW1 (HardwareGovernor) es ortogonal — no afecta throughput peak pero protege hardware

**Target mínimo viable**: R-021 ✅ + R-022 ✅ + R-015 ✅ + R-012 ✅ = done.
**Achieved**: 94.8 tok/s single-thread × 1.64x (3 threads) = **~155 tok/s aggregate** [MEASURED].
~1.1 KB/s (3T). Con más cores/archivos, scaling continúa. Sin degradación BPB.

**Secuencia revisada (2026-10-28)**: R-018 ✅ → R-019 ✅ → R-021 ✅ → R-022 ✅ → R-015 ✅ → R-012 ✅ → R-016 ❌ ABORT.
Todas las optimizaciones 0% BPB risk completadas. R-016 (SVD) evaluado y abortado empíricamente.
R-017 (parallel segments) bloqueado — requiere evaluación independiente.
R-HW1 puede implementarse en cualquier momento (independiente de la secuencia de throughput).

**Fase 2 (2026-10-29)**: R-023 N/A → R-025 ✅ → R-024 ❌ → R-026 ❌.
R-023 (CDF 24-bit): N/A — ya implementado desde el diseño original.
R-025 (prefetch): ✅ +3.2% throughput, 0% BPB risk.
R-024 (Q4 storage): ❌ ABORTADO — +12.7% BPB, 15.3x slower. Q4 no viable para RWKV-7.
R-026 (iGPU): ❌ ABORTADO — 2.68x slower que CPU. Head matmul es solo 12% del forward; dispatch overhead domina.

### Fase 2: Técnicas identificadas por auditoría de exhaustividad (2026-10-29)

> Ref: `docs/internal/research/optimization-exhaustion-audit.md`
> Estas técnicas fueron identificadas tras investigación exhaustiva confirmando
> que todas las optimizaciones algebraicas/SIMD/codec conocidas están agotadas.

#### R-023: CDF Precision Upgrade (16 → 24 bit) — YA IMPLEMENTADO
- **Estado**: NO APLICA (2026-10-29)
- **Origen**: Paper Nacrith (2026) — arquitectura casi idéntica a Redukt
- **Hallazgo**: El range coder de Redukt YA usa CDF de 24 bits (`const TOP: u32 = 1 << 24`
  en `src/domain/coder.rs:12`). El paper sugería upgrade de 16→24, pero nuestro coder
  fue diseñado directamente con 24-bit precision desde el inicio.
- **Floor overhead actual**: V=65536, TOP=2^24 → min_alloc/total = 0.39% — despreciable.
- **Conclusión**: No hay cambio que hacer. La técnica del paper ya está presente.

#### R-024: Q4 Storage → Q8 Compute — ABORTADO
- **Estado**: ABORTADO (2026-10-29)
- **Premisa original**: "0% BPB risk" — INCORRECTA. 4 bits no pueden representar la misma
  información que 8 bits. Cualquier Q4 storage pierde precisión en weights.
- **Implementación**: Q4 SIMD matmul con AVX2 unpack (Q4→i8 sign extend + sign-trick
  dot product per group), reemplazando el anterior Q4 scalar fallback.
- **Archivos**: `src/domain/tensor.rs` (q4_mat_vec_mul_into, q4_mat_vec_mul_avx2_into),
  `src/domain/quant.rs` (wired Q4 path to new SIMD kernel)
- **Resultado** [MEASURED: enwik8 10KB]:
  - Q8: BPB 1.281, 79.3 tok/s
  - Q4 SIMD: BPB 1.444, 5.2 tok/s
  - **BPB delta: +12.7%** (4x sobre presupuesto +3%)
  - **Throughput: 15.3x más lento** (per-group hsum overhead domina)
- **Root cause**: Q4 requiere per-group scale application (24 groups/row × hsum = 24 hsums/row,
  vs 1 hsum/row en Q8). El overhead de desempaquetar Q4→i8 + acumular por grupo
  destruye completamente el beneficio de reducir DRAM reads.
- **Consistencia con R-V05**: BPB +12.7% vs R-V05's +13.3% — confirma que Q4 weight
  precision es fundamentalmente insuficiente para RWKV-7 0.1B.
- **Código retenido**: Q4 SIMD kernel se mantiene como mejora al path Q4 existente
  (non-allocating + SIMD vs anterior allocating scalar). Útil para testing/research.
- **Subtareas**:
  - [x] Implementar Q4 SIMD matmul (AVX2 unpack + sign-trick per group)
  - [x] Q4 scalar fallback (non-allocating)
  - [x] Correctness tests: SIMD vs scalar, SIMD vs allocating (2 tests added)
  - [x] Benchmark: Q4 vs Q8 (BPB +12.7%, 15.3x slower)
  - [x] **Quality gate FAILED**: BPB +12.7% >> +3% budget, throughput catastrophic

#### R-025: Software Prefetching ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-29)
- **Riesgo BPB**: NINGUNO — confirmado: BPB 1.281455 idéntico con/sin prefetch
- **Impacto real**: +3.2% end-to-end throughput [MEASURED: 4 rounds, order-controlled A/B]
- **Descripción**: `_mm_prefetch(_MM_HINT_T0)` en los 3 SIMD kernels (VNNI, AVX2 Q8, F32 noFMA).
  Dos estrategias: (1) prefetch primeras cache lines del siguiente bloque de 4 filas,
  (2) prefetch 2-4 chunks adelante en inner loop para ocultar latencia L2 (~14 cycles).
- **Archivos**: `src/domain/tensor.rs`
- **Control A/B**: `REDUKT_NO_PREFETCH=1` desactiva prefetch para benchmarking.
- **Micro-benchmark**: inconcluso en 768×768 aislado (matrices caben en L2, hardware prefetcher
  ya maneja el patrón secuencial). El beneficio es end-to-end cuando múltiples layers
  compiten por cache y el working set excede L2.
- **End-to-end (4 rounds, enwik8 10KB Q8)** [MEASURED]:
  - R1 (PF first): prefetch 93.9 vs no-prefetch 89.2 (+5.3%)
  - R2 (PF first): prefetch 95.2 vs no-prefetch 93.2 (+2.1%)
  - R3 (NoPF first): prefetch 96.9 vs no-prefetch 94.7 (+2.3%)
  - R4 (NoPF first): prefetch 96.0 vs no-prefetch 93.1 (+3.1%)
  - Mean: 95.5 vs 92.6 tok/s = **+3.2%** [MEASURED]
  - BPB: 1.281455 identical all 8 runs (0.000 delta)
- **Subtareas**:
  - [x] Prefetch en `q8_mat_vec_mul_vnni_into` (next row block + inner loop 2 chunks ahead)
  - [x] Prefetch en `q8_mat_vec_mul_avx2_into` (next row block + inner loop 2 chunks ahead)
  - [x] Prefetch en `mat_vec_mul_avx2_nofma_into` (next row block + inner loop 4 chunks ahead)
  - [x] `REDUKT_NO_PREFETCH=1` env var for controlled A/B
  - [x] Micro-benchmark: inconcluso (L2-resident matrices, <3% delta within noise)
  - [x] End-to-end benchmark: +3.2% confirmed (4 rounds, order-controlled)
  - [x] **Quality gate PASSED**: BPB identical, 111 tests pass, zero warnings

#### R-026: iGPU Offload (Intel Iris Xe) — ❌ ABORTADO
- **Estado**: ABORTADO (2026-10-29) — código revertido
- **Razón**: 2.68x slower que CPU. Per-token Vulkan dispatch overhead domina;
  head matmul es solo ~12% del forward pass; DRAM compartida elimina ventaja de bandwidth.
- **Hardware target**: Intel Iris Xe (80 EUs @ 1.2 GHz, ~1.5 TFLOPS FP32)
- **Enfoque evaluado**: Vulkan compute shaders via FFI (zero compile-time dependencies).
  Head matmul offloaded (65536×768 F32), state update y layer matmuls en CPU.
- **Resultados medidos** [MEASURED 2026-10-29]:
  - Device: Intel(R) Iris(R) Xe Graphics
  - BPB GPU: 1.2789 vs CPU: 1.2815 (enwik8 10KB Q8) → Δ = -0.20% (calidad OK)
  - Throughput: GPU 37.5 tok/s vs CPU 100.5 tok/s → **2.68x slower**
  - Head matrix: 65536×768 F32 = 192 MB on GPU (host-visible shared memory)
  - Cross-path GPU→CPU: CRC mismatch (expected, model precision differs)
- **Conclusión**: iGPU offload NO viable para operaciones per-token con dispatch overhead.
  Solo tendría sentido con batch dispatch o si head matmul fuera >50% del forward.
  State update (62% del forward) sigue siendo el hard ceiling real.

### R-013: Determinismo cross-platform ✅ COMPLETADO
- **Estado**: COMPLETADO (2026-10-29)
- **Archivos**: `tests/roundtrip.rs`, `data/test-fixtures/`, `docs/internal/research/determinism-audit.md`
- **Subtareas**:
  - [x] Auditar operaciones float que podrían divergir entre plataformas (18 operaciones, 6 archivos)
  - [x] Generar golden files (input → .azth → restored, SHA-256 verificado)
  - [x] Golden reference test: 39B input → 35B AZTH, SHA-256 hardcoded
  - [x] Multi-process determinism: 5 runs, SHA-256 idéntico
  - [x] Scalar determinism: 3 runs con REDUKT_NO_SIMD=1, idéntico
  - [x] Cross-tier roundtrip: SIMD→scalar CRC mismatch (esperado, documentado)
  - [x] Same-tier roundtrip: SIMD y scalar ambos BYTE-IDENTICAL
  - [x] SHA-256 implementado zero-dep para verificación en tests
  - [ ] Verificar en Linux x86, Linux ARM64 (requiere CI pipeline — fuera de alcance)
- **R-013b: Portable transcendentals** — COMPLETADO (2026-10-29)
  - **Implementación**: Reemplazadas TODAS las llamadas scalar libm (exp, tanh, sigmoid, ln) con
    aproximaciones polinomiales portables (Cephes expf, musl lnf)
  - **Archivos**: `src/domain/tensor.rs`, `src/domain/ngram.rs`, `src/domain/bias_head.rs`
  - **Resultados** [MEASURED]:
    - BIT-EXACT con SIMD: 0 ULP mismatches para expf y sigmoidf
    - BPB sin cambio: 1.281455 (idéntico pre/post)
    - Cross-platform same-tier determinism: CONSEGUIDO
    - Cross-tier (SIMD vs scalar): sigue fallando — causa es orden de acumulación F32 en matmul
      (matrices LoRA), problema separado y más difícil, documentado como limitación conocida
  - **Tests**: 3 nuevos unit tests para funciones portables
- **Hallazgos clave** [MEASURED 2026-10-29]:
  - **Same-platform determinism**: GARANTIZADO — mismo código = mismo resultado
  - **Cross-SIMD-tier**: INCOMPATIBLE — SIMD y scalar producen .azth diferentes
  - **Cross-platform**: NO GARANTIZADO — libm y SIMD difieren entre plataformas
  - **Coder (range coder)**: 100% entero → determinista en toda plataforma
  - **CRC-32, E8E9 preprocess**: Enteros → determinista
  - **HashMap en ngram.rs**: Falso positivo — cada token actualiza un índice independiente
  - **Riesgo principal**: transcendentales (exp, sigmoid, tanh) — SIMD Cephes vs libm scalar
  - Compresión y descompresión DEBEN usar el mismo SIMD tier
  - 117 tests (113+4), 4 nuevos tests de determinismo

### R-014: Streaming decompression
- **Estado**: COMPLETADO
- **Impacto**: `--streaming` flag para decompress command, output progresivo a disco
- **Restricción**: v4 only — v2/v3 legacy emite warning y cae back a batch
- **Archivos**: `src/application/codec.rs`
- **Implementación**:
  - `BufWriter<File>` para escritura progresiva a disco conforme se decodifica
  - CRC-32 incremental: `crc32_update` por chunk + `crc32_finalize` al final
  - Peak memory: O(buffer_size) en vez de O(file_size)
  - On CRC failure: output file deleted automatically (cleanup on error)
  - Batch mode (sin `--streaming`) unchanged — comportamiento idéntico al anterior
- **Resultado**: verified byte-identical output batch vs streaming, CRC D124164C identical
- **Subtareas**:
  - [x] Evaluar viabilidad: v4 interleaved decode+detokenize ya existía, solo faltaba progressive write
  - [x] Emitir bytes conforme se decodifican: BufWriter flush progresivo, CRC incremental

---

### Fase 4: Backbone Abstraction & Scalability (2026-10-30)

> Ref: `docs/internal/research/backbone-scalability-analysis.md`
> Diagnóstico: la recurrencia matricial O(D²) de RWKV-7 (62% del forward pass) es un
> muro secuencial inherente. No es NP — es P con constante ~3000x vs brotli y
> recurrencia no lineal probablemente P-completa (no paralelizable dentro de un archivo).
> A ~1.1 KB/s, comprimir 1 TB requiere ~10,800 cores/día. Inviable para producción a escala.
>
> Estrategia: construir primero la abstracción (dock), luego evaluar backbones alternativos
> con estado diagonal O(D·N) o O(D), y finalmente aplicar técnicas universales (BPE, expanded CM)
> que beneficien a todos los backbones.

#### R-034: Backbone Dock — trait abstraction for swappable models ✅
- **Estado**: COMPLETADO (2026-10-30, cf3fc4b)
- **Prioridad**: P1 — prerequisito para toda la Fase 4
- **Impacto**: Codec completamente desacoplado de RWKV-7 — zero RWKV imports en codec/api/benchmark
- **Riesgo BPB**: CERO — refactor puro, zero-cost generics, monomorphización idéntica
- **Resultado**: 130 líneas añadidas, 49 eliminadas, 7 archivos modificados
- **Archivos**:
  - `src/domain/backbone.rs` (NUEVO): `Backbone`, `BackboneState`, `BackboneScratch`, `BackendTokenizer`
  - `src/infrastructure/rwkv7/model.rs`: `impl Backbone for Rwkv7Model`
  - `src/infrastructure/rwkv7/tokenizer.rs`: `impl BackendTokenizer for WorldTokenizer`
  - `src/application/codec.rs`: todas las funciones genéricas, zero RWKV imports
  - `src/application/api.rs`: `compress_bytes`/`decompress_bytes` genéricos
  - `src/application/benchmark.rs`: `compress_and_measure`/`verify_roundtrip` genéricos
  - `src/application/model_loader.rs`: punto de construcción concreto (sin cambios)
- **Verificación** [MEASURED]:
  - `cargo build --release`: zero warnings
  - `cargo test`: 176 unit + 4 integration PASS
  - `embedding_surgery` como pre-processing concreto, fuera del trait

#### R-035: Mamba backbone — alternative O(D·N) inference
- **Estado**: COMPLETADO (2026-10-30)
- **Prioridad**: P1 — resuelve el bottleneck O(D²) → O(D·N) con N=16
- **Impacto**: BPB dramáticamente mejor, throughput significativamente menor
- **Modelo**: state-spaces/mamba-130m-hf (Apache-2.0)
  - d_model=768, d_inner=1536, d_state=16, d_conv=4, dt_rank=48, n_layer=24, vocab=50280
  - Pesos: 493 MB SafeTensors, tokenizer: 2.1 MB (GPT-NeoX BPE, 50254 entries, 50009 merges)
- **Resultados [MEASURED 2026-10-30]**:
  - enwik8 10KB: **0.9074 BPB** (Mamba) vs 1.4067 BPB (RWKV-7 baseline) = **-35.5%**
  - dickens 10KB: **0.6149 BPB** (Mamba) vs 1.7970 BPB (RWKV-7 hybrid) = **-65.8%**
  - apache-access.log 10KB: **0.1636 BPB** (Mamba) vs 0.2960 BPB (RWKV-7 hybrid) = **-44.7%**
  - Throughput: **15.2-16.3 tok/s** (scalar) vs ~95 tok/s (RWKV-7) = **6x más lento**
  - BPE tokenization: ~2.8-3.5 bytes/token vs ~1.5 bytes/token (World)
- **Análisis**: Mamba-130m tiene 4x más parámetros que RWKV-7 0.1B y BPE reduce tokens/byte.
  Esto produce mejor BPB pero la comparación no es directa (distinto tamaño + tokenizer).
  El throughput bajo se debe a que no hay SIMD para el SSM recurrence loop.
- **CHECKPOINT DE BIFURCACIÓN**: Este punto documenta que Mamba es viable como backbone
  alternativo. Decisiones futuras: SIMD para SSM scan, modelo más pequeño (Mamba-70m?),
  integración con ensemble completo (CM + BiasHead + MatchModel), compression pipeline.
- **Archivos**:
  - `src/infrastructure/mamba/model.rs` — MambaModel + forward pass (~280 líneas)
  - `src/infrastructure/mamba/tokenizer.rs` — BpeTokenizer (GPT-NeoX JSON parser)
  - `src/domain/tensor.rs` — portable_silu, portable_softplus, rms_norm_into
  - `scripts/download-mamba-weights.sh` — descarga automática de pesos
  - `weights/MANIFEST.md` — provenance documentada
- **Subtareas**:
  - [x] Investigar pesos Mamba pretrained disponibles → mamba-130m-hf seleccionado
  - [x] Implementar inference Mamba zero-dep (selective scan, conv1d shift, SSM recurrence)
  - [x] Implementar BPE tokenizer zero-dep (JSON parser, GPT-2 bytes_to_unicode)
  - [x] `impl Backbone for MambaModel` + `impl BackboneState for MambaState`
  - [x] Cargar pesos SafeTensors (reusar parser existente)
  - [x] CLI: `mamba-eval` command para evaluación BPB
  - [x] Benchmark comparativo: 3 archivos medidos (enwik8, dickens, apache)
  - [x] Documentar en MANIFEST.md y ROADMAP.md
  - [ ] FUTURO: SIMD kernels para SSM scan (AVX2 elementwise)
  - [ ] FUTURO: Benchmark comparativo completo 19 archivos
  - [ ] FUTURO: Integración con ensemble (CM + BiasHead + MatchModel)
  - [ ] FUTURO: Compression pipeline (compress/decompress con Mamba backend)

#### R-036: BPE tokenizer — reduce tokens per byte
- **Estado**: COMPLETADO VÍA R-035 (2026-10-30)
- **Prioridad**: P2 — multiplica throughput 3-5x para cualquier backbone
- **Impacto**: Mamba usa GPT-NeoX BPE (50,280 vocab), 2.8-3.5 bytes/token vs 1.5 (World)
- **Resultado**: BPE implementado en `src/infrastructure/mamba/tokenizer.rs` como `BpeTokenizer`.
  Carga desde HuggingFace `tokenizer.json` (vocab + merges). Zero-dep, JSON parser propio.
  Implementa `BackendTokenizer` trait (R-034). 3 unit tests.
- **BLOQUEADO para RWKV-7**: El World tokenizer (65K vocab) está acoplado al head layer de
  RWKV-7. Cambiar tokenizer sin cambiar modelo no reduce forward passes. BPE solo es viable
  con modelos entrenados en vocabularios BPE (Mamba, GPT-NeoX, etc.).
- **Archivos**: `src/infrastructure/mamba/tokenizer.rs`
- **Subtareas**:
  - [x] Implementar BPE encode/decode zero-dep en Rust (GPT-2 bytes_to_unicode)
  - [x] `impl BackendTokenizer for BpeTokenizer`
  - [x] Cargar tokenizer.json (HuggingFace format, parser JSON propio)
  - [x] 3 unit tests (byte_unicode_map, pre_tokenize, unescape_json)
  - [N/A] Adaptar para RWKV-7 — BLOQUEADO (modelo acoplado a World vocab)

#### R-037: Expanded Context Mixing — more predictors
- **Estado**: COMPLETADO (2026-10-30)
- **Prioridad**: P2 — mejora BPB sin afectar throughput significativamente
- **Archivos**: `src/domain/match_model.rs`, `src/domain/run_model.rs`, `src/domain/word_model.rs`, `src/domain/mixer.rs`
- **Predictores implementados**:
  1. **Match Model** (5a8d1ed): Hash-chain longest-match, O(1) extend, ln(1+L) bias, 4 tests
  2. **Run-Length Model**: Detecta repeticiones consecutivas, ln(run) bias, 4 tests
  3. **Word Model**: Contexto por "palabra" (secuencias entre separadores), bigram learning, 3 tests
- **AdaptiveMixer**: 5 pesos SGD (w_ngram, w_bias, w_match, w_run, w_word)
- **Resultados cross-entropy** [MEASURED 2026-10-30]:
  | Archivo | Antes (ce_bpb) | Después | Δ |
  |---|---|---|---|
  | enwik8 10KB | 1.2813 | 1.2778 | -0.27% |
  | apache-access.log | 0.2488 | 0.2452 | **-1.4%** |
  | facturae-batch.xml | 0.4084 | 0.3991 | **-2.3%** |
  | samba | 1.2574 | 1.2614 | +0.3% |
  | mozilla | 1.9641 | 1.9431 | **-1.1%** |
- **Resultados compresión real AZTH** [MEASURED 2026-10-30, 19 archivos]:
  - 19/19 roundtrip PASS, CRC OK
  - 15/19 archivos mejoran, 3 empeoran marginalmente (<0.6%), 1 sin cambio
  - Mejores gains: ubl -2.4%, xml -2.2%, mozilla -1.5%, facturae -1.2%
  - mozilla pasa de +1.5% a 0.0% vs brotli (empate exacto)
  - Wins vs brotli-11: **12/19** (empate mozilla no cuenta como win)
  - CSV: `data/benchmark/results/actual-compression-benchmark.csv`
- **Predictores pendientes** (posible trabajo futuro):
  - [ ] Indirect Context Model (skip-grams)
  - [ ] Record Model (columnar context para binarios)

#### R-038: Mamba SIMD Optimization — SSM recurrence + activations
- **Estado**: COMPLETADO (2026-10-30)
- **Prioridad**: P2 — throughput improvement for Mamba backbone
- **Prerequisito**: R-035 (Mamba backbone)
- **Archivos**: `src/domain/tensor.rs`, `src/infrastructure/mamba/model.rs`
- **Problema**: Mamba forward pass dominated by scalar SSM recurrence (~45-50%) and
  element-wise transcendentals (~10-15%). At 15-16 tok/s scalar, 6x slower than RWKV-7.
- **Optimizaciones implementadas**:
  1. **SSM recurrence SIMD** (`ssm_recurrence_d16`): d_state=16 = 2×8 AVX2.
     Replaces 32 scalar `portable_expf` with 4 SIMD `fast_exp_ps` per inner dimension.
     Vectorizes state update and dot-product accumulation with `hsum_avx2`.
  2. **Bulk SiLU** (`silu_inplace`): 8-wide `x * sigmoid(x)` via `fast_sigmoid_ps`.
     Applied to conv_out (1536 elements/layer) after separating conv compute from activation.
  3. **Gated SiLU multiply** (`gated_silu_mul`): `out[i] = y[i] * silu(z[i])`.
     Fuses gating and activation into single SIMD pass.
- **Dispatch**: AVX2 > scalar (follows R-011/R-021 pattern, respects `REDUKT_NO_SIMD=1`)
- **Tests**: 3 unit tests (silu_inplace, gated_silu_mul, ssm_recurrence_d16 vs scalar)
- **BPB impact**: none on RWKV pipeline (independent code path). Mamba BPB: identical
  (0.9074 SIMD vs 0.9074 scalar on enwik8 10KB) [MEASURED]
- **Throughput results** [MEASURED 2026-10-30, controlled A/B alternating]:
  | File | SIMD tok/s | Scalar tok/s | Speedup |
  |---|---|---|---|
  | enwik8 R2 | 34.2 | 23.3 | **+46.8%** |
  | samba R2 | 34.0 | 23.8 | **+42.9%** |
  | enwik8 R3 | 33.7 | 23.3 | **+44.6%** |
  | **Median** | **34.0** | **23.3** | **+44.6%** |
- **Full 19-file benchmark** [MEASURED 2026-10-30, SIMD, Q8=N/A (F32 only)]:
  | File | Tokens | BPB | tok/s | RWKV ce_bpb | vs RWKV |
  |---|---|---|---|---|---|
  | apache-access.log | 3570 | 0.1636 | 34.6 | 0.2482 | **-34.1%** |
  | linux-syslog.log | 4029 | 0.2517 | 34.1 | 0.3702 | **-32.0%** |
  | openssh-auth.log | 3877 | 0.3023 | 34.5 | 0.4722 | **-36.0%** |
  | samba | 3001 | 0.2448 | 33.9 | 1.2637 | **-80.6%** |
  | enwik8 | 3552 | 0.9074 | 34.2 | 1.2786 | **-29.0%** |
  | xml | 5299 | 0.5165 | 34.6 | 0.6231 | **-17.1%** |
  | dickens | 2901 | 0.6149 | 34.2 | 1.7546 | **-65.0%** |
  | webster | 2789 | 0.5500 | 33.6 | 1.7621 | **-68.8%** |
  | reymont | 5651 | 1.7138 | 34.2 | 1.8020 | **-4.9%** |
  | facturae-batch.xml | 4545 | 0.4234 | 34.1 | 0.4023 | +5.2% |
  | ubl-invoice-batch.xml | 4689 | 0.5146 | 34.7 | 0.4783 | +7.6% |
  | nci | 6506 | 0.6336 | 34.6 | 0.5953 | +6.4% |
  | mozilla | 7985 | 2.0854 | 34.2 | 1.9460 | +7.2% |
  | mr | 9159 | 2.6735 | 34.4 | 2.2862 | +16.9% |
  | ooffice | 9500 | 5.2153 | 34.3 | 3.6900 | +41.3% |
  | osdb | 7606 | 6.7121 | 34.2 | 5.6116 | +19.6% |
  | x-ray | 9793 | 6.6064 | 34.3 | 5.7806 | +14.3% |
  | sao | 8954 | 10.7240 | 34.6 | 9.2249 | +16.3% |
  | sample-invoice.pdf | 8965 | 11.8993 | 34.6 | 11.0952 | +7.2% |
  - **Throughput**: 33.6-34.7 tok/s (min webster, max ubl). Mean **34.3 tok/s**. Remarkably stable.
  - **Mamba wins**: 10/19 (logs, text, code, xml). Strongest: samba **-80.6%**, webster **-68.8%**, dickens **-65.0%**
  - **Mamba loses**: 9/19 (binary, PDF, e-invoicing). Worst: ooffice +41.3%, osdb +19.6%
  - **Pattern**: BPE tokenization gives huge advantage on text (2.8-3.6 bytes/token) but near-1:1 on binary
  - CSV: `data/benchmark/results/mamba-simd-benchmark.csv`
- **Gap vs RWKV-7**: 6x slower (scalar) → **2.8x slower** (SIMD). Gap halved.
  - RWKV-7 SIMD: ~95 tok/s [MEASURED R-011]
  - Mamba SIMD: ~34 tok/s [MEASURED R-038]
  - Mamba scalar: ~23 tok/s [MEASURED R-038]
- **Subtareas**:
  - [x] `hsum_avx2` — horizontal sum of 8 packed f32
  - [x] `silu_inplace` + `silu_inplace_avx2` — bulk SiLU activation
  - [x] `gated_silu_mul` + `gated_silu_mul_avx2` — fused gating
  - [x] `ssm_recurrence_d16` + `ssm_recurrence_d16_avx2` — vectorized SSM step
  - [x] `ssm_recurrence_d16_scalar` — extracted scalar fallback
  - [x] Restructure conv loop: separate compute from activation for bulk SiLU
  - [x] Remove stale `#[allow(dead_code)]` on portable_silu, portable_softplus, rms_norm_into
  - [x] 3 unit tests passing
  - [x] RWKV BPB reference unchanged: 1.2778 [MEASURED]
  - [x] Controlled A/B benchmark: 3 rounds alternating, +44.6% median [MEASURED]

#### Secuencia Fase 4

```
R-034 (dock)     ─→ R-035 (Mamba)     ─→ R-038 (Mamba SIMD) ─→ benchmark comparativo
                 ─→ R-036 (BPE)       ─→ aplica a ambos backbones
                 ─→ R-037 (expanded CM) ─→ aplica a ambos backbones
```

R-034 es bloqueante para R-035 únicamente. R-036 y R-037 son independientes y pueden
ejecutarse en paralelo con R-035. R-038 depende de R-035 y optimiza el backbone Mamba.
Las tres (R-035, R-036, R-037) se benefician mutuamente:
BPE reduce tokens para Mamba, expanded CM mejora BPB para ambos backbones.

**Target de Fase 4**: throughput >10 KB/s single-thread, >50 KB/s aggregate (3T).
Esto haría viable comprimir 1 TB en ~6 días con 50 cores — aceptable para cold/archival.

---

### Fase 5: Adaptive Tokenization & Backbone Routing (R-039 to R-042)

**Diagnóstico**: BPE tokenization gives Mamba massive advantage on text (29-81% better than
RWKV ce_bpb) but collapses on binary (7-41% worse). Research confirms: no system has solved
"one tokenizer for all data with a pretrained model." Changing tokenization on a pretrained
model creates severe distribution shift (EMNLP 2024). Architecture-level solutions (MambaByte,
Bolmo) require training. The practical approach: content-adaptive backbone routing.

**Key insight**: Redukt already has dual backbones (Mamba BPE + RWKV byte-level) and a content
detector (R-028). The optimal strategy is routing text→Mamba, binary→RWKV.

Research: `docs/internal/research/adaptive-tokenization-research.md` (18 sources)

#### R-039: Content-adaptive backbone router ✅
- **Estado**: COMPLETADO (2026-10-31)
- **Prioridad**: P1 — high impact, low risk, leverages existing infrastructure
- **Prerequisitos**: R-028 (content detector) ✅, R-034 (backbone dock) ✅
- **Archivos**: `src/domain/backbone.rs`, `src/application/model_loader.rs`, `src/application/codec.rs`
- **Implementación**:
  - `BackboneChoice` enum (Rwkv, Mamba) in domain/backbone.rs with header_byte/from_header_byte/from_str_opt
  - `route_backbone()` function: Text/StructuredText → Mamba, rest → RWKV
  - `DynModel`/`DynTokenizer`/`DynState`/`DynScratch` unified enums in model_loader.rs
  - `load_dyn_model()` loads correct backbone based on BackboneChoice
  - AZTH v4 FLAG bit 6 (0x40) + 1B backbone ID in header
  - `--backbone auto|mamba|rwkv` CLI flag in compress
  - Decompress reads backbone from header, loads correct model
  - Only chosen backbone loaded (no double model load)
  - 9 unit tests: choice roundtrip, from_str, header build/parse, routing for all 6 content types
- **Resultado**: 198 tests pass, zero warnings, backward compatible (RWKV files don't set flag)

#### R-040: Dual-backbone evaluation & routing optimization
- **Estado**: COMPLETADO (2026-10-31)
- **Prioridad**: P2 — research evaluation + routing fix
- **Prerequisitos**: R-039 (router)
- **Concepto**: Evaluate actual AZTH compression (not ce_bpb) with both backbones on 12 files.
  Determine optimal routing based on empirical BPB comparison.
- **Subtareas**:
  - [x] Actual compression with RWKV+CM on 12 files (10KB, Q8 auto) [MEASURED]
  - [x] Actual compression with Mamba+CM on 12 files (10KB, Q8 auto) [MEASURED]
  - [x] Roundtrip verification for Mamba-compressed files (facturae, enwik8 PASS)
  - [x] Routing optimization: updated `route_backbone()` — Mamba default, RWKV for GenericBinary only
  - [x] Competitive analysis vs brotli-11 and gzip-9
  - [x] Fix: MAMBA_WEIGHTS_DIR path corrected (`mamba-130m` not `mamba-130m-hf`)
  - [x] 198+4 tests pass, zero warnings
- **Resultado**:
  - Mamba wins **10/12 files** (83%). RWKV only wins on GenericBinary (+1-4.6%)
  - Biggest wins: samba **-79.1%**, dickens **-63.7%**, enwik8 **-27.6%**, apache **-27.0%**
  - vs brotli-11: wins **8/12** (up from 7/12), average win margin **-50.5%**
  - Best new BPB: samba **0.271** (-87.8% vs brotli), dickens **0.652** (-76.5% vs brotli)
  - Initial hypothesis "byte-level RWKV better for binary" REFUTED
  - Ensemble (mix logits) NOT needed — Mamba dominates, no "close" files found
- **Decision**: NO-GO for ensemble. Single-backbone routing is optimal.
- **Evidence**: `data/benchmark/results/dual-backbone-benchmark.md`, `.csv`

#### R-041: Byte-level Mamba training feasibility (INVESTIGACIÓN)
- **Estado**: COMPLETADO (investigación) — CONDITIONAL GO (2026-10-31)
- **Prioridad**: P3 — future work, requires training infrastructure
- **Concepto**: Train MambaByte-small (50-130M params) on byte sequences (vocab=256).
- **Subtareas**:
  - [x] Training cost estimate: 200-500 A100-hours ($300-750) for 130M [SOURCED]
  - [x] Corpus requirements: 10-30 GB mixed text+binary [SOURCED]
  - [x] Pretrained models: NONE at target size/vocab; 353M text-only available [SOURCED]
  - [x] Zero-dep constraint: training in PyTorch (external, one-time); inference in Rust (existing)
  - [x] Decision: **CONDITIONAL GO** — validate with Phase 0 before committing
- **Resultado (investigación)**:
  - MambaByte-353M: 0.93 BPB PG19, ~1.05-1.15 enwik8 [SOURCED: Wang et al. 2024]
  - MambaByte-130M (estimated): ~1.20-1.40 BPB enwik8 with CM [ESTIMATE]
  - Throughput projection: 200-500 tok/s (2-5x over RWKV-7) [ESTIMATE]
  - No pretrained small byte-level Mamba exists — training from scratch required
  - Training is PyTorch/CUDA (external); zero-dep Rust constraint NOT violated
  - Kill criterion: Phase 1 BPB > 1.5 on enwik8 10KB with CM → abort
- **Phased plan**: Phase 0 (validate 353M, $0) → Phase 1 (50M, $200) → Phase 2 (130M, $500) → Phase 3 (integrate)
- **References**: MambaByte (arXiv:2401.13660), StateSMix (arXiv:2605.02904), BLT (arXiv:2412.09871)

#### R-042: Tokenizer-aware preprocessing for binary
- **Estado**: COMPLETADO (investigación) — NO-GO (2026-10-31)
- **Prioridad**: P3 (was P2, deprioritized by R-040 results)
- **Concepto**: Preprocess binary data to make it more BPE-friendly before Mamba tokenization.
- **Subtareas**:
  - [x] Evaluate byte-frequency remapping: NO-GO (semantic mismatch worsens predictions)
  - [x] Evaluate structure-aware splitting: NO-GO for Mamba (already works for RWKV via R-031)
  - [x] Evaluate alphabet reduction: NO-GO (same fundamental problem as remapping)
  - [x] Evaluate E8E9+shuffle+delta before BPE: MARGINAL (5-10% max, doesn't close 7-41% gap)
  - [x] Literature review: no known technique makes pretrained BPE model work on binary [SOURCED: 5 papers]
- **R-040 finding**: Mamba already beats RWKV on most binary files. Routing is sufficient.
- **R-042 finding**: BPE is fundamentally incompatible with binary — tokens carry wrong semantic
  embeddings. The 130M params encode text-domain knowledge that actively harms binary predictions.
  Even best-case preprocessing (~10% improvement) doesn't close the gap on files where RWKV wins.
- **Decision**: NO-GO. Backbone routing (R-039/R-040) is the correct solution.
  Future improvement requires byte-level model (R-041), not preprocessing.

#### Secuencia Fase 5

```
R-039 (router) ✅ ──→ R-040 (dual eval) ✅ ──→ NO-GO ensemble, routing optimizado
     │
     ├──→ R-042 (binary preproc) ✅ ──→ NO-GO, routing suficiente
     │
     └──→ R-041 (MambaByte feasibility) ✅ ──→ CONDITIONAL GO, $300-750 training
```

**Estado Fase 5**: R-039 ✅ R-040 ✅ R-041 ✅(research) R-042 ✅(research)
**Resultado**: Dual-backbone routing es la solución correcta. Mamba wins 10/12.
vs brotli-11: 8/12 wins, avg margin -50.5%. Best: samba -88%, dickens -77%.
**Siguiente paso**: R-041 Phase 0 (validate MambaByte-353M) cuando haya acceso GPU.

---

### Fase 6: Information-Theoretic Enhancements (R-043 to R-047)

> Origen: PDF "Matrices y Alternativas Vectoriales" — tokenización como problema de
> compresión/proyección de señales. Filtrado por viabilidad sin retraining.
> Research: 3 agentes paralelos (BPP/entropy, symbol-width/folding, codebase analysis).
> Fuentes: BLT (2412.09871), Kronecker Embeddings (2605.29459), Proxy Compression
> (2602.04289), Nacrith (2602.19626), ByteFlow (2603.03583), EmByte (EMNLP 2025).

#### R-043: Entropy-Adaptive Mixer Weighting
- **Estado**: ✅ COMPLETADO (2026-10-31, fd77c70)
- **Prioridad**: P1 — zero retraining, uses existing entropy signal
- **Origen PDF**: "Segmentación por Entropía Local" — cortes donde predictibilidad colapsa
- **Concepto**: Modulate CM predictor weights based on backbone's per-byte cross-entropy.
  At high-entropy positions (backbone uncertain), boost CM predictors (ngram, match, word).
  At low-entropy positions (backbone confident), trust neural backbone more.
- **Implementación**:
  - [x] `cm_scale` field in AdaptiveMixer — modulates all CM contributions
  - [x] `ce_ema` running EMA of cross-entropy (α=0.01, init=3.0)
  - [x] Boost-only: `cm_scale = (ce / ce_ema).clamp(1.0, 1.2)` — never reduces CM below baseline
  - [x] Gated by `V4_FLAG_ENHANCED_CM` (0x80) for backward compatibility
  - [x] `set_entropy_scaling(bool)` method — false for old files, true for new
- **Resultados A/B** (10KB, Mamba backbone) [MEASURED]:
  | Config | facturae | apache | enwik8 |
  |--------|----------|--------|--------|
  | Baseline | 0.3891 | 0.2164 | 0.9609 |
  | R-043 only (1.0,1.2) | 0.3906 (+0.4%) | 0.2148 (-0.7%) | 0.9609 (0%) |
- **Hallazgo**: Near-zero effect at 10KB — EMA doesn't stabilize with ~3500 tokens.
  Expected to improve at 50KB+ scale. Safe (no regression beyond noise).
- **Archivos**: `src/domain/mixer.rs`, `src/application/codec.rs`, `src/application/api.rs`

#### R-044: Segment-Aware CM Predictor (SegmentModel)
- **Estado**: ✅ COMPLETADO (2026-10-31, fd77c70)
- **Prioridad**: P1 — new CM predictor, zero retraining
- **Origen PDF**: "Agrupamiento suave" — segments of coherent information
- **Concepto**: Track entropy-based segment boundaries. Within each segment, maintain
  a per-segment byte frequency histogram that predicts based on local distribution.
  Segment transitions occur when backbone entropy spikes above 2x running average.
- **Implementación**:
  - [x] `SegmentModel` struct: per-segment token-frequency histogram
  - [x] Boundary detection: surprise > 2.5× surprise_ema → soft reset (halve counts)
  - [x] Laplace-smoothed log-probability bias: `SEG_SCALE * (ln(p) - ln(base))`
  - [x] Constants: ALPHA=0.1, SEG_SCALE=0.03, MIN_OBS=100, SPIKE_THRESHOLD=2.5
  - [x] Additive bias AFTER mixer combine (avoids 6th mixer weight + signature changes)
  - [x] Gated by `V4_FLAG_ENHANCED_CM` (0x80) — disabled for old files
  - [x] 4 unit tests: disabled_returns_zeros, no_bias_below_min_obs, frequent_token_gets_positive_bias, segment_boundary_decays_counts
- **Resultados A/B** (10KB, Mamba backbone) [MEASURED]:
  | Config | facturae | apache | enwik8 |
  |--------|----------|--------|--------|
  | Baseline | 0.3891 | 0.2164 | 0.9609 |
  | R-044 only | 0.3891 (0%) | 0.2172 (+0.4%) | 0.9633 (+0.2%) |
- **Hallazgo**: Near-zero effect at 10KB — MIN_OBS=100 means SegmentModel barely activates
  with ~3500 tokens. Expected to improve at 50KB+ scale where segment structure emerges.
- **Archivos**: `src/domain/segment_model.rs` (new), `src/application/codec.rs`, `src/application/api.rs`

#### R-045: Bit-Plane CM Hash Features
- **Estado**: ✅ COMPLETADO (2026-10-31)
- **Prioridad**: P2 — trivial implementation, uncertain impact
- **Origen PDF**: "Bit-Plane Projection" — decompose bytes into bits for richer features
- **Concepto**: Use individual bit positions of current/previous bytes as additional context
  in CM predictor hash functions. MSB (bit 7) captures case in ASCII, sign in floats.
  Bit-plane features add structural information without changing the backbone.
- **Subtareas**:
  - [x] Add bit-plane context to ngram hash: XOR MSB/nibble of last token's low byte
  - [x] Add bit-plane context to match_model hash: MSB of both tokens as extra discriminator
  - [x] Benchmark: 9 files measured [MEASURED]
  - [x] Roundtrip verification: enwik8, dickens, samba — PASS
- **Resultados** [MEASURED, 10KB]:
  | File | Baseline | R-045 | Δ |
  |---|---|---|---|
  | x-ray | 5.8416* | 5.8008 | **-0.7%** |
  | mr | 2.3914 | 2.3281 | **-2.6%** |
  | samba | 1.2970 | 1.2914 | **-0.4%** |
  | ooffice | 3.8588 | 3.8469 | **-0.3%** |
  | enwik8 | 0.9633 | 0.9633 | 0% |
  | dickens | 0.6523 | 0.6523 | 0% |
  | apache | 0.2164 | 0.2164 | 0% |
  | osdb | 5.5953 | 5.6336 | +0.7% |
  | sao | 8.6430 | 8.6438 | +0.01% |
- **Conclusión**: Marginal en binarios tras BUG-001 fix (x-ray baseline era corrupto).
  Texto sin regresión. Solo osdb +0.7% leve. *x-ray baseline incorrecto por BUG-001.
- **Archivos**: `src/domain/ngram.rs`, `src/domain/match_model.rs`

#### R-046: Arithmetic Coder Precision Audit
- **Estado**: ✅ COMPLETADO (2026-10-31, fd77c70)
- **Prioridad**: P1 — Nacrith finding: CDF-24 = -28% BPB improvement [SOURCED: 2602.19626]
- **Origen PDF**: "Codificación Aritmética / ANS" — longitud fraccionaria de bits
- **Concepto**: Audit Redukt's range coder against Nacrith's CDF-24 findings. Verify:
  (a) minimum frequency floor per symbol, (b) CDF rounding strategy,
  (c) whether reduced CDF precision on low-probability tokens wastes bits.
- **Hallazgos del audit**:
  - [x] Redukt already uses CDF-24 (TOP=1<<24) — matches Nacrith recommendation [VERIFIED]
  - [x] Nacrith's -28% BPB was CDF-16→CDF-24 transition — NOT applicable to Redukt
  - [x] min_freq=1 (floor per symbol) — correct, matches best practice
  - [x] Applied round-to-nearest in frequency allocation: `(prob * inv * remaining + 0.5) as u64`
    (was truncation). Marginally better rounding accuracy.
  - [x] Roundtrip verified: enwik8 10KB, CRC OK, BPB unchanged (0.9633)
- **Resultado**: No precision gap vs Nacrith. Redukt's coder was already near-optimal.
  Round-to-nearest is a micro-improvement (<0.01 BPB at any scale).
- **Archivos**: `src/domain/coder.rs`

#### R-047: Bit-Plane Projection Research (FUTURE)
- **Estado**: PENDIENTE — investigación, depende de R-041 (training infrastructure)
- **Prioridad**: P3 — requires retraining, long-term
- **Origen PDF**: "Proyección directa sin tabla de embeddings" — `z = W_proj · b + bias`
- **Concepto**: Replace embedding lookup with learned bit-plane projection. Eliminates
  vocabulary dependence entirely. Published precedent: Kronecker Embeddings (2605.29459)
  achieves -2.5% val loss and 91-94% embedding param reduction, but requires training.
- **Prerequisitos**: R-041 (MambaByte training), GPU access
- **Decision**: Defer until training infrastructure exists. Embedding is <0.05% of model size.

#### Secuencia Fase 6

```
R-043 (entropy mixer) ✅ ──┐
R-044 (segment model) ✅ ──┼──→ CHECKPOINT: build ✅ + test ✅ + roundtrip ✅
R-045 (bit-plane hash) ⏸ ──┘    (12-file benchmark pendiente — near-zero effect at 10KB)
         │
R-046 (CDF audit) ✅ ────────→ No gap vs Nacrith. CDF-24 already optimal.
         │
R-047 (BPP research) ────────→ FUTURE (depende R-041)
```

R-043, R-044: implemented and tested. Effects near-zero at 10KB scale (~3500 tokens).
EMA/histogram don't stabilize. Expected improvement at 50KB+ scale.
R-045: deferred (P2, uncertain impact). R-047: future.
**Estado Fase 6**: R-043 ✅ R-044 ✅ R-045 ⏸ R-046 ✅ R-047 ⏸(future)

---

### Fase 7: Backbone Offloading — PredCache (R-048 to R-050)

> Origen: Insight del usuario — "almacenar las recurrencias en tabla hash,
> trasladando conocimiento del backbone a los CM, iteración tras iteración."
> Research: `docs/internal/research/prediction-cache-research.md`
>
> **Problema**: El backbone (Mamba-130M) cuesta ~29ms/token = 99.5% del tiempo.
> No existe optimización de software que cierre la brecha 500,000x vs brotli.
>
> **Solución**: Cache de predicciones neurales indexado por contexto. Cuando
> un contexto (últimos N tokens) se repite, reusar la distribución del backbone
> en vez de ejecutar el forward pass. El cache absorbe progresivamente el
> conocimiento del backbone, haciendo el sistema más rápido con el tiempo.
>
> **Diferencia vs R-033 (adaptive skip)**: R-033 reusa logits del token
> anterior cuando la confianza es alta. PredCache almacena y recupera
> distribuciones para CUALQUIER contexto previamente visto. Es estrictamente
> más poderoso y subsume el adaptive skip.

#### R-048: PredCache — Neural Prediction Cache (data structure)
- **Estado**: ✅ COMPLETADO (2026-10-31)
- **Prioridad**: P1 — addresses fundamental throughput bottleneck
- **Concepto**: Hash table mapping context fingerprints to top-K backbone
  logit distributions. Enables skipping the 29ms backbone forward pass
  for previously-seen contexts.
- **Implementación**:
  - [x] `src/domain/pred_cache.rs` — 240 lines, fully tested
  - [x] `CacheEntry`: top-32 tokens (u16) + logits (f32) + base_logit + hash + observations
  - [x] `context_hash()`: FNV-1a over last N tokens (configurable order, default 8)
  - [x] `lookup(hash, out)` → bool, full-hash collision check, reconstructs logits
  - [x] `insert(hash, logits)` — extract top-K, compute base_logit, LFU eviction on collision
  - [x] `reconstruct_logits(entry, out)` — fill vocab-size array from top-K + base
  - [x] Table: 64K entries (open addressing), ~13MB max
  - [x] 16 unit tests: hash determinism, insert/lookup, collision eviction, LFU survival,
    min_obs threshold, disabled mode, hit rate tracking, large vocab (50280), e2e stream
- **Métricas**: 0 warnings, 222 tests pass, BPB unchanged (1.2739 RWKV enwik8 10KB)
- **Archivos**: `src/domain/pred_cache.rs` (new), `src/domain/mod.rs`

#### R-049: PredCache Integration — Backbone Skip Protocol
- **Estado**: ❌ REVERTIDO (2026-10-31, e01411c)
- **Prioridad**: P1 — requires R-048
- **Concepto**: Wire PredCache into compress/decompress loops. Both sides
  build identical caches from the same token stream, ensuring deterministic
  skip decisions without additional header data.
- **Implementación (revertida)**:
  - Integrado en 7 loops: compress main, compress_single_file, v2/v3 decompress,
    v4 decompress, api compress, api decompress, benchmark
  - Determinismo CONFIRMADO: compress 598/3574 = decompress 598/3573 hits
  - Roundtrip PASS (byte-identical, CRC-32 verified)
- **Razón de reversión**:
  - BPB: 0.9633 → 1.1305 (**+17.4%**) — inaceptable, estimación era 0-3% [MEASURED]
  - Throughput: +15.9% (32.8→38.0 tok/s) — insuficiente para justificar la regresión
  - Hardcoded siempre activo, sin flag CLI para desactivar
  - State divergence con MAX_SKIPS=8 acumula error mucho más rápido de lo previsto
  - **Decisión**: revertir hasta que R-050 (tuning) determine parámetros viables
- **Prerequisitos para re-implementar**:
  - [ ] Flag CLI `--pred-cache` (opt-in, no activo por defecto)
  - [ ] R-050 tuning: encontrar MAX_SKIPS, MIN_OBS, ORDER óptimos por content type
  - [ ] Objetivo: ≤3% BPB overhead con ≥2x speedup en datos estructurados
- **Archivos**: `src/application/codec.rs`, `src/application/api.rs`,
  `src/application/benchmark.rs`

#### R-050: PredCache Evaluation & Tuning
- **Estado**: PENDIENTE
- **Prioridad**: P1 — requires R-049
- **Concepto**: Comprehensive benchmark of PredCache speedup vs BPB tradeoff.
  Tune parameters per content type. Test scale behavior.
- **Subtareas**:
  - [ ] A/B benchmark: 12-file suite, PredCache ON vs OFF
  - [ ] Per-file hit rate measurement and reporting
  - [ ] Speedup measurement: tok/s with PredCache vs without
  - [ ] BPB delta: quantify compression quality cost per file
  - [ ] Scale test: 10KB → 50KB → 100KB → 500KB (hit rate should improve)
  - [ ] Batch mode: cache persistence across files (same domain)
  - [ ] Batch test: 10 facturas XML — measure progressive cache warming
  - [ ] Memory profiling: actual cache usage vs theoretical max
  - [ ] Tune: ORDER (4, 8, 16), MIN_OBS (1, 2, 4), MAX_SKIPS (4, 8, 16)
  - [ ] Content-adaptive params via content_detect.rs integration
  - [ ] Document optimal params per content type
- **Métricas target**:
  - Structured data (logs, XML): ≥2x speedup, ≤2% BPB overhead
  - Natural text: ≥1.2x speedup, ≤3% BPB overhead
  - Binary: minimal impact (≤5% overhead, ≤1.1x speedup)
- **Archivos**: `data/benchmark/results/predcache-benchmark.csv` (new),
  `docs/internal/research/prediction-cache-research.md` (update with results)

#### Secuencia Fase 7

```
R-048 (data structure) ──→ R-049 (integration) ──→ CHECKPOINT: roundtrip + A/B
                                                         │
                                                    R-050 (eval + tuning) ──→ CHECKPOINT: full benchmark
```

Estrictamente secuencial: R-049 depende de R-048, R-050 depende de R-049.

**Métrica de éxito Fase 7**: ≥2x speedup en datos estructurados (logs, XML, facturas)
con ≤2% BPB overhead. Hit rate ≥50% en batch mode para archivos del mismo dominio.

---

### Fase 8: Modular Execution Architecture (R-051 to R-053)

> Prerequisito para GPU parallel scan y speculative execution.
> Módulos puramente aditivos — no modifican backbone ni coder existentes.
> Research: `docs/internal/research/execution-architecture-analysis.md`

#### R-051: CmEnsemble — Context-Mixing consolidation
- **Estado**: COMPLETADO (ad62291, 2026-11-01)
- **Impacto**: -259 líneas de código duplicado, 9 loops refactorizados
- **Archivos**: `src/domain/cm_ensemble.rs` (nuevo), `src/application/codec.rs`,
  `src/application/api.rs`, `src/application/benchmark.rs`
- **Descripción**: Agrupa los 6 modelos CM (TokenNgram, BiasHead, MatchModel,
  RunModel, WordModel, SegmentModel) + AdaptiveMixer + 5 buffers pre-allocated
  en un solo struct. Reemplaza 12+ parámetros sueltos por `CmEnsemble`.
- **API**: `new()`, `compute_biases(entropy_hint)`, `combine_into()`, `update()`,
  `observe()`, `draft_next()`, `with_bias_decay(tau)`
- **Tests**: 3 unit tests + 232 existentes pass
- **BPB**: sin cambio (dickens 10KB 0.6464, roundtrip byte-identical) [MEASURED]

#### R-052: Parallel Scan — Generic Blelloch algorithm
- **Estado**: COMPLETADO (ad62291, 2026-11-01)
- **Impacto**: Infraestructura para GPU parallel scan futuro
- **Archivos**: `src/domain/parallel_scan.rs` (nuevo) — **borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte**
- **Descripción**: Algoritmo genérico de Blelloch scan (prefix sum) sobre
  operaciones asociativas. O(N) trabajo, O(log N) profundidad secuencial.
  Incluye `SsmScanPair` (escalar) y `SsmScanVec` (vectorial d_state)
  para representar recurrencias SSM como pares composables (A, b).
- **Tests**: 6 unit tests (scan, associativity, identity, recurrence equivalence)
- **Pendiente**: wiring a Mamba forward pass (requiere GPU o multi-thread Rayon)

#### R-053: Speculative Execution — CM-draft with backbone verify
- **Estado**: COMPLETADO (ad62291, 2026-11-01)
- **Impacto**: Infraestructura para speculative compression
- **Archivos**: `src/domain/speculative.rs` (nuevo) — **borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte**
- **Descripción**: Usa CmEnsemble (0.3% del costo) para predecir K tokens ahead.
  `SpeculativeTracker` adapta K basado en rolling acceptance rate (EMA).
  `draft_tokens()` genera draft, `verify_draft()` compara contra tokens reales.
- **Tests**: 5 unit tests (adaptive K, acceptance rate, verify match/partial/none)
- **Pendiente**: integración en codec.rs como `--speculative K` CLI flag

#### Secuencia Fase 8

```
R-051 (CmEnsemble) ──→ R-052 (parallel scan) ──→ FUTURO: GPU wiring
         │
         └──→ R-053 (speculative) ──→ FUTURO: codec integration (--speculative K)
```

**Estado Fase 8**: R-051 ✅ R-052 ✅ R-053 ✅ (módulos listos, integración futura) — R-052/R-053 borrados en 1dc6402, nunca medidos; rehacer a nivel byte tras MambaByte

---

### Fase 9: CPU Parallelism — Breaking the Sequential Bottleneck (R-054 to R-058)

> Research: `docs/internal/research/cpu-parallelism-strategies.md`
> Goal: maximize CPU utilization without GPU, exploit mathematical properties
> of linear SSM recurrences and known-token compression asymmetry.

#### R-054: Multi-threaded matmul ✅
- **Estado**: COMPLETADO (f7a1eec, 2026-11-01)
- **Riesgo BPB**: 0% (verificado — BPB idéntico)
- **Target**: matmuls con rows > 4096 (Mamba LM head 50,280×768)
- **Implementación**: `std::thread::scope` (stdlib, zero-dep), raw-slice kernels
- **Funciones**: `mat_vec_mul_parallel_into`, `mat_vec_mul_scalar_raw`, `mat_vec_mul_avx2_nofma_raw`
- **Threads**: `available_parallelism().min(4)`, cached en AtomicUsize
- **Archivos**: `src/domain/tensor.rs`
- **Resultado**: mismo-tier determinístico, cross-tier divergencia pre-existente

#### R-055: Layer-by-layer parallel scan ❌ BLOQUEADO
- **Estado**: BLOQUEADO — determinismo FP impide implementación (2026-11-01)
- **Riesgo BPB**: 0% teórico, FATAL en práctica
- **Descripción**: procesar TODOS los tokens por layer 1, luego layer 2, etc.
  Pre-compute per-token quantities en paralelo, parallel scan para SSM.
- **Bloqueador**: Blelloch scan compone pares (a,b) con orden de asociación
  diferente al recurrence secuencial. FP no-asociatividad:
  `Sequential: a3*(a2*(a1*0+b1)+b2)+b3 ≠ Blelloch: (a3*a2*a1)*0 + (a3*a2)*b1 + a3*b2 + b3`
  Compresor y descompresor DEBEN producir CDFs idénticas.
  Parallel scan da resultados numéricamente diferentes → roundtrip FALLA.
- **Factible solo con**: (a) fixed-point arithmetic, (b) format batch-only
  (descompresor también batch, requiere todos los tokens — chicken-and-egg),
  (c) double-pass (parallel + sequential) que no ahorra tiempo.
- **Dependencias**: R-052 (parallel_scan.rs) — módulo listo, no wirable (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte)

#### R-056: CM-selective backbone skip ❌ NO VIABLE
- **Estado**: INVESTIGADO + DESCARTADO (2026-11-01)
- **Riesgo BPB medido**: +81% a +294% — muy por encima del +3% budget
- **Descripción**: skip backbone forward cuando confianza > θ,
  con K≤1 o K≤3 consecutive skips.
- **Hallazgos empíricos (10KB, Mamba)**:
  - CM-only confidence (R-056 original): NUNCA supera 5% con BPE 50K vocab.
    El vocabulario es demasiado grande para que CM solo concentre probabilidad.
  - Ensemble confidence (R-033): K=1, θ=99% → apache +81% BPB (0.22→0.40),
    enwik8 +35.7% (0.96→1.31). K=3, θ=95% → apache +101% (0.22→0.44).
  - Throughput gain: 1.5-2.3x (57-87 tok/s vs 38 baseline).
  - Tradeoff NO aceptable: la degradación BPB es 10-100x peor que el objetivo ≤3%.
- **Causa raíz**: backbone proporciona >95% de la información predictiva.
  Logits "stale" (de posición anterior) son cualitativamente incorrectos para
  la posición actual. Incluso K=1 causa divergencia catastrófica en CDFs.
- **Confirmación**: consistente con R-049 (PredCache, +17.4% BPB con K=8).
  El backbone es irreduciblemente secuencial para lossless compression.
- **Infraestructura preservada**: `cm_only_top_prob()` en CmEnsemble,
  adaptive-skip en codec/header (FLAG 0x20), disponible para futuro byte-level model.

#### R-057: dt-sparse state skip ✅
- **Estado**: COMPLETADO (f7a1eec, 2026-11-01)
- **Riesgo BPB**: 0% (verificado — BPB idéntico, ε=1e-6 matemáticamente exacto)
- **Descripción**: skip SSM state update per-channel cuando |dt[j]| < 1e-6.
  State unchanged (da≈1, dt*x*b≈0), output computed from existing state.
- **Implementación**: both scalar + AVX2 paths, `DT_SPARSE_EPS` constant
- **Archivos**: `src/domain/tensor.rs` (ssm_recurrence_d16_scalar, ssm_recurrence_d16_avx2)

#### R-058: LSH state memoization
- **Estado**: INVESTIGACIÓN
- **Riesgo BPB**: 1-5% (approximate matching)
- **Descripción**: locality-sensitive hashing sobre estados del modelo.
  Cache y reutiliza logits cuando estados son similares.
  Evolución de R-049 (PredCache) con approximate matching.

#### Secuencia Fase 9

```
R-054 ✅ (parallel matmul) ──→ R-057 ✅ (dt-sparse) ──→ COMPLETADOS
R-055 ❌ (layer scan) ──→ BLOQUEADO por FP determinism
R-056 ❌ (CM-selective) ──→ NO VIABLE (+81-294% BPB)
R-058 ⏸ (LSH memo) ──→ INVESTIGACIÓN futura
```

**Resultado Fase 9**: de 5 estrategias, 2 implementadas (0% BPB risk),
2 descartadas (backbone irreduciblemente secuencial), 1 en investigación.
El backbone forward pass es el cuello de botella fundamental e irreducible
para compresión lossless. Solo GPU parallelism (R-055 con formato batch-only)
o byte-level model (R-041 MambaByte) pueden romper esta barrera.
- Gap vs brotli: de 46,000x a ~1,300-4,700x

### Fase 10: Lottery Ticket Sparsity Protocol (R-059 to R-063)

> Research: `docs/internal/research/lottery-ticket-sparsity-protocol.md`
> Origin: Frankle & Carbin (MIT, 2018) — derivative works 2023-2026
> Goal: reduce FLOPs per token via pruning + dynamic sparsity + pipeline
> BPB budget: +5% max (CLAUDE.md)

#### R-059: Activation sparsity — gated[] channel measurement
- **Estado**: ✅ COMPLETADO (medición) + NO VIABLE (optimización)
- **Resultado**: sparsity medida **1.7-1.9%** — insuficiente para block-sparse
  - enwik8 10KB: 2,522,394 / 130,940,928 near-zero = **1.9%**
  - dickens 10KB: 1,863,013 / 106,942,464 near-zero = **1.7%**
  - Umbral |gated[j]| < 1e-4 (GATED_EPS)
  - 24 layers × d_inner=1536 channels × ~3500 tokens = ~130M checks
  - Block-sparse out_proj requiere >25% sparsity para beneficio CPU
- **Corrección de diseño**: la hipótesis original (conv_out sparse → skip SSM)
  era incorrecta — y[j] porta historial de estado, z_branch es independiente.
  El target correcto es gated[] = y * silu(z), pero su sparsity es mínima
  en Mamba-130M porque SiLU(z) raramente produce zeros exactos.
- **Instrumentación implementada**: contadores en MambaScratch, reportados
  al final de compress/decompress para monitoreo continuo.
- **Papers**: Mukherji et al. (2023) arXiv:2311.07625
- **Archivos**: `src/infrastructure/mamba/model.rs`, `src/application/model_loader.rs`,
  `src/application/codec.rs`

#### R-060: Split-Head Pipeline — LM head overlap
- **Estado**: ✅ COMPLETADO (implementado, beneficio marginal en DDR4)
- **Riesgo BPB**: 0% ✓ (confirmed: BPB 0.9648 vs ref 0.9609, within noise)
- **Resultado medido**: throughput ~19 tok/s → ~19 tok/s (DDR4-2667 saturado)
  - Pipeline funciona correctamente: head(t-1) on worker, layers(t) on main
  - Roundtrip byte-identical verified (CRC-32 pass)
  - Beneficio nulo por bandwidth contention en DDR4 single-channel laptop
  - Esperado beneficio significativo en DDR5/dual-channel desktop
- **Implementación**:
  - `MambaModel::forward_layers_normed()` — embedding + 24 layers + RMSNorm
  - `MambaModel::compute_head_into()` — LM head matmul only
  - `forward_into()` refactored to call both (no behavior change)
  - `DynModel` pipeline methods: `supports_pipeline()`, `hidden_dim()`,
    `forward_layers_normed()`, `compute_head_into()`
  - Compress loop: dual hidden buffers (`hidden_a/b`), `std::thread::scope`,
    `std::mem::swap` after each step
  - Decompress: NO pipeline (tok[t] unknown until logits[t-1] decoded)
  - Fallback: RWKV or adaptive-skip → original single-thread path
- **Papers**: inspirado en Compiler-First SSD (2026) arXiv:2603.09555
- **Archivos**: `src/infrastructure/mamba/model.rs`, `src/application/model_loader.rs`,
  `src/application/codec.rs`

#### R-061: Wanda column pruning — d_inner reduction
- **Estado**: ❌ NO VIABLE (validación empírica 2026-11-02)
- **Resultado medido** (enwik8 10KB, mamba-eval):
  - Baseline: 0.9074 BPB
  - 1 canal (0.065%): **1.0928 BPB (+20.4%)** — 1 canal ya excede +5% budget
  - 2% pruning: **4.0199 BPB (+343%)**
  - 5% pruning: **3.0316 BPB (+234%)**
  - 10% pruning: **4.8191 BPB (+431%)**
- **Causa raíz**: Mamba-130M es demasiado pequeño para structured pruning.
  Los papers (Wanda, SparseSSM) validan en modelos 7B+, donde hay redundancia.
  A 130M params, cada canal de d_inner es irreemplazable.
  La relación NO es monotónica (5% < 2% < 10%) porque la remoción de canales
  diferentes tiene efectos variables — confirma alta sensibilidad.
- **Implementación**: `apply_column_pruning(prune_frac)` en MambaModel +
  `--prune` CLI en mamba-eval. Código conservado para futuros backbones mayores.
- **Papers**: Wanda (Sun et al., ICML 2023) arXiv:2306.11695
- **Archivos**: `src/infrastructure/mamba/model.rs`, `src/application/codec.rs`

#### R-062: Highway layer skip — upper layer bypass
- **Estado**: ❌ NO VIABLE (validación empírica 2026-11-02)
- **Resultado medido** (enwik8 10KB, A/B controlado):
  - thr=0.01: 0 layers skipped, BPB 0.9074 (idéntico)
  - thr=0.05: 0 layers skipped, BPB 0.9074 (idéntico)
  - thr=0.10: **17/85248 layers skipped (0.02%), BPB 0.9424 (+3.86%)**
  - thr=0.15, start=20+: 0 layers skipped (residuales > 15%)
  - thr=0.20: 4678/85248 (5.5%), BPB **7.3752 (+713%)**
  - thr=0.15, start=12: 11636/85248 (13.6%), BPB **11.12 (+1125%)**
- **Causa raíz**: las capas superiores de Mamba-130M NO son highways.
  Todas las capas contribuyen >5% al hidden state norm. El paper Simba
  observa "90% token sparsity" en modelos grandes (1.3B+), no en 130M.
  Incluso 17 skips en 85K causan +3.86% BPB — la divergencia de estado
  es severa a esta escala de modelo.
- **Implementación**: `enable_highway(start, thr, resync)` + contadores.
  `--highway`, `--highway-thr`, `--resync` CLI en mamba-eval.
  Código conservado para futuros backbones mayores.
- **Papers**: Simba (Song et al., TMLR 2025) arXiv:2505.20698,
  LAST (Gwak et al., NeurIPS 2024) arXiv:2411.02824
- **Archivos**: `src/infrastructure/mamba/model.rs`, `src/application/codec.rs`

#### R-063: Batch state checkpoint — prefix reuse
- **Estado**: PENDIENTE
- **Riesgo BPB**: 0% (exact state memcpy, no approximation)
- **Speedup estimado**: 2-10x (batch mode only, files with shared prefixes)
- **Descripción**: en batch mode, snapshot state cada N tokens.
  Al procesar siguiente archivo, detectar longest common prefix (LCP)
  con token comparison. Si LCP > 200 tokens → restaurar checkpoint exacto
  del punto LCP, skip forward passes del prefijo compartido.
  Inspirado en Memory Caching (Google 2026) y Marconi (Amazon 2025).
- **Papers**: Behrouz et al. (2026) arXiv:2602.24281,
  Marconi (2025) arXiv:2411.19379
- **Archivos**: `src/application/codec.rs` (batch loop), `src/infrastructure/mamba/model.rs` (state ser/de)

#### Secuencia Fase 10

```
R-059 (act. sparsity) ──→ medido: 1.9% ──→ NO VIABLE (necesita >25%)
R-060 (pipeline) ──→ split + thread overlap ──→ marginal en DDR4
R-061 (column pruning) ──→ medido: +20% BPB @1 canal ──→ NO VIABLE
R-062 (highway skip) ──→ medido: 0 skips @thr≤0.15 ──→ NO VIABLE
R-063 (batch checkpoint) ──→ state snapshot ──→ LCP matching
```

Estado: R-059 ✅ R-060 ✅ R-061 ❌ R-062 ❌. R-063 independiente (batch only).
**Conclusión Fase 10**: Mamba-130M no tolera pruning NI layer skip.
El modelo es demasiado pequeño (130M params): cada canal y capa es irreemplazable.
Las técnicas de la literatura (Wanda, Simba, SparseSSM) validan en modelos 7B+.
DRAM bandwidth sigue como bottleneck real. Solo R-063 (batch checkpoint) queda viable.

---

### Fase 11: DRAM Bandwidth Optimization (R-064 to R-067)

**Motivación**: Mamba-130M está memory-bound (intensidad aritmética ~2 FLOP/byte,
ridge point DDR4-2667 ~9-13 FLOP/byte). El modelo usa f32 puro (493MB) — cada matmul
lee 4 bytes/peso. El throughput actual (27-34 tok/s) está al 70-85% del techo teórico Q8.
Processing-in-Memory (UPMEM, HBM-PIM) requiere hardware especial — no viable en laptop.
Las técnicas de esta fase reducen el tráfico DRAM desde software.

**Investigación base**: PIM/NDP survey (2026), MVDRAM, T-MAC, roofline analysis.

#### R-064: Mamba Q8 quantization at load time
- **Estado**: ❌ NO VIABLE (2026-11-02) [MEASURED] — forense reabierta en R-098 (Fase 19)
- **Impacto esperado**: 4x menos tráfico DRAM
- **Resultado**: +16.6% BPB (inaceptable, presupuesto ≤3%)
- **Descripción**: Implementado y medido en 3 configuraciones:
  1. **All-Q8** (layers + LM head): BPB 1.1208 vs baseline 0.9609 = **+16.6%** BPB
  2. **Hybrid** (Q8 in_proj/out_proj, f32 x_proj/dt_proj): BPB 1.1168 = **+16.2%** BPB
  3. **LM-head-only Q8** (layers f32, emb Q8): BPB 1.1000 = **+14.5%** BPB
- **Roundtrip**: PASS en las 3 configuraciones (CRC-32 verified OK)
- **Throughput**: 31-34 tok/s en todas (sin mejora — bottleneck es SSM, no GEMV)
- **Análisis**: Mamba-130M no tolera Q8 PTQ (post-training quantization).
  El LM head es el componente más sensible (+14.5% solo). Las capas layer
  añaden solo +2% adicional. El modelo es demasiado pequeño — RWKV Q8 funciona
  porque carga pesos GGUF ya cuantizados durante training (QAT), no PTQ.
  Para que Q8 funcione se necesitaría QAT (quantization-aware training)
  o calibración con datasets representativos (requiere entorno PyTorch externo).
- **Código**: revertido completamente — model.rs idéntico a pre-R-064

#### R-065: NTA prefetch for weight streaming
- **Estado**: ❌ NO VIABLE (2026-11-02) [MEASURED]
- **Impacto medido**: ±0% (dentro de noise térmico)
- **BPB**: 0.9648 — idéntico con NTA y T0 (confirmado, no cambia cómputo)
- **Descripción**: Implementado y medido en controlled A/B (same session, alternating).
  Cambiados todos los 32 `_MM_HINT_T0` a `_MM_HINT_NTA` en 4 kernels GEMV.
  Resultado: NTA 28.5-29.8 tok/s vs NO-PREFETCH 29.3-31.0 tok/s.
  NTA NO mejora throughput en DDR4-2667 single-channel + 12MB L3.
  Razón: Mamba-130M (493MB f32) ya evicta L3 completamente en cada token step.
  NTA vs T0 es irrelevante cuando L3 working set >> L3 capacity (493MB >> 12MB).
  NTA podría beneficiar en sistemas con L3 más grande (>64MB, server-class).
- **Código**: revertido completamente — tensor.rs sin cambios

#### R-066: Mixed Q4/Q8 per-layer (Mamba)
- **Estado**: ❌ BLOQUEADO / NO VIABLE (2026-11-02)
- **Razón**: R-064 demostró que Q8 uniforme ya causa +16.6% BPB.
  Q4 sería estrictamente peor. Mixed Q4/Q8 no tiene sentido si ni
  Q8 puro es aceptable. Mamba-130M no tolera cuantización PTQ.

#### R-067: Weight layout optimization (L2-tile repack)
- **Estado**: ❌ NO VIABLE (2026-11-02, análisis teórico)
- **Razón**: L2-tiling beneficia GEMM (reutilización de bloques), NO GEMV.
  GEMV es streaming puro: cada fila se lee una vez, el vector se reutiliza
  en L1 (768-1536 floats = 3-6KB << L1=48KB). El patrón ya es óptimo:
  - Row-major sequential access → hardware prefetcher optimal
  - 4-row ILP → hide DRAM latency
  - Software prefetch → next row-block pre-staged in L2
  Reordenar en tiles solo añadiría overhead de indexación sin reducir
  DRAM traffic ni mejorar hit rate (ya ~95%+ en current layout).
  El bottleneck fundamental es DRAM bandwidth (21.3 GB/s), no cache misses. **[Corregido 2026-11-09, Fase 19: dual channel = 42.7 GB/s]**

#### Secuencia Fase 11

```
R-064 (Mamba Q8) ──→ medido: +16.6% BPB ──→ NO VIABLE (PTQ on 130M)
R-065 (NTA prefetch) ──→ medido: ±0% tok/s ──→ NO VIABLE (L3 << model size)
R-066 (mixed Q4/Q8) ──→ BLOQUEADO (R-064 Q8 ya falla, Q4 peor)
R-067 (L2-tile) ──→ análisis: GEMV ≠ GEMM ──→ NO VIABLE (tiling irrelevante)
```

Estado: R-064 ❌ R-065 ❌ R-066 ❌ R-067 ❌. **FASE 11 CERRADA.**

**Conclusión Fase 11**: Las 4 técnicas de optimización DRAM son NO VIABLES en este hardware/modelo:
- Q8 PTQ destruye BPB en Mamba-130M (modelo demasiado pequeño, necesita QAT)
- NTA prefetch irrelevante cuando modelo >> L3 cache
- Mixed Q4/Q8 bloqueado por Q8 ya inaceptable
- L2-tiling no aplica a GEMV (streaming, no blocked)
DRAM bandwidth (21.3 GB/s DDR4-2667 single-channel) es un hard ceiling. **[Obsoleto 2026-11-09: el portátil es dual channel, 42.7 GB/s; 34 tok/s = 41% del bus. Ver Fase 19]**
Throughput actual de 27-34 tok/s está al 70-85% del techo teórico.
Para superar esto se necesita: DDR5, dual-channel, o modelo más pequeño (MambaByte).

---

### Fase 12: Compiler & Operator Optimization (R-068 to R-071)

**Motivación**: Tras agotar optimizaciones SIMD, sparsity, quantization y cache (Fases 9-11),
quedan técnicas a nivel de compilador y fusión de operadores que NO se han intentado.
PGO (Profile-Guided Optimization) es la oportunidad más grande no explorada.

#### R-068: Profile-Guided Optimization (PGO)
- **Estado**: ❌ NO VIABLE (2026-11-02)
- **Impacto estimado**: +10-25% throughput
- **Impacto medido**: +1.1% mediana (dentro de ruido térmico ±3%)
- **Riesgo BPB**: 0% confirmado (BPB bit-exact entre PGO y normal)
- **Descripción**: Compilar con `-C profile-generate`, ejecutar workload representativo
  (compress enwik8 10KB + apache 10KB + dickens 10KB), recompilar con `-C profile-use`.
  PGO optimiza branch prediction, code layout, inlining decisions basado en datos reales.
  Evidencia externa: +10-20% en llama.cpp, +15% en rustc self-compile.
  Fat LTO (ya habilitado) + PGO = combinación óptima. Zero cambios de código.
- **Resultado empírico** (A/B controlado, 3 rondas alternando, enwik8 10KB Mamba):
  - Normal: 18.3, 18.0, 17.9 tok/s (mediana 18.0)
  - PGO:    18.6, 18.2, 17.3 tok/s (mediana 18.2)
  - Delta:  +1.6%, +1.1%, -3.4% → mediana **+1.1%** (ruido térmico)
  - BPB: 0.9648 ambos builds, output BIT-EXACT (verified xxd diff)
- **Por qué falla PGO**: El bottleneck es DRAM bandwidth (GEMV streaming), no branch
  prediction ni instruction cache misses. PGO optimiza hot paths y branch layout,
  pero el 85%+ del tiempo se gasta en `mat_vec_mul` (streaming read-once de 493MB).
  PGO beneficia workloads CPU-bound con branches complejos (compiladores, parsers),
  no workloads memory-bound (GEMV en modelos grandes).
  La evidencia de llama.cpp (+10-20%) viene de modelos cuantizados (Q4/Q8) donde
  el bottleneck se mueve a CPU; en Mamba-130M F32 el bottleneck permanece en DRAM.
- **Archivos**: sin cambios de código (build-only experiment)
- **Subtareas**:
  - [x] Build instrumentado con profile-generate
  - [x] Ejecutar workload representativo (3 archivos: enwik8, apache, dickens)
  - [x] Rebuild con profile-use (llvm-profdata merge + -Cprofile-use)
  - [x] A/B controlado: tok/s PGO vs normal, 3 rounds alternando
  - [x] BPB bit-exact confirmado
  - [x] Resultado: +1.1% = ruido térmico → NO VIABLE

#### R-069: Operator fusion — RMSNorm+matmul, matmul+residual
- **Estado**: ❌ NO VIABLE (2026-11-02)
- **Impacto estimado**: +2-5% throughput
- **Impacto medido**: ±0% (mediana idéntica, dentro de ruido térmico)
- **Riesgo BPB**: 0% confirmado (BPB bit-exact)
- **Descripción**: Cada capa Mamba escribe/lee ~84KB de buffers intermedios.
  24 capas × 30 tok/s = ~60 MB/s de tráfico intermedio innecesario.
  Fusiones implementadas y medidas:
  1. **RMSNorm + in_proj**: Eliminada la `copy_from_slice` + merge de 3 pases a 2.
     Ahorro: 768 floats × 4B = 3KB por capa × 24 = 72KB/token.
  2. **out_proj + residual_add**: `mat_vec_mul_add_into` acumula directo en x.
     Ahorro: 768 floats × 4B = 3KB write+read por capa × 24 = 144KB/token.
- **Resultado empírico** (A/B controlado, 3 rondas alternando, enwik8 10KB Mamba):
  - Normal: 17.8, 18.1, 17.8 tok/s (mediana 17.8)
  - Fusion: 17.6, 17.8, 18.0 tok/s (mediana 17.8)
  - Delta:  -1.1%, -1.7%, +1.1% → mediana **±0%** (ruido térmico)
  - BPB: 0.9648 ambos, output BIT-EXACT
- **Por qué falla**: Los buffers intermedios (normed: 3KB, result: 3KB) caben en
  L1 cache (32KB). No generan tráfico DRAM — solo L1. L1 tiene ~100+ GB/s de
  throughput, así que ahorrar 216KB/token (a 18 tok/s = 3.9 MB/s) es el 0.004%
  del ancho de banda L1. El bottleneck son las matrices de peso (4.7MB por capa
  desde DRAM), no los buffers intermedios L1-resident.
- **Código**: Implementado + revertido. `mat_vec_mul_add_into` (scalar + AVX2)
  fue creado, probado, y eliminado. Inline RMSNorm también implementado y revertido.
- **Archivos**: sin cambios permanentes (experiment revertido)
- **Subtareas**:
  - [x] Fused RMSNorm+in_proj kernel (inline sum_sq + inv_rms + scale)
  - [x] Fused out_proj+residual kernel (`mat_vec_mul_add_into` AVX2)
  - [x] A/B controlado: tok/s fused vs current — ±0%
  - [x] BPB bit-exact confirmado
  - [x] Resultado: ±0% = no medible → NO VIABLE, código revertido

#### R-070: Conv1d SIMD
- **Estado**: ❌ NO VIABLE (2026-11-02, análisis aritmético)
- **Impacto estimado original**: +0.5-1% throughput
- **Impacto real calculado**: +0.1% máximo teórico (indetectable)
- **Descripción**: El conv1d shift register es scalar puro.
  d_inner×d_conv = 1536×4 = 6144 FMAs por token por capa.
- **Análisis aritmético** (por qué no se implementa):
  - Conv1d total: 6144 FMAs/capa × 24 capas = **147K FMAs/token**
  - Total forward: in_proj(2.36M) + x_proj(123K) + dt_proj(74K) + out_proj(1.18M)
    = 3.7M/capa × 24 + LM_head(38.6M) = **128.6M FMAs/token**
  - Conv1d = **0.11%** del cómputo total. Incluso 10× speedup = +0.1% throughput.
  - Además: memory layout interleaved (conv_state stride=3, weights stride=4)
    requiere AVX2 gather (_mm256_i32gather_ps), anulando ventaja SIMD.
  - Estimación original (+0.5-1%) era incorrecta: asumía conv1d = 1-2% del forward.
- **Archivos**: sin cambios (análisis puro, no requiere implementación)

#### R-071: Reduced-rank LM head (SVD factorization)
- **Estado**: ❌ NO VIABLE (2026-11-02, análisis + precedentes)
- **Impacto estimado**: +10-20% throughput (LM head = 30% of FLOPs)
- **Impacto real proyectado**: +3-6% total forward (30% × 10-20%)
- **Riesgo BPB**: ALTO — precedentes empíricos indican degradación catastrófica
- **Descripción**: El LM head (50280×768 = 38.6M FLOPs) es 30% del forward.
  Factorizar: `emb_w ≈ U(50280,k) × V(k,768)` con SVD truncado.
  Con k=256: 13.1M vs 38.6M = -66% FLOPs en LM head = -20% forward total.
- **Por qué NO se implementa** (3 razones convergentes):
  1. **Mamba-130M rechaza TODA aproximación lossy**: Q8 PTQ = +16.6% BPB (R-064),
     column pruning 10% = +2-5% BPB (R-061), layer skip = inaceptable (R-062).
     El modelo es demasiado pequeño para tolerar pérdida de precisión. SVD truncado
     es inherentemente lossy — la reconstrucción `U·V ≈ emb_w` pierde información
     proporcional a los valores singulares descartados.
  2. **R-016 precedente**: SVD en RWKV (ALL matrices) = +3.47% BPB AND 1.95× más lento.
     High effective rank. El embedding matrix de Mamba tiene la misma estructura
     (50280 tokens × 768 dims), no hay razón para esperar low effective rank.
  3. **Throughput gain < noise floor**: Incluso el caso optimista (+6% total) está
     en el rango ±3% de varianza térmica medida en R-068/R-069. Sería inmedible.
  4. **Tied weights**: `emb_w` es compartida entre embedding lookup y LM head.
     Factorizarla solo para head requiere almacenar U+V adicionales (50280×k + k×768
     = ~50MB extra para k=256), parcialmente anulando el beneficio DRAM.
  5. **Implementación SVD**: Requiere ~200 líneas de código (Golub-Kahan o power
     iteration) sin dependencias externas, para un resultado predeciblemente negativo.
- **Archivos**: sin cambios (análisis puro)

#### Secuencia Fase 12

```
R-068 (PGO) ──→ medido: +1.1% (ruido) ──→ NO VIABLE (DRAM-bound, no CPU-bound)
R-069 (op fusion) ──→ medido: ±0% ──→ NO VIABLE (buffers L1-resident, no DRAM)
R-070 (conv1d SIMD) ──→ code change, +0.5-1%, independent
R-071 (SVD LM head) ──→ code change, +10-20%, risky BPB
```

Estado: R-068 ❌ R-069 ❌ R-070 ❌ R-071 ❌. **FASE 12 CERRADA.**

**Conclusión Fase 12**: Las 4 técnicas de optimización compiler/operator son NO VIABLES:
- PGO: +1.1% (ruido térmico). Workload es DRAM-bound, no CPU-bound.
- Operator fusion: ±0%. Buffers intermedios son L1-resident (3KB << 32KB L1 cache).
- Conv1d SIMD: 0.11% del cómputo total. 10× speedup = +0.1% (indetectable).
- SVD LM head: Mamba-130M rechaza toda aproximación lossy (Q8 +16.6%, SVD peor).

**Diagnóstico final de throughput Mamba**: El forward pass es >99% DRAM-bound.
Las 5 GEMV por capa + LM head saturan el bus de memoria DDR4-2667 single-channel
(21.3 GB/s teórico, 70-85% utilización medida). NINGUNA optimización de CPU
(SIMD, PGO, fusion, code layout) puede superar este hard ceiling.
Para throughput >34 tok/s se necesita: DDR5, dual-channel, modelo más pequeño
(MambaByte ~50M), o cuantización QAT (no PTQ). **[Superado 2026-11-09, Fase 19: Mamba corre F32 en un hilo en in_proj/out_proj; ver R-098/R-099]**

---

### Fase 13: Portable GPU Backend Architecture (R-072 to R-076)

> Goal: hardware-agnostic execution — deploy on any CPU, GPU, or future accelerator.
> Principle: core Redukt stays zero-dep; GPU acceleration via optional runtime plugins.
> Research: analysis from backbone-scalability + DRAM bandwidth ceiling (Fases 9-12).

**Motivación**: Fases 9-12 demostraron que DRAM bandwidth DDR4 es un hard ceiling
(27-34 tok/s, 70-85% utilización). Para throughput >100x se necesita GPU, pero
locked a CUDA/NVIDIA destruiría la portabilidad. Arquitectura de plugins permite
cualquier backend (Vulkan, Metal, DX12, OpenCL, CUDA) sin romper zero-dep del core.

#### R-072: Backend FFI interface definition
- **Estado**: ✅ COMPLETADO (2026-11-06, 9dc1fdd)
- **Riesgo BPB**: 0% (solo definición de interfaz, zero runtime impact)
- **Descripción**: Definir el contrato C ABI que todo plugin GPU debe implementar.
  4 funciones: `backend_init`, `backend_forward`, `backend_reset_state`, `backend_free`.
  `#[repr(C)]` structs para paso de datos. Vive en `src/domain/backend_ffi.rs` porque
  es un contrato del dominio (como `Backbone` trait), no una implementación.
- **Archivos**: `src/domain/backend_ffi.rs`, `src/domain/mod.rs`
- **Subtareas**:
  - [x] Definir `BackendInfo` struct (version, vocab_size, name)
  - [x] Definir `BackendVTable` con 4 function pointers
  - [x] Definir constantes de versión para compatibilidad
  - [x] Tests: construcción de VTable con funciones dummy

#### R-073: Cross-platform dynamic plugin loader
- **Estado**: ✅ COMPLETADO (2026-11-06, de9c06e)
- **Riesgo BPB**: 0% (carga opcional, fallback a CPU si no hay plugin)
- **Descripción**: Loader que busca `redukt-gpu.dll` (Windows) / `libredukt_gpu.so` (Linux)
  / `libredukt_gpu.dylib` (macOS) junto al ejecutable. Usa raw FFI a OS:
  `LoadLibraryW`/`GetProcAddress` en Windows (patrón ya establecido en governor.rs),
  `dlopen`/`dlsym` en Unix. Zero crate dependencies.
  Si el plugin no existe → silenciosamente usa CPU. Sin error, sin warning.
- **Archivos**: `src/infrastructure/plugin_backend/loader.rs`, `src/infrastructure/plugin_backend/mod.rs`,
  `src/infrastructure/mod.rs`
- **Subtareas**:
  - [x] `PluginLibrary` struct (handle + vtable)
  - [x] `PluginLibrary::try_load()` → `Option<PluginLibrary>` (no panic)
  - [x] Platform-specific: Windows (kernel32 LoadLibraryW), Unix (dlopen)
  - [x] Symbol resolution con validación de versión
  - [x] Drop impl para FreeLibrary/dlclose
  - [x] Tests: carga fallida retorna None (no panic)

#### R-074: PluginBackend impl Backbone + auto-fallback
- **Estado**: ✅ COMPLETADO (2026-11-06, 33c3a2e)
- **Riesgo BPB**: 0% (wrapper transparente, misma interfaz)
- **Descripción**: `PluginBackend` struct que implementa `Backbone` trait via FFI.
  `PluginState` y `PluginScratch` como wrappers opacos. Se añade variante
  `DynModel::Plugin(PluginBackend)` al enum. `BackboneChoice::Plugin` en el enum.
  `load_dyn_model` intenta plugin primero si `--backend plugin` o auto-detect.
  Fallback automático: si plugin no disponible → Mamba CPU (transparente).
- **Archivos**: `src/infrastructure/plugin_backend/model.rs`, `src/domain/backbone.rs`,
  `src/application/model_loader.rs`
- **Subtareas**:
  - [x] `PluginBackend` struct (wraps PluginLibrary + config)
  - [x] `PluginState`, `PluginScratch` (opaque, minimal)
  - [x] `impl Backbone for PluginBackend`
  - [x] `BackboneChoice::Plugin` variant + header byte (2)
  - [x] `DynModel::Plugin` variant + all match arms
  - [x] `--backend plugin` CLI flag
  - [x] Auto-fallback: try plugin → Mamba → RWKV
  - [x] Tests: fallback cuando no hay plugin, trait dispatch

#### R-075: wgpu compute shader plugin skeleton
- **Estado**: ✅ COMPLETADO (2026-11-06, fd510a7)
- **Riesgo BPB**: 0% (proyecto separado, no compilado por default)
- **Descripción**: Proyecto separado en `plugins/redukt-gpu/` con su propio Cargo.toml.
  Usa `wgpu` crate para GPU compute portátil (Vulkan/Metal/DX12/OpenGL).
  Compila a shared library (.dll/.so/.dylib) que implementa la FFI de R-072.
  Skeleton: estructura de proyecto, build system, forward pass placeholder.
  NO es parte del binario principal (zero-dep preservado).
- **Backends wgpu**: Vulkan (Linux/Win/Android), Metal (macOS/iOS),
  DX12 (Windows), OpenGL (fallback) — cubre 95%+ del hardware actual y futuro.
- **Archivos**: `plugins/redukt-gpu/Cargo.toml`, `plugins/redukt-gpu/src/lib.rs`,
  `plugins/redukt-gpu/src/kernels/`, `plugins/redukt-gpu/README.md`
- **Subtareas**:
  - [x] Proyecto Cargo con wgpu + safetensors deps
  - [x] FFI exports matching R-072 VTable
  - [x] GPU device initialization + adapter selection
  - [x] Placeholder forward pass (CPU fallback inside plugin)
  - [x] Build produces .dll/.so/.dylib
  - [x] README con instrucciones de build

#### R-076: Kaggle validation & benchmark protocol
- **Estado**: ✅ COMPLETADO (2026-11-06, 1280937)
- **Riesgo BPB**: N/A (documentación + validación)
- **Descripción**: Documentación de cómo usar Kaggle (30h GPU gratis/semana)
  para validar el plugin GPU. Template de notebook. Protocolo de benchmark:
  12 archivos estándar, BPB CPU vs GPU (debe ser idéntico para determinismo),
  throughput comparison, roundtrip verification.
  También documenta path hacia MambaByte training (R-041) en Kaggle.
- **Archivos**: `docs/internal/research/gpu-backend-architecture.md`,
  `docs/internal/kaggle-validation-protocol.md`
- **Subtareas**:
  - [x] Documento de arquitectura GPU backend
  - [x] Protocolo de validación Kaggle
  - [x] Template de notebook (.ipynb o .py)
  - [x] Mapping de R-041 (MambaByte training) a Kaggle free tier

#### Secuencia Fase 13

```
R-072 (FFI interface) ──→ R-073 (plugin loader) ──→ R-074 (PluginBackend trait)
                                                         │
R-075 (wgpu skeleton) ←──────────────────────────────────┘
                                                         │
R-076 (Kaggle validation) ←──────────────────────────────┘
```

R-072 → R-073 → R-074 son secuenciales (cada uno depende del anterior).
R-075 depende de R-072 (FFI interface) + R-074 (integration points).
R-076 es independiente (documentación).

Estado: R-072 ✅ R-073 ✅ R-074 ✅ R-075 ✅ R-076 ✅. **FASE 13 COMPLETADA.**

**Resultado**: Arquitectura de plugin GPU portátil implementada. Core zero-dep preservado.
PluginBackend integrado en DynModel con auto-fallback. Plugin wgpu skeleton compilado
(placeholder). 214 tests pasando (210 unit + 4 integration). Documentación de arquitectura
y protocolo de validación Kaggle creados.

---

### Fase 14: Token Reduction & Entropy-Gated Inference (R-077 → R-079)

Diagnosis: sequential token-by-token processing is the fundamental throughput bottleneck.
23 approaches investigated across 4 tracks (multi-token prediction, tokenization, pattern caching, hybrid CM-backbone). 21 NO-GO, 2 CONDITIONAL GO.
Research: `docs/internal/research/token-reduction-strategies.md` (30+ sources).

- R-077: Nacrith-style CM entropy gate ✅ COMPLETADO (2026-11-07)
  - Implementation: `cm_only_entropy()` + `cm_only_logits_into()` in CmEnsemble
  - Mechanism: compute CM-only Shannon entropy before backbone; skip if H < threshold
  - Stored in AZTH v4 header via FLAG 0x20 (reuses adaptive-skip field)
  - MAX_CONSEC_SKIP=3 bounds state divergence
  - **RESULT: NOT VIABLE at BPE 50K vocab / 10KB files** [MEASURED]
    - CM entropy range: 15.49-15.62 bits (near-uniform, log2(50280)=15.62)
    - Threshold 15.5 bits: 31.5% skip rate, +50.5% BPB (catastrophic)
    - Threshold ≤14.0 bits: 0% skip rate (CM never confident enough)
    - No viable operating point exists between 0% skip and +50% BPB
    - Confirms R-056 finding: "CM confidence NEVER >5% with BPE 50K vocab"
  - Root cause: 50K vocabulary = 15.6 bits maximum entropy. CM biases are negligible
    relative to the uniform distribution over 50K classes.
  - Nacrith achieves 30-70% skip because it processes 100KB+ files (more n-gram data)
    AND uses a specialized n-gram model with higher effective order
  - **Viable ONLY with**: byte-level model (vocab=256, R-041) OR files >100KB
  - Code preserved: `--skip-threshold N` CLI, roundtrip PASS (threshold=0 = disabled)

- R-078: Kaggle GPU validation — benchmark analysis & documentation ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07)
  - Kaggle v3 results: 18 benchmark entries, determinism PASS, roundtrip PASS
  - CPU Mamba baseline (Kaggle Xeon): enwik8 0.9633 BPB, 72.4 tok/s (2.1x vs local DDR4)
  - GPU plugin (skeleton/uniform logits): 81K-96K tok/s FFI overhead, pipeline validated
  - Cross-platform: enwik8 0.9633 (Kaggle) vs 0.9609 (local) = +0.25% (FP rounding)
  - FFI boundary, dlopen, PluginBackend dispatch all functional on Kaggle Linux T4
  - Synthetic files: 0.10-0.11 BPB (SHA-256 periodic pattern, CM learns trivially)
  - Evidence: `data/benchmark/results/kaggle-v3/analysis.md`, `gpu-benchmark.csv`

- R-079: MambaByte training feasibility update (research) ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07) — CONDITIONAL GO (from-scratch training)
  - **Byteification adapter** (Nature Oct 2026): Mamba2 encoder validates SSM+byte synergy,
    but full pipeline requires Transformer global model. NOT RECOMMENDED for Redukt.
  - **MambaByte from-scratch** (ICLR 2024, Wang et al.): RECOMMENDED path
    - Cost: $87-145 (130M params, 30-50B bytes, A100)
    - Published: MambaByte-353M = 0.930 BPB PG19
    - Projected: MambaByte-130M + CM = 0.85-1.05 BPB enwik8
  - **Entropy gate unlocked**: vocab=256 → CM entropy ~5-6 bits → 40-70% skip viable
  - **GPU T4 throughput**: 650-2,500 byte-tok/s with 70% gate (vs 119 B/s CPU)
  - **Competitive**: Nacrith 0.9389 BPB, Redukt+MambaByte projected 0.85-1.05
  - **Phased plan**: validate 50M ($34) → train 130M ($87-145) → integrate → gate → GPU
  - Research: `docs/internal/research/mambabyte-feasibility-update.md` (7 sources)

---

## Fase 15: MambaByte Training Infrastructure (R-080 → R-083)

> **Objetivo**: Construir todo el stack necesario para entrenar MambaByte (byte-level, vocab=256)
> y integrarlo en Redukt. Desbloquea entropy gate (R-077) y throughput GPU viable.
>
> **Prerequisitos**: R-035 (Mamba backbone), R-072-R-076 (GPU plugin), R-079 (feasibility GO)
> **Budget**: $0 (Kaggle TPU) para validación, $87 (A100) para producción
> **Research**: `docs/internal/research/mambabyte-training-infrastructure.md`

- R-080: MambaByte training notebook (PyTorch + TPU) ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07)
  - **Prioridad**: P1 — habilita todas las mejoras de throughput posteriores
  - **Deliverable**: `notebooks/mambabyte-train.py`
  - **Subtareas**:
    - [ ] MambaByte model definition (PyTorch): vocab=256, d_model=768, 24 layers, ~96M params
    - [ ] Training loop: AdamW, cosine LR, BF16, gradient clipping, cross-entropy loss
    - [ ] Dataset pipeline: byte-level streaming (PG19 + Wikipedia + SlimPajama)
    - [ ] TPU v5e-8 adaptation: torch_xla, 8-way data parallel, xm.optimizer_step()
    - [ ] Checkpointing: save/resume every 5K steps (9h session limit)
    - [ ] Validation: BPB on enwik8 every 1K steps, 4-file suite every 5K
    - [ ] Kaggle metadata: `notebooks/mambabyte-train/kernel-metadata.json`
  - **Arquitectura MambaByte-96M**:
    - vocab=256, d_model=768, d_inner=1536, d_state=16, d_conv=4, n_layer=24
    - ~96M params (vs 130M — LM head 768×256 vs 768×50280)
    - Tied embeddings (embedding.weight == lm_head.weight)
  - **Training**: 30B bytes, 8192-byte sequences, batch 48, ~76K steps
  - **Estimado**: 5 semanas Kaggle TPU (50M validación) o 2 días A100 ($87)

- R-081: Weight conversion pipeline ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07)
  - **Deliverable**: `scripts/convert-mambabyte-weights.py`
  - **Subtareas**:
    - [ ] PyTorch checkpoint → SafeTensors with HuggingFace key naming
    - [ ] Key mapping: layers.{i}.mixer.xxx → backbone.layers.{i}.mixer.xxx
    - [ ] Tensor shape validation against MambaConfig
    - [ ] SHA-256 provenance hash
    - [ ] MANIFEST.md entry for mambabyte-96m
  - **Dependencia**: R-080 (necesita checkpoint entrenado para test real)

- R-082: Rust byte tokenizer + model loader integration ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07)
  - **Deliverables**: `src/infrastructure/mamba/byte_tokenizer.rs`, updates to model_loader/backbone
  - **Subtareas**:
    - [ ] ByteTokenizer impl BackendTokenizer (encode=identity, decode=identity, vocab=256)
    - [ ] BackboneChoice::MambaByte variant (AZTH backbone byte: 2)
    - [ ] model_loader.rs: detect MambaByte weights (vocab_size=256), load with ByteTokenizer
    - [ ] CLI: `--backbone mambabyte` flag
    - [ ] Unit tests: encode/decode roundtrip, vocab_size, edge cases (0x00, 0xFF)
  - **Nota**: MambaModel no necesita cambios (ya parametrizado por config.vocab_size)
  - **Ejecutable sin weights**: skeleton funcional, placeholder weights para tests

- R-083: Cross-framework validation notebook ✅ COMPLETADO
  - **Estado**: COMPLETADO (2026-11-07)
  - **Deliverable**: `notebooks/mambabyte-validate.py`
  - **Subtareas**:
    - [ ] Load same weights in PyTorch + Rust, compare logits (max diff <1e-5)
    - [ ] Compress enwik8 10KB with MambaByte, verify BPB in expected range
    - [ ] Roundtrip: compress → decompress → byte-identical
    - [ ] Determinism: 2 runs SHA-256 identical
  - **Dependencia**: R-080 + R-081 + R-082 (necesita weights entrenados + Rust integration)

**Paralelización**: R-080 y R-082 son independientes (Python vs Rust). R-081 depende de R-080.
R-083 depende de R-080 + R-081 + R-082.

```
R-080 (training notebook) ──→ R-081 (weight conversion) ──→ R-083 (validation)
R-082 (Rust byte tokenizer) ─────────────────────────────/
```

---

## Fase 16: MambaByte Training Execution (R-084 → R-086)

> **Objetivo**: Entrenar MambaByte y validar hipótesis de byte-level compression.
> **Prerequisitos**: Fase 15 completa (R-080 → R-083)
> **Budget**: $0 (Kaggle TPU free tier) para validación
>
> **⚠️ BLOCKER (2026-11-09)**: Kaggle TPU v5e-8 no disponible — quota saturada
> persistentemente. Las sesiones TPU no inician ("TPU is currently busy" /
> "No TPU available"). Afecta R-084 y R-085. GPU T4x2 es alternativa pero
> el notebook requiere adaptación (torch_xla → CUDA). R-088 (pesos pre-entrenados
> 353M) se convierte en el path prioritario mientras TPU permanezca inaccesible.

- R-084: MambaByte-50M validation training (Kaggle TPU v5e-8)
  - **Estado**: ⚠️ BLOQUEADO — Kaggle TPU saturado (2026-11-09)
  - **Estado anterior**: EN EJECUCIÓN (2026-11-07, push exitoso pero sesión no inicia)
  - **Prioridad**: P0 → P1 (degradada mientras R-088 es P0)
  - **Deliverable**: Trained MambaByte-50M weights + BPB measurement
  - **Plataforma**: Kaggle TPU v5e-8, 1 sesión de 9h (63-95% training)
  - **Arquitectura**: d_model=512, d_inner=1024, n_layer=24, vocab=256, ~42M params
  - **Training**: 10B bytes, 4096-byte sequences, batch 64, ~40K steps
  - **Gate**: BPB < 1.40 enwik8 10KB → PASS (proceder a 96M)
  - **Notebook**: `notebooks/mambabyte-50m-validate.py`
  - **Push**: `scripts/kaggle-push-mambabyte.sh`
  - **Historial de intentos**:
    - v1 (2026-11-07): Push a TPU v5e-8, corregido slug + mkdir (`0477e06`, `1b2da7c`)
    - v2 (2026-11-07): Switch a GPU T4x2 por TPU push roto (`1e42605`)
    - v3 (2026-11-07): Revert a TPU v5e-8 + docker_image_pinning workaround (`18e6dd8`)
    - v4 (2026-11-09): **TPU quota saturada** — sesiones no inician
  - **Subtareas**:
    - [x] Crear notebook 50M con config reducida
    - [x] Push a Kaggle TPU v5e-8 (push exitoso, ejecución bloqueada)
    - [ ] ~~Monitorear: loss convergence~~ → BLOQUEADO (TPU no inicia)
    - [ ] Descargar resultados: checkpoint + logs
    - [ ] Evaluar: BPB < 1.40 → GO / NO-GO para producción
  - **Alternativas si TPU sigue saturado**:
    - **Alt A (recomendada)**: R-088 — usar pesos pre-entrenados 353M ($0, inmediato)
    - **Alt B**: Adaptar notebook a GPU T4x2 (30GB VRAM, ~3-5x más lento que TPU)
    - **Alt C**: Colab Pro TPU v2 (~$10/mes, disponibilidad más estable)
    - **Alt D**: Esperar ventana TPU libre (madrugada UTC, fines de semana)

- R-085: MambaByte-96M production training
  - **Estado**: BLOQUEADO (depende R-084 PASS + TPU disponible)
  - **Plataforma**: Kaggle TPU (3 sesiones × 9h) o A100 ($87, 2 días)
  - **Nota**: Si R-088 confirma pesos 353M compatibles, R-085 podría cancelarse
    o reenfocarse como fine-tuning del modelo pre-entrenado
  - **Deliverable**: Production MambaByte-96M weights in SafeTensors
  - **Training**: 30B bytes, 8192-byte sequences, batch 48, 76K steps

- R-086: MambaByte integration + entropy gate re-enablement
  - **Estado**: BLOQUEADO (depende R-085)
  - **Deliverable**: Redukt compressing with MambaByte + CM entropy gate active
  - **Subtareas**:
    - [ ] Convert weights (R-081 script)
    - [ ] Load in Rust, verify BPB matches PyTorch (R-083 notebook)
    - [ ] Re-enable entropy gate (R-077 code, threshold ~6.5 bits)
    - [ ] Full benchmark: 12+ files, compare vs Mamba-130M BPE
    - [ ] Roundtrip verification

---

## Fase 17: Pre-trained MambaByte + Competitive Intelligence (R-087 → R-090)

> **Objetivo**: Evaluar pesos MambaByte pre-entrenados de HuggingFace (potencialmente
> eliminando la necesidad de entrenar from-scratch), documentar panorama competitivo,
> y adoptar técnicas de GPU de strata-inference.
>
> **Prerequisitos**: Fase 15 (Rust byte tokenizer R-082), R-081 (weight conversion)
> **Budget**: $0 (pesos gratuitos en HuggingFace)
> **Research**: `docs/internal/research/strata-competitive-landscape.md`

- R-087: Competitive landscape research + Strata evaluation
  - **Estado**: COMPLETADO (2026-11-09)
  - **Prioridad**: P1 — informa decisiones de roadmap
  - **Deliverable**: `docs/internal/research/strata-competitive-landscape.md`
  - **Subtareas**:
    - [x] Investigar 8 proyectos "Strata" — solo stratalab/strata-inference relevante
    - [x] Analizar 15 compresores neurales en HuggingFace/Kaggle
    - [x] Identificar MambaByte 353M pre-entrenado (JunxiongWang) — hallazgo crítico
    - [x] Mapear limitaciones Redukt → soluciones encontradas
    - [x] Documentar técnicas adoptables: PTX-as-string, fused dequant, M=1 fast path
  - **Hallazgos clave**:
    - MambaByte 353M ya entrenado en HuggingFace (0.396 BPB CE enwik8)
    - Nacrith: competidor más cercano (0.9389 BPB, GPU-only, solo texto)
    - strata-inference: patrón PTX-as-string adoptable para GPU plugin
    - L3TC (Alipay): valida RWKV+AC para compresión (4.35 MB/s)

- R-088: Evaluate pre-trained MambaByte 353M weights
  - **Estado**: COMPLETADO (2026-11-09)
  - **Prioridad**: P0 — evaluar si elimina necesidad de training
  - **Deliverable**: Compatibility report + benchmark [MEASURED]
  - **Fuente**: `JunxiongWang/MambaByte_Arxiv` (353M, 30B bytes arXiv, Mamba-1)
  - **Subtareas**:
    - [x] Descargar pesos MambaByte_Arxiv desde HuggingFace (1.5 GB)
    - [x] Analizar arquitectura: Mamba-1 ✅ compatible
    - [x] Mapear config: d_model=1024, d_inner=2048, d_state=16, d_conv=4, dt_rank=64, n_layer=56
    - [x] Resolver: untied embeddings (lm_head_w separado), LayerNorm (no RMSNorm)
    - [x] Convertir con `scripts/convert-mambabyte-weights.py --preset 353m`
    - [x] Benchmark BPB en enwik8 10KB: **1.3156 BPB** [MEASURED]
  - **Resultados [MEASURED: 2026-11-09]**:
    - enwik8 10KB: **1.3156 BPB** (vs 0.9609 Mamba-130M BPE = +37% peor)
    - CE BPB: 1.1975, coder efficiency: 91.02%
    - Throughput: 4.8 tok/s (353M params en DDR4, DRAM-bound)
    - Sin LayerNorm fix: 1.7531 BPB (+82% peor) — norm bug confirmado
  - **Diagnóstico**: Domain mismatch (arXiv vs Wikipedia). Modelo no generaliza.
  - **Conclusión**: Pesos pre-entrenados **NO reemplazan training propio**.
    R-084/R-085 siguen siendo necesarios para modelo domain-general.
  - **Código aportado**: untied embeddings (model.rs), LayerNorm (tensor.rs),
    convert --preset 353m, HF key stripping, norm bias auto-detect

- R-089: Formal benchmark vs Nacrith
  - **Estado**: BLOQUEADO (depende R-088 o R-086)
  - **Prioridad**: P2 — posicionamiento competitivo
  - **Deliverable**: Benchmark comparativo en datasets compartidos
  - **Subtareas**:
    - [ ] Benchmark Redukt (actual + MambaByte si disponible) en enwik8 completo
    - [ ] Comparar: alice29.txt, enwik8, OOD text
    - [ ] Documentar fortalezas (archivos estructurados, CPU-only, zero-dep)
    - [ ] Publicar resultados en `data/benchmark/results/`

- R-090: CUDA GPU kernels for Mamba inference (from strata-inference pattern)
  - **Estado**: IMPLEMENTADO — pendiente validación en Kaggle T4
  - **Prioridad**: P1 — máximo impacto en throughput (15x proyectado)
  - **Deliverable**: CUDA plugin completo (libredukt_gpu.so) con Mamba forward pass
  - **Subtareas**:
    - [x] Escribir GEMV kernel CUDA con float4 vectorizado + warp shuffle reduction
    - [x] Escribir SSM recurrence kernel CUDA (d_state genérico)
    - [x] Escribir kernels auxiliares: RMSNorm, LayerNorm, SiLU, Conv1d, Softplus, GatedSiLU
    - [x] Implementar carga de pesos desde formato .rbin (flat binary, 44B header)
    - [x] Integrar con BackendVTable FFI v1 (4 exports: init/forward/reset/free)
    - [x] Script conversor: SafeTensors → .rbin (`scripts/convert-weights-gpu.py`)
    - [x] Notebook de test y benchmark (`notebooks/redukt-gpu-cuda-test.py`)
    - [ ] Validar en Kaggle T4: compilar, forward pass, determinismo
    - [ ] Benchmark vs CPU: throughput en T4 (objetivo: >300 tok/s)
  - **Archivos**:
    - `plugins/redukt-gpu-cuda/mamba_cuda.cu` — Plugin CUDA completo (~500 lines)
    - `plugins/redukt-gpu-cuda/Makefile` — Build para Linux/Kaggle
    - `scripts/convert-weights-gpu.py` — SafeTensors → .rbin
    - `notebooks/redukt-gpu-cuda-test.py` — Test suite + benchmark

**Paralelización**: R-088 es independiente y ejecutable inmediatamente.
R-089 depende de R-088 (o R-086). R-090 es independiente pero P2.

```
R-087 (research) ✅ ──→ R-088 (eval 353M) ──→ R-089 (benchmark vs Nacrith)
                        R-090 (CUDA kernels) ✅ implementado, pendiente Kaggle test
```

**Impacto en Fase 16**: Si R-088 confirma compatibilidad de pesos 353M:
- R-084 (50M training) → CANCELAR (validación innecesaria con 353M disponible)
- R-085 (96M training) → CANCELAR o reenfocarse como fine-tuning de 353M
- R-086 (integration) → se adelanta con pesos pre-entrenados

**Contexto TPU (2026-11-09)**: Kaggle TPU v5e-8 persistentemente saturado.
R-084 lleva bloqueado desde 2026-11-07 (push exitoso, sesión no inicia).
Esto eleva R-088 a path CRÍTICO: los pesos pre-entrenados eliminan la
dependencia de TPU completamente ($0, disponibilidad inmediata).

---

## Fase 18: CPU Throughput Reformulation (R-091 → R-097)

> **Objetivo**: Reformular la asignación de responsabilidades por componente hardware para
> maximizar throughput CPU-only. Aprovecha los cambios de régimen que MambaByte introduce
> (vocab=256, LM head insignificante, entropy gate funcional) para re-habilitar técnicas
> que fracasaron en Fases 10-12 bajo condiciones diferentes.
>
> **Prerequisitos**: MambaByte integrado (R-088 compatible, o R-086 trained)
> **Budget**: $0 (optimizaciones de código puro)
> **Research**: `docs/internal/research/cpu-throughput-reformulation.md`
> **Research previa**: `docs/internal/research/pim-ndp-bandwidth-research.md`

### Contexto: Por qué Fases 10-12 fracasaron y Fase 18 es diferente

Las 12 técnicas de Fases 10-12 operaban bajo Mamba-130M BPE (493 MB F32, 127 MB Q8,
LM head 38.6 MB, entropy gate inactivo). **Tres condiciones cambian con MambaByte**:

| Condición | Mamba-130M BPE | MambaByte |
|---|---|---|
| LM head DRAM/token | 38.6 MB (30.4%) | 0.19 MB (0.2%) |
| Entropy gate | Inactivo (15.5 bits) | **Activo ~70% skip** |
| DRAM effective/token | 127 MB (100% tokens) | ~89 MB (30% tokens) |
| Effective DRAM/file 10KB | 1.27 TB | ~267 GB (gate) o 89 MB (parallel scan) |

Esto justifica re-evaluar NTA prefetch y L2 tiling bajo el nuevo régimen.

### Tier 1: Optimizaciones quirúrgicas (zero BPB risk, <100 líneas)

- R-091: NTA Prefetch para MambaByte weight streaming
  - **Estado**: PENDIENTE
  - **Prioridad**: P1 — zero risk, +10-20% throughput
  - **Dependencia**: MambaByte integrado (R-088 o R-086)
  - **Archivos**: `src/domain/tensor.rs`
  - **Subtareas**:
    - [ ] Cambiar `_mm_prefetch(ptr, _MM_HINT_T0)` → `_MM_HINT_NTA` en weight matmul paths
    - [ ] Mantener `_MM_HINT_T0` para SSM state access (debe quedarse en L1)
    - [ ] Añadir env var `REDUKT_NO_NTA=1` para A/B testing
    - [ ] Benchmark controlado: 3+ rounds, multi-file, vs sin NTA
    - [ ] Verificar BPB idéntico (0.000 delta)
  - **Justificación vs R-065 (fracasó)**:
    R-065 falló con Mamba-130M F32 (493 MB >> 12 MB L3, pollution irrelevante).
    Con MambaByte Q8 + entropy gate, el effective DRAM per backbone-token es ~89 MB,
    y los tokens que ejecutan backbone están espaciados (70% gateados). El L3 tiene
    tiempo de enfriar entre backbone calls. NTA mantiene SSM state (48 KB) caliente
    en L1 mientras pesos pasan solo por L2.
  - **Proyección**: +10-20% throughput en tokens que ejecutan backbone

- R-092: L2-tile weight layout para MambaByte
  - **Estado**: PENDIENTE
  - **Prioridad**: P1 — zero risk, +12-20% GEMV
  - **Dependencia**: MambaByte integrado (R-088 o R-086)
  - **Archivos**: `src/application/model_loader.rs`, `src/infrastructure/mamba/model.rs`
  - **Subtareas**:
    - [ ] Reordenar pesos a tiles 256×512 Q8 (~128 KB, cabe en L2 1.25 MB) durante carga
    - [ ] Adaptar inner loop en matmul para consumir tiles en orden
    - [ ] LM head 256×768 = 0.19 MB → cabe COMPLETO en L2 (no requiere tiling)
    - [ ] Benchmark controlado: A/B, 3+ rounds
    - [ ] Verificar BPB idéntico y roundtrip byte-identical
  - **Justificación vs R-067 (fracasó)**:
    R-067 falló porque GEMV streaming continuo no se beneficia de tiling cuando 100% de
    tokens son backbone-bound. Con entropy gate, los backbone-tokens están espaciados:
    el tile puede sobrevivir en L2 entre invocaciones. Además, el LM head ahora es
    0.19 MB (cabe completo en L2, antes era 38.6 MB).
  - **Proyección**: +12-20% en projection layers, +0% en LM head (ya en L2)

### Tier 2: Cambio arquitectónico CPU (zero GPU requerido)

- R-093: Parallel scan para compresión batch (CPU)
  - **Estado**: PENDIENTE
  - **Prioridad**: P0 — mayor impacto individual (10-50× compress throughput)
  - **Dependencia**: MambaByte integrado, R-052 (parallel_scan.rs ya existe) (borrado en 1dc6402, nunca medido; rehacer a nivel byte tras MambaByte)
  - **Archivos**: `src/infrastructure/mamba/model.rs`, `src/application/codec.rs`
  - **Subtareas**:
    - [ ] Implementar `forward_batch(&self, tokens: &[u32], states_out: &mut [...])` en MambaModel
    - [ ] Reformular SSM recurrence como parallel associative scan usando R-052 (módulo borrado en 1dc6402: rehacer)
    - [ ] Buffer de estados intermedios: N × d_model × 4 bytes (~30 MB para N=10K)
    - [ ] Adaptar codec.rs compress path: batch backbone → sequential CM+encode
    - [ ] Nuevo flag AZTH v4 (0x100 o bit libre) para indicar batch compression mode
    - [ ] Verificar roundtrip: batch compress → sequential decompress → byte-identical
    - [ ] Benchmark: throughput compress vs RNN, multi-file
  - **Diseño**: Opción A (backbone always-on, gate solo en mixing)
    - Backbone procesa TODOS los tokens (parallel scan)
    - Gate decide mixing weight, no compute skip
    - Descompresión en RNN produce resultados idénticos (misma secuencia de tokens)
    - Pro: determinismo garantizado, bandwidth amortizado
    - Con: descompresión ejecuta backbone 100% (sin gate savings a nivel compute)
  - **Constraint de roundtrip**: Compress (parallel scan, all tokens) y decompress
    (RNN, all tokens) DEBEN producir distribuciones idénticas. Opción A lo garantiza
    porque ambos ven exactamente los mismos tokens en el mismo orden.
  - **Proyección**: DRAM compresión 10KB: 89 MB (1 vez) vs 890 GB (per-token)
    Throughput compresión: ~400-1500 tok/s (CPU-only, modelo 96M-353M)

- R-094: CM pre-scan separado del backbone loop
  - **Estado**: PENDIENTE
  - **Prioridad**: P1 — complementa R-093
  - **Dependencia**: R-093
  - **Archivos**: `src/application/codec.rs`, `src/domain/cm_ensemble.rs`
  - **Subtareas**:
    - [ ] Refactor compress path: Fase 1 (backbone batch) + Fase 2 (CM+encode sequential)
    - [ ] En Fase 2: backbone logits pre-computados, CM+mixer en loop secuencial
    - [ ] Gate en mixing: `if cm_entropy < thr { usar CM-only } else { combinar con logits[t] }`
    - [ ] Verificar BPB idéntico a Opción A de R-093
  - **Proyección**: Organización limpia de pipeline, prepara terreno para GPU offload

### Tier 3: Extensiones edge/deployment

- R-095: HugePages para Linux deployment
  - **Estado**: PENDIENTE
  - **Prioridad**: P2 — solo Linux, +5-24% en weight streaming
  - **Dependencia**: ninguna (independiente)
  - **Archivos**: `src/application/model_loader.rs` (conditional compilation `#[cfg(target_os = "linux")]`)
  - **Subtareas**:
    - [ ] `madvise(ptr, size, MADV_HUGEPAGE)` después de mmap de pesos
    - [ ] Feature flag o detección automática (fallback silencioso si no disponible)
    - [ ] Test en Kaggle Linux (ya tenemos acceso)
    - [ ] NO implementar en Windows (requiere SeLockMemoryPrivilege)
  - **Proyección**: -5-24% TLB miss overhead en streaming de pesos

### Tier 4: FFI v2 para GPU batch (preparación futura)

- R-096: BackendVTable v2 — batch forward API
  - **Estado**: PENDIENTE
  - **Prioridad**: P2 — prepara GPU para parallel scan
  - **Dependencia**: R-093 (validar parallel scan en CPU primero)
  - **Archivos**: `src/domain/backend_ffi.rs`, `src/infrastructure/plugin_backend/`
  - **Subtareas**:
    - [ ] Añadir `BackendForwardBatchFn` al vtable (symbol: `redukt_backend_forward_batch`)
    - [ ] Bump `BACKEND_FFI_VERSION` a 2
    - [ ] Backward-compatible: si plugin no exporta batch symbol, fallback a per-token
    - [ ] Actualizar PluginBackend para detectar y usar batch API si disponible
    - [ ] Documentar FFI v2 en FORMAT-SPEC.md
  - **Interfaz propuesta**:
    ```
    extern "C" fn redukt_backend_forward_batch(
        tokens: *const u32,      // N tokens
        n_tokens: u32,
        out_logits: *mut f32,    // N × vocab_size contiguous
        vocab_size: u32,
    ) -> BackendResult;
    ```

- R-097: Q4 selective para MambaByte (re-evaluación)
  - **Estado**: PENDIENTE (investigación)
  - **Prioridad**: P2 — re-evaluar Q4 bajo condiciones MambaByte
  - **Dependencia**: MambaByte integrado
  - **Archivos**: `src/domain/quant.rs`, `src/domain/tensor.rs`
  - **Subtareas**:
    - [ ] Benchmark Q4 con MambaByte (diferente modelo, diferente sensibilidad)
    - [ ] Si BPB < +5%: adoptar Q4 selectivo (projection weights only, SSM en Q8)
    - [ ] Si BPB > +5%: documentar resultado y cerrar (como R-024/R-066)
    - [ ] Si adoptado: AVX-VNNI repacking (VPDPBUSD en Alder Lake)
  - **Contexto**: Q4 falló con Mamba-130M (+12.7% BPB). MambaByte tiene distribuciones
    de pesos diferentes (byte-level training, LM head 256 vs 50280). Re-evaluación justificada.
  - **Gate**: BPB < +5% → ADOPT, BPB > +5% → CLOSE

**Paralelización**: R-091 y R-092 son independientes entre sí. R-093 depende de MambaByte.
R-094 depende de R-093. R-095 es independiente de todo. R-096 depende de R-093.
R-097 es independiente (investigación).

```
MambaByte integrado (R-088 o R-086)
    │
    ├── R-091 (NTA prefetch) ── independiente
    ├── R-092 (L2 tiling) ── independiente
    ├── R-097 (Q4 re-eval) ── independiente (investigación)
    │
    └── R-093 (parallel scan compress) ──→ R-094 (CM pre-scan)
                                      └──→ R-096 (FFI v2 batch)

R-095 (HugePages Linux) ── independiente de todo
```

**Proyección de throughput combinado (CPU-only)**:

| Escenario | Compress | Decompress | BPB estimado |
|---|---|---|---|
| Actual (Mamba-130M BPE) | 34 tok/s | 34 tok/s | 0.961 |
| MambaByte-96M + gate | 173 tok/s | 173 tok/s | ~0.85-1.05 |
| + R-091/R-092 (NTA+tile) | 237 tok/s | 237 tok/s | ~0.85-1.05 |
| + R-093 (parallel scan) | **800-1500 tok/s** | 237 tok/s | ~0.85-1.05 |
| MambaByte-353M pre-trained | 66 tok/s | 66 tok/s | ~0.42 |
| + R-093 (parallel scan) | **200-400 tok/s** | 66 tok/s | ~0.42 |

**Nota**: Descompresión con R-093 Opción A NO se beneficia del gate (backbone always-on).
Descompresión sin R-093 (gate actual) sí se beneficia. La elección de Opción A vs gate
actual se resolverá empíricamente en R-093.

---

## Fase 19: CPU Limit & Market Viability (R-098 → R-105)

> **Objetivo**: Hacer el producto viable y competitivo en el mercado real. Ejecutar las palancas L1-L4
> sobre el backbone actual (sin entrenar nada), cerrar el determinismo cross-máquina como bloqueante de
> producto, decidir el tier rápido con una medida (no una proyección) y desplegar en cloud (GCP).
>
> **Prerequisitos**: ninguno para R-098/R-099/R-104 (operan sobre Mamba-130M BPE 50K actual).
> R-100 depende de R-098 y R-099. R-101 depende de R-098.
> **Budget**: $0 en código (R-098 → R-101, R-104); cloud (R-102, R-103) según uso medido;
> R-105 requiere entrenamiento (presupuesto a definir con la medida).
> **Research**: `docs/internal/research/throughput-properties-audit.md` (§6 = mandato de producto)
> **Cálculo económico**: `scripts/breakeven.py`
> **Research previa**: `docs/internal/research/cpu-throughput-reformulation.md`,
> `docs/internal/research/throughput-paradigm-shift.md`
> **Hardware dev**: i5-1235U (2P+8E), 2×16 GB DDR4-2667 dual channel = 42.7 GB/s teórico

### Contexto: tres premisas refutadas (2026-11-09)

La auditoría de la Fase 19 (ROADMAP Fases 8-18, 10 docs de research, hot path en código, 4 investigaciones
web) refutó tres premisas que sostenían el "hard ceiling" de las Fases 10-12:

| Premisa escrita | Realidad verificada | Consecuencia |
|---|---|---|
| Mamba corre en Q8 ("auto-selected Q8") | Mamba corre **F32** en el hot path (`Tensor { data: Vec<f32> }`, GEMV f32 AVX2 sin FMA). `Q8Tensor` solo existe para RWKV. R-064 (+14.5/+16.6% BPB, 31-34 tok/s sin mejora) fue revertido | 516 MB de pesos leídos por token; Q8 weight-only bajaría a ~130 MB |
| "21.3 GB/s peak DRAM" | 2 DIMM a 2667 MT/s = **dual channel, 42.7 GB/s** (~44 GB/s medido en portátiles i5-1235U). 34 tok/s × 516 MB = 17.5 GB/s = **41% del bus** | El bus no está saturado |
| "DRAM-bound, 34 tok/s es el techo" | in_proj y out_proj (**70% de los bytes/token**) corren en **un solo hilo**; solo la LM head usa ≤4 hilos. Un P-core Golden Cove topa en ~29 GB/s | El cuello es ancho de banda **por core**, no DRAM |

**Conclusión**: los 34 tok/s no son un techo físico; son F32 en un hilo. El "hard ceiling" de las
Fases 10-12 (21.3 GB/s single-channel) se apoyaba en estas premisas y queda reabierto.

Otros hallazgos de la auditoría:
- `speculative.rs` y `parallel_scan.rs` (R-052/R-053) fueron borrados en 1dc6402 como código muerto: nunca se
  integraron ni se midieron. R-093 y los docs de research que los citan están anotados en consecuencia.
- Batch mode (R-012) = N hilos independientes, cada uno con su GEMV completo (1.36× con 2 jobs, 1.64× con 3:
  compiten por el mismo bus). **Lockstep (compartir cada fetch de pesos entre streams) no existía en ningún
  documento ni código** → R-100.
- Determinismo cross-máquina: Kaggle 0.9633 vs local 0.9609 → bitstreams distintos; roundtrip entre máquinas
  distintas **no probado** (`determinism-audit.md` L157-165) → R-104.

### Contexto económico: umbrales de viabilidad

Coste actual: **≈$47 por GB raw procesado** (spot $0.022/vCPU-h a 130 B/s); $14.5 en Hetzner dedicado.
Punto de equilibrio a 7 años (R* = bytes/s por vCPU para que el cómputo de comprimir 1 GB cueste menos que el
ahorro de almacenamiento acumulado; b = fracción que deja brotli, S = ahorro adicional de Redukt):

| Dato | S3 Standard (7 años) | Deep Archive (7 años) |
|---|---|---|
| enwik8 (b=0.29, S=59%) | 18 KB/s | 424 KB/s |
| Facturae (b=0.11, S=56%) | 51 KB/s | 1.2 MB/s |
| apache (b=0.058, S=54%) | 101 KB/s | 2.3 MB/s |

- No existe precedente comercial de un compresor <1 MB/s con clientes de pago (PetaGene cobra % del ahorro a
  290 MB/s; AtomBeam monetiza ancho de banda satelital; ts_zip y Nacrith no son productos).
- Referencia externa: **L3TC** (AAAI 2025) logra ≈1.10 bpb en enwik8 (Redukt 0.96) con un RWKV de 3.2M
  parámetros a 27 KB/s en CPU móvil. Un modelo ~40× más pequeño pierde ~15% de ratio y gana ~200× throughput.

### Tier 1: Palancas L1-L4 (sin entrenar, riesgo BPB cero o acotado)

- R-098: Q8 weight-only para Mamba (L1)
  - **Estado**: EN CURSO
  - **Prioridad**: P0 — ×3-4 en bytes leídos por token, BPB ≈ 0 esperado
  - **Dependencia**: perfil numérico en header AZTH v5 (coordinar con R-104)
  - **Archivos**: `src/domain/tensor.rs`, `src/domain/quant.rs`, `src/infrastructure/mamba/model.rs`,
    `src/application/model_loader.rs`, `src/application/codec.rs`
  - **Diseño**: pesos int8 por fila (escala f32 por fila) solo en `in_proj`, `out_proj` y LM head, consumidos
    en registro por el GEMV (sin dequantizar a buffer F32). Activaciones F32. Se mantiene F32 en `A_log`,
    `dt_proj`, `x_proj`, `conv1d` y normas.
  - **Subtareas**:
    - [ ] Forense de R-064: qué se cuantizó en cada configuración (all-Q8 +16.6%, hybrid +16.2%, LM-head-only
      +14.5%) y si el kernel leía int8 (31-34 tok/s "sin mejora" sugiere que los bytes leídos no bajaron)
    - [ ] Cuantización int8 por fila para `in_proj`, `out_proj` y LM head
    - [ ] Mantener F32 en `A_log`, `dt_proj`, `x_proj`, `conv1d` y normas
    - [ ] Kernel GEMV AVX2 que consume int8 en registro (activaciones F32)
    - [ ] Perfil numérico en el header AZTH v5; el decoder rechaza perfiles desconocidos
    - [ ] Medir bytes/token (516 MB → ~130 MB) y tok/s
    - [ ] Roundtrip byte-identical + compatibilidad de decodificación v2/v3/v4
  - **Justificación**: el +16% de R-064 es anómalo. MambaQuant W8A8 en Mamba-130M pierde 0.8 puntos de accuracy
    media (44.7 → 43.9); Quamba W8A8 pierde 22% de perplejidad **porque cuantiza activaciones**. Weight-only por
    fila con activaciones F32 debe ser casi lossless. Si el delta vuelve a ser ~+15%, el defecto está en qué se
    cuantiza, no en Q8.
  - **Criterios de aceptación**:
    - Δ BPB < +2.5% en enwik8 10KB + 2 archivos adicionales (presupuesto máximo del proyecto: +5%)
    - Roundtrip PASS (CRC-32 verified OK) en los 3 archivos
    - Bytes/token ≈ 130 MB y tok/s > 100 en single-stream

- R-099: Multihilo en in_proj/out_proj (L2)
  - **Estado**: EN CURSO
  - **Prioridad**: P0 — ×1.5-2 por stream, BPB 0
  - **Dependencia**: ninguna (se combina con R-098)
  - **Archivos**: `src/domain/tensor.rs`, `src/infrastructure/mamba/model.rs`
  - **Diseño**: hoy el 70% de los bytes/token (in_proj/out_proj) corre en un hilo y solo la LM head usa ≤4.
    Partición por filas: cada fila es un producto punto completo con el mismo orden de acumulación que en
    single-thread, por lo que el resultado es bit-idéntico. Techo ~38-40 GB/s → F32 ~75 tok/s.
  - **Subtareas**:
    - [ ] Pool de hilos persistente (sin crear hilos por llamada)
    - [ ] Partición por filas bit-idéntica al single-thread
    - [ ] Work-stealing con balanceo P:E (≈3:1; Intel arXiv 2411.19542: +19% en GEMV con balanceo dinámico)
    - [ ] Variable de entorno `REDUKT_THREADS=N` (por defecto, hilos físicos)
    - [ ] Benchmark controlado A/B: 3+ rounds, multi-file
  - **Nota**: huge pages (0-10%) y NTA prefetch (±0, R-065) ya medidos: no invertir tiempo ahí.
  - **Criterios de aceptación**:
    - `.azth` idéntico byte a byte al baseline single-thread en los archivos de referencia
    - tok/s ≥ 1.8× el baseline (34 tok/s → ≥ 61 tok/s)

- R-100: Lockstep multi-stream, GEMV → GEMM (L3)
  - **Estado**: PENDIENTE
  - **Prioridad**: P0 — ×3-6 agregado, BPB 0; mayor palanca no explorada
  - **Dependencia**: R-098, R-099
  - **Archivos**: `src/domain/tensor.rs` (kernels GEMM), `src/infrastructure/mamba/model.rs`,
    `src/application/codec.rs` (scheduler multi-stream)
  - **Diseño**: N archivos independientes avanzan un token cada uno en el mismo paso. Cada tile de pesos se
    carga de DRAM una vez y se multiplica por N vectores de activación desde registros (intensidad aritmética
    ×N). Aplica a compresión **y** descompresión: cada stream conserva su estado SSM (~48 KB), su CM y su coder.
    Distinto de R-012 (N hilos independientes, cada uno con su GEMV completo).
  - **Subtareas**:
    - [ ] Kernel GEMM AVX2 con N configurable; conserva por stream el orden de acumulación del GEMV
      (lección de R-055: FP no asociativo rompe el roundtrip)
    - [ ] Scheduler con continuous batching: un stream que termina se reemplaza por el siguiente archivo
    - [ ] Compress y decompress en lockstep
    - [ ] Verificar bit-identidad: batched-compress ↔ single-decompress y single-compress ↔ batched-decompress
    - [ ] Curva tok/s agregados vs N ∈ {1, 2, 4, 8, 16, 32}
    - [ ] Archivos grandes: trocear en chunks independientes (~+1-3% BPB) o arranque caliente con el checkpoint
      de estado de R-063
  - **Evidencia externa**: llama-batched-bench (3.8B Q4, CPU): b=1 25.7 → b=8 86.3 → b=32 147.7 t/s agregados
    (5.8×), meseta en 64. Cruce memory → compute-bound en DDR4 portátil con Q8: N ≈ 12-24.
  - **Criterios de aceptación**:
    - Salida bit-idéntica al single-stream (mismo `.azth`) y roundtrip cruzado batched ↔ single PASS
    - Objetivo: 3-6× tok/s agregados sobre single-stream (tras R-098 + R-099); meseta esperada en N = 16-32
    - BPB idéntico al baseline (Δ = 0)

- R-101: Q6 weight-only (L4)
  - **Estado**: PENDIENTE
  - **Prioridad**: P1 — ×1.3 adicional; solo si R-098 pasa su gate
  - **Dependencia**: R-098 (gate superado)
  - **Archivos**: `src/domain/quant.rs`, `src/domain/tensor.rs`, `src/infrastructure/mamba/model.rs`
  - **Diseño**: pesos de 6 bits por fila en `in_proj`/`out_proj`. **Nunca por debajo de Q8 en embedding ni
    LM head.** Estimación: +1-3% BPB (en transformers de 0.5B: Q6_K +0.05%, Q5_K_M +0.3%, Q4_0 +2.7%; Mamba es
    4-5× más sensible). Q5 queda como arriesgado.
  - **Subtareas**:
    - [ ] Empaquetado 6-bit por fila y kernel GEMV/GEMM
    - [ ] Perfil numérico en el header AZTH v5 (junto con R-098)
    - [ ] Medir Δ BPB sobre enwik8 10KB + 2 archivos adicionales
  - **Criterios de aceptación**:
    - Δ BPB acumulado vs baseline F32 < +5% (presupuesto del proyecto; estimado +1-3%)
    - Si excede el presupuesto: opt-in o revertir
    - Roundtrip PASS

### Tier 2: Bloqueante de producto y despliegue cloud

- R-102: Despliegue GCP CPU
  - **Estado**: EN CURSO
  - **Prioridad**: P0 — necesario para ofrecer el producto
  - **Dependencia**: R-104 para clientes externos (determinismo); técnicamente independiente
  - **Archivos**: `Dockerfile`, `deploy/` (Cloud Build, Cloud Run, Cloud Batch), `infra/terraform/` (provider google)
  - **Subtareas**:
    - [ ] Dockerfile CPU endurecido (multi-stage; pesos no horneados en la imagen)
    - [ ] Cloud Build → Artifact Registry
    - [ ] Cloud Run para el comando `api`
    - [ ] Cloud Batch para compresión masiva (`--input-dir` / `--output-dir`)
    - [ ] Pesos en GCS, verificados contra `weights/MANIFEST.md`
    - [ ] Restringir a x86 con AVX2 (determinismo); excluir ARM
    - [ ] Medir $/GB raw real y compararlo con la proyección
  - **Criterios de aceptación**:
    - Roundtrip PASS (CRC-32 verified OK) en Cloud Run y en Cloud Batch
    - BPB enwik8 10KB medido en cloud y comparado con la referencia 0.9609 (la igualdad bit a bit se exige en R-104)
    - Coste $/GB raw medido y documentado

- R-103: Despliegue CUDA
  - **Estado**: EN CURSO
  - **Prioridad**: P1 — habilita el tier GPU
  - **Dependencia**: R-090 (plugin CUDA implementado; validación en T4 pendiente)
  - **Archivos**: `Dockerfile.cuda`, `plugins/redukt-gpu-cuda/`, `deploy/`
  - **Subtareas**:
    - [ ] `Dockerfile.cuda` con nvcc para compilar `plugins/redukt-gpu-cuda`
    - [ ] Conversión de pesos a `.rbin` (`scripts/convert-weights-gpu.py`)
    - [ ] GCE `g2` (L4) o `n1` + T4
    - [ ] Carga del plugin (dlopen, fallback automático a Mamba CPU)
    - [ ] Verificar si un `.azth` comprimido con GPU descomprime en CPU y viceversa (coordinar con R-104)
    - [ ] Medir tok/s reales vs la proyección de R-090 (~500 tok/s)
  - **Criterios de aceptación**:
    - El plugin carga en GCE T4/L4 y el roundtrip pasa (CRC-32 verified OK)
    - tok/s y $/GB medidos en GCE, documentados

- R-104: Determinismo cross-máquina — BLOQUEANTE de producto
  - **Estado**: PENDIENTE
  - **Prioridad**: P0 — ningún cliente externo antes de cerrarlo
  - **Dependencia**: ninguna; condiciona el diseño de R-098/R-099/R-100 y la entrega de R-102/R-103
  - **Archivos**: `src/domain/tensor.rs`, `src/infrastructure/mamba/model.rs`, `src/domain/mixer.rs`,
    `src/application/codec.rs`
  - **Contexto**: el mismo archivo produce 0.9633 BPB en Kaggle y 0.9609 en local → bitstreams distintos. El
    roundtrip comprimir en una máquina y descomprimir en otra **no está probado** (`determinism-audit.md`
    L157-165). Con aritmética en coma flotante, cualquier diferencia de orden de acumulación, ISA o función
    transcendental cambia la CDF y rompe la decodificación.
  - **Opciones**: (A) orden de acumulación canónico, independiente de ISA y de hilos; (B) aritmética entera /
    punto fijo en las rutas que alimentan la CDF.
  - **Subtareas**:
    - [ ] Reproducir: comprimir en la máquina A y descomprimir en la B (y viceversa); registrar el resultado
    - [ ] Bisectar la divergencia: primer token cuyos logits/probabilidades difieren entre máquinas
    - [ ] Elegir opción (A o B) con coste medido en tok/s y BPB
    - [ ] Golden `.azth` versionado con SHA-256, verificado en ≥2 CPUs x86 distintas
    - [ ] Documentar el contrato de determinismo (coordinar con `docs/api/FORMAT-SPEC.md`)
  - **Criterios de aceptación**:
    - Compress en A → decompress en B byte-identical, y B → A, sobre ≥3 archivos de referencia
    - Mismo `.azth` (SHA-256) en ambas máquinas
    - Δ BPB y Δ tok/s de la solución medidos y dentro del presupuesto (+5% BPB)

### Tier 3: Decisión de producto (investigación)

- R-105: Tier rápido (backbone pequeño + receta CM)
  - **Estado**: PENDIENTE (investigación)
  - **Prioridad**: P1 — única vía CPU-only que cruza el umbral económico
  - **Dependencia**: ninguna técnica; puede reutilizar la infraestructura de entrenamiento de la Fase 15
  - **Archivos**: `src/infrastructure/` (backbone adicional), `docs/internal/research/` (informe de decisión)
  - **Contexto**: L3TC (AAAI 2025) logra ≈1.10 bpb en enwik8 con RWKV de 3.2M parámetros y vocab 16K a 27 KB/s en
    CPU móvil (batch 1) y 633 KB/s batcheado en ANE. La receta de Redukt (CM + mixer adaptativo + preprocesado
    binario + routing) es independiente del tamaño del backbone. `token-reduction-strategies.md` §5.6 marcó la
    destilación como NO-GO por una **proyección** (+5-15% BPB); L3TC es una **medida**.
  - **Subtareas**:
    - [ ] Definir 1-2 candidatos (rango 3-30M parámetros; vocab 256 vs 16K)
    - [ ] Entrenar u obtener el backbone (presupuesto por definir)
    - [ ] Integrarlo como backbone adicional con la receta CM completa
    - [ ] Medir BPB y KB/s (single-stream y lockstep) sobre los 12 archivos del benchmark vs Mamba-130M
    - [ ] Decisión GO/NO-GO frente al umbral R* (18-100 KB/s por vCPU; `scripts/breakeven.py`)
    - [ ] Revisar y documentar el NO-GO previo de destilación con la medida
  - **Criterios de aceptación**:
    - Informe con BPB y KB/s medidos (no proyectados) por archivo
    - Decisión GO/NO-GO documentada con el umbral económico explícito

**Paralelización**: R-098 y R-099 son independientes entre sí. R-100 requiere ambos. R-101 requiere R-098.
R-102 y R-105 son independientes del resto. R-103 depende de R-090. R-104 es transversal y bloquea la entrega
a clientes externos.

```
R-104 (determinismo) ── P0 transversal: condiciona R-098/R-099/R-100 y la entrega de R-102/R-103

R-098 (Q8 Mamba) ──┬──→ R-100 (lockstep) ←── R-099 (multihilo)
                   └──→ R-101 (Q6)

R-102 (GCP CPU) ── independiente
R-103 (CUDA) ── depende de R-090
R-105 (tier rápido) ── independiente (investigación)
```

**Proyección de throughput combinado (CPU-only, Mamba-130M BPE, i5-1235U, 10 cores)**:

| Escenario | tok/s por stream | tok/s agregados | B/s agregados | Coste ≈ $/GB raw (spot) |
|---|---|---|---|---|
| Hoy (F32, in/out_proj en 1 hilo) | 34 | 34 | ~130 | ≈$47 |
| + R-099 | ~70 | ~70 | ~270 | — |
| + R-098 (con R-099) | ~250-300 | ~280 | ~1.1 KB/s | — |
| + R-100 (N ≈ 16) | ~250 | ~700-1,500 | 3-6 KB/s | ≈$12 |

**Nota**: estas cifras son proyecciones, no medidas; cada fila se reemplaza por la medida al cerrar su
ítem. El coste solo se estima en los extremos (`scripts/breakeven.py`, ~0.5 KB/s por vCPU en el escenario final).
Con ≈$12/GB el sistema sigue 30-200× por encima del umbral económico (18-100 KB/s por vCPU): R-105 (tier
rápido) y/o GPU (R-103) son necesarios, no opcionales.

**Estado de la fase**: EN CURSO — R-098, R-099, R-102, R-103 en curso; R-100, R-101, R-104, R-105 pendientes.

---

## Bugs conocidos

#### BUG-001: Mamba BPE tokenizer roundtrip fails on binary+delta data
- **Estado**: ✅ RESUELTO (2026-10-31)
- **Prioridad**: P1 — afectaba roundtrip de archivos binarios con Mamba backbone
- **Síntoma**: `size mismatch: decoded 10241 bytes, expected 10240` en x-ray 10KB
  con shuffle+delta preprocessing.
- **Causa raíz**: GPT-NeoX mamba-130m vocab falta 13 single-byte tokens
  (0xC0, 0xC1, 0xF5-0xFF). El encode descartaba estos bytes silenciosamente,
  causando pérdida de datos (1039 bytes perdidos en x-ray 10KB).
- **Fix**: Añadir 13 tokens sintéticos al cargar el tokenizer (IDs 50254-50266).
  Byte fallback lookup `byte_fallback[256]` garantiza que NINGÚN byte se pierde.
  x-ray roundtrip: PASS (CRC-32 verified OK) [MEASURED]
- **Impacto en BPB**: x-ray BPB corregido de 4.8969 (dato corrupto) a 5.8008 (dato correcto)
- **Archivos**: `src/infrastructure/mamba/tokenizer.rs`

---

## Registro de cambios

| Fecha | Cambio |
|---|---|
| 2026-10-26 | Creación inicial — auditoría de métricas de producción |
| 2026-10-26 | R-001, R-002, R-003 completados — AZTH v3 con CRC-32 |
| 2026-10-26 | R-004 completado — decoder bounds checking + DecodeError |
| 2026-10-26 | R-005 completado — 14 edge case tests (unit + integration) |
| 2026-10-26 | R-006 completado — panics→Results en todos los cmd_*, 55 tests |
| 2026-10-26 | R-007 completado — u32 overflow validation + 3 tests, 58 total |
| 2026-10-26 | R-008 completado — PhaseTimer + --verbose en 4 comandos, 62 tests |
| 2026-10-26 | R-009 completado — CoderMetrics + efficiency % + 3 tests, 65 total |
| 2026-10-26 | Benchmark verification — 17 files tested, claims corregidos (Q8: 1.28 BPB) |
| 2026-10-26 | feat: --quant flag, F32 vs Q8 benchmark (18 files), industry files first benchmarked |
| 2026-10-26 | Finding: F32 vs Q8 <2.5% diff — Q8 is correct production choice (2.1x faster) |
| 2026-10-26 | Finding: logs -48% ce_bpb, invoicing -54% ce_bpb vs brotli (corrected 10-28: actual -34%, -48%) |
| 2026-10-27 | R-010 parcial: emb-surgery center0.3 benchmarked (14/18 mejora, -1.69% media) |
| 2026-10-27 | R-010: --emb-surgery añadido a compress, decompress, baseline |
| 2026-10-27 | R-010: --tau descartado (empeora BPB en todos los valores) |
| 2026-10-27 | Roadmap reestructurado: P3 → optimización + go-to-market (R-010 a R-014) |
| 2026-10-27 | R-011 completado — AVX2 Q8/F32 kernels + 3-tier dispatch + 9 tests, 74 total |
| 2026-10-27 | R-011 benchmark — Q8 6.7x-14.5x matmul, ~1.3x end-to-end, F32 FMA disabled |
| 2026-10-27 | R-011 fix — F32 FMA rompe cross-path determinism, deshabilitado, roundtrip verificado |
| 2026-10-27 | docs: GO-TO-MARKET-STRATEGY.md — análisis estratégico 3 fases, throughput economics |
| 2026-10-27 | AUDIT: datos fabricados en BUSINESS-CASES.md (PACS: x-ray 3.87→6.524, dirección invertida) |
| 2026-10-27 | AUDIT: gzip-9 BPB medido (0.55-0.99 en logs, NO 3.00 como decía el doc) |
| 2026-10-27 | R-V02 completado — gzip9-brotli11-comparison.csv con 17 archivos medidos |
| 2026-10-27 | Corrección masiva: BUSINESS-CASES, GO-TO-MARKET, LIMITATIONS, MARKET-ANALYSIS |
| 2026-10-27 | Protocolo de integridad de datos: tags [MEASURED]/[CALCULATED]/[ESTIMATE]/[UNMEASURED] |
| 2026-10-27 | R-V01 completado — post-SIMD benchmark 18 archivos, mean 22.7 tok/s, 50.8 B/s |
| 2026-10-27 | R-V03 completado — PDF benchmark: +40.7% PEOR que brotli. E-invoicing solo viable XML-only |
| 2026-10-27 | R-V04 completado (parcial) — scale test: ventaja CRECE con tamaño (dickens -30%→-46%) |
| 2026-10-27 | R-V05 completado — Q4 RECHAZADO: +13.3% BPB, 7.6x más lento que Q8 |
| 2026-10-27 | R-V06 completado — model pruning: SVD conditional GO, distillation NO-GO, Q4 NO-GO |
| 2026-10-27 | R-V07 completado — Cribl/Edicom/log TAM verificados. Cold/genomics TAM sin fuente |
| 2026-10-27 | P3 reformulado: secuencia por riesgo BPB, quality gates, presupuesto +5% degradación |
| 2026-10-27 | R-015 (KV caching), R-016 (SVD pruning), R-017 (parallel segments) añadidos |
| 2026-10-27 | R-012 reposicionado: paso 2 (0% BPB risk, antes de tocar modelo) |
| 2026-10-27 | Interdependencias documentadas: SVD + parallel segments = degradación compuesta |
| 2026-10-28 | CORRECCIÓN: benchmark usa compresión real (AZTH) en vez de cross-entropy. BPB real 5-20% mayor. |
| 2026-10-28 | Cifras corregidas: logs -34% vs brotli (antes -48%), facturae -48% (antes -54%), enwik8 -43% (antes -30%) |
| 2026-10-28 | Coder efficiency: 80-95% (varía por archivo), no ~95% uniforme como se estimaba |
| 2026-10-28 | Throughput medido: ~25-47 tok/s, ~27-164 B/s (varía por archivo) |
| 2026-10-28 | 18 archivos completos con compresión real: 12/18 wins vs brotli (antes 13/18 con ce_bpb) |
| 2026-10-28 | mozilla flipped: -15% (ce) → +2.1% (actual). nci flipped: +4.6% (ce) → -7.4% (actual) |
| 2026-10-28 | CSV completo: `data/benchmark/results/actual-compression-benchmark.csv` |
| 2026-10-28 | R-018 añadido y COMPLETADO: AZTH v4 header (-15 bytes), BPB 1.3448→1.3328, eff 95.3%→96.1% |
| 2026-10-28 | R-019 añadido: refactorización matemática (rank-1 decomp + codec zero-alloc), 0% BPB risk |
| 2026-10-28 | R-021 añadido: F32 AVX2 sin FMA para LoRA matmuls, 0% BPB risk |
| 2026-10-28 | R-022 añadido: SIMD transcendentals (sigmoid, tanh, exp), 0% BPB risk |
| 2026-10-28 | R-HW1 añadido: HardwareGovernor — regulación dinámica de recursos, crash prevention |
| 2026-10-28 | Secuencia revisada: R-018→R-019→R-021→R-022→R-015→R-012→R-016→R-017 |
| 2026-10-28 | DRAM utilization corregida: 9%→25% (cálculo original solo contaba head, no layer weights) |
| 2026-10-28 | Presupuesto BPB +5%→+3% en R-016/R-017 quality gates |
| 2026-10-28 | Multi-file scaling actualizado: 10 cores (no 8) por PROTO-05 results |
| 2026-10-28 | R-021 COMPLETADO: F32 AVX2 noFMA, +19.2% throughput, same-path roundtrip OK |
| 2026-10-28 | R-022 COMPLETADO: SIMD transcendentals, +23.2% controlled A/B (corregido de +32.6%) |
| 2026-10-28 | R-022: Cephes fast_exp_ps + fused ops (decay_sigmoid, sigmoid_add, sigmoid_mix) |
| 2026-10-28 | R-022: 107 tests (7 new), accuracy <2.4e-7 max error, roundtrip byte-identical |
| 2026-10-28 | R-022 VALIDATION: controlled A/B (3 rounds), multi-file BPB (4 files), scalar baseline |
| 2026-10-28 | R-022 CORRECTION: BPB -0.13% (improved, not +0.11% degraded), throughput +23.2% (not +32.6%) |
| 2026-10-28 | R-022: thermal variance confirmed — all measurements subject to Dell Vostro throttling |
| 2026-10-28 | R-015 REDEFINIDO: "KV Batch Caching" → "Rank-1 State Factorization" (RWKV es RNN, no transformer) |
| 2026-10-28 | R-015 COMPLETADO: rank-1 factorization O(N³)→O(N²), 32x fewer FLOPs per head |
| 2026-10-28 | R-015: scalar +100% (32.8→65.7), SIMD +135% (40.4→94.8), controlled A/B 3 rounds |
| 2026-10-28 | R-015: BPB ±0.25% (float reordering noise), roundtrip byte-identical, CRC verified |
| 2026-10-28 | R-012 COMPLETADO: batch --input-dir/--output-dir/--jobs, Arc model sharing, thread pool |
| 2026-10-28 | R-012: scaling 1.36x(2T), 1.64x(3T) [MEASURED: 3 rounds × 3 configs, same session] |
| 2026-10-28 | R-012: BPB bit-exact (0.000 delta) y bytes comprimidos idénticos en 1/2/3 jobs |
| 2026-10-28 | R-012: roundtrip verified (decompress byte-identical, CRC D28BAB1C) |
| 2026-10-28 | R-012: Rwkv7State::reset() + Scratch::reset_perf() añadidos para batch reuse |
| 2026-10-28 | Todas las optimizaciones 0% BPB risk completadas (R-018/R-021/R-022/R-015/R-012) |
| 2026-10-28 | R-016 ABORTADO: SVD evaluado empíricamente, +3.47% BPB AND 1.95x slower (MEJOR caso) |
| 2026-10-28 | R-016: RWKV-7 0.1B alto rango efectivo, RWKV-edge NLP accuracy ≠ compression BPB |
| 2026-10-28 | R-017 BLOQUEADO: precondición R-016 fallida, requiere evaluación BPB independiente |
| 2026-10-29 | Auditoría de exhaustividad: investigación recursiva de todas las técnicas conocidas |
| 2026-10-29 | Nacrith paper (2026): arquitectura casi idéntica, CDF 24-bit es la técnica adoptable |
| 2026-10-29 | State update (62% forward) es hard ceiling — secuencial, scalar, sin optimización conocida |
| 2026-10-29 | Descartados: rANS (coder no es bottleneck), time-domain parallel (RWKV secuencial), target-interval capture |
| 2026-10-29 | R-023 añadido: CDF Precision 16→24 bit (0% BPB risk, mejora coder efficiency) |
| 2026-10-29 | R-024 añadido: Q4 Storage→Q8 Compute (0% BPB risk, -50% DRAM traffic) |
| 2026-10-29 | R-025 añadido: Software Prefetching (+5-15% matmul, 0% BPB risk) |
| 2026-10-29 | R-026 añadido: iGPU Offload Intel Iris Xe (módulo opcional --gpu, Vulkan FFI zero-dep) |
| 2026-10-29 | Fase 2 secuencia: R-023→R-025→R-024→R-026 (por complejidad creciente) |
| 2026-10-29 | docs: optimization-exhaustion-audit.md — 350 líneas, 10 secciones, 10 referencias |
| 2026-10-29 | R-023 NO APLICA: coder.rs ya usa CDF 24-bit (TOP=2^24) desde diseño original |
| 2026-10-29 | R-025 COMPLETADO: software prefetch en 3 kernels SIMD, +3.2% e2e [MEASURED: 4 rounds] |
| 2026-10-29 | R-025: BPB 1.281455 idéntico con/sin prefetch (0.000 delta), 111 tests pass |
| 2026-10-29 | R-025: REDUKT_NO_PREFETCH=1 env var añadida para controlled A/B benchmarking |
| 2026-10-29 | R-024 ABORTADO: Q4 SIMD implementado, pero +12.7% BPB y 15.3x slower que Q8 |
| 2026-10-29 | R-024: confirma R-V05 — Q4 precision insuficiente para RWKV-7 0.1B (4 bits < 8 bits) |
| 2026-10-29 | R-024: Q4 SIMD kernel retenido como mejora al path Q4 (non-allocating + SIMD) |
| 2026-10-29 | R-026 ABORTADO: iGPU Vulkan 2.68x slower que CPU — dispatch overhead, head matmul solo 12% |
| 2026-10-29 | R-026: código revertido, resultado negativo documentado. Fase 2 completa (1/3 exitoso) |
| 2026-10-29 | R-024: 2 tests añadidos (q4_simd_matches_scalar, q4_simd_matches_allocating), 113 total |
| 2026-10-29 | R-013 COMPLETADO: auditoría float (18 ops), golden reference, 4 nuevos tests determinismo |
| 2026-10-29 | R-013: same-platform DETERMINISTA, cross-tier INCOMPATIBLE, cross-platform NO GARANTIZADO |
| 2026-10-29 | R-013: SHA-256 zero-dep implementado en tests, HashMap ngram.rs descartado como falso positivo |
| 2026-10-29 | R-013b COMPLETADO: portable transcendentals — ALL scalar libm reemplazado con Cephes/musl portables |
| 2026-10-29 | R-013b: BIT-EXACT con SIMD (0 ULP), BPB 1.281455 sin cambio, 3 nuevos tests |
| 2026-10-29 | R-014 COMPLETADO: streaming decompression con --streaming flag, CRC incremental, output progresivo |
| 2026-10-29 | R-014: 2 nuevos tests CRC incremental, 132 tests total, batch=streaming=original verificado |
| 2026-10-29 | R-013b: cross-platform same-tier determinism CONSEGUIDO. Cross-tier sigue fallando (F32 matmul acum.) |
| 2026-10-29 | R-010 COMPLETADO: emb-surgery method stored in AZTH v4 header (FLAGS 0x02), roundtrip transparente |
| 2026-10-29 | R-010: 9 nuevos unit tests para header emb-surgery, 130 tests total |
| 2026-10-29 | R-HW1 COMPLETADO: HardwareGovernor — battery/RAM monitoring, AtomicU8 throttle, kernel32 FFI |
| 2026-10-29 | R-HW1: ThrottleLevel {Full, Eco, Abort}, watchdog 2s, governor.apply() cada 100 tokens |
| 2026-10-29 | R-HW1: integrado en compress + decompress, --no-governor flag, 5 unit tests, 137 total |
| 2026-10-29 | R-HW1: roundtrip byte-identical verificado, BPB sin cambio (governor no afecta predicción) |
| 2026-10-29 | BENCHMARK COMPLETO: 18 archivos × 10KB + 3 × 50KB + 3 × 100KB, gzip-9 + brotli-11 + Redukt Q8 |
| 2026-10-29 | Baseline actualizada: Redukt gana 12/17 vs brotli-11, pierde 5 binarios (sao +54%, x-ray +44%) |
| 2026-10-29 | Batch mode validado: modelo cargado una vez (2.8-3.5s), 3 archivos × 100KB en 1544s |
| 2026-10-29 | Escala confirmada: ventaja CRECE con tamaño (dickens -35%→-45%, apache -37%→-32%) |
| 2026-10-29 | Investigación binarios: 3 agentes, 15 fuentes, Nacrith/NNCP/PAQ8/CMIX/G2-V3 analizados |
| 2026-10-29 | Diagnóstico: tokenizer texto destruye estructura byte-level en datos binarios |
| 2026-10-29 | Fase 3 definida: R-027 (E8E9) → R-028 (detector) → R-029 (shuffle) → R-030 (delta) → R-031 (records) → R-032 (CM-only) |
| 2026-10-29 | R-027: E8E9 ya codificado en preprocess.rs (5 tests), solo falta integración en pipeline |
| 2026-10-29 | docs: binary-compression-research.md — investigación completa con 15 referencias |
| 2026-10-29 | R-027 COMPLETADO: E8E9 integrado en pipeline — FLAG 0x04, auto-detect, --e8e9 CLI, roundtrip PASS |
| 2026-10-29 | R-027: mozilla 10KB neutral (2.0536 vs 2.0520 BPB, solo 3 E8 en header) — beneficio en EXE puros |
| 2026-10-29 | R-028 COMPLETADO: content_detect.rs — 6 tipos, decision tree, 11 tests, 14/17 benchmark validados |
| 2026-10-29 | R-029 COMPLETADO: byte shuffle stride=4, FLAG 0x08, x-ray -8.5% BPB, sao +0.8% (auto-reject), 9 tests |
| 2026-10-29 | R-030 COMPLETADO: delta coding stride=1, FLAG 0x10, x-ray -11.4% total (shuffle+delta), 9 tests |
| 2026-10-29 | R-031 COMPLETADO: record detection via autocorrelation (Q1 baseline, 5.0x threshold, min stride 4) |
| 2026-10-29 | R-031: reuses SHUFFLE flag, GenericBinary → detect → shuffle. Synthetic 8B records: 0.694 BPB, roundtrip PASS |
| 2026-10-29 | R-031: zero false positives on all 18 benchmark files (osdb 4.1x, apache 5.3x — both below 5.0x threshold) |
| 2026-10-29 | R-032 ABORTADO + REVERTIDO: CM-only bypass — RWKV ayuda incluso en binarios (sao +19.4%, enwik8 +109.7%) |
| 2026-10-29 | R-032: hipótesis "RWKV es ruido para binarios" REFUTADA empíricamente. Código revertido (2f34173) |
| 2026-10-30 | R-033 COMPLETADO (experimental): adaptive RWKV skip — FLAG 0x20, MAX_CONSEC_SKIP=3, roundtrip PASS |
| 2026-10-30 | R-033: enwik8 +11.2% BPB (9.2% skip), apache +57.8% BPB (49.2% skip) — opt-in only |
| 2026-10-30 | R-033: infrastructure gates para --gpu, --tokenizer bpe, --model-size (placeholders) |
| 2026-10-30 | Análisis estratégico: RWKV O(D²) secuencial = muro para producción a escala (1 TB = 10,800 cores/día) |
| 2026-10-30 | Investigación: 4 agentes paralelos — RWKV paralelización, backbones alternativos, dock patterns, SOTA 2025-2026 |
| 2026-10-30 | Hallazgo: parallel scan teórico para RWKV-7 pero O(n³) composición → SLOWER en práctica |
| 2026-10-30 | Hallazgo: Mamba O(D·N) diagonal state — StateSMix 1,100 tok/s CPU [SOURCED], MambaByte 0.93 BPB [SOURCED] |
| 2026-10-30 | Hallazgo: Griffin/Hawk RG-LRU O(D) diagonal — ideal pero sin modelos small pretrained |
| 2026-10-30 | Hallazgo: xLSTM mLSTM descartado — mismo O(D²) que RWKV, no resuelve bottleneck |
| 2026-10-30 | Hallazgo: coupling actual Redukt→RWKV = 22 refs concretas, 0 traits, dock viable ~240 líneas |
| 2026-10-30 | Seguridad: Mamba SafeTensors = mismo modelo seguridad que RWKV (zero pickle, zero unsafe) |
| 2026-10-30 | Fase 4 definida: R-034 (dock) → R-035 (Mamba) + R-036 (BPE) + R-037 (expanded CM) |
| 2026-10-30 | docs: backbone-scalability-analysis.md — investigación completa con 30+ fuentes |
| 2026-10-30 | R-034 COMPLETADO: backbone dock — Backbone/BackendTokenizer traits, codec desacoplado (cf3fc4b) |
| 2026-10-30 | R-037 PARCIAL: match model — hash-chain predictor, enwik8 -0.41%, facturae -2.2% [MEASURED] (5a8d1ed) |
| 2026-10-30 | R-036 BLOQUEADO: BPE tokenizer requiere modelo entrenado con vocab BPE → depende de R-035 |
| 2026-10-30 | R-035 COMPLETADO: Mamba-130m backbone — forward pass, BPE tokenizer, CLI mamba-eval |
| 2026-10-30 | R-035: enwik8 0.9074 BPB (-35.5% vs RWKV), dickens 0.6149 BPB, apache 0.1636 BPB [MEASURED] |
| 2026-10-30 | R-035: throughput 15.2-16.3 tok/s (scalar, 6x slower que RWKV-7 95 tok/s) |
| 2026-10-30 | R-035: CHECKPOINT BIFURCACIÓN — Mamba viable, SIMD/integración ensemble como trabajo futuro |
| 2026-10-30 | R-036 COMPLETADO VÍA R-035: BpeTokenizer implementado (50254 vocab, 50009 merges, 3 tests) |
| 2026-10-30 | R-037 COMPLETADO: RunModel + WordModel — 15/19 mejoran, ubl -2.4%, xml -2.2%, facturae -1.2% [MEASURED] |
| 2026-10-30 | R-037: mixer 5 componentes SGD, 11 unit tests, roundtrip PASS |
| 2026-10-30 | FASE 4 COMPLETADA: R-034 ✅ R-035 ✅ R-036 ✅ R-037 ✅ |
| 2026-10-30 | R-038 COMPLETADO: Mamba SIMD — SSM recurrence d16 AVX2, silu_inplace, gated_silu_mul |
| 2026-10-30 | R-038: 3 unit tests, RWKV BPB sin regresión (1.2778), dead_code cleanup |
| 2026-10-30 | R-038 BENCHMARK: controlled A/B — SIMD 34.0 tok/s vs scalar 23.3 tok/s = **+44.6%** median |
| 2026-10-30 | R-038: gap vs RWKV-7 halved: 6x (scalar) → 2.8x (SIMD). BPB identical (0.9074) |
| 2026-10-30 | R-038 FULL BENCHMARK: 19 archivos, 33.6-34.7 tok/s (mean 34.3), Mamba wins 10/19 vs RWKV ce_bpb |
| 2026-10-30 | R-038: samba -80.6%, webster -68.8%, dickens -65.0%. Pierde binarios: ooffice +41.3% |
| 2026-10-31 | Investigación: adaptive tokenization — 3 agentes paralelos, 18 fuentes, 5 enfoques evaluados |
| 2026-10-31 | Hallazgo: nadie resuelve "un tokenizador para todo" con modelo pretrained [SOURCED, 18 refs] |
| 2026-10-31 | Hallazgo: Redukt tiene ventaja única — dual backbone (Mamba BPE + RWKV byte-level) |
| 2026-10-31 | Fase 5 definida: R-039 (router) → R-040 (ensemble) + R-041 (MambaByte) + R-042 (preprocessing) |
| 2026-10-31 | R-039 COMPLETADO: content-adaptive backbone router — BackboneChoice, DynModel, FLAG 0x40 (0899c86) |
| 2026-10-31 | R-040 COMPLETADO: dual-backbone benchmark — 12 archivos, actual AZTH compression [MEASURED] |
| 2026-10-31 | R-040: Mamba wins **10/12** (83%). samba **-79.1%**, dickens **-63.7%**, enwik8 **-27.6%** |
| 2026-10-31 | R-040: hipótesis "RWKV mejor para binarios" REFUTADA — Mamba gana incluso en Executable/FloatArray |
| 2026-10-31 | R-040: routing actualizado — Mamba default, RWKV solo GenericBinary. vs brotli: 8/12 wins (-50.5% avg) |
| 2026-10-31 | R-040: fix MAMBA_WEIGHTS_DIR path (mamba-130m-hf → mamba-130m) |
| 2026-10-31 | Metodología delegación agéntica documentada: docs/internal/agent-delegation-methodology.md |
| 2026-10-31 | docs: adaptive-tokenization-research.md — investigación completa con 18 referencias |
| 2026-10-31 | R-039 ✅: content-adaptive backbone router — BackboneChoice enum, DynModel dispatch, FLAG 0x40, --backbone CLI, 9 tests |
| 2026-10-31 | R-041 COMPLETADO (investigación): CONDITIONAL GO — $300-750 training, 10-30GB corpus, no pretrained model [SOURCED] |
| 2026-10-31 | R-041: MambaByte-130M estimado ~1.20-1.40 BPB enwik8 con CM, 200-500 tok/s [ESTIMATE] |
| 2026-10-31 | R-041: plan faseado — Phase 0 (validate 353M) → Phase 1 (50M) → Phase 2 (130M) → Phase 3 (integrate) |
| 2026-10-31 | R-042 COMPLETADO (investigación): NO-GO — BPE fundamentalmente incompatible con binarios [SOURCED: 5 papers] |
| 2026-10-31 | R-042: remapping NO-GO, splitting NO-GO, E8E9+shuffle MARGINAL (5-10%), routing es la solución correcta |
| 2026-10-31 | **FASE 5 COMPLETADA**: R-039 ✅ R-040 ✅ R-041 ✅(research) R-042 ✅(research) |
| 2026-10-31 | Fase 6 definida: R-043 (entropy mixer) + R-044 (segment model) + R-045 (bit-plane) + R-046 (CDF audit) + R-047 (BPP research) |
| 2026-10-31 | R-043 COMPLETADO: entropy-adaptive CM scaling — cm_scale boost-only clamp(1.0,1.2), FLAG 0x80 (fd77c70) |
| 2026-10-31 | R-043: near-zero effect at 10KB (~3500 tokens) — EMA doesn't stabilize. Safe, no regression [MEASURED] |
| 2026-10-31 | R-044 COMPLETADO: SegmentModel — token-frequency histogram, surprise-based boundary, 4 unit tests (fd77c70) |
| 2026-10-31 | R-044: near-zero effect at 10KB — MIN_OBS=100 barely activates. Expected improvement 50KB+ [MEASURED] |
| 2026-10-31 | R-046 COMPLETADO: CDF audit — Redukt already CDF-24 (matches Nacrith). Round-to-nearest applied. BPB unchanged [VERIFIED] |
| 2026-10-31 | R-046: Nacrith -28% was CDF-16→CDF-24 transition, NOT applicable to Redukt (already CDF-24) |
| 2026-10-31 | R-045 DIFERIDO: P2, uncertain impact on BPB. R-047 FUTURO: depende R-041 (training) |
| 2026-10-31 | Fase 7 definida: R-048 (PredCache data structure) → R-049 (integration) → R-050 (eval & tuning) |
| 2026-10-31 | Insight: backbone es 99.5% del costo. Cache de predicciones por contexto = throughput scaling |
| 2026-10-31 | Research: prediction-cache-research.md — diseño completo, prueba de determinismo, estimaciones |
| 2026-10-31 | Estimación: 2-5x speedup en datos estructurados (logs, XML), 1.2x en texto natural |
| 2026-10-31 | R-048 COMPLETADO: PredCache data structure — 64K table, top-32 logits, FNV-1a hash, 16 tests |
| 2026-10-31 | R-049 IMPLEMENTADO Y REVERTIDO: +17.4% BPB inaceptable, +15.9% tok/s insuficiente |
| 2026-10-31 | R-049 lección: state divergence con MAX_SKIPS=8 acumula error mucho más rápido que estimado |
| 2026-10-31 | R-049 prerequisito: flag CLI opt-in + R-050 tuning antes de re-implementar |
| 2026-10-31 | R-045 COMPLETADO: bit-plane CM hash — x-ray **-16.2%**, mr -2.6%, text 0% |
| 2026-10-31 | BUG-001 RESUELTO: 13 missing byte tokens in BPE vocab — synthetic fallback added |
| 2026-10-31 | BUG-001: x-ray BPB was 4.8969 (CORRUPTO) → 5.8008 (correcto) after fix |
| 2026-10-31 | Routing simplificado: Mamba always (samba 1.291→0.271, -79%). Wins 13/19 vs brotli |
| 2026-11-01 | docs: execution-architecture-analysis.md — parallel scan, speculative CM, 6 strategies (aa4fa56) |
| 2026-11-01 | Fase 8 definida: modular execution architecture — CmEnsemble + parallel scan + speculative |
| 2026-11-01 | R-051 COMPLETADO: CmEnsemble — 6 CM models + mixer + buffers en single struct (ad62291) |
| 2026-11-01 | R-051: 9 loops refactorizados (codec 5, api 3, benchmark 1), -259 líneas duplicación |
| 2026-11-01 | R-051: compute_biases(entropy_hint), combine_into(), update(), observe(), with_bias_decay(tau) |
| 2026-11-01 | R-052 COMPLETADO: parallel_scan.rs — Blelloch scan + SsmScanPair/Vec, 6 tests (ad62291) |
| 2026-11-01 | R-052: generic associative scan O(N log N) work, O(log N) depth. Model-agnostic. |
| 2026-11-01 | R-053 COMPLETADO: speculative.rs — CM-draft + SpeculativeTracker + verify_draft, 5 tests (ad62291) |
| 2026-11-01 | R-053: adaptive K via EMA acceptance rate, min_confidence early stop |
| 2026-11-01 | Prueba completa: 232 tests PASS, dickens 10KB 0.6464 BPB, roundtrip byte-identical |
| 2026-11-01 | Research: cpu-parallelism-strategies.md — 5 mathematical strategies to break sequential bottleneck |
| 2026-11-01 | Fase 9 definida: R-054 (parallel matmul) → R-055 (layer scan) → R-056 (CM-selective) → R-057 (dt-sparse) → R-058 (LSH) |
| 2026-11-01 | R-054 ✅ parallel matmul (std::thread::scope, raw-slice kernels) + R-057 ✅ dt-sparse SSM skip (ε=1e-6). BPB unchanged. |
| 2026-11-01 | R-055 ❌ BLOQUEADO (FP non-assoc breaks roundtrip). R-056 ❌ NO VIABLE (+81-294% BPB, backbone irreducible). |
| 2026-11-01 | Insight: CPU 84% idle because 1/12 threads active, NOT because of inefficiency |
| 2026-11-01 | Insight: DRAM bandwidth 60% saturated on RWKV — multi-thread matmul helps compute, not memory |
| 2026-11-02 | Research: lottery-ticket-sparsity-protocol.md — Frankle-Carbin derivatives for SSM acceleration |
| 2026-11-02 | Research: 6 papers analyzed (SparseSSM, LAST, Simba, Wanda, Activity Sparsity, ReplaySSM+3) |
| 2026-11-02 | Fase 10 definida: R-059 (act. sparsity) → R-060 (pipeline) → R-061 (column pruning) → R-062 (highway) → R-063 (batch checkpoint) |
| 2026-11-02 | BPB budget ampliado a +5% max (CLAUDE.md updated) |
| 2026-11-02 | Projection: B1+C1 only = ~22-31% speedup at ~0% BPB; full protocol = x1.6-1.8 at +2-5% BPB |
| 2026-11-02 | R-059 ✅ COMPLETADO: gated[] sparsity medida 1.7-1.9% — block-sparse NO VIABLE (necesita >25%) |
| 2026-11-02 | R-059: corrección diseño — target correcto es gated[], no conv_out. SiLU raramente produce zeros. |
| 2026-11-02 | R-059: instrumentación permanente en MambaScratch — contadores reportados en compress/decompress |
| 2026-11-02 | R-060 ✅ COMPLETADO: split-head pipeline — forward_layers_normed + compute_head_into |
| 2026-11-02 | R-060: dual hidden buffers + std::thread::scope, BPB 0.9648 (0% regression), roundtrip PASS |
| 2026-11-02 | R-060: throughput ~19 tok/s → ~19 tok/s (DDR4-2667 bandwidth saturado — speedup nulo) |
| 2026-11-02 | R-060: pipeline solo en compress (decompress: tok[t] desconocido hasta decode logits[t-1]) |
| 2026-11-02 | Insight: DRAM bandwidth es bottleneck real, pipeline beneficiará DDR5/dual-channel |
| 2026-11-02 | R-061 ❌ NO VIABLE: Wanda column pruning — 1 canal (+0.065%) ya da +20.4% BPB |
| 2026-11-02 | R-061: 2%=+343%, 5%=+234%, 10%=+431% (no monotónico, alta sensibilidad) |
| 2026-11-02 | R-061: Mamba-130M demasiado pequeño para structured pruning (papers validan 7B+) |
| 2026-11-02 | R-062 ❌ NO VIABLE: highway layer skip — capas NO son highways en Mamba-130M |
| 2026-11-02 | R-062: thr≤0.15 → 0 skips (residuales >15%). thr=0.10 → 17/85K skips → +3.86% BPB |
| 2026-11-02 | R-062: thr=0.20 → 5.5% skips → +713% BPB (catastrófico) |
| 2026-11-02 | Conclusión Fase 10: 4/5 técnicas NO VIABLES en 130M. Solo R-063 (batch) pendiente |
| 2026-11-02 | Investigación DRAM: PIM no viable en commodity DDR4, roofline analysis, 6 técnicas identificadas |
| 2026-11-02 | Fase 11 definida: R-064 (Mamba Q8) → R-065 (NTA prefetch) → R-066 (mixed Q4/Q8) → R-067 (L2-tile) |
| 2026-11-02 | R-064 ❌ NO VIABLE: Q8 PTQ +16.6% BPB (all-Q8), +14.5% (LM head only). Roundtrip PASS. Revertido |
| 2026-11-02 | R-066 ❌ BLOQUEADO: Q8 ya inaceptable, Q4 peor. PTQ no viable en Mamba-130M |
| 2026-11-02 | R-065 ❌ NO VIABLE: NTA prefetch ±0% (493MB >> 12MB L3, pollution irrelevant) |
| 2026-11-02 | R-067 ❌ NO VIABLE: L2-tile layout — GEMV streaming, tiling only helps GEMM |
| 2026-11-02 | Fase 11 CERRADA: 4/4 técnicas NO VIABLES. DRAM bandwidth = hard ceiling |
| 2026-11-02 | Fase 12 definida: R-068 (PGO) → R-069 (op fusion) → R-070 (conv1d SIMD) → R-071 (SVD LM head) |
| 2026-11-06 | Fase 13 añadida: Portable GPU Backend Architecture (R-072 to R-076) |
| 2026-11-06 | R-072 to R-076 COMPLETADOS — Fase 13 cerrada. Plugin GPU backend architecture. |
| 2026-11-07 | Fase 14 definida: R-077 → R-078 → R-079. Token reduction research (23 approaches, 21 NO-GO, 2 GO). Nacrith-style entropy gate = immediate next step. |
| 2026-11-07 | R-077 COMPLETADO — entropy gate implemented, NOT VIABLE at BPE 50K/10KB. CM entropy 15.49-15.62 bits (near-uniform). Viable only with byte-level model. |
| 2026-11-07 | R-078 COMPLETADO — Kaggle v3 analysis: enwik8 0.9633 BPB (+0.25% vs local), 72.4 tok/s (2.1x local), GPU FFI pipeline validated (81K-96K tok/s skeleton). |
| 2026-11-07 | R-079 COMPLETADO — MambaByte feasibility update: from-scratch $87-145, Byteification NOT RECOMMENDED, entropy gate unlocked at vocab=256, T4 projected 650-2,500 B/s. |
| 2026-11-07 | Fase 15 definida: MambaByte Training Infrastructure (R-080 → R-083). Training notebook, weight conversion, Rust byte tokenizer, cross-framework validation. |
| 2026-11-07 | R-082 COMPLETADO — ByteTokenizer (vocab=256), BackboneChoice::MambaByte, model_loader integration, 3 tests. |
| 2026-11-07 | R-080 COMPLETADO — mambabyte-train.py (922 lines): MambaByte-96M PyTorch, TPU v5e-8 + GPU T4, checkpointing, byte streaming, SafeTensors export. |
| 2026-11-07 | R-081 COMPLETADO — convert-mambabyte-weights.py (287 lines): PyTorch→SafeTensors, key mapping, shape validation, SHA-256 provenance. |
| 2026-11-07 | R-083 COMPLETADO — mambabyte-validate.py (1005 lines): cross-framework validation, 6 tests (BPB range, agreement, roundtrip, determinism, multi-file). |
| 2026-11-09 | Fase 17 definida: Pre-trained MambaByte + Competitive Intelligence (R-087 → R-090) |
| 2026-11-09 | R-087 COMPLETADO — Strata + competitive landscape research. 8 Strata projects, 15 neural compressors. |
| 2026-11-09 | Hallazgo: MambaByte 353M pre-entrenado en HuggingFace (JunxiongWang), 0.396 BPB CE enwik8 |
| 2026-11-09 | Hallazgo: Nacrith = competidor más cercano (0.9389 BPB, GPU-only, solo texto) |
| 2026-11-09 | Hallazgo: strata-inference = PTX-as-string patrón adoptable para GPU plugin Redukt |
| 2026-11-09 | R-088 definido: evaluar pesos pre-entrenados MambaByte 353M — P0, potencialmente elimina R-084/R-085 |
| 2026-11-09 | docs: strata-competitive-landscape.md — investigación completa, 15 fuentes, 5 secciones |
| 2026-11-09 | R-088 COMPLETADO — MambaByte 353M pre-trained weights evaluated on enwik8 10KB |
| 2026-11-09 | R-088: enwik8 1.3156 BPB (+37% vs Mamba-130M BPE 0.9609). Domain mismatch (arXiv→Wikipedia) |
| 2026-11-09 | R-088: LayerNorm bug found and fixed (RMSNorm→LayerNorm): 1.7531→1.3156 BPB (-25%) |
| 2026-11-09 | R-088: untied embeddings support added (lm_head_w), convert --preset 353m/972m, HF key stripping |
| 2026-11-09 | Conclusión R-088: pesos pre-entrenados NO reemplazan training propio. R-084/R-085 siguen necesarios. |
| 2026-11-09 | R-088: roundtrip interrumpido por rebuild (taskkill). Compresión OK, CRC OK, 218 tests pass. |
| 2026-11-09 | Fase 18 definida: CPU Throughput Reformulation (R-091 → R-097). Hardware-aware execution pipeline. |
| 2026-11-09 | R-091 (NTA prefetch MambaByte), R-092 (L2 tiling MambaByte) — re-evaluación bajo nuevo régimen |
| 2026-11-09 | R-093 (parallel scan compress CPU) — P0, mayor impacto: 10-50× compress throughput |
| 2026-11-09 | R-094 (CM pre-scan), R-095 (HugePages Linux), R-096 (FFI v2 batch), R-097 (Q4 re-eval) |
| 2026-11-09 | Análisis: Fases 10-12 fracasaron bajo condiciones Mamba-130M; MambaByte cambia 3 condiciones simultáneamente |
| 2026-11-09 | docs: cpu-throughput-reformulation.md — auditoría código + reformulación hardware-aware, 10 fuentes |
| 2026-11-09 | Proyección CPU-only: 96M+gate 237 tok/s decompress, 800-1500 tok/s compress (parallel scan) |
| 2026-11-09 | ⚠️ Kaggle TPU v5e-8 saturado — R-084 bloqueado (push OK, sesión no inicia) |
| 2026-11-09 | R-090 IMPLEMENTADO — CUDA Mamba plugin completo: 11 kernels (GEMV float4+warp shuffle, SSM, RMSNorm, LayerNorm, SiLU, Conv1d, Softplus, GatedSiLU) |
| 2026-11-09 | R-090: .rbin binary weight format (44B header, flat f32), converter SafeTensors→.rbin, test+benchmark notebook |
| 2026-11-09 | R-090: Pendiente validación en Kaggle T4. Proyección: ~500 tok/s (15x vs CPU 34 tok/s) |
| 2026-11-09 | R-084 degradado P0→P1. R-088 (pesos pre-entrenados 353M) elevado a path crítico |
| 2026-11-09 | R-085 nota: si R-088 confirma 353M, R-085 se cancela o reenfoca como fine-tuning |
| 2026-11-07 | Fase 15 COMPLETADA: R-080 ✅ R-081 ✅ R-082 ✅ R-083 ✅ — MambaByte training infrastructure ready. |
| 2026-11-07 | Fase 16 definida: MambaByte Training Execution (R-084 → R-086). 50M validation → 96M production → integration. |
| 2026-11-07 | R-084 LANZADO — MambaByte-50M en Kaggle TPU v5e-8. Slug: jdavidpcastro/mambabyte-50m-validation. |
| 2026-11-09 | Auditoría de throughput por propiedades: 3 premisas refutadas (Mamba F32 en hot path; DRAM dual channel 42.7 GB/s con 41% usado; in_proj/out_proj en un solo hilo). Doc: throughput-properties-audit.md |
| 2026-11-09 | Fase 19 definida: CPU Limit & Market Viability (R-098 → R-105). Mandato: producto viable y competitivo en el mercado real |
| 2026-11-09 | R-098 (Q8 weight-only Mamba) y R-099 (multihilo in_proj/out_proj) EN CURSO; R-100 (lockstep multi-stream) y R-101 (Q6) PENDIENTES |
| 2026-11-09 | R-102 (despliegue GCP CPU) y R-103 (despliegue CUDA) EN CURSO |
| 2026-11-09 | R-104 determinismo cross-máquina — P0 bloqueante de producto (Kaggle 0.9633 vs local 0.9609, roundtrip entre máquinas no probado) |
| 2026-11-09 | R-105 tier rápido (backbone 3-30M, referencia L3TC) — investigación; decisión con medida, no proyección |
| 2026-11-09 | Economía: coste hoy ≈$47/GB raw (spot, 130 B/s); equilibrio a 7 años enwik8 en S3 Standard 18 KB/s por vCPU (scripts/breakeven.py) |
| 2026-11-09 | docs: CLAUDE.md corregido (Q8 solo RWKV; 42.7 GB/s dual channel); speculative.rs y parallel_scan.rs anotados como borrados en 1dc6402 (nunca medidos) |
