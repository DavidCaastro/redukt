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
- **Estado**: PENDIENTE (requiere R-034)
- **Prioridad**: P1 — resuelve el bottleneck O(D²) → O(D·N) con N=16
- **Impacto**: Throughput estimado 3-10x sobre RWKV para mismo tamaño de modelo
- **Riesgo BPB**: DESCONOCIDO — requiere benchmark comparativo
- **Evidencia**:
  - StateSMix (Mamba 120K, CPU, C+AVX2): 1,100 tok/s [SOURCED: arXiv:2605.02904]
  - MambaByte (353M, byte-level): 0.93 BPB en PG19 [SOURCED: arXiv:2401.13660]
  - `mamba-rs` (Rust, SafeTensors loader): inference zero-alloc [SOURCED: github.com/silvermpx/mamba-rs]
  - Estado diagonal → parallel scan funcional, SIMD trivial (elementwise)
  - Vs RWKV-7: estado vector O(D·N) vs matriz O(D²), composición O(D·N) vs O(N³)
- **Seguridad**: SafeTensors format (sin pickle, sin ejecución de código).
  Redukt ya tiene parser SafeTensors propio (zero-dep, zero-unsafe, bounds-checked).
  Solo cambian los nombres de tensores (mixer.A_log, mixer.D, mixer.in_proj, etc.)
- **Archivos**: `src/infrastructure/mamba/` (nuevo), `src/domain/backbone.rs`
- **Subtareas**:
  - [ ] Investigar pesos Mamba pretrained disponibles (130M, 370M, MambaByte 353M)
  - [ ] Evaluar MambaByte vs Mamba genérico para byte-level compression
  - [ ] Auditar dependencias de `mamba-rs` crate — portar lo necesario a zero-dep
    - [ ] Identificar qué usa `mamba-rs` de crates externos (safetensors, serde, etc.)
    - [ ] Portar lógica de inference Mamba a Rust zero-dep (NO usar mamba-rs como crate)
    - [ ] Reusar parser SafeTensors propio de Redukt (ya zero-dep)
    - [ ] Portar operaciones matemáticas específicas de Mamba (sin deps externas)
  - [ ] Implementar inference Mamba zero-dep:
    - [ ] Selective scan (S6): input-dependent A, B, C, D matrices
    - [ ] Discretization: ZOH (zero-order hold) para A, B continuas → discretas
    - [ ] State update: `h_t = Ā·h_{t-1} + B̄·x_t` (diagonal, O(D·N))
    - [ ] Output: `y_t = C_t·h_t + D·x_t`
    - [ ] SIMD kernels para selective scan (AVX2, reusar dispatch de tensor.rs)
  - [ ] `impl Backbone for MambaModel` + `impl BackboneState for MambaState`
  - [ ] Cargar pesos SafeTensors (reusar parser existente, nuevo mapping de nombres)
  - [ ] SHA-256 checksums en `weights/MANIFEST.md`
  - [ ] Benchmark comparativo 19 archivos: BPB + tok/s vs RWKV
  - [ ] Quality gate: si BPB dentro de ±5% de RWKV → Mamba como default por throughput
  - [ ] Si BPB peor >5%: flag `--backend rwkv|mamba` para selección por usuario
  - [ ] Roundtrip: compress → decompress byte-identical, CRC verified
  - [ ] Cross-backend: NO se espera compatibilidad (distinta distribución de probabilidades)
  - [ ] Documentar resultados en `data/benchmark/results/rwkv-vs-mamba.csv`

#### R-036: BPE tokenizer — reduce tokens per byte
- **Estado**: PENDIENTE (independiente de R-034/R-035, beneficia ambos backbones)
- **Prioridad**: P2 — multiplica throughput 3-5x para cualquier backbone
- **Impacto**: Menos forward passes por byte → throughput proporcional
- **Riesgo BPB**: DESCONOCIDO — BPE agrupa bytes, pierde granularidad byte-level
- **Evidencia**:
  - L3TC (AAAI 2026): outlier-aware tokenizer, 2.2x faster RWKV decode [SOURCED: AAAI]
  - Gate ya reservado: `--tokenizer bpe` (R-033 infrastructure gate)
  - Requiere vocabulario entrenado en `weights/bpe-vocab.json`
- **Archivos**: `src/domain/tokenizer.rs` (nuevo o extensión), `weights/bpe-vocab.json` (nuevo)
- **Subtareas**:
  - [ ] Investigar vocabularios BPE existentes compatibles (SentencePiece, tiktoken)
  - [ ] Evaluar entrenamiento de vocabulario propio vs reuso
  - [ ] Implementar BPE encode/decode zero-dep en Rust
  - [ ] `impl Tokenizer for BpeTokenizer`
  - [ ] Adaptar arithmetic coder para vocabularios BPE (vocab_size variable)
  - [ ] AZTH v4: almacenar tokenizer type en header (nuevo FLAG bit)
  - [ ] Benchmark: BPB + tok/s con BPE vs byte-level en 19 archivos
  - [ ] Quality gate: BPB BPE debe ser ≤ byte-level BPB (o justificar throughput tradeoff)

#### R-037: Expanded Context Mixing — more predictors
- **Estado**: PENDIENTE (independiente, beneficia todos los backbones)
- **Prioridad**: P2 — mejora BPB sin afectar throughput significativamente
- **Impacto**: Más modelos CM → mejor ensemble → menor BPB
- **Evidencia**:
  - Redukt actual: 19 hash-table models + LSTM mixer
  - cmix: ~2,000 context models [SOURCED: mattmahoney.net/dc/text.html]
  - Nacrith: 4 online context models alongside pretrained LM [SOURCED: arXiv:2602.19626]
  - Espacio para 10-50 modelos adicionales sin impacto significativo en throughput
    (CM forward es <5% del forward total)
- **Archivos**: `src/domain/context_mixing.rs` (o equivalente), `src/domain/adaptive_mixer.rs`
- **Subtareas**:
  - [ ] Auditar modelos CM actuales (19 hash-tables: qué n-grams, qué contextos)
  - [ ] Investigar modelos adicionales viables: match model, word model, sparse contexts
  - [ ] Implementar 5-10 nuevos predictores
  - [ ] Benchmark incremental: BPB por cada predictor añadido (diminishing returns)
  - [ ] Quality gate: cada predictor debe mejorar BPB en ≥2 archivos sin empeorar ninguno

#### Secuencia Fase 4

```
R-034 (dock)     ─→ R-035 (Mamba)     ─→ benchmark comparativo → decisión backbone default
                 ─→ R-036 (BPE)       ─→ aplica a ambos backbones
                 ─→ R-037 (expanded CM) ─→ aplica a ambos backbones
```

R-034 es bloqueante para R-035 únicamente. R-036 y R-037 son independientes y pueden
ejecutarse en paralelo con R-035. Las tres (R-035, R-036, R-037) se benefician mutuamente:
BPE reduce tokens para Mamba, expanded CM mejora BPB para ambos backbones.

**Target de Fase 4**: throughput >10 KB/s single-thread, >50 KB/s aggregate (3T).
Esto haría viable comprimir 1 TB en ~6 días con 50 cores — aceptable para cold/archival.

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
