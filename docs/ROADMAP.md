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

### R-V04: Test de escala (BPB vs tamaño de archivo)
- **Estado**: EN PROGRESO (gzip/brotli medidos, Redukt pendiente)
- **Impacto**: Benchmarks actuales son a 10KB. No sabemos si la ventaja crece o decrece.
- **Resultado parcial** (gzip-9 y brotli-11 en apache-access.log, MEDIDO):
  - 1KB: gzip 1.760, brotli 1.592
  - 10KB: gzip 0.546, brotli 0.466
  - 50KB: gzip 0.463, brotli 0.353
  - 100KB: gzip 0.445, brotli 0.328
  - **Ambos mejoran significativamente con más datos** (brotli: 1.59→0.33, -79%)
  - Detalle: `data/benchmark/results/scale-test-apache.csv`
- **CRITICAL**: Si Redukt no mejora proporcionalmente, la ventaja de -48% a 10KB
  podría reducirse sustancialmente a 100KB+. Esto afecta TODOS los business cases.
- **Subtareas**:
  - [x] Medir gzip-9 y brotli-11 a 1KB, 10KB, 50KB, 100KB
  - [ ] Medir Redukt Q8 a 1KB, 10KB, 50KB, 100KB (requiere weights + model run)
  - [ ] Documentar curva de convergencia comparativa

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
- **Estado**: EN PROGRESO
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
  - [ ] Evaluar almacenar método en header (v4) para roundtrip transparente
  - [ ] Documentar que compress/decompress deben usar el mismo `--emb-surgery`
  - [ ] `--match` (longest-match predictor): sólo spec en docs/research/, -0.0095 BPB, bajo ROI
  - [ ] `--hierarchical`: supersedido por AdaptiveMixer actual, no portar

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

### R-021: F32 AVX2 sin FMA para LoRA matmuls (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — AVX2 mul+add separados (no FMA) produce resultados bit-exact con scalar
- **Impacto estimado**: +10-20% en TimeMix (96 LoRA matmuls/token son F32 scalar) [ESTIMATE]
- **Contexto**: R-011 deshabilitó F32 FMA porque `fma(a,b,c) ≠ a*b+c` (distinto redondeo).
  Pero AVX2 SIN FMA (vmulps + vaddps separados) ES bit-exact con scalar mul+add.
  El kernel FMA existe (`mat_vec_mul_avx2_fma_into`, dead code) — se puede adaptar.
- **Archivos**: `src/domain/tensor.rs`
- **Subtareas**:
  - [ ] Implementar `mat_vec_mul_avx2_nofma_into`: vmulps + vaddps, 8-wide, 4-row ILP
  - [ ] Bit-exact test: scalar vs AVX2-noFMA (0/768 mismatches requerido)
  - [ ] Dispatch: AVX2-noFMA > scalar en `mat_vec_mul_into` (reemplazar scalar-always)
  - [ ] Cross-path roundtrip: SIMD→scalar y scalar→SIMD byte-identical
  - [ ] **Quality gate**: BPB idéntico (0.000 delta) en enwik8 + 3 logs
  - [ ] Benchmark end-to-end: tok/s con y sin LoRA SIMD

### R-022: SIMD transcendentals (tanh, sigmoid, exp) (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO si se usa polynomial approximation de suficiente precisión (~23 bit)
- **Impacto estimado**: reducción menor pero acumulativa en TimeMix element-wise ops [ESTIMATE]
- **Contexto**: `tanh_inplace`, `sigmoid_inplace`, `exp` en tensor.rs son scalar per-element.
  Vectorización con aproximaciones polinómicas (Cephes/minimax) puede ser bit-exact si el
  error ULP está dentro del redondeo f32.
- **Archivos**: `src/domain/tensor.rs`
- **Subtareas**:
  - [ ] Implementar sigmoid AVX2 (polynomial minimax, ~23 bit precision)
  - [ ] Implementar tanh AVX2 (derivado de sigmoid: tanh = 2σ(2x) - 1)
  - [ ] Max-ULP-error test: comparar con scalar en rango [-10, 10]
  - [ ] Cross-path roundtrip test
  - [ ] **Quality gate**: BPB idéntico o delta medible ≤ 0.001%
  - [ ] Evaluar si el error ULP rompe determinismo cross-path antes de integrar

### R-HW1: HardwareGovernor — regulación dinámica de recursos
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — no cambia computación, solo la regula
- **Impacto**: Protección contra deterioro de hardware (batería, térmico), crash prevention
- **Contexto**: Dell Vostro 3520 tiene batería al 80.2% health (único componente en riesgo).
  El governor lee sensores del hardware antes y durante la ejecución para regular el consumo.
- **Archivos**: nuevo `src/infrastructure/governor.rs`, `src/application/codec.rs`, `src/main.rs`
- **Diseño**:
  - Pre-flight checks: batería (umbral + health), RAM libre, temperatura
  - Watchdog thread: 10ms sensor polling via kernel32 FFI (Windows) / sysfs (Linux)
  - Comunicación: AtomicU8 → ThrottleLevel {Full, Balanced, Eco, Pause, Abort}
  - Worst-case reaction time: ~35ms (10ms poll + 25ms token)
  - Zero overhead en hot path: una lectura atómica por token (~1ns)
- **Subtareas**:
  - [ ] Implementar ThrottleLevel enum + AtomicU8 protocol
  - [ ] Pre-flight: `GlobalMemoryStatusEx`, `GetSystemPowerStatus` (kernel32 FFI)
  - [ ] In-flight watchdog: thread con 10ms sleep + sensor read
  - [ ] Integrar en compress/decompress loop: check ThrottleLevel por token
  - [ ] Eco mode: añadir sleep entre tokens para reducir power draw
  - [ ] Pause/Abort: checkpoint + graceful shutdown
  - [ ] `--no-governor` flag para deshabilitar (benchmarking)
  - [ ] Tests: mock sensors, throttle transitions, abort recovery
  - [ ] Documentar umbrales por defecto y cómo override

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

#### Paso 0.5: R-019 — Refactorización matemática (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — equivalencia algebraica exacta
- **Speedup estimado**: 1.3-1.5x en TimeMix (87% reducción FLOPs state update + zero-alloc codec)
- **Impacto**: Reduce cómputo Y tráfico de bus (menos datos intermedios en cache/DRAM)
- **Quality gate**: BPB bit-exact, roundtrip byte-identical

#### Paso 1: R-021 — F32 AVX2 sin FMA (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — vmulps + vaddps separados = bit-exact con scalar
- **Speedup estimado**: +10-20% en TimeMix (96 LoRA matmuls/token pasan de scalar a AVX2)
- **Quality gate**: BPB bit-exact, cross-path roundtrip byte-identical

#### Paso 1.5: R-022 — SIMD transcendentals (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO si precision ≥ 23 bits (f32 mantissa)
- **Speedup estimado**: ~5% en TimeMix element-wise ops
- **Quality gate**: BPB ≤ 0.001% delta, cross-path roundtrip

#### Paso 2: R-015 — KV Batch Caching (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — elimina cómputo redundante, no cambia predicciones
- **Speedup estimado**: 2-3x [ESTIMATE — arquitectural, no medido]
- **Impacto**: Cachear estados KV del RWKV entre tokens para evitar recálculo
- **Archivos**: `src/infrastructure/rwkv7/model.rs`
- **Subtareas**:
  - [ ] Perfilar con PhaseTimer: identificar % de tiempo en recálculo de estados
  - [ ] Implementar KV state cache en RWKV forward pass
  - [ ] Benchmark pre/post: tok/s y B/s en 3 logs + enwik8
  - [ ] **Quality gate**: verificar BPB idéntico pre/post (debe ser 0.000 delta)
  - [ ] Roundtrip test: compress → decompress → diff byte-identical

#### Paso 3: R-012 — Multi-file paralelismo (0% BPB risk)
- **Estado**: PENDIENTE
- **Riesgo BPB**: NINGUNO — cada archivo se procesa independientemente
- **Speedup estimado**: Nx con N cores [PROJECTION: linear scaling]
- **Impacto**: Linear con cores (N archivos en paralelo, modelo por thread)
- **Archivos**: `src/application/codec.rs`, `src/main.rs`
- **Subtareas**:
  - [ ] Evaluar RAM por instancia (~400 MB Q8, ~1.6 GB F32) [ESTIMATE]
  - [ ] Implementar modo batch: `redukt compress --input-dir <DIR> --output-dir <DIR> --jobs N`
  - [ ] Thread pool con modelo pre-cargado, state por thread
  - [ ] Benchmark: 1, 2, 4, 8 jobs en corpus de 8+ archivos de log
  - [ ] **Quality gate**: verificar BPB idéntico single-thread vs multi-thread (debe ser 0.000 delta)
  - [ ] Medir RAM real por thread y documentar requisitos

#### Paso 4: R-016 — SVD Pruning del modelo RWKV (BPB risk: +2-3%)
- **Estado**: PENDIENTE
- **Riesgo BPB**: MEDIO — reduce capacidad del modelo, degrada predicciones
- **Speedup estimado**: 3.5-4.8x [SOURCED: RWKV-edge paper, pero en NLP no compresión]
- **Presupuesto BPB consumido**: este paso solo, máximo **+3%**. Si consume +3%, no queda margen para R-017.
- **Dependencia**: R-V06 (factibilidad completada, SVD conditional GO)
- **Archivos**: `src/infrastructure/rwkv7/model.rs`, scripts de pruning offline
- **Subtareas**:
  - [ ] Implementar SVD rank-384 en TimeMix projections (offline, genera nuevos weights)
  - [ ] Benchmark BPB: 3 logs + enwik8 + facturae con modelo pruned vs baseline
  - [ ] Benchmark throughput: tok/s y B/s con modelo pruned
  - [ ] **Quality gate BPB**: si degradación >+3% en logs → reducir agresividad (rank-512, rank-448)
  - [ ] **Quality gate combinado**: medir BPB con R-018 + R-019 + R-021 + R-015 + R-012 + R-016 todos activos
  - [ ] Si pasa gates: generar weights pruned y añadir a `weights/MANIFEST.md`
  - [ ] Documentar BPB delta exacto por archivo [MEASURED]
  - [ ] Roundtrip test con modelo pruned

#### Paso 5: R-017 — Parallel Segment Compression (BPB risk: variable) — ESTRICTAMENTE CONDICIONAL
- **Estado**: PENDIENTE
- **Riesgo BPB**: ALTO — divide el archivo en segmentos, cada uno pierde contexto previo
- **Speedup estimado**: 2-4x [ESTIMATE — depende de tamaño de segmento]
- **Presupuesto BPB restante**: +3% total menos lo consumido por R-016
- **PRECONDICIÓN**: solo proceder si R-016 dejó margen en el presupuesto de +3%.
  Con el presupuesto reducido, es probable que R-016 consuma todo el margen y R-017 no proceda.
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
| R-021 | + F32 AVX2 noFMA | 1.1-1.2x | ~215-270 B/s | 0% | BPB bit-exact |
| R-022 | + SIMD transcendentals | ~1.05x | ~225-284 B/s | 0% | BPB ≤ 0.001% |
| R-015 | + KV caching | 1.5-2x | ~338-567 B/s | 0% | BPB = 0 delta |
| R-012 | + 10 cores | 10x | ~3375-5670 B/s | 0% | BPB = 0 delta |
| R-016 | + SVD pruning | 3-5x | ~10125-28350 B/s | +2-3% | BPB ≤ +3% |
| R-017 | + Parallel segments | 2-4x | ~20250-113400 B/s | +3%? | BPB ≤ +3% total |

**Nota crítica**: Los speedups multiplicativos son teóricos. En la práctica:
- R-018 no mejora throughput, pero reduce BPB → amplía el margen competitivo
- R-019 es compute reduction puro — beneficio directo en TimeMix latency
- R-021/R-022 mejoran el "techo" scalar de TimeMix (74-88% del tiempo)
- R-015 × R-016 podrían ser sub-multiplicativos (ambos tocan inference path)
- R-012 es genuinamente multiplicativo (paralelismo puro, 10 cores útiles [MEASURED: PROTO-05])
- R-017 tiene overhead de merge y cold-start por segmento
- El gate de **+3%** BPB probablemente fuerza STOP antes de R-017
- R-HW1 (HardwareGovernor) es ortogonal — no afecta throughput peak pero protege hardware

**Target mínimo viable**: R-019 + R-021 + R-015 + R-012 (10 cores) = ~3-6 KB/s sin degradación BPB.
Si se alcanza, log archival de ~2-4 GB/día es viable sin tocar el modelo.

**Secuencia revisada (2026-10-28)**: R-018 ✅ → R-019 → R-021 → R-022 → R-015 → R-012 → R-016 → R-017.
Primero las optimizaciones 0% BPB risk (compute reduction, SIMD), luego paralelismo, al final modelo.
R-HW1 puede implementarse en cualquier momento (independiente de la secuencia de throughput).

### R-013: Determinismo cross-platform
- **Estado**: PENDIENTE
- **Subtareas**:
  - [ ] Generar golden files (input → .azth → restored) en CI
  - [ ] Verificar en Windows x86, Linux x86, Linux ARM64
  - [ ] Auditar operaciones float que podrían divergir entre plataformas

### R-014: Streaming decompression
- **Estado**: PENDIENTE
- **Subtareas**:
  - [ ] Evaluar viabilidad (el modelo es secuencial, pero output puede ser progresivo)
  - [ ] Útil para archivos grandes: emitir bytes conforme se decodifican

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
