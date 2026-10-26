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
- **Estado**: PENDIENTE
- **Impacto**: `panic!()` en I/O no es aceptable para librería/API
- **Archivos**: `src/application/codec.rs`
- **Subtareas**:
  - [ ] Reemplazar `unwrap_or_else(|e| panic!())` por propagación de `Result<>`
  - [ ] Devolver `AzError` desde compress/decompress
  - [ ] CLI: catch en main, exit(1) con mensaje limpio
  - [ ] API: mapear a HTTP status codes apropiados

### R-007: Límite u32 TOTAL_BYTES — documentar y validar
- **Estado**: PENDIENTE
- **Impacto**: Archivos >4GB corrompen silenciosamente (overflow u32)
- **Coste**: 0 bytes (validar), o +4 bytes (upgrade a u64 en futuro)
- **Subtareas**:
  - [ ] Validar en compresión: si input > u32::MAX, error explícito
  - [ ] Documentar en FORMAT-SPEC.md como limitación conocida (ya documentado)
  - [ ] Evaluar upgrade a u64 en futura versión

---

## P2 — Métricas y Observabilidad

### R-008: Timing por fase
- **Estado**: PENDIENTE
- **Impacto**: Necesario para identificar cuellos de botella en optimización
- **Subtareas**:
  - [ ] Medir: tokenización, carga modelo, forward pass, encoding, escritura
  - [ ] Reportar en JSON telemetry
  - [ ] Opcional: `--verbose` flag para mostrar breakdown

### R-009: Métricas de eficiencia del coder
- **Estado**: PENDIENTE
- **Subtareas**:
  - [ ] Overhead ratio: bits_encoded vs entropía teórica (cross-entropy)
  - [ ] Ya existe parcialmente (ce_bpb vs compressed_bpb) — exponer en report

---

## P3 — Futuro

### R-010: Determinismo cross-platform
- **Subtareas**:
  - [ ] Generar golden files (input → .azth → restored) en CI
  - [ ] Verificar en Windows x86, Linux x86, Linux ARM64
  - [ ] Auditar operaciones float que podrían divergir entre plataformas

### R-011: Streaming decompression
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
