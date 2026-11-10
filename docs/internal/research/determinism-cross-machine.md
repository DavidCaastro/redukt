# Determinismo entre máquinas: evidencia y diseño de R-104

> Fecha: 2026-11-10 · Commit medido: `7ba6373` (worktree limpio de HEAD, sin los cambios en curso)
> Hardware: i5-1235U (Alder Lake, AVX2 + FMA + AVX-VNNI). Windows 11 + Docker Desktop 28 (WSL2)
> Toolchain: rustc 1.98.1 en los dos lados · Windows = MSVC UCRT · Linux = Debian bookworm, glibc 2.36
> Complementa a `determinism-audit.md` (R-013/R-013b), que solo cubría una misma máquina.

---

## 1. Resumen

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Un AZTH comprimido en Windows se descomprime en Linux, y al revés? | **Sí**, en las 9 celdas probadas | §3.2: 18/18 roundtrips OK |
| ¿Los bitstreams son iguales entre sistemas operativos? | **Sí**, sha256 idéntico en 9/9 | §3.2 |
| ¿Y entre CPUs distintas (el Xeon de Kaggle)? | **Sí**. El `enwik8.cpu.azth` generado en Kaggle es **idéntico byte a byte** al que genera HEAD en Windows, y se descomprime en Windows con CRC OK | §3.3 |
| ¿Por qué Kaggle dio 0.9633 y local 0.9609? | Porque la **referencia estaba obsoleta**, no por la plataforma. HEAD da 0.9633 en Windows, en Linux y en Kaggle; 0.9609 se midió el 31-10 con un binario anterior | §3.3 |
| H1: libm (`exp`, `ln`, `tanh`, `powf`) | **Las libm sí difieren**: el 0.5 % de los `f64::exp` en 1 ulp y el 4.8 % de los `f32::tanh` hasta en 2 ulp. Pero la ruta crítica f32 **no usa libm** (R-013b). Solo queda un uso que entra en el bitstream, `f64::exp` en `cdf_from_logits`, y el redondeo a f32 lo enmascara: 0 cambios en 1.5·10⁹ valores. Es un riesgo latente, no observado | §4 |
| H2: orden de acumulación distinto entre tiers SIMD | **Confirmada**. Es la única incompatibilidad observada: un archivo AVX2 decodificado con `REDUKT_NO_SIMD=1` falla | §3.2, §2 |
| H3: FMA implícito según `target-cpu` | **Refutada**. `native` (Windows), `x86-64-v3` (Windows) y `x86-64-v3` (Docker) dan el mismo sha. rustc/LLVM nunca fusiona `a*b+c` sin `mul_add` o un intrínseco explícito | §3.2 |
| H4: número de hilos | **Irrelevante**. `REDUKT_THREADS=1` y 12 hilos dan el mismo sha | §3.2 |

**Conclusión**: en HEAD es falsa la premisa de que los bitstreams cambian de una máquina a otra (x86-64 con
AVX2). El riesgo real para el producto es más acotado:

- (a) el tier SIMD frente al escalar: CPU sin AVX2, ARM y `REDUKT_NO_SIMD`;
- (b) dos llamadas residuales a libm f64, que hoy no rompen nada por suerte numérica;
- (c) un header que no declara el contrato numérico ni la cuantización interna de RWKV.

---

## 2. Inventario de funciones matemáticas no triviales (HEAD `7ba6373`)

Leyenda de la columna **Impl**:

- **libm**: std, depende de la plataforma.
- **portable**: Rust puro, solo `+ − × ÷` IEEE y operaciones de bits.
- **IEEE**: operación correctamente redondeada por norma, idéntica en todas partes.
- **int**: aritmética entera exacta.

La columna **Bitstream** indica si el resultado entra en la CDF que ve el range coder.

### 2.1 Inferencia (backbone)

| Función | Archivo:línea | Impl | Bitstream | Comportamiento por tier |
|---|---|---|---|---|
| `portable_expf` | tensor.rs:765 | portable (Cephes, grado 5) | sí | bit-exacta con `fast_exp_ps` (test tensor.rs:2641) |
| `portable_lnf` | tensor.rs:799 | portable (`logf` de musl/FreeBSD) | sí (softplus, mixer, bias, ngram) | sin versión SIMD |
| `portable_sigmoidf` / `tanhf` / `silu` | tensor.rs:841-855 | portable (vía `expf`) | sí | bit-exactas con `fast_sigmoid_ps` / `fast_tanh_ps` |
| `portable_softplus` (dt de Mamba) | tensor.rs:859, mamba/model.rs:161 | portable | sí | escalar en los dos tiers |
| `fast_exp_ps` / `fast_sigmoid_ps` / `fast_tanh_ps` | tensor.rs:887-935 | portable AVX2 (mul+add, sin FMA) | sí | — |
| `ssm_recurrence_d16` (producto y = Σ ss·c) | tensor.rs:1174 / 1194 / 1226 | IEEE | sí | **diverge**: el escalar suma en secuencia n=0..15; AVX2 suma lo+hi y luego `hsum` en árbol |
| `mat_vec_mul_avx2_nofma` / `_scalar` (GEMV F32: LoRA de RWKV, x_proj/dt_proj de Mamba) | tensor.rs:1330-1407, 1496 | IEEE | sí | **diverge**: 8 carriles + `hsum` frente a suma secuencial |
| `gemv_q8w` (R-099, Mamba Q8 W8A32) | gemv.rs | IEEE | sí | **invariante al tier** por contrato: el escalar emula los 8 carriles |
| `q8_mat_vec_mul` VNNI / AVX2 / escalar (RWKV) | tensor.rs:237, 1700 | int + 1 mul f32 | sí | exacta: `dpbusd` = `maddubs` (sin saturación con \|q\| ≤ 127) = escalar |
| `quantize_vec_i8` (`round`) | tensor.rs:~200 | IEEE (`roundf` es exacta) | sí | — |
| `rms_norm_into` / `layer_norm_into` / `group_norm` (`sqrt`) | tensor.rs:869, 1291, 1306 | IEEE (`sqrt` correctamente redondeada) | sí | escalar en los dos tiers |
| `mat_vec_mul_small_into` (estado RWKV) | tensor.rs:1772 | IEEE | sí | escalar en los dos tiers |
| `mat_vec_mul_avx2_fma_into` | tensor.rs:1413 | FMA | — | **no se despacha** (`allow(dead_code)`, solo para bench) |
| GEMV multihilo (`par_pool`, `mat_vec_mul_parallel_into`) | gemv.rs:79, tensor.rs:1520 | — | sí | partición por filas: el hilo que calcula una fila no cambia el cálculo |

### 2.2 CM, mixer y coder

| Función | Archivo:línea | Impl | Bitstream | Nota |
|---|---|---|---|---|
| **`cdf_from_logits`: `((l as f64) - max).exp()`** | **codec.rs:502** | **libm f64** | **sí** | **único uso de libm en la ruta crítica**. El resultado se redondea a f32 antes de cuantizar |
| **`cm_only_entropy`: `f64::exp`, `f64::ln`** | **cm_ensemble.rs:133-139** | **libm f64** | sí, **solo con `--adaptive-skip`**, porque decide el gate. La ruta RWKV lo calcula siempre, pero sin el flag no se usa | el gate solo cambia si \|H − umbral\| < ~1e-7 |
| `softmax_into` (probabilidades para la actualización de CM y mixer) | tensor.rs:1797 | portable + suma f32 secuencial | sí (vía la actualización) | — |
| `AdaptiveMixer::update` (`-portable_lnf(p)`) | mixer.rs:104 | portable | sí | gradiente en el orden fijo del Vec |
| `BiasHead::update` (sorpresa) | bias_head.rs:90 | portable | sí | — |
| `ngram` (`portable_lnf(prob/base)`) | ngram.rs:128 | portable | sí | iterar el HashMap es inocuo porque las claves son únicas |
| `Cdf::from_probs` (suma f64, `+0.5` y `as u64`) | coder.rs:43-81 | IEEE + int | sí | determinista |
| Range coder, CRC-32 | coder.rs, crc32.rs | int | sí | exactos |
| `content_detect` (`log2`) | content_detect.rs:123 | libm | no: la decisión queda escrita en FLAGS/BACKBONE del header | — |
| `ce_bits` (`log2`), telemetría, benchmark | codec.rs:850…, telemetry.rs | libm | no (solo informes) | — |
| Carga de pesos: `powi(-14)` en f16→f32 | quant.rs:585 | exacta (potencia de 2) | sí (pesos) | — |

---

## 3. Experimento

### 3.1 Montaje

- Worktree limpio: `git worktree add <scratchpad>/wt-det HEAD`, para dejar fuera el trabajo en curso de
  codec/model/gemv/tensor.
- Linux: `docker build -f deploy/gcp/Dockerfile -t redukt:det .` (rust 1.98.1, `RUSTFLAGS=-C target-cpu=x86-64-v3`,
  perfil `maxperf`, runtime debian bookworm-slim con glibc 2.36). `weights/` montado en solo lectura.
- Windows: `cargo build --profile maxperf` en el mismo worktree (`.cargo/config.toml` fija `target-cpu=native`) y
  una segunda build con `RUSTFLAGS="-C target-cpu=x86-64-v3"`. Los dos `.exe` son distintos (sha `e5634a94…` y
  `27ec99c5…`).
- Entradas: los primeros 10 240 B de `data/benchmark/enwik8`, `industry/facturae-batch.xml` e
  `industry/apache-access.log`.
- En cada celda: comprimir en Windows (W) y en Linux (L); descomprimir W→L y L→W; `cmp` contra el original; sha256
  de los dos `.azth`.

### 3.2 Matriz (misma CPU, distinto sistema operativo y libm)

| Archivo | Backbone / perfil | W→L | L→W | sha W = sha L | sha (16 primeros) | BPB |
|---|---|---|---|---|---|---|
| enwik8 | Mamba F32 (v4) | OK | OK | **sí** | `384c30427270db79` | 0.9633 |
| enwik8 | Mamba Q8 (v5) | OK | OK | **sí** | `c251c8df87da2e36` | 0.9688 |
| enwik8 | RWKV (auto: 73 tensores en Q8) | OK | OK | **sí** | `daf838c8b66d7f3c` | 1.3258 |
| facturae | Mamba F32 | OK | OK | **sí** | `4de41c97181572ba` | 0.3883 |
| facturae | Mamba Q8 | OK | OK | **sí** | `6867e5fbcf1c8ec6` | 0.3930 |
| facturae | RWKV | OK | OK | **sí** | `5baf150382a7c6a1` | 0.4406 |
| apache | Mamba F32 | OK | OK | **sí** | `2dd4762b17706529` | 0.2164 |
| apache | Mamba Q8 | OK | OK | **sí** | `090c41efc4a555e8` | 0.2188 |
| apache | RWKV | OK | OK | **sí** | `0c4b49e7fa060d4a` | 0.2930 |

Controles en Windows:

| Variante | sha (16) | Resultado |
|---|---|---|
| facturae Mamba Q8, `target-cpu=native`, 12 hilos (por defecto) | `6867e5fbcf1c8ec6` | referencia |
| facturae Mamba Q8, `target-cpu=x86-64-v3` (binario distinto) | `6867e5fbcf1c8ec6` | **idéntico**: H3 refutada |
| facturae Mamba Q8, `REDUKT_THREADS=1` (18.6 tok/s) | `6867e5fbcf1c8ec6` | **idéntico**: H4, los hilos no influyen |
| apache Mamba Q8: archivo AVX2 descomprimido con `REDUKT_NO_SIMD=1` | — | **FALLO**: `integrity error: size mismatch: decoded 10242 bytes, expected 10240`. H2 confirmada (era lo esperado) |

Con Q8 el GEMV ya es invariante al tier (R-099), así que el fallo sin SIMD viene de los kernels F32 que siguen
ligados al tier: `ssm_recurrence_d16`, x_proj y dt_proj.

### 3.3 Otra CPU: Kaggle (Intel Xeon, Linux, binario compilado allí, 2026-11-06)

`data/benchmark/results/kaggle-v3/benchmark-data/enwik8.cpu.azth` (1 233 B, 0.9633 BPB) tiene el sha256
`384c30427270db79d07b12af71ebbb3993bc984f8ef14a5a8353734cf6355a94`. Es **idéntico** al que produce HEAD en
Windows y en Docker (tabla 3.2, enwik8 Mamba F32).

Descomprimido en Windows con HEAD da `CRC-32 1C94A6FE (verified OK)` y coincide byte a byte con el original. Lo
mismo pasa con `dickens.cpu.azth` (CRC `4D838BAC` OK; ese "dickens" es sintético, con periodo de 32 B). Por tanto,
el roundtrip **Kaggle → Windows ya está demostrado** con archivos reales y sobre otra microarquitectura (otra
caché, otro número de núcleos y probablemente sin AVX-VNNI).

La diferencia entre 0.9609 y 0.9633 no viene de la plataforma: entre el 31-10 y el 06-11 cambió el código (por
ejemplo `ecfe46d`, BUG-001 del tokenizer, y R-054/R-057).

**Acción documental**: en HEAD, la referencia correcta de CLAUDE.md y README para Mamba F32 sobre enwik8 10 KB es
**0.9633**. Mamba Q8 (R-099) da **0.9688**, un +0.57 %, dentro del presupuesto del +5 %.

---

## 4. Diagnóstico de libm (tarea C)

Programa independiente `sweep.rs`, fuera del repo: 2²⁴ entradas pseudoaleatorias con la misma semilla en los dos
sistemas operativos, mismo rustc 1.98.1 y `-O`. Se vuelcan los bits de cada salida y se comparan.

| Función std | Dominio | Valores distintos (UCRT frente a glibc) | Máx. ulp |
|---|---|---|---|
| `f64::exp` | [−60, 0] (el de `cdf_from_logits`) | 84 130 de 16.8 M (**0.50 %**) | 1 |
| `f64::exp` | [−700, 700] | 83 472 (0.50 %) | 1 |
| `f64::ln` | (0, 10⁶] | 919 (0.0055 %) | 1 |
| `f64::log2` | (0, 1] | 90 425 (0.54 %) | 1 |
| `f32::exp` | [−87, 88] | 11 872 (0.071 %) | 1 |
| `f32::ln` | (0, 10⁶] | 3 271 (0.020 %) | 1 |
| `f32::tanh` | [−10, 10] | 800 059 (**4.77 %**) | **2** |
| `f32::powf(x, 0.7)` | [0, 100] | 11 767 (0.070 %) | 1 |
| `f32::sqrt` | [0, 10⁶] | **0** | 0 |
| De extremo a extremo: réplica de `cdf_from_logits` + `Cdf::from_probs` (V = 50 277, 400 pasos) | — | **0** probabilidades f32 y **0** frecuencias distintas | 0 |

**H1 se confirma en la libm, pero hoy se refuta en el bitstream.** Si algún kernel f32 llamara a `f32::exp` o
`f32::tanh` de std, Windows y Linux producirían archivos distintos en pocos tokens; R-013b lo evitó.

El único uso de libm que sobrevive en la ruta crítica es `f64::exp` en `cdf_from_logits`, y queda enmascarado por
dos motivos:

- `(e * inv_sum) as f32` solo cambia si el valor f64 cae a 1 ulp f64 o menos de un punto medio f32. Eso tiene una
  probabilidad de ~2⁻²⁸ por cada valor alterado.
- La suma apenas cambia, porque los términos alterados casi siempre son minúsculos.

Monte Carlo (`rare.rs`): 30 000 pasos × 50 277 = 1.5·10⁹ softmax, con un 0.5 % de los `exp` alterados en 1 ulp.
Resultado: **0** probabilidades f32 distintas y **0** CDF enteras distintas.

- Cota al 95 %: menos de 1·10⁻⁴ pasos con alguna probabilidad f32 distinta.
- La estimación analítica de que cambie la CDF entera ronda 10⁻⁸–10⁻⁶ por token.

A escala de producto (1 GB ≈ 2.5·10⁸ tokens BPE) esa probabilidad **no es despreciable**. Es un fallo latente: si
ocurre, corrompe el resto del archivo. El CRC lo detecta, pero los datos se pierden. Hay que cerrarlo antes de
prometer portabilidad entre sistemas operativos por contrato.

---

## 5. Diseño de R-104

### 5.1 Opciones

| Opción | Qué resuelve | Coste | Impacto en BPB / tok/s | Veredicto |
|---|---|---|---|---|
| (a) Trascendentales propias en Rust puro | libm. Para f32 ya está hecho (R-013b). Faltan `exp`/`ln` f64 en codec.rs:502 y cm_ensemble.rs:133-139 | 0.5–1 día: portar `exp`/`log` de musl (tabla de 128 entradas + polinomio, error < 0.52 ulp, solo `+ − ×` y bits) | BPB ±0 (diferencia ≤ 1 ulp f64). 50 K `exp` por token ≈ 0.2 ms frente a ~30 ms por token: < 1 % | **Hacer (P0)** |
| (a′) Variante de (a): CDF en f32 con `portable_expf` y suma en f64 | lo mismo que (a) | 2 h | hay que demostrar que no empeora la cola de 65 K de vocabulario, que es el motivo original del f64 | alternativa si (a) se complica |
| (b) Orden de acumulación canónico, igual en escalar y en AVX2 | tier SIMD/escalar; habilita ARM/NEON | 2–4 días: el escalar emula 8 carriles y el mismo árbol `hsum` en `mat_vec_mul_*_raw`, en `ssm_recurrence_d16` (lo/hi + hsum) y en cualquier kernel futuro. El patrón ya existe en `gemv_q8w` (R-099) | la ruta SIMD no cambia: BPB y tok/s idénticos. El escalar puede incluso acelerar (8 cadenas independientes) | **Hacer (P1)**: necesario para ARM y para CPU sin AVX2 |
| (c) Prohibir el FMA implícito y fijar `target-cpu` | H3 | 0.5 días. FMA implícito ya no hay (rustc no contrae). Añadir un lint de CI basado en `grep` que falle con `mul_add`, `_mm256_fmadd` o `f32/f64::exp/ln/tanh/powf` en `src/domain` y `src/application/codec.rs`, salvo una lista blanca (tests, bench, telemetría). La release de distribución sigue en `x86-64-v3` (ya lo está en Docker) **por portabilidad del binario** (evitar SIGILL), no por la numérica | 0 | **Hacer** (guardarraíl) |
| (d) Contrato numérico en el header (byte PROFILE de v5, bits 4-7) | fallar con un error claro ante un mismatch, en vez de "CRC mismatch" después de decodificar todo | 1 día. Hoy `parse_profile_byte` ya rechaza los bits reservados, así que un decoder viejo falla rápido | 0 | **Hacer (P0)**, junto con (a) |
| (e) CM y mixer en punto fijo | nada que esté roto: el CM solo usa aritmética IEEE básica y ya es determinista | 1–2 semanas, con riesgo de BPB (el mixer SGD es sensible) | posible empeoramiento de BPB | **Descartar**; reconsiderar solo para un target sin FPU IEEE |

Propuesta para el byte PROFILE de v5 (los bits 0-3 siguen siendo la cuantización, como ahora):

```
bits 4-5  NUMERIC_CONTRACT   0 = legacy  (HEAD: tier AVX2, libm f64 en la CDF)
                             1 = R-104   (sin libm, orden canónico, independiente del tier)
                             2-3 reservados -> error "contract N not supported by this build"
bit  6    TIER_SCALAR        solo con contract 0: 1 = comprimido con REDUKT_NO_SIMD (tier escalar)
bit  7    reservado (0)
```

- **Con contract 0**, el decoder compara `TIER_SCALAR` con su propio tier y aborta **antes** de decodificar, con
  "archive requires AVX2 tier" o "scalar tier".
- **Con contract 1**, el tier no importa.
- **Cuantización de RWKV**: también hay que declararla. Hoy el header dice `profile F32` mientras el loader carga 73
  tensores en Q8 (`auto`). Si alguien comprime con `--quant f32` y descomprime con la opción por defecto, el error
  solo aparecería como CRC mismatch. Los códigos de `WeightProfile` deben aplicarse también a RWKV.

### 5.2 Plan por fases

| Fase | Contenido | Criterio de aceptación | Estimación |
|---|---|---|---|
| F0 ✅ | Esta evidencia: 9 celdas W↔L, Kaggle→W, hilos, `target-cpu`, libm | — | hecha |
| F1 | (a) `portable_exp_f64` / `portable_ln_f64` (port de musl), sustituyendo codec.rs:502 y cm_ensemble.rs:133-139. (d) NUMERIC_CONTRACT=1 en el PROFILE. (c) lint de CI | un test de golden hash de las funciones portables (barrido de 2²⁴ como el de `sweep.rs`) da lo mismo en Windows y en Docker; los roundtrips con archivos v4 y v5-legacy siguen intactos | 1.5–2 días; BPB ±0.01 %, tok/s −<1 % |
| F2 | (b) orden canónico en los kernels F32 que tienen versión SIMD (`mat_vec_mul_*`, `ssm_recurrence_d16` y los de R-098 en curso). Cuantización de RWKV en el header | `REDUKT_NO_SIMD=1` produce el mismo sha que AVX2 en los 12 archivos; test unitario por kernel que compruebe SIMD = escalar bit a bit | 2–4 días; BPB 0, tok/s SIMD 0 |
| F3 | Matriz de CI: Windows MSVC, Linux glibc (Docker), Linux musl estático, Kaggle Xeon, GCP N2 (Intel) y N2D (AMD Zen) | **roundtrip cruzado y sha idéntico en los 12 archivos del benchmark** (10 KB; Mamba F32, Mamba Q8 y RWKV) para todas las parejas; golden hashes versionados en `tests/` | 1–2 días + un coste cloud mínimo |
| F4 (opcional) | NEON con el mismo orden de carriles (2×float32x4 = 8 carriles lógicos), para ARM64 | sha ARM = sha x86 en los 12 archivos | 3–5 días |

**Riesgos residuales a vigilar**:

1. El trabajo en curso (R-098/R-100 en codec.rs, model.rs, gemv.rs y tensor.rs) puede introducir kernels nuevos
   con otro orden de acumulación. El lint y el test SIMD = escalar deben bloquearlo.
2. "Debug ≠ Release" (`determinism-audit.md`) sigue sin causa documentada. Conviene investigarlo en F2, porque ni
   `debug_assert` ni los overflow checks deberían cambiar la numérica.
3. Hay que actualizar `determinism-audit.md`: la fila "Windows x86 → Linux x86" pasa de afirmación a
   **verificada** (y Kaggle Xeon → Windows también).

---

## 6. Reproducción

Los scripts están en el scratchpad de la sesión y no se versionan:

- `step.sh`: `cw|cl|dl|dw|rep FILE PROFILE [native|v3]`.
- `sweep/sweep.rs`: barrido de libm. `sweep <dir>` vuelca los resultados y `sweep cmp a b {4|8}` los compara.
- `sweep/rare.rs`: Monte Carlo de la CDF.

En Linux:

```
docker run --rm -v <sweep>:/s -w /s rust:1.98-bookworm sh -c "rustc -O --edition 2021 -o s sweep.rs && ./s lin"
```

Para hacer la sweep reproducible en CI, en F1 conviene versionarla como test o como ejemplo.
