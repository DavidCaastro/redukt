# CPU Architecture Map — Intel Core i5-1235U (Alder Lake-U)

> Mapa bidimensional del hardware donde corre Redukt.
> Referencia para optimizacion de throughput (R-012 a R-017).
> Fecha: 2026-10-28. Corregido con mediciones reales de PROTO-01 a PROTO-08.

---

## 1. Vista General del Die

```
+==============================================================================+
|                        Intel i5-1235U — Alder Lake-U                         |
|                     Intel 7 (10nm ESF) — FCBGA1744                           |
+==============================================================================+
|                                                                              |
|  +------------------+    +------------------+                                |
|  |   P-CORE 0       |    |   P-CORE 1       |                               |
|  |  Golden Cove      |    |  Golden Cove      |                              |
|  |  2 threads (HT)   |    |  2 threads (HT)   |                             |
|  |  max 4.4 GHz      |    |  max 4.4 GHz      |                             |
|  |                    |    |                    |                             |
|  |  L1i: 32KB 8-way  |    |  L1i: 32KB 8-way  |                             |
|  |  L1d: 48KB 12-way |    |  L1d: 48KB 12-way |                             |
|  |  L2: 1.25MB priv  |    |  L2: 1.25MB priv  |                             |
|  +--------+---------+    +--------+---------+                               |
|           |                        |                                         |
+===========|========================|=========================================+
|           v                        v                                         |
|  +-----------------------------------------------------------------------+  |
|  |                     RING BUS INTERCONNECT                              |  |
|  |          ~4.7 GHz (P-only) / ~3.6 GHz (cuando E activos)             |  |
|  +---+-------+-------+-------+-------+-------+-------+-------+------+---+  |
|      |       |       |       |       |       |       |       |      |       |
|      v       v       v       v       v       v       v       v      v       |
|  +-----------------------------------------------------------------------+  |
|  |              L3 INTEL SMART CACHE: 12 MB TOTAL                        |  |
|  |              Non-inclusive, 12+ way, 64B line, ~40-55cy (P) ~60cy (E) |  |
|  +-----------------------------------------------------------------------+  |
|      |                                                       |              |
|      v                                                       v              |
|  +-----------------------------+    +-----------------------------+         |
|  |     E-CORE CLUSTER 0       |    |     E-CORE CLUSTER 1       |         |
|  |        Gracemont            |    |        Gracemont            |        |
|  | +------+ +------+ +------+ |    | +------+ +------+ +------+ |        |
|  | |E-C 0 | |E-C 1 | |E-C 2 | |    | |E-C 4 | |E-C 5 | |E-C 6 | |       |
|  | |1 thrd| |1 thrd| |1 thrd| |    | |1 thrd| |1 thrd| |1 thrd| |       |
|  | |3.3GHz| |3.3GHz| |3.3GHz| |    | |3.3GHz| |3.3GHz| |3.3GHz| |       |
|  | +------+ +------+ +------+ |    | +------+ +------+ +------+ |        |
|  | +------+  L1d: 32KB/core   |    | +------+  L1d: 32KB/core   |        |
|  | |E-C 3 |  L2: 2MB/cluster  |    | |E-C 7 |  L2: 2MB/cluster  |       |
|  | +------+                    |    | +------+                    |        |
|  +-----------------------------+    +-----------------------------+         |
|                                                                              |
+==============================================================================+
|                    MEMORY CONTROLLER (Dual Channel DDR4/DDR5)                |
+==============================+===============================================+
                               |
   +---------------------------+---------------------------+
   |  DIMM A: Crucial 16GB    |  DIMM B: Crucial 16GB    |
   |  DDR4-3200 (rated)       |  DDR4-3200 (rated)       |
   |  Running: 2667 MT/s [!]  |  Running: 2667 MT/s [!]  |
   +---------------------------+---------------------------+
   |  TOTAL: 32 GB DDR4 @ 2667 MT/s                       |
   |  BW DRAM medido: 20.6 GB/s [MEASURED]                 |
   |  (seria ~24.8 GB/s a 3200 con XMP)                   |
   +-------------------------------------------------------+
```

---

## 2. Jerarquia de Cache — Latencia y BW Medidos

```
  Registros (256-bit YMM)  <-- 0 cy
       |
       v
  +--L1d: 48KB (P) / 32KB (E)--+  <-- 5 cy (P) / 3 cy (E)
  | 96 B/cy (P)                 |
  +-------------+---------------+
                |
                v
  +--L2: 1.25MB (P) / 2MB shared (E)--+  <-- 15 cy / ~3.4 ns
  | BW medido 1MB buffer: 63 GB/s     |  [MEASURED]
  +-------------+----------------------+
                |
                v
  +--L3: 12MB shared--+  <-- 40-55 cy / ~12-17 ns
  | BW medido 4MB: 57 GB/s  [MEASURED]
  +--------+-----------+
           |
           v
  +--DRAM: 32GB DDR4 @ 2667--+  <-- ~76 ns
  | BW medido 64MB: 20.6 GB/s |  [MEASURED]
  | Demanda Redukt: ~5.1 GB/s  |  [CALCULATED: ~130MB/tok * 39 tok/s]
  | Utilizacion Redukt: ~25%   |  [CALCULATED: 5.1/20.6]
  +----------------------------+
```

---

## 3. Cuello de Botella Real — Mediciones PROTO-03

```
HIPOTESIS PREVIA (INCORRECTA):
  Head matmul (DRAM): 70-80%   <-- EQUIVOCADO, solo 12%

MEDICION REAL (PROTO-03):
  +================================================================+
  |  TimeMix  (12 capas)     [=============================] 74%   |
  |  ChannelMix (12 capas)   [=====] 14%                          |
  |  Head (65536x768)        [====] 12%                            |
  +================================================================+

  TimeMix por token: ~20ms (apache), ~16ms (facturae) [MEASURED]
  ChannelMix por tok: ~3.5ms                           [MEASURED]
  Head por token:     ~3.0ms                           [MEASURED]

  DENTRO DE TimeMix (por capa, 12 capas):
  - 5 Q8 matmuls 768x768: R, K, V, output_w (SIMD VNNI)
  - 6 F32 LoRA matmuls: w1,w2,a1,a2,g1,g2 (SCALAR!! no SIMD)
  - 1 state matrix update: N*N per head, 12 heads (SCALAR)
  - Element-wise: exp, sigmoid, lerp, multiply (SCALAR)
  - LayerNorm (SCALAR)

  => La mayoria de TimeMix es SCALAR, no SIMD.
  => Esto explica por que P-cores y E-cores rinden IGUAL.
```

---

## 4. P-core vs E-core — Resultado Empirico

```
HIPOTESIS PREVIA: E-core ~3-4x mas lento que P-core
MEDICION REAL: E-core ~8% mas lento que P-core (<< 3-4x)

  Razon: workload dominado por operaciones escalares (74% TimeMix)
  SIMD solo ayuda en ~12% del tiempo (head + Q8 layer matmuls)
  Para operaciones escalares, P y E son mucho mas similares

  Throughput por LP [MEASURED, enwik8 2KB, affinity pinned]:
  LP  0: 34.7 tok/s    LP  6: 36.3 tok/s
  LP  1: 36.4 tok/s    LP  7: 37.6 tok/s  <-- fastest
  LP  2: 36.8 tok/s    LP  8: 37.4 tok/s
  LP  3: 37.1 tok/s    LP  9: 36.6 tok/s
  LP  4: 36.7 tok/s    LP 10: 36.5 tok/s
  LP  5: 36.8 tok/s    LP 11: 37.4 tok/s

  IMPLICACION: Los 10 cores son utiles para R-012 multi-file.
  Scaling esperado: ~10x (no solo 2x como se asumia con P-only).
```

---

## 5. Thermal & Power Profile

```
  Plan de energia: Equilibrado (Balanced)
  PL1: 15W sostenido | PL2: 55W burst (28-56s Tau)

  PROTO-08: Sustained test 330s (5.5 min) [MEASURED]:
  - tok/s: 38.8-39.2 ESTABLE (0% degradacion)
  - Thermal throttling: NO detectado
  - PL1=15W suficiente para single-thread sostenido
  - Razon: single-thread a 15W permite ~3.5-4.0 GHz

  SIMD vs Scalar [MEASURED]:
  - SIMD: 36.8 tok/s
  - Scalar: 32.0 tok/s
  - Delta: +15% (solo, porque SIMD solo afecta ~12% del tiempo)
```

---

## 6. Donde Vive Cada Dato de Redukt

```
+=========================+=========+==========+==========================+
| Dato                    | Tamano  | Reside   | Demanda DRAM             |
+=========================+=========+==========+==========================+
| Vector token (input)    | 768 B   | L1d      | 0                        |
| Acumuladores i32        | 3 KB    | Regs/L1  | 0                        |
| RWKV state (per layer)  | ~24 KB  | L1d      | 0                        |
| LayerNorm params        | ~6 KB   | L1d      | 0                        |
| Mixer + Coder           | ~5 KB   | L1d      | 0                        |
+-------------------------+---------+----------+--------------------------+
| TimeMix weights (Q8)    | 576 KB  | L2       | 0 (cabe en L2 1.25MB)    |
| ChannelMix weights (Q8) | 576 KB  | L2       | 0 (cabe en L2 1.25MB)    |
| LoRA matrices (F32)     | ~96 KB  | L2       | 0                        |
| Ngram hash tables       | ~512 KB | L2       | 0                        |
+-------------------------+---------+----------+--------------------------+
| Head weights (Q8)       | 48 MB   | DRAM     | 48MB * 39 tok/s = 1.87GB/s|
| Layer weights (Q8)*     | ~82 MB  | DRAM     | 82MB * 39 tok/s = 3.2GB/s |
| Embedding table         | 48 MB   | DRAM     | 768B/tok (negligible)    |
| Total modelo Q8         | 130 MB  | DRAM     | ~5.1 GB/s = ~25% of 20.6 |
*Layer weights > L2 (1.25MB) → streamed from DRAM each token
+=========================+=========+==========+==========================+
```

---

## 7. Oportunidades de Optimizacion (Revisadas Post-Profiling)

```
+=====+==============================+============+============+===============+
| Pri | Segmento                     | Bottleneck | Speedup    | Tarea         |
+=====+==============================+============+============+===============+
|  1  | Multi-file 10 cores          | PARALLELISM| ~10x       | R-012         |
|     | TODOS los cores son utiles   |            | (no 2x)    |               |
+-----+------------------------------+------------+------------+---------------+
|  2  | TimeMix scalar ops (74%)     | COMPUTE    | 1.5-2x?    | R-015 (KV)    |
|     | LoRA F32 scalar, state upd.  |            | [ESTIMATE] | + nuevo R-018 |
+-----+------------------------------+------------+------------+---------------+
|  3  | XMP RAM 2667->3200 MT/s      | CONFIG     | +5-10%     | BIOS (manual) |
|     | Gratis, sin cambio de codigo |            |            |               |
+-----+------------------------------+------------+------------+---------------+
|  4  | SVD pruning (reduce matmuls) | COMPUTE    | ~2x?       | R-016         |
|     | Reduce 768->384 en layers    |            | [ESTIMATE] |               |
+-----+------------------------------+------------+------------+---------------+
|  5  | F32 LoRA SIMD (determinista) | COMPUTE    | +10-20%?   | Nuevo R-018   |
|     | w1,w2,a1,a2,g1,g2 con SIMD  |            | [ESTIMATE] |               |
+-----+------------------------------+------------+------------+---------------+

DESCARTADOS/DEPRIORIZADOS:
- Thread affinity P-core: innecesario (<8% diferencia P vs E)
- Ring bus optimization: irrelevante para este workload
- Head optimization: solo 12% del tiempo, ya SIMD-optimizado
- Thermal management: 0% degradacion sostenida
```

---

## 8. Resumen Cuantitativo

```
+=================================+==================+=========================+
| Metrica                         | Valor            | Tag                     |
+=================================+==================+=========================+
| TimeMix % del forward pass      | 73-88%           | [MEASURED: PROTO-03]    |
| ChannelMix % del forward pass   | 13-15%           | [MEASURED: PROTO-03]    |
| Head % del forward pass         | 12%              | [MEASURED: PROTO-03]    |
| P vs E core delta               | <8%              | [MEASURED: PROTO-05]    |
| SIMD vs scalar delta (e2e)      | +15%             | [MEASURED: PROTO-06]    |
| DRAM BW (64MB sequential)       | 20.6 GB/s        | [MEASURED: PROTO-04]    |
| DRAM BW (L3, 4MB)               | 57.2 GB/s        | [MEASURED: PROTO-04]    |
| DRAM BW (L2, 1MB)               | 63.4 GB/s        | [MEASURED: PROTO-04]    |
| Thermal degradation (330s)      | 0%               | [MEASURED: PROTO-08]    |
| Sustained tok/s (330s)          | 38.8             | [MEASURED: PROTO-08]    |
| Cores usables para multi-file   | 10 (ALL)         | [MEASURED: PROTO-05]    |
| Scaling esperado (10 cores)     | ~10x             | [PROJECTION]            |
| RAM underclock penalty          | -17% BW          | [CALCULATED: 2667/3200] |
| Power plan                      | Equilibrado      | [MEASURED: PROTO-01]    |
+=================================+==================+=========================+
```

---

## Fuentes

- Intel ARK i5-1235U: https://ark.intel.com/content/www/us/en/ark/products/226266
- Chips and Cheese: Golden Cove, Gracemont, Alder Lake cache analysis
- WikiChip Fuse: Golden Cove, Gracemont microarch
- HWCooling: Alder Lake microarch analysis
- Measurements: Redukt PROTO-01 to PROTO-08 (this document)
