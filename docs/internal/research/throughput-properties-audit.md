# Throughput por propiedades, no por fórmulas — síntesis 2026-11-09

> R-098 → R-105 (Fase 19) — Auditoría de throughput por propiedades del proceso y viabilidad de mercado.
> Fecha: 2026-11-09. Método: auditoría interna de solo lectura + 4 investigaciones web independientes.
> Estado: INVESTIGACIÓN. La auditoría no cambió código; el mandato de la §6 la convierte en la Fase 19 del ROADMAP.
> Script de cálculo: `scripts/breakeven.py`. Ejecución: `python scripts/breakeven.py`.

---

Pregunta: ¿qué no hemos visto o intentado para el throughput? ¿Tiene sentido formalizar el
proceso por sus propiedades (no por la estructura de las fórmulas) para que el hardware
"se ponga de acuerdo" sin pagar en la fuente del problema (ancho de banda DRAM)?

Método: 1 auditoría interna de solo lectura (ROADMAP Fases 8-18, 10 docs de research, hot
path en código) + 4 investigaciones web independientes (decodificación, inferencia
"consultiva", hardware, economía). Sin cambios en el repo.

---

## 0. Tres premisas del proyecto que resultaron falsas

| Premisa escrita | Realidad verificada | Fuente |
|---|---|---|
| "Quantization: auto-selected Q8" para Mamba | Mamba corre **F32** en el hot path: `Tensor { data: Vec<f32> }`, `mat_vec_mul_into` f32 AVX2 sin FMA. Q8Tensor existe solo para RWKV. R-064 (Q8 PTQ Mamba) dio +14.5/+16.6% BPB y 31-34 tok/s "sin mejora" y fue revertido | tensor.rs:8-11, model.rs:208-243, rwkv7/model.rs:78-108, ROADMAP L1728-1744 |
| "21.3 GB/s peak DRAM" | El portátil tiene **2 DIMM (A y B) a 2667 MT/s, 64 bits cada uno → dual channel = 42.7 GB/s**. Medidas en portátiles i5-1235U: ~44 GB/s (AIDA64). 34 tok/s × 516 MB = 17.5 GB/s = **41% del bus** | Win32_PhysicalMemory; notebookcheck VAIO FE14 |
| "DRAM-bound, 34 tok/s es el techo" | in_proj y out_proj (**70% de los bytes/token**) corren en **un solo hilo**; solo la LM head (50280 filas) usa ≤4 hilos. Un P-core Golden Cove tope en ~29 GB/s. El cuello es ancho de banda **por core**, no DRAM | tensor.rs:1477-1515, chipsandcheese |

Consecuencia: los 34 tok/s no son un techo físico. Son F32 en un hilo.

Otros hechos del inventario:
- `speculative.rs` y `parallel_scan.rs` (R-052/R-053) **fueron borrados** en el commit 1dc6402 como código muerto. Nunca se integraron ni se midieron. throughput-paradigm-shift.md L370-374 y ROADMAP L2408 afirman que existen: desactualizados (anotados con nota el 2026-11-09).
- Batch mode (R-012) = N hilos independientes, cada uno con su propio GEMV completo. Speedup 1.36× (2 jobs), 1.64× (3): compiten por el mismo bus. **Lockstep (compartir cada fetch de pesos entre streams) no aparece en ningún documento ni código.**
- Entropy gate R-077: entropía del CM sobre tokens BPE = 15.49-15.62 bits (log2 50280 = 15.62). El CM es casi uniforme a nivel token. Threshold 15.5 → 31.5% skip → +50.5% BPB. Nunca probado en apache/XML.
- Q5/Q6 weight-only: no aparece. Compresión lossless de pesos: no aparece. Shortlist/MIPS/hierarchical head: no aparece. Gating decidido por el encoder con side-info en el stream: no aparece.
- Determinismo cross-máquina: Kaggle 0.9633 vs local 0.9609 → bitstreams distintos; roundtrip entre máquinas distintas **no probado**. Riesgo comercial abierto (determinism-audit.md L157-165).

---

## 1. Formalización por propiedades

El pipeline por token: emb → 24 × {in_proj (GEMV), conv1d, scan SSM, out_proj (GEMV)} → LM head (GEMV) → softmax → CM mix → coder.

| Propiedad | Qué permite | Técnica | Estado |
|---|---|---|---|
| P1. Linealidad del estado: h_t = A_t h_{t-1} + B_t x_t | Scan asociativo; rollback gratis (estado del prefijo aceptado); superposición | Parallel scan (encoder); speculative verify; rollback | Planificado (R-093); speculative borrado |
| P2. Pesos independientes del dato | Un fetch de DRAM sirve a N streams a la vez: GEMV → GEMM | **Lockstep multi-stream** | **No visto** |
| P3. Determinismo encoder = decoder | Cualquier atajo determinista es admisible aunque sea aproximado: solo arriesga BPB, nunca corrección | Q8/Q6, shortlist head, draft models, gate | Q8 mal medido; resto no visto |
| P4. Asimetría de información: el encoder conoce el futuro | El encoder compra velocidad para el decoder transmitiendo decisiones | Side-info flags + catch-up por parallel scan | **No visto**; solo rentable con vocab=256 (ver §3) |
| P5. Normalización local del coder | CDF jerárquica → normalizador barato | Hierarchical/adaptive softmax destilado | No visto; exige entrenar la cabeza |
| P6. Redundancia estadística de los pesos | Cómputo ocioso a cambio de menos bytes | Pesos ANS en DRAM (DFloat11, ZipNN) | No visto; **≈0% neto en CPU** (decodificador 1.3-2.5 GB/s/core < DRAM) |
| P7. Separabilidad por capas (weight-stationary) | En forma paralela, cada capa (3.8 MB Q8) cabe en L2 → DRAM O(modelo) por chunk | Parallel scan encoder; lockstep decoder | R-093 planificado |

Tu "consultor del valor en vez de buscador" corresponde a P2 (consultar un fetch ya hecho),
P3 (consultar una aproximación determinista) y P4 (consultar decisiones precomputadas por el
encoder). No es una idea estúpida: P2 es la palanca más grande que no se ha tocado.

---

## 2. Palancas sin tocar el modelo (todas con BPE 50K, hoy)

Multiplicativas entre sí. Riesgo BPB: cero salvo donde se indica.

### L1. Q8 bien hecho (×3-4, BPB ≈0)
El +16% BPB de R-064 es anómalo. Evidencia externa: MambaQuant W8A8 Mamba-130M pierde 0.8 puntos de accuracy media (44.7→43.9); Quamba W8A8 pierde 22% de perplejidad **porque cuantiza activaciones**. Weight-only Q8 por fila, con activaciones F32, debe ser casi lossless. Y "31-34 tok/s sin mejora" indica que la implementación no redujo bytes leídos (dequant a buffer F32 o kernel que seguía leyendo F32).
Diseño: pesos int8 por fila (escala f32 por fila) consumidos en registro (vpmovsxbw/vpmaddubsw o conversión a f32 en registro), activaciones F32; mantener F32 en A_log, dt_proj bias, conv1d, norms; embedding/head Q8 (Quamba2: cabeza y embedding son las más sensibles por debajo de Q8). Bytes/token: 516 MB → ~130 MB.
Validación: BPB enwik8 10KB + 2 archivos, Δ < 2.5% como RWKV. Si el delta vuelve a ser +15%, el bug está en qué se cuantiza, no en Q8.

### L2. Multihilo en in_proj/out_proj (×1.5-2, BPB 0)
70% de los bytes en un hilo. Reparto de filas entre 2P+8E con balanceo P:E ≈ 3:1 (Intel arXiv 2411.19542: +19% GEMV INT4 con balanceo dinámico; >90% del ancho MLC alcanzado). Hilos físicos, pinning, evitar que un E-core baje el ring clock innecesariamente. Techo: ~38-40 GB/s → F32 ~75 tok/s; con L1 ~250-300 tok/s.
Nota: huge pages 0-10%, NTA prefetch ±0 (ya medido R-065). No perder tiempo ahí.

### L3. Lockstep multi-stream: GEMV → GEMM (×3-6 agregado, BPB 0)
N archivos independientes avanzan un token cada uno en el mismo paso; cada tile de pesos se carga de DRAM una vez y se multiplica por N vectores de activación desde registros. Intensidad aritmética ×N. Datos CPU: llama-batched-bench 3.8B Q4: b=1 25.7 → b=8 86.3 → b=32 147.7 t/s agregados (5.8×), meseta en 64; Qwen3-0.6B Q8 GEMM vs GEMV 6-14×. Cruce memory→compute-bound en DDR4 portátil con Q8: B ≈ 12-24.
Aplica a compresión Y descompresión (cada stream lleva su estado SSM de ~48 KB, su CM, su coder). Es exactamente "poner de acuerdo" DRAM (stream único de pesos) y SIMD (reutilización ×N).
Límite: un archivo grande solo se beneficia si se trocea en chunks independientes (BOA Constrictor, +1-3% BPB) o con el checkpoint de estado de R-063 para arranque caliente.
Encaja con el producto batch/archivo: siempre hay N archivos.

### L4. Q6 weight-only (×1.3 adicional, BPB +1-3% estimado, dentro del +5%)
Hueco real no medido. Transformers 0.5B: Q6_K +0.05% en bits, Q5_K_M +0.3%, Q4_0 +2.7%. Mamba es 4-5× más sensible (Q4 +12.7% en RWKV) → Q6 estimado +1-3%, Q5 arriesgado. Nunca por debajo de Q8 en embedding/head. Solo tiene sentido tras L1.

### Proyección CPU-only, Mamba-130M, i5-1235U
| Escenario | tok/s por stream | agregado (10 cores) | B/s agregado |
|---|---|---|---|
| Hoy (F32, 1 hilo en 70%) | 34 | 34 | ~130 |
| L2 | ~70 | 70 | ~270 |
| L1+L2 | ~250-300 | ~280 | ~1.1 KB/s |
| L1+L2+L3 (B≈16, compute-bound ~200 GFLOPS AVX2 sostenidos / 0.26 GFLOP/token) | ~250 | **~700-1,500** | **3-6 KB/s** |
| +L4 | — | +30% hasta el cruce compute-bound | — |

Entre 20× y 40× sobre hoy, con el mismo modelo y el mismo bus.

---

## 3. Lo que NO funciona a nivel BPE y por qué (cierra debates)

- **Cualquier "consultor" que sustituya al backbone en un token** (gate, draft sin verificar, side-info del encoder): a BPE 50K el CM está a 15.5 bits de entropía. Saltarse el backbone cuesta ~12 bits por token frente a ~2-4 con él. Ningún criterio de skip, ni del decoder ni del encoder, puede compensarlo. R-056 en apache: +81%. Esto solo cambia con vocab=256 (MambaByte), donde el CM sí es informativo. P4 (side-info) queda **aparcada hasta MambaByte**, pero ahí sí es nueva: el encoder decide con el resultado real (¿ayudó el backbone en este byte?) y transmite flags RLE; el decoder hace catch-up del estado por parallel scan en bloque (P1+P7). Coste: entropía de los flags (0.03-0.07 BPB estimado).
- **Speculative decoding a BPE**: aceptación 5-15% (CM casi uniforme). En SSM el verify de 6 tokens cuesta 1.2-2× la llamada (Snakes & Ladders, STree) y los speedups publicados son 1.5-2.6× con drafters n-gram sobre modelos de 1.3B+. Para 130M el overhead relativo es mayor. **Rehacer speculative.rs solo a nivel byte**, con el CM/match model como draft (coste cero).
- **LUT (T-MAC, MADDNESS, bitnet.cpp)**: speedup lineal con la reducción de bits; a 8 bits no hay ganancia. Requieren ≤4 bits o ternario. Descartado para el backbone.
- **Pesos lossless en DRAM**: entropía real de int8 4-5 bits (−40% teórico), pero el único experimento CPU con decodificación fusionada en el GEMV da 1.006-1.02× e2e. Descartado en CPU.
- **Probabilidades byte exactas desde el modelo BPE** (Phan ICLR'25, Vieira ICML'25): da −16/−19% de bpb pero cuesta K≈8 forwards por byte → ~30× más llamadas. Herramienta de ratio, no de throughput.
- **iGPU Iris Xe**: para modelos <1.5B la CPU gana 2× (OpenVINO, Vulkan). Descartado.
- **Hierarchical softmax destilado / shortlist MIPS**: ahorra el 30% de bytes de la LM head, pero exige entrenar la cabeza o un modelo de escape para el 5% fuera del shortlist. Tras L1-L3 la LM head deja de ser el problema (lockstep la amortiza igual). Prioridad baja.

---

## 4. La parte comercial: el throughput no es solo un problema técnico

Punto de equilibrio (spot $0.022/vCPU-h; R* = bytes/s por vCPU necesarios para que comprimir 1 GB cueste menos que el ahorro de almacenamiento acumulado):

| Dato (b = fracción que deja brotli, S = ahorro adicional Redukt) | S3 Standard, 7 años | Deep Archive, 7 años |
|---|---|---|
| enwik8 (b=0.29, S=59%) | 18 KB/s | 424 KB/s |
| Facturae (b=0.11, S=56%) | 51 KB/s | 1.2 MB/s |
| apache (b=0.058, S=54%) | 101 KB/s | 2.3 MB/s |

Hoy: 130 B/s → **$47 por GB de entrada (raw)** comprimido (spot), $14.5 en Hetzner dedicado. Tras L1-L4 (3-6 KB/s agregado por máquina de 10 cores, ~0.5 KB/s por vCPU): ~$12/GB. Sigue 30-200× por encima del umbral.

Hechos que reordenan la tesis:
1. Cuanto mejor comprime brotli (logs, XML), **menos bytes absolutos quedan por ahorrar**: el archivo profundo ($0.001/GB-mes) es el peor caso de negocio, no el mejor.
2. Todo el mercado español de factura electrónica B2B (~1,200 M facturas × 15 KB ≈ 18 TB/año raw, ~2 TB tras brotli) ahorraría ≈1.1 TB/cohorte = **$300/año en S3 Standard**. El archivo de facturas no es un nicho de ahorro de storage; su valor sería cumplimiento/inmutabilidad.
3. No existe precedente comercial de un compresor <1 MB/s con clientes de pago. PetaGene cobra % del ahorro a 290 MB/s; AtomBeam monetiza ancho de banda satelital, no storage; ts_zip y Nacrith no son productos.
4. Egress: una descarga equivale a 3.9 meses de S3 Standard, pero el CM es simétrico: el cliente descomprime a la misma velocidad. Sin decode rápido no hay caso CDN.
5. Hardware propio con memoria unificada (Mac Studio M3 Ultra 819 GB/s, ~$0.05/PB de tráfico de pesos) es ~10× más barato por PB que cualquier CPU cloud y comparable a T4 spot. Pero incluso ahí Mamba-130M queda en ~$1/GB.

Dato externo decisivo: **L3TC** (AAAI 2025): RWKV de **3.2M parámetros** + vocab 16K logra ≈1.10 bpb en enwik8 (Redukt 0.96) a 27 KB/s en CPU de iPhone con batch 1, y 633 KB/s batcheado en ANE. Un modelo 40× más pequeño que Mamba-130M pierde 15% de ratio y gana 200× en throughput. La "receta" de Redukt (CM + mixer adaptativo + preprocesado binario + routing) es independiente del tamaño del backbone. token-reduction §5.6 marcó la destilación como NO-GO por una *proyección* (+5-15% BPB); L3TC es una *medida*.

---

## 5. Recomendación ordenada

**Ahora (BPE, sin entrenar nada, sin riesgo BPB):**
1. L1 Q8 weight-only correcto en Mamba. Primero diagnosticar R-064: qué se cuantizó y si el kernel leía int8. Gate: Δ BPB < 2.5%, bytes/token ~130 MB, tok/s > 100.
2. L2 multihilo en in_proj/out_proj con balanceo P/E.
3. L3 lockstep multi-stream (compress y decompress) con N configurable; medir curva tok/s agregados vs N (esperado meseta en 16-32).
4. L4 Q6 solo si L1 pasa el gate.
Resultado esperado: 20-40× agregado. Corregir CLAUDE.md (Q8, 42.7 GB/s) y los docs que citan speculative.rs.

**Tras MambaByte (vocab 256):**
5. Rehacer speculative con CM/match model como draft, verify en forma paralela (P1).
6. Side-info del encoder + catch-up por parallel scan (P4+P7): nuevo, solo viable aquí.

**Decisión de producto (no técnica):**
7. Evaluar un tier "rápido": backbone de 3-30M entrenado (clase L3TC) + la receta CM de Redukt, con Mamba-130M como tier "máximo ratio". Es la única vía CPU-only que cruza el umbral de 18-100 KB/s por vCPU. Requiere revisar el NO-GO de destilación con una medida, no una proyección.
8. Cerrar el determinismo cross-máquina antes de cualquier cliente externo (orden de acumulación canónico o enteros).

---

## 6. Mandato de producto (2026-11-09)

El objetivo declarado es **hacer el producto viable y competitivo en el mercado real**. Eso deja de ser
una opción de investigación y se convierte en mandato de ejecución. Implica cuatro decisiones:

**(a) Ejecutar las palancas L1-L4 ahora.** Operan sobre el backbone actual (Mamba-130M, BPE 50K), no requieren
entrenar nada. Riesgo BPB: cero (L2, L3), ≈0 con gate Δ < +2.5% (L1) y +1-3% estimado (L4), siempre dentro
del presupuesto de +5%. En el ROADMAP:
L1 = R-098 (Q8 weight-only Mamba), L2 = R-099 (multihilo in_proj/out_proj), L3 = R-100 (lockstep multi-stream),
L4 = R-101 (Q6 weight-only, condicionado a R-098). Resultado proyectado: 20-40× agregado sobre hoy.

**(b) Cerrar el determinismo cross-máquina como bloqueante de producto (R-104, P0).** Kaggle 0.9633 vs local
0.9609 implica bitstreams distintos; el roundtrip comprimir en una máquina y descomprimir en otra **no está
probado**. Ningún cliente externo antes de cerrarlo (orden de acumulación canónico o aritmética entera).
Las palancas L2 y L3 se diseñan con partición por filas para conservar el orden de acumulación por fila y
mantener la salida bit-idéntica al baseline.

**(c) Decidir el tier rápido con una medida, no con una proyección (R-105).** Incluso con L1-L4 (~$12/GB raw)
el sistema queda 30-200× por encima del umbral económico del §4. La única vía CPU-only que lo cruza es un
backbone de 3-30M parámetros (clase L3TC) con la receta CM de Redukt. El NO-GO de destilación de
token-reduction-strategies §5.6 se basó en una proyección (+5-15% BPB); L3TC es una medida. La decisión se toma
entrenando el backbone pequeño y midiendo BPB y KB/s sobre el mismo set de 12 archivos.

**(d) Desplegar en cloud (GCP).** Cloud Run para el comando `api`, Cloud Batch para compresión masiva
(`--input-dir`) y GCE con T4/L4 para el plugin CUDA (R-090). Restricción: solo x86 con AVX2, por determinismo (ver R-104).
Hitos: R-102 (CPU) y R-103 (CUDA).

Plan de ejecución: Fase 19 del ROADMAP (`docs/ROADMAP.md`, sección
"Fase 19: CPU Limit & Market Viability (R-098 → R-105)").

---

## Fuentes principales
- Intel GEMV híbrido: https://arxiv.org/html/2411.19542 · Alder Lake BW: https://chipsandcheese.com/p/popping-the-hood-on-golden-cove · i5-1235U AIDA64: notebookcheck VAIO FE14
- Batching CPU: https://github.com/ggml-org/llama.cpp/discussions/18030 · https://github.com/kekzl/malkin
- Quamba https://arxiv.org/abs/2410.13229 · MambaQuant https://arxiv.org/abs/2501.13484 · Quamba2 https://arxiv.org/abs/2503.22879 · Qwen2-0.5B GGUF ppl table
- T-MAC https://arxiv.org/abs/2407.00088 · DFloat11 https://arxiv.org/abs/2504.11651 · ZipNN https://arxiv.org/abs/2411.05239 · bf16-exponent-gemv (GitHub cvs0)
- L3TC https://arxiv.org/abs/2412.16642 · FineZip https://arxiv.org/abs/2409.17141 · Invocation Bottleneck https://arxiv.org/abs/2611.05871 · ts_zip https://bellard.org/ts_zip/ · nncp LTCB
- Phan https://arxiv.org/abs/2410.09303 · Vieira https://arxiv.org/abs/2412.03719 · BLT https://arxiv.org/abs/2412.09871 · HAT https://arxiv.org/abs/2603.15953
- Snakes & Ladders (PMLR v262) · STree https://arxiv.org/abs/2505.14969 · Mamba Drafters https://arxiv.org/abs/2506.01206 · Spec-Bench
- Precios: instances.vantage.sh, prices.azure.com, developers.cloudflare.com/r2/pricing · PetaGene (TechCrunch, NetApp) · AEAT calendario factura electrónica · SERES estudio 2024
- Script de breakeven: `scripts/breakeven.py`

## 7. Resultados (2026-11-10)

Medido en i5-1235U (portátil térmicamente limitado), `maxperf`, 10 KB. Commits 7ba6373 (R-098/R-099) y 365ef68 (R-100).

| Escenario | Proyectado (2026-11-09) | Medido | Nota |
|---|---|---|---|
| Línea base F32 (1 hilo en in/out_proj) | 34 tok/s | 34 tok/s | 516.5 MB/token |
| + R-099 multihilo F32 | ~70 tok/s | 53.6 tok/s (partiendo de 39.0 en la medida A/B) | `.azth` byte-idéntico (sha 384c3042) |
| + R-098 Q8 weight-only bloque-64 | ~250-300 tok/s | 127 tok/s en frío, 111 en caliente | 153.6 MB/token; BPB +0.57% / +1.21% / +1.11% |
| + R-100 lockstep Q8, N=32 | ~700-1,500 tok/s agregados | 315 tok/s agregados | N=1/2/4/8/16/32 = 111/134/236/266/292/315 |
| + R-100 lockstep F32, N=32 | — | 322.1 tok/s agregados | 57.7/96.5/154.6/206.5/262.6/322.1 |

Conclusiones:
- **~9x agregado** respecto a los 34 tok/s de partida (315/34 = 9.3x) en un portátil térmicamente limitado; muy por debajo
  de la proyección de 20-40x del §6(a).
- La premisa refutada se confirma en lo cualitativo (los 34 tok/s no eran un techo de DRAM), pero el nuevo límite es el
  **cómputo**: compute-bound desde N~8 con 130-190 GFLOPS sin FMA, antes del cruce N=12-24 previsto. A N>=16 F32 iguala a Q8
  en throughput, así que el ancho de banda ya no es la palanca.
- Escala por fila falló (+7% BPB); el diseño final es bloque-64 con activaciones F32. R-064 falló por W8A8 y escala por fila
  en la cabeza atada.
- **Siguiente palanca**: FMA/kernels de cómputo (nuevo perfil numérico, coordinado con R-104) y GPU batch (R-103,
  compilado pero sin ejecutar en GPU real). R-101 (Q6) aporta poco mientras el límite sea cómputo.
- **Coste** (corregido 2026-11-10): la métrica correcta es por vCPU, no por máquina. El lockstep satura 12 hilos para
  dar ~1,180 B/s (3.8 B/token), es decir ~100 B/s por vCPU frente a ~65-130 B/s por vCPU de la línea base (que usaba
  1-2 cores): la eficiencia mejora ~1.5-2x, no 9x. Con C = P * 1e9 / (3600 * R): 12 vCPU spot ($0.26/h) ≈ **$50-60/GB
  raw**; Hetzner dedicado ≈ $19/GB; GPU T4 N=128 (4.2 KB/s, $0.13-0.25/h) ≈ **$9-17/GB**. La estimación previa de
  "≈$5/GB" asumía 1 vCPU y queda retirada. Umbral: $0.06-0.33/GB (CLAUDE.md "Commercial Target").
- El umbral económico (18-100 KB/s por vCPU) sigue ~15-80x por encima del throughput actual (1.2 KB/s por máquina): R-105
  (tier rápido) sigue siendo necesario.
- **Re-medición con la máquina en reposo** (`data/benchmark/results/lockstep-throughput-2026-11-10.csv`): Q8 N=8 270.8,
  N=16 286.9, N=32 309.9 tok/s agregados; F32 N=16 259.6. Confirma la ronda previa (266/292/315; F32 262.6). 12 archivos =
  75,398 tokens; 32 archivos = 198,844 tokens. La verificación independiente confirmó los gates BPB exactos y la identidad
  lote vs single (N=1,2,3, ambas direcciones).
- Determinismo (R-104): roundtrip Windows<->Linux<->Kaggle idéntico 9/9; F1 ya sustituyó libm por `fmath` (commit 51dafd9);
  pendiente SIMD vs escalar (F2).
