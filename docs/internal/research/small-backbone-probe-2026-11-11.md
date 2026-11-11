# ¿Existe un backbone más pequeño que conserve el ratio? — sonda de candidatos preentrenados (2026-11-11)

> R-105 (Fase 19), paso 1 — medida, no proyección.
> Fecha: 2026-11-11. Plataforma: Kaggle T4 + 4 vCPU. Kernel `small-backbone-probe` v1 (modelos HF) y v3 (L3TC).
> Script: `notebooks/small-backbone-probe.py`. Datos crudos: `data/benchmark/results/small-backbone-probe/`.
> Estado: MEDIDO. Decisión al final (§7).

## 1. Pregunta y respuesta corta

**Pregunta**: ¿podemos sustituir Mamba-130M por un backbone más pequeño ya entrenado sin perder calidad, y con
eso acercarnos al umbral económico ($0.06-0.33/GB)?

**Respuesta**: no con nada de lo que existe publicado.

1. **Modelos generales pequeños (Pythia 14M-70M, mismo tokenizador y mismo corpus que Mamba-130M)**: el ratio se
   degrada de forma continua con el tamaño (+27% bpb a un tercio del coste, +54% a un 6% del coste en texto) y en
   XML y logs dejan de ganar a brotli/xz.
2. **Especialistas diminutos (L3TC 200K-12M, entrenados con enwik8)**: igualan o superan a Mamba-130M **solo sobre
   su propio fichero de entrenamiento** (1.04-1.10 bpb el de 3.2M en enwik8). Sobre Wikipedia que no han visto
   pierden un 46% (1.46 vs 1.00), sobre prosa inglesa apenas ganan a bzip2 (1.90 vs 2.19) y sobre XML y logs son
   inservibles (4.3-7.3 bpb, 12-25x peor que brotli).
3. **A igual tamaño sí hay un backbone mejor**: SmolLM2-135M mejora a Mamba-130M un 5-11% en texto y un 19-21% en
   XML, y pierde un 3-8% en logs. No abarata nada (mismo coste por token), así que no toca el problema económico.
4. **Ni siquiera el más barato cruza el umbral**: proyectando el coste por byte sobre la eficiencia medida del
   runtime actual, L3TC-200K (72x más barato que Mamba) quedaría en ~$0.77/GB frente a $0.18/GB de valor del
   ahorro en texto Wikipedia no visto: sigue 4x por encima, y no sirve para ningún otro tipo de dato.

Lo que la medida sí deja abierto: un especialista **entrenado por nosotros sobre el dominio del cliente** (logs,
factura electrónica) con receta CM. Eso no existe publicado y hay que entrenarlo (§7).

## 2. Qué se había hecho antes (para no repetirlo)

| Intento | Qué era | Resultado |
|---|---|---|
| R-088 (2026-11-09) | MambaByte preentrenado de 353M (más grande, a nivel de byte) | 1.3156 bpb en enwik8, 4.8 tok/s: peor y más lento (dominio arXiv) |
| R-084 (2026-11-07/09) | Entrenar MambaByte propio de 42M | Nunca entrenó: la TPU de Kaggle no arrancó (los 3 logs se cortan tras la configuración) |
| Fase 10 | Poda, Q4 y sparsity sobre Mamba-130M | No viable; es encoger el mismo modelo, no probar uno pequeño |
| `token-reduction-strategies.md` §5.6 | Destilación | NO-GO por una proyección (+5-15% bpb), sin medida |

Hasta hoy no había **ninguna medida** de un backbone más pequeño que Mamba-130M.

## 3. Candidatos

Búsqueda (2026-11-11): modelos causales preentrenados por debajo de ~130M, con pesos públicos.

| Candidato | Tamaño | Tokenizador | Entrenado con | Licencia | Papel en la sonda |
|---|---|---|---|---|---|
| `state-spaces/mamba-130m-hf` | 129M | GPT-NeoX BPE 50K | The Pile | Apache-2.0 | referencia (backbone del producto) |
| `EleutherAI/pythia-14m/31m/70m/160m` | 14-162M | GPT-NeoX BPE 50K (el mismo) | The Pile (el mismo) | Apache-2.0 | curva tamaño-calidad controlada |
| `distilgpt2`, `gpt2` | 82M, 124M | GPT-2 BPE 50K | WebText | Apache-2.0 / MIT | control (otra receta) |
| `HuggingFaceTB/SmolLM2-135M` | 135M | BPE 49K | mezcla curada 2024 | Apache-2.0 | control (mismo tamaño, receta moderna) |
| L3TC 200K / 800K / 3.2M / 12M | 3.3-19M totales | SentencePiece BPE 16K | **enwik8** | **sin licencia en el repo** | especialista diminuto (solo referencia) |
| `pszemraj/nGPT-enwiki8` | ~25M, vocab 256 | bytes | enwik8 | MIT | nivel byte — **no evaluado** (ver §6) |
| `ijktech/ByteGPT-small` | 86M, bytes | bytes | no declarado | CC-BY-NC-4.0 | descartado: licencia no comercial |
| RWKV-7 < 0.1B | — | — | — | — | no existe: el más pequeño publicado es 0.1B |

Notas de procedencia:
- **L3TC** (`alipay/L3TC-...`, AAAI 2025): el repositorio solo trae un `LEGAL.md` sobre el idioma de los comentarios,
  sin licencia de código ni de pesos. No es utilizable en el producto; se mide únicamente como referencia de la
  relación tamaño-ratio. Los checkpoints están en Google Drive (989 MB, cuatro `.pth`) **sin el tokenizador**; se
  reconstruyó con los parámetros de su `scripts/preprocessor.py` (BPE 16384, cobertura 0.999, entrenado sobre
  enwik8 crudo, `pad=0 unk=1 bos=2`, símbolos `\n` y `\t`). La reconstrucción se valida sola: los cuatro
  checkpoints cargan con todas las claves y dan en enwik8 las cifras que el artículo publica (~1.10 bpb el de 3.2M).
- Los nombres "200K / 3.2M / 12M" de L3TC no cuentan los embeddings: con ellos son 3.3M / 10.4M / 19.3M parámetros.

## 4. Protocolo

- **Métrica**: entropía cruzada del backbone solo, en bits por byte de entrada (`sum -log2 p(token) / bytes`).
  Sin CM, sin mixer, sin overhead del coder. Mide el modelo, no el producto.
- **Reinicio**: el flujo de tokens se corta en trozos independientes de C tokens (BOS + C-1), que es lo que hace
  el producto con el reinicio R-108 (C = 4096 en Mamba). Los transformers se evalúan en el mayor C que admiten
  (2048 Pythia, 1024 GPT-2) y la referencia además en C = 2048. L3TC usa sus segmentos de 2048 caracteres y sus
  tokens fuera de vocabulario se cobran como bytes crudos.
- **Datos**: 8 ventanas de 10 KB de enwik8; primeros 500 KB de enwik8; 500 KB de enwik9 en el desplazamiento
  600 MB (fuera de enwik8: no visto por los modelos entrenados con enwik8); primeros 500 KB de dickens (Silesia);
  los cinco ficheros de industria enteros. Compresores clásicos sobre los mismos bytes.
- **Coste**: (a) analítico — parámetros tocados por token (todo menos la consulta del embedding de entrada),
  x2 = MFLOP/token, dividido por bytes/token; (b) medido — PyTorch en CPU, **un hilo**, decodificación con caché.
  (b) es PyTorch, no el runtime de Redukt: solo vale la relación entre modelos.
- **Verificación del script**: antes de lanzarlo, los tres evaluadores se comprobaron en local contra un cálculo
  de referencia independiente con modelos aleatorios pequeños (bits idénticos a 1e-4).

**Límites de la medida** (importan para leer las tablas):
1. Es solo backbone. En el producto el CM aporta mucho en datos repetitivos: UBL da 0.414 bpb con el backbone solo
   y 0.239 en el `.azth` real. Un backbone débil con CM puede quedar mejor de lo que sugiere esta tabla; no está
   medido.
2. enwik8 es dato de entrenamiento de L3TC (y Wikipedia está en The Pile). La fila honesta para L3TC es
   "enwik9 tail", y aun así es Wikipedia.
3. Los ficheros de industria de XML son sintéticos (generados por `scripts/prepare-benchmark-data.py`).
4. El coste proyectado en $/GB supone que un modelo pequeño corre con la misma eficiencia por FLOP que
   Mamba-130M en nuestro runtime. No está medido; con modelos diminutos el CM, la CDF y el coder pasan a dominar.

## 5. Resultados

### 5.1 bpb por conjunto (mejor C de cada modelo; negrita = mejor que el mejor compresor clásico)

| Model | Active params | Cost/byte vs Mamba | enwik8 8x10K | enwik8 500K | enwik9 tail | dickens | facturae | UBL | apache | syslog | openssh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mamba-130m | 129.1M | 1.000x | **1.028** | **0.938** | **0.998** | **1.018** | **0.231** | 0.414 | **0.175** | **0.246** | **0.214** |
| smollm2-135m | 134.5M | 1.103x | **0.919** | **0.834** | **0.915** | **0.964** | **0.183** | 0.336 | **0.182** | **0.253** | **0.231** |
| pythia-160m | 123.7M | 0.958x | **1.149** | **1.026** | **1.094** | **1.112** | **0.292** | 0.599 | **0.236** | **0.308** | **0.302** |
| gpt2 | 124.4M | 1.014x | **1.287** | **1.173** | **1.270** | **1.766** | 0.865 | 1.136 | 0.769 | 0.702 | 0.747 |
| distilgpt2 | 81.9M | 0.667x | **1.437** | **1.311** | **1.406** | **1.953** | 1.175 | 1.375 | 0.734 | 0.800 | 0.848 |
| pythia-70m | 44.7M | 0.346x | **1.313** | **1.193** | **1.269** | **1.292** | **0.369** | 0.740 | **0.263** | **0.362** | 0.402 |
| pythia-31m | 17.6M | 0.136x | **1.420** | **1.283** | **1.368** | **1.377** | 0.470 | 0.832 | **0.289** | 0.412 | 0.441 |
| pythia-14m | 7.6M | 0.059x | **1.593** | **1.448** | **1.541** | **1.551** | 0.629 | 0.985 | 0.356 | 0.502 | 0.529 |
| l3tc-12m | 13.0M | 0.104x | **0.902** | **0.850** | **1.467** | **1.909** | 5.164 | 6.413 | 4.791 | 5.107 | 5.345 |
| l3tc-3.2m | 6.2M | 0.049x | **1.102** | **1.039** | **1.455** | **1.900** | 5.146 | 6.210 | 4.314 | 4.573 | 4.912 |
| l3tc-800k | 3.4M | 0.027x | **1.233** | **1.176** | **1.587** | **2.005** | 5.396 | 6.544 | 4.676 | 4.683 | 4.700 |
| l3tc-200k | 1.7M | 0.014x | **1.378** | **1.318** | **1.628** | **2.076** | 6.562 | 7.281 | 4.396 | 4.617 | 4.608 |
| best classical | — | — | 2.735 | 2.253 | 2.361 | 2.189 | 0.408 | 0.252 | 0.313 | 0.370 | 0.346 |

Lectura:
- **Pythia (control limpio de tamaño)**: de 160M a 14M el bpb en texto sube 1.03 -> 1.19 -> 1.28 -> 1.45
  (enwik8 500K). Cada reducción de ~2.5x en coste cuesta un 7-16% de bpb. A igual tamaño Pythia-160M ya es un
  9-12% peor que Mamba-130M: la arquitectura y la receta pesan tanto como el tamaño.
- **En industria los pequeños pierden el margen**: Pythia-31M y 14M pierden contra el clásico en facturae, syslog y
  openssh; todos los Pythia pierden en UBL (igual que Mamba sin CM).
- **L3TC**: excelente sobre enwik8 (su fichero de entrenamiento; el de 12M baja a 0.85, memorización), mediocre
  sobre Wikipedia no vista (1.46-1.63; el de 12M no mejora al de 3.2M: sobreajuste) y catastrófico fuera de
  dominio. Un especialista de este tamaño **no generaliza**.
- **SmolLM2-135M**: mejor que Mamba en texto y XML, algo peor en los tres logs.

### 5.2 Coste por token y por byte

| Model | Total params | Active params/token | MFLOP/token | B/token | MFLOP/byte | Cost/byte vs Mamba | PyTorch CPU 1 thread tok/s |
|---|---|---|---|---|---|---|---|
| mamba-130m | 129,135,360 | 129,135,360 | 258.3 | 3.620 | 71.35 | 1.0000x | 14.0 |
| smollm2-135m | 134,515,008 | 134,515,008 | 269.0 | 3.418 | 78.70 | 1.1030x | 13.3 |
| pythia-160m | 162,322,944 | 123,689,472 | 247.4 | 3.620 | 68.34 | 0.9578x | 17.9 |
| gpt2 | 124,439,808 | 124,439,808 | 248.9 | 3.440 | 72.35 | 1.0140x | 19.2 |
| distilgpt2 | 81,912,576 | 81,912,576 | 163.8 | 3.440 | 47.62 | 0.6673x | 31.2 |
| pythia-70m | 70,426,624 | 44,670,976 | 89.3 | 3.620 | 24.67 | 0.3457x | 46.5 |
| pythia-31m | 30,494,720 | 17,616,896 | 35.2 | 3.620 | 9.72 | 0.1363x | 91.6 |
| pythia-14m | 14,067,712 | 7,628,800 | 15.3 | 3.620 | 4.23 | 0.0592x | 154.6 |
| l3tc-12m | 19,287,552 | 12,996,096 | 25.99 | 3.507 | 7.41 | 0.1039x | 147.9 |
| l3tc-3.2m | 10,364,928 | 6,170,624 | 12.34 | 3.507 | 3.52 | 0.0493x | 289.8 |
| l3tc-800k | 6,278,976 | 3,395,392 | 6.79 | 3.507 | 1.94 | 0.0271x | 508.4 |
| l3tc-200k | 3,295,872 | 1,723,008 | 3.45 | 3.507 | 0.98 | 0.0138x | 560.6 |

- En los modelos pequeños con vocabulario de 50K la cabeza de salida domina: Pythia-14M toca 7.6M de parámetros
  por token y 6.4M son la matriz de salida (50,304 x 128). **El vocabulario pone un suelo al coste** que el número
  de capas no puede bajar. L3TC, con 16K, baja ese suelo a 1.7M.
- La velocidad medida en PyTorch crece menos que el coste analítico (40x frente a 72x para L3TC-200K) porque a
  ese tamaño manda el overhead del framework; es la misma advertencia que aplica a nuestro runtime.

### 5.3 El periodo de reinicio importa mucho más en XML y logs que en texto

| Dataset | Mamba C=4096 | Mamba C=2048 | delta | SmolLM2 C=4096 | SmolLM2 C=2048 | delta |
|---|---|---|---|---|---|---|
| enwik8 8x10K | 1.0282 | 1.0451 | +1.6% | 0.9187 | 0.9371 | +2.0% |
| enwik8 500K | 0.9384 | 0.9539 | +1.7% | 0.8341 | 0.8543 | +2.4% |
| enwik9 tail | 0.9983 | 1.0135 | +1.5% | 0.9150 | 0.9425 | +3.0% |
| dickens | 1.0177 | 1.0315 | +1.4% | 0.9644 | 0.9785 | +1.5% |
| facturae | 0.2309 | 0.3059 | +32.5% | 0.1830 | 0.2681 | +46.5% |
| UBL | 0.4145 | 0.5999 | +44.7% | 0.3360 | 0.5875 | +74.9% |
| apache | 0.1746 | 0.2170 | +24.3% | 0.1824 | 0.2483 | +36.1% |
| syslog | 0.2461 | 0.2911 | +18.3% | 0.2533 | 0.3248 | +28.2% |
| openssh | 0.2140 | 0.2825 | +32.0% | 0.2308 | 0.3483 | +50.9% |

Pasar de 4096 a 2048 tokens de contexto cuesta un 1.5-3% en texto y un **18-75% en XML y logs**. El contexto es de
primer orden en datos repetitivos. Consecuencia para R-108: el periodo por defecto (4096) se fijó con enwik8
100 KB; en los ficheros de industria conviene medir 8192 (el colapso de Mamba empieza hacia los 16-18K tokens)
o un reinicio con solapamiento. Es una palanca de ratio a coste cero que no estaba en el plan.

### 5.4 Economía: valor del ahorro frente a coste proyectado (texto Wikipedia no visto)

Valor = (bpb del mejor clásico - bpb del modelo) / 8 x $0.023/GB-mes x 84 meses (S3 Standard, 7 años).
Coste proyectado = $56/GB (CPU spot medido con Mamba-130M Q8 en lockstep) x coste por byte relativo.

| Model | enwik9-tail bpb | saving vs best classical (bpb) | 7y S3 value $/GB | projected CPU $/GB | cost / value |
|---|---|---|---|---|---|
| mamba-130m | 0.998 | 1.362 | 0.329 | 56.00 | 170x |
| smollm2-135m | 0.915 | 1.446 | 0.349 | 61.77 | 177x |
| pythia-160m | 1.094 | 1.266 | 0.306 | 53.64 | 175x |
| gpt2 | 1.270 | 1.091 | 0.263 | 56.79 | 216x |
| distilgpt2 | 1.406 | 0.955 | 0.231 | 37.37 | 162x |
| pythia-70m | 1.269 | 1.091 | 0.264 | 19.36 | 73x |
| pythia-31m | 1.368 | 0.993 | 0.240 | 7.63 | 32x |
| pythia-14m | 1.541 | 0.820 | 0.198 | 3.32 | 17x |
| l3tc-12m | 1.467 | 0.894 | 0.216 | 5.82 | 27x |
| l3tc-3.2m | 1.455 | 0.905 | 0.219 | 2.76 | 13x |
| l3tc-800k | 1.587 | 0.774 | 0.187 | 1.52 | 8x |
| l3tc-200k | 1.628 | 0.732 | 0.177 | 0.77 | 4x |

- Achicar el backbone acerca el coste al valor (de 170x a 4-17x) pero **ningún candidato lo cruza**, ni siquiera
  suponiendo eficiencia perfecta del runtime y sobre el único tipo de dato en que el pequeño funciona.
- El valor baja a la vez que el coste: un modelo 72x más barato ahorra un 46% menos de bytes.

### 5.5 Compresores clásicos sobre los mismos bytes

| Dataset | bytes | gzip-9 | bz2-9 | xz-9e | brotli-11 | zstd-19 |
|---|---|---|---|---|---|---|
| enwik8 8x10K | 81,920 | 3.5309 | 3.3798 | 3.4645 | 2.7354 | 3.4501 |
| enwik8 500K | 512,000 | 2.8190 | 2.2624 | 2.3741 | 2.2532 | 2.4533 |
| enwik9 tail | 512,000 | 2.9205 | 2.4017 | 2.4897 | 2.3608 | 2.5650 |
| dickens | 512,007 | 2.9536 | 2.1892 | 2.4650 | 2.4031 | 2.4925 |
| facturae | 38,371 | 0.5631 | 0.5283 | 0.4428 | 0.4078 | 0.4731 |
| UBL | 81,558 | 0.3850 | 0.3506 | 0.2809 | 0.2519 | 0.2934 |
| apache | 171,239 | 0.4284 | 0.3326 | 0.3151 | 0.3132 | 0.3345 |
| syslog | 216,485 | 0.5444 | 0.5261 | 0.3697 | 0.3789 | 0.4185 |
| openssh | 225,216 | 0.5522 | 0.4269 | 0.3460 | 0.3592 | 0.3855 |

## 6. Lo que no se pudo medir

- **nGPT-enwiki8** (nivel byte, ~25M): el paquete `nGPT-pytorch` falla al construir el modelo con el PyTorch de
  Kaggle (`unhashable type: 'NormLinear'`). No se insistió: es un modelo "de prueba" entrenado solo con enwik8, con
  contexto de 512 bytes; por lo visto en L3TC, tampoco generalizaría.
- **Backbone pequeño + CM**: requiere integrar el candidato en el runtime de Rust. Ningún candidato lo justifica.
- **Estirpe Hutter 2026** (cmix + transformer ~6M): son compresores completos bajo GPL-3.0, ajustados a enwik9; no
  son un backbone que se pueda enchufar, y la licencia es incompatible con Apache-2.0.

## 7. Decisión

1. **NO-GO a sustituir Mamba-130M por un backbone preentrenado más pequeño.** No existe ninguno que conserve el
   ratio: los generales pequeños lo pierden de forma continua y los especialistas publicados solo sirven sobre su
   fichero de entrenamiento.
2. **El tier rápido (R-105) solo tiene sentido como especialista por dominio entrenado por nosotros**, y su
   pregunta abierta es de generalización, no de tamaño: ¿un modelo de 1-10M entrenado con logs (o con factura
   electrónica) de unos sistemas comprime los de otros sistemas que no ha visto? L3TC demuestra que dentro del
   dominio 3-12M bastan para igualar a 130M; también demuestra que fuera del dominio no sirven de nada.
   Experimento siguiente, barato (un modelo de este tamaño se entrena en horas en una T4):
   - corpus real y público por dominio (Loghub: entrenar con N sistemas, evaluar con sistemas retenidos);
   - vocabulario pequeño (bytes o BPE <= 16K) para no pagar el suelo de la cabeza de 50K;
   - gate: bpb sobre sistemas retenidos frente a xz/brotli **y** frente a Mamba-130M, con el coste por byte medido.
   Los XML de factura del repo son sintéticos y no sirven para este gate: hace falta corpus real.
3. **A coste cero**: medir el periodo de reinicio 8192 (y reinicio con solapamiento) en los cinco ficheros de
   industria (§5.3).
4. **A igual coste**: SmolLM2-135M es un backbone mejor para texto y XML. Exigiría un backbone transformer en el
   runtime (atención con caché) y no mejora el $/GB; se anota como opción de calidad, no se prioriza.
5. **Corrección de una referencia del repo**: "L3TC-3.2M 1.105 bpb en enwik8" es cierto, pero enwik8 es su
   fichero de entrenamiento. Sobre Wikipedia no vista medimos 1.455 bpb, y 4.3-6.2 bpb en XML y logs. No debe
   citarse como prueba de que un modelo de 3M iguala a uno de 130M en general.

## 8. Reproducción

```
cp notebooks/small-backbone-probe.py notebooks/small-backbone-probe/
python -m kaggle kernels push -p notebooks/small-backbone-probe
python -m kaggle kernels output <user>/small-backbone-probe -p <dir> --file-pattern '^small-backbone-'
```

`PROBE_ONLY=l3tc` (o una lista de etiquetas) limita los modelos; `PROBE_SMOKE=1` hace una pasada mínima.
Ejecución completa: 0.61 h (v1, modelos HF) + 0.06 h (v3, L3TC) en T4.

## Fuentes

- L3TC: https://arxiv.org/abs/2412.16642 y
  https://github.com/alipay/L3TC-leveraging-rwkv-for-learned-lossless-low-complexity-text-compression
- Pythia: https://huggingface.co/EleutherAI/pythia-14m (y -31m, -70m, -160m)
- Mamba-130M: https://huggingface.co/state-spaces/mamba-130m-hf
- SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- nGPT-enwiki8: https://huggingface.co/pszemraj/nGPT-enwiki8
- ByteGPT-small: https://huggingface.co/ijktech/ByteGPT-small
- Compression via Pre-trained Transformers (Heurtel-Depeiges et al., ICML 2025): https://arxiv.org/abs/2410.05078
- Hutter Prize 2026 (cmix-obias, GPL-3.0): https://huggingface.co/dfreelan/cmix-obias
