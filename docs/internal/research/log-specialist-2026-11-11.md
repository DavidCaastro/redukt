# Especialista por dominio para logs — ¿generaliza a sistemas no vistos? (2026-11-11)

> R-112 (Fase 20; R-105 paso 2). Entrenamiento en Kaggle T4 (kernel `log-specialist-train` v1, 3.14 h) y
> evaluación de los LSTM en local (CPU) con los modelos descargados del kernel.
> Script: `notebooks/log-specialist-train.py`. Datos: `data/benchmark/results/log-specialist/run-1/`.
> Estado: MEDIDO. Decisión: **NO-GO** (§5).

## 1. Pregunta

La sonda de backbones preentrenados (R-105 paso 1) dejó una sola puerta abierta para el tier rápido: un modelo
pequeño **entrenado por nosotros sobre el dominio del cliente**. La pregunta es de generalización, no de tamaño:
¿un modelo de 0.3-4M entrenado con logs de unos sistemas comprime los logs de sistemas que no ha visto nunca?

## 2. Diseño

- **Corpus**: Loghub (He et al., CC-BY-4.0, Zenodo 3227177), real y público. División **por sistema**:
  - entrenamiento (11 sistemas, 456.6 MB): BGL, HDFS, Hadoop, Spark, HPC, Mac, Android, Windows, Thunderbird,
    OpenStack, HealthApp — hasta 48 MB de cada uno;
  - "vistos": 300 KB de cada sistema de entrenamiento, de una zona posterior que no se usó para entrenar
    (mismo sistema, otro tramo temporal);
  - "no vistos" (5 sistemas, nunca en entrenamiento): Apache, Linux y OpenSSH — los tres ficheros de log del
    benchmark de Redukt, byte a byte — más Zookeeper y Proxifier (primeros 300 KB).
- **Modelos** a nivel de byte (vocabulario 256, sin el suelo de coste de una cabeza de 50K): tres LSTM de dos
  capas (XS/S/M) y un transformer causal pequeño. La arquitectura es la que entrena rápido y fiable en una T4 con
  PyTorch puro; la pregunta es la generalización a un tamaño dado. Cada modelo tiene un presupuesto de tiempo.
- **Evaluación** (bits por byte, backbone solo, sin CM):
  - *estática*: modelo congelado, trozos independientes de C bytes desde estado cero (LSTM) o ventana
    deslizante de 1024 bytes (transformer);
  - *dinámica*: el LSTM sigue entrenando con los bytes que acaba de codificar (un paso de Adam cada 256 bytes,
    estado arrastrado), como hacen NNCP o cmix. Un decodificador puede reproducirlo, a ~3x de cómputo y con un
    contrato de determinismo muy exigente (entrenamiento en coma flotante bit a bit idéntico en ambos extremos);
  - *dinámica desde cero* (control): el mismo procedimiento con pesos aleatorios, para medir cuánto aporta el
    preentrenamiento;
  - referencias sobre los mismos bytes: xz -9e, brotli -11, zstd -19, bzip2 -9 y Mamba-130M (HF, solo backbone,
    trozos de 4096 tokens).
- **Coste**: 2 x parámetros tocados por byte = MFLOP/byte, frente a 71.35 MFLOP/byte de Mamba-130M.

Incidencia: dentro del kernel la evaluación de los LSTM abortó con un error de cuDNN (entrada no contigua); el
transformer, los clásicos y la referencia Mamba-130M sí se evaluaron allí. Los cuatro modelos quedaron guardados;
los LSTM se evaluaron después en CPU con las mismas funciones del script sobre 8 de los 16 conjuntos (los 5 no
vistos y 3 vistos: HPC, Mac, HealthApp). Los bytes son los mismos: xz -9e da idéntico bpb en local y en Kaggle
en los 8 conjuntos. El script ya lleva la corrección.

## 3. Modelos entrenados

| Modelo | Parámetros | Tocados por byte | MFLOP/byte | Coste vs Mamba-130M | Minutos | Pasos | Bytes vistos |
|---|---|---|---|---|---|---|---|
| lstm-xs | 280,832 | 264,448 | 0.529 | 0.0074x | 22.0 | 9,437 | 1,237 MB |
| lstm-s | 938,240 | 921,856 | 1.844 | 0.0258x | 30.0 | 4,589 | 601 MB |
| lstm-m | 3,580,160 | 3,547,392 | 7.095 | 0.0994x | 45.0 | 2,199 | 288 MB |
| tf-s | 990,208 | 826,368 | 1.653 | 0.0232x | 30.0 | 14,497 | 950 MB |

La pérdida de entrenamiento seguía bajando al acabar el presupuesto (0.46-0.74 bpb), pero **el bpb sobre los
sistemas no vistos empeoraba con el entrenamiento** en los cuatro modelos:

| Modelo | Paso | Minutos | bpb entrenamiento | bpb no vistos (estático 4096) |
|---|---|---|---|---|
| lstm-xs | 1 | 0.1 | 7.9970 | 7.9942 |
| lstm-xs | 2,872 | 6.6 | 0.9659 | 5.1913 |
| lstm-xs | 5,691 | 13.2 | 0.7787 | 5.4980 |
| lstm-xs | 8,500 | 19.8 | 0.7356 | 5.5997 |
| lstm-s | 1 | 0.0 | 7.9957 | 7.9932 |
| lstm-s | 1,386 | 9.0 | 0.9813 | 4.8077 |
| lstm-s | 2,759 | 18.0 | 0.7305 | 4.9477 |
| lstm-s | 4,131 | 27.0 | 0.6703 | 4.9648 |
| lstm-m | 1 | 0.1 | 7.9903 | 7.9922 |
| lstm-m | 665 | 13.6 | 1.0468 | 4.6447 |
| lstm-m | 1,324 | 27.1 | 0.7151 | 4.8525 |
| lstm-m | 1,982 | 40.6 | 0.6446 | 4.8220 |
| tf-s | 1 | 0.0 | 8.2990 | 8.3009 |
| tf-s | 4,339 | 9.0 | 0.8278 | 5.9184 |
| tf-s | 8,686 | 18.0 | 0.5211 | 6.6202 |
| tf-s | 13,041 | 27.0 | 0.4579 | 7.2193 |

Más entrenamiento con el mismo corpus no arregla la generalización: la empeora. El modelo aprende los formatos
de los 11 sistemas, no "cómo son los logs".

## 4. Resultados (bpb)

### 4.1 Los 8 conjuntos con evaluación completa

| Modelo | apache | linux | openssh | zookeeper | proxifier | HPC | Mac | HealthApp | media no vistos | media vistos |
|---|---|---|---|---|---|---|---|---|---|---|
| Mejor clásico | 0.313 | 0.370 | 0.346 | 0.204 | 0.242 | 0.274 | 0.583 | 0.480 | 0.295 | 0.446 |
| Mamba-130M (solo backbone, C=4096 tokens) | 0.175 | 0.246 | 0.214 | 0.170 | 0.205 | 0.205 | 0.785 | 0.523 | 0.202 | 0.504 |
| lstm-xs estático (C=4096 B) | 6.067 | 5.288 | 5.104 | 5.180 | 6.357 | 1.569 | 1.203 | 0.972 | 5.599 | 1.248 |
| lstm-s estático | 5.592 | 4.710 | 4.638 | 4.452 | 5.500 | 0.998 | 1.079 | 0.944 | 4.979 | 1.007 |
| lstm-m estático | 5.432 | 4.033 | 4.786 | 4.189 | 5.514 | 0.931 | 0.971 | 0.977 | 4.791 | 0.959 |
| tf-s estático (ventana 1024 B) | 7.391 | 6.991 | 7.106 | 6.264 | 9.233 | 0.530 | 0.684 | 0.567 | 7.397 | 0.594 |
| lstm-xs dinámico | 0.433 | 1.351 | 0.559 | 0.328 | 0.282 | 0.366 | 1.109 | 0.590 | 0.591 | 0.688 |
| lstm-s dinámico | 0.375 | 1.189 | 0.467 | 0.265 | 0.249 | 0.340 | 0.986 | 0.548 | 0.509 | 0.625 |
| lstm-m dinámico | 0.329 | 1.055 | 0.411 | 0.235 | 0.229 | 0.326 | 0.930 | 0.521 | 0.452 | 0.592 |
| lstm-xs dinámico desde cero | 3.252 | 3.987 | 3.284 | 2.165 | 1.804 | 1.851 | 4.252 | 3.050 | 2.898 | 3.051 |
| lstm-s dinámico desde cero | 2.151 | 3.186 | 2.397 | 1.442 | 1.144 | 1.222 | 3.666 | 2.393 | 2.064 | 2.427 |

### 4.2 Los 16 conjuntos (transformer estático y referencias)

| Conjunto | Tipo | Mejor clásico | tf-s estático | Mamba-130M |
|---|---|---|---|---|
| BGL | visto | 0.179 | 0.141 | 0.223 |
| HDFS_1 | visto | 0.542 | 0.496 | 0.616 |
| Hadoop | visto | 0.454 | 0.391 | 0.444 |
| Spark | visto | 0.310 | 0.253 | 0.243 |
| HPC | visto | 0.274 | 0.530 | 0.205 |
| Mac | visto | 0.583 | 0.684 | 0.785 |
| Android | visto | 0.736 | 0.942 | 1.012 |
| Windows | visto | 0.210 | 0.311 | 0.455 |
| Thunderbird | visto | 0.387 | 0.845 | 0.578 |
| OpenStack | visto | 0.539 | 0.519 | 0.862 |
| HealthApp | visto | 0.480 | 0.567 | 0.523 |
| apache | no visto | 0.313 | 7.391 | 0.175 |
| linux | no visto | 0.370 | 6.991 | 0.246 |
| openssh | no visto | 0.346 | 7.106 | 0.214 |
| zookeeper | no visto | 0.204 | 6.264 | 0.170 |
| proxifier | no visto | 0.242 | 9.233 | 0.205 |
| **media vistos (11)** | | 0.427 | 0.516 | 0.541 |
| **media no vistos (5)** | | 0.295 | 7.397 | 0.202 |

### 4.3 Lectura

1. **Un especialista estático no generaliza a sistemas no vistos.** 4.0-9.2 bpb en los cinco: de 11 a 38 veces
   peor que el mejor clásico (0.20-0.37) y, en Proxifier con el transformer, peor que no comprimir (9.2 bpb).
   El contexto no lo arregla: de 1024 bytes al fichero entero el LSTM mejora menos de un 1%.
2. **En los sistemas con los que se entrenó sí funciona, pero no gana.** El transformer de 1M (2.3% del coste por
   byte de Mamba-130M) da 0.516 bpb de media en los 11 sistemas vistos: mejor que el backbone de Mamba-130M
   (0.541) y peor que el mejor clásico (0.427). Gana al clásico en 5 de 11 (BGL, HDFS, Hadoop, Spark, OpenStack).
3. **La adaptación en línea es lo que rescata los sistemas no vistos**: de ~5 bpb a 0.23-1.06 con el LSTM de
   3.6M. Y el preentrenamiento aporta de verdad: el mismo procedimiento desde pesos aleatorios se queda en
   1.1-4.0 bpb en esos mismos ficheros. Lo que se transfiere entre sistemas no es el contenido, es la capacidad de adaptarse rápido.
4. **Aun adaptándose, pierde contra xz.** Media en no vistos: 0.452 (LSTM 3.6M dinámico) frente a 0.295 del mejor
   clásico y 0.202 de Mamba-130M. Solo gana en Proxifier (0.229 vs 0.242), por la mínima.
5. **Mamba-130M es el único que gana a los clásicos en los cinco sistemas no vistos** (-15% a -44%), y eso sin
   CM. Es exactamente lo que un generalista grande compra y uno pequeño no: un prior que sirve para datos que
   no ha visto.

## 5. Decisión

**NO-GO al especialista de logs como tier rápido.**

- Estático: inservible fuera de los sistemas de entrenamiento. Exigiría entrenar un modelo por cliente y por
  formato, y aun así no gana a xz de media.
- Dinámico: no alcanza a xz, cuesta ~3x más cómputo que el estático y obliga a un contrato de determinismo
  (entrenamiento bit a bit idéntico en codificador y decodificador) mucho más duro que el actual.
- **La economía de los logs no deja margen ni en el mejor caso.** xz y brotli ya dejan los logs en 0.2-0.4 bpb:
  aunque un modelo los llevara a 0 bpb, el ahorro máximo es ~0.3 bpb = 0.04 GB por GB crudo = **$0.07/GB** a
  7 años en S3 Standard. Para que el cómputo cueste menos que eso hacen falta >55 KB/s por vCPU, que con el
  rendimiento por FLOP actual solo permite modelos de unos pocos cientos de miles de parámetros tocados por
  byte, justo los que aquí pierden contra xz. No es un problema de ingeniería pendiente: el techo de valor del
  segmento está por debajo de lo que cuesta cualquier red neuronal útil.

Con esto, **la vía 2 (tier rápido) queda cerrada** con medidas en sus dos variantes: preentrenados publicados
(R-105 paso 1) y especialista propio (este documento).

## 6. Límites de la medida

- **Pocos sistemas y poco entrenamiento**: 11 sistemas y 22-45 minutos por modelo (0.3-1.2 GB vistos). Un corpus
  con cientos de fuentes distintas podría enseñar "cómo son los logs" en vez de 11 formatos. No está medido, y
  no cambia el techo económico del punto anterior.
- **Sin CM.** En el producto, el ensamble CM ya es el componente que se adapta en línea; un backbone pequeño con
  CM quedaría mejor que estas cifras estáticas. Tampoco cambia el techo económico.
- **Arquitectura**: LSTM y un transformer pequeño, no RWKV ni Mamba. A este tamaño la diferencia esperable entre
  arquitecturas es de decenas de puntos porcentuales, no los 11-38x que separan a los modelos estáticos de xz.
- Los LSTM se evaluaron en 8 de los 16 conjuntos (los 5 no vistos, que son los que deciden, y 3 vistos).
- La evaluación dinámica no es bit a bit reproducible entre ejecuciones en CPU (0.3292 vs 0.3293 en apache):
  ilustra el problema de determinismo que tendría en producto.

## 7. Reproducción

```
cp notebooks/log-specialist-train.py notebooks/log-specialist-train/
python -m kaggle kernels push -p notebooks/log-specialist-train
python -m kaggle kernels output <user>/log-specialist-train -p <dir> --file-pattern '^specialist-|\.pt$'
python data/benchmark/results/log-specialist/run-1/local_eval.py notebooks/log-specialist-train.py <dir> out.csv \
    lstm-s static-4096,dynamic-256,dynamic-256-random-init
```

## Fuentes

- Loghub: https://github.com/logpai/loghub , https://zenodo.org/records/3227177 (CC-BY-4.0)
- Informe previo: `docs/internal/research/small-backbone-probe-2026-11-11.md`
