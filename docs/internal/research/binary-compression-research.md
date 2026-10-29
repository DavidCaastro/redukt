# Investigación: Mejora de compresión en datos binarios

> Fecha: 2026-10-29
> Contexto: Redukt gana 12/17 archivos vs brotli-11 en texto/estructurado,
> pero pierde significativamente en 5 archivos binarios (sao +54%, x-ray +44%,
> ooffice +37%, osdb +22%, mozilla +1.5%).

## Diagnóstico

El tokenizer de RWKV-7 está entrenado en texto. Cuando procesa datos binarios
(floats IEEE 754, ejecutables x86, registros de base de datos), los byte
sequences se mapean a tokens de texto sin significado. El modelo no puede
predecir estos tokens, resultando en BPB cercano a la entropía máxima.

Los 3 vectores de investigación convergen en el mismo patrón arquitectónico:

```
Estado actual:   Input -> RWKV-7 + CM -> Arithmetic Code
Estado del arte: Input -> Detect Type -> Preprocess -> Model -> Arithmetic Code
```

La pieza faltante es **detect + preprocess**.

## Fuentes investigadas

- Nacrith (DCC 2026): segmentación text/binary, SmolLM2-135M + LZMA fallback
- NNCP (Bellard, 2019/2021): byte-level 256-symbol model para binarios
- PAQ8PX v215: catálogo completo de ~30 transforms por tipo de archivo
- CMIX: LSTM mixer + PAQ8 transforms + Precomp
- G2-V3 (AIT Challenge 2026): pipeline detect->preprocess->transform->compress
- L3TC (2024): RWKV específicamente para compresión
- DZip: adaptive online learning sin datos de entrenamiento
- Burtscher FPC: predicción XOR para floats científicos
- Elf/Chimp (VLDB): erasing-based float compression
- ZipNN: byte shuffle para embeddings
- HDF5/Blosc: shuffle filter para arrays numéricos
- Turbo-Transpose: byte-plane splitting optimizado

## Técnicas identificadas

### Tier 1: Inmediatas (ya codificado o <1 día)

| Técnica | Descripción | Archivos | BPB mejora |
|---|---|---|---|
| E8E9 integración | Convertir CALL/JMP relativos a absolutos. YA IMPLEMENTADO en `preprocess.rs` | mozilla, ooffice | -5-10% |
| Delta coding | `out[i] = in[i] - in[i-stride]`. 5 líneas de código | sao, x-ray | -10-25% |

### Tier 2: Corto plazo (2-5 días)

| Técnica | Descripción | Archivos | BPB mejora |
|---|---|---|---|
| Byte shuffle | Transponer N-byte values: agrupar todos byte-0, byte-1, etc. | sao, x-ray | -15-30% |
| Data-type detector | Entropy + ASCII ratio + float patterns → clasificar tipo | dispatcher | habilita todo |
| Column transposition | Detectar record-length R, transponer R columnas | osdb | -15-30% |

### Tier 3: Medio plazo (1-3 semanas)

| Técnica | Descripción | Archivos | BPB mejora |
|---|---|---|---|
| Record-length detection | Autocorrelación para encontrar stride fijo | osdb, sao | -5-15% |
| CM-only bypass | Rutear binarios solo por 19 CM models (ya byte-level) | todos binarios | -20-40% |
| XOR float prediction | Hash predictor + XOR residual (estilo FPC/Burtscher) | sao | -20-40% |

### Tier 4: Largo plazo (3-6 semanas)

| Técnica | Descripción | Archivos | BPB mejora |
|---|---|---|---|
| Byte-level auxiliary model | LSTM 256-symbol, ~500K params, entrenado offline | todos binarios | -30-50% |
| OLE2 stream extraction | Parsear compound documents, separar streams | ooffice | -10-25% |
| Embedded zlib decompression | Descomprimir deflate embebido (Precomp-style) | ooffice | -15-30% |

## Análisis por archivo problemático

### sao (astronomía, +54% vs brotli)
- Datos FITS: headers de texto + arrays de float64 IEEE 754
- **Mejor approach**: byte shuffle (stride-8) + delta coding
- Exponent bytes (~2-3 bits entropy) se separan de mantissa (near-random)
- Esperado: -30-50% BPB con byte shuffle solo

### x-ray (imagen médica, +44% vs brotli)
- Pixel data (probablemente 16-bit grayscale)
- **Mejor approach**: byte shuffle (stride-2) + left-pixel delta prediction
- Similar a filtro PNG sub(1)
- Esperado: -20-35% BPB

### ooffice (Office binario, +37% vs brotli)
- OLE2 compound document: múltiples streams internos
- Puede contener zlib/JPEG embebido
- **Mejor approach**: E8E9 (tiene código x86 embebido) + futuro OLE2 parsing
- Esperado: -10-20% con E8E9, más con OLE2

### osdb (base de datos, +22% vs brotli)
- Registros de longitud fija con campos numéricos
- **Mejor approach**: record detection + column transposition
- Esperado: -15-30% BPB

### mozilla (ejecutable, +1.5% vs brotli)
- Binario x86 con strings embebidos
- **Mejor approach**: E8E9 transform (ya codificado)
- Esperado: -5-10% BPB (suficiente para superar brotli)

## Compatibilidad con Redukt

Todas las técnicas Tier 1-3 son:
- **Zero-dependency**: solo aritmética entera y manipulación de arrays
- **Reversibles**: transform + inverse transform = identidad
- **Integrables**: AZTH v4 FLAGS byte tiene 6 bits libres para señalar preprocessing
- **Sin riesgo BPB en texto**: los transforms solo se activan cuando se detecta
  el tipo correcto de datos, por lo que archivos de texto/log no se ven afectados

## Arquitectura propuesta

```
compress:
  1. Detect content type (entropy, ASCII ratio, magic bytes, float patterns)
  2. Select preprocessing pipeline:
     - text/structured: no preprocess (actual)
     - executable: E8E9 transform
     - float array: byte shuffle + delta
     - fixed records: record detection + column transpose
     - mixed: segmentation (text neural, binary CM-only)
  3. Store preprocessing ID in AZTH v4 FLAGS
  4. RWKV-7 + CM + Arithmetic Code (sin cambios)

decompress:
  1. Read FLAGS → preprocessing ID
  2. Decompress normally (sin cambios)
  3. Apply inverse transform(s)
```

## Benchmark actual (2026-10-29)

### 10KB — 18 archivos, Redukt Q8 vs brotli-11

| Archivo | Redukt BPB | brotli-11 | Δ |
|---|---|---|---|
| apache-access.log | 0.296 | 0.466 | -36.5% |
| linux-syslog.log | 0.412 | 0.582 | -29.2% |
| facturae-batch.xml | 0.453 | 0.891 | -49.2% |
| openssh-auth.log | 0.527 | 0.783 | -32.7% |
| ubl-invoice-batch.xml | 0.542 | 0.889 | -39.0% |
| nci | 0.669 | 0.735 | -9.0% |
| xml | 0.689 | 0.902 | -23.6% |
| samba | 1.298 | 2.221 | -41.6% |
| enwik8 | 1.333 | — | — |
| dickens | 1.797 | 2.779 | -35.3% |
| webster | 1.805 | 2.810 | -35.8% |
| reymont | 1.899 | 2.136 | -11.1% |
| mozilla | 2.052 | 2.021 | +1.5% |
| mr | 2.389 | 2.373 | +0.7% |
| ooffice | 3.767 | 2.742 | +37.4% |
| osdb | 5.706 | 4.676 | +22.0% |
| x-ray | 6.590 | 4.593 | +43.5% |
| sao | 9.243 | 5.988 | +54.3% |

### Escala (batch mode, modelo cargado una vez)

| Archivo | 10KB | 50KB | 100KB | brotli 100KB | Δ 100KB |
|---|---|---|---|---|---|
| apache-access.log | 0.296 | 0.227 | 0.224 | 0.328 | -31.7% |
| dickens | 1.797 | 1.503 | 1.457 | 2.649 | -45.0% |
| samba | 1.298 | 1.356 | 1.206 | 1.944 | -38.0% |

### Rendimiento

| Métrica | 10KB | 50KB batch | 100KB batch |
|---|---|---|---|
| Mean tok/s | 64.0 | 61.3 | 63.7 |
| Model load | per-file | 3.5s (once) | 2.8s (once) |
| Roundtrip | 2/2 PASS | — | — |

## Referencias

1. Nacrith (2026) - arxiv.org/abs/2602.19626
2. NNCP - bellard.org/nncp/nncp.pdf
3. PAQ8PX - github.com/hxim/paq8px
4. CMIX - github.com/byronknoll/cmix
5. G2-V3 (2026) - arxiv.org/pdf/2606.17712
6. L3TC (2024) - arxiv.org/html/2412.16642v2
7. DZip - arxiv.org/abs/1911.03572
8. FPC (Burtscher) - userweb.cs.txstate.edu/~burtscher/papers/dcc06.pdf
9. Elf (VLDB) - vldb.org/pvldb/vol16/p1763-li.pdf
10. Chimp (VLDB) - vldb.org/pvldb/vol15/p3058-liakos.pdf
11. Precomp - github.com/schnaader/precomp-cpp
12. HDF5 Shuffle - support.hdfgroup.org/documentation
13. Turbo-Transpose - github.com/powturbo/Turbo-Transpose
14. FITS Compression - arxiv.org/pdf/1201.1340
15. ZipNN - arxiv.org/html/2602.00079v1
