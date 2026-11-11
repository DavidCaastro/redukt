# Periodo de reinicio del estado en ficheros enteros (2026-11-11)

> R-111 (Fase 20). Kaggle T4 + 4 vCPU, kernel `reset-period-sweep` v1 (4.80 h), código en e932f38.
> Script: `notebooks/reset-period-sweep.py`. Datos: `data/benchmark/results/reset-period-sweep/run-1/`.
> Estado: MEDIDO. Decisión en §4.

## 1. Pregunta

R-108 fijó el reinicio del estado de Mamba en 4096 tokens usando enwik8 100 KB. La sonda de backbones midió
(backbone solo) que bajar de 4096 a 2048 cuesta 18-75% de bpb en XML y logs y solo 1.5-3% en texto. ¿Conviene
subirlo en los ficheros de industria? ¿Y un reinicio con solapamiento, que recalienta el estado con los últimos K
tokens en vez de arrancar en frío?

## 2. A — `.azth` reales por periodo (backbone + CM + coder)

Siete ficheros enteros (los cinco de industria, enwik8 500 KB, dickens 500 KB) comprimidos con
`--reset-every` 2048 / 4096 / 8192 / 16384 / 32768 por el lockstep del plugin (un lote de 7 flujos por periodo,
576.010 tokens, ~343 tok/s agregados). Roundtrip en lote de los archivos de 8192: **7/7 idénticos**.

| Fichero | Bytes | 2048 | **4096 (actual)** | 8192 | 16384 | 32768 | 8192 vs 4096 | 2048 vs 4096 | 16384 vs 4096 |
|---|---|---|---|---|---|---|---|---|---|
| facturae-batch.xml | 38,371 | 0.2729 | 0.2408 | 0.2514 | 0.4230 | 0.4460 | +4.4% | +13.3% | +75.7% |
| ubl-invoice-batch.xml | 81,558 | 0.2457 | 0.2392 | 0.2355 | 0.2906 | 0.4429 | -1.5% | +2.7% | +21.5% |
| apache-access.log | 171,239 | 0.2047 | 0.1849 | 0.1830 | 0.2577 | 0.3723 | -1.0% | +10.7% | +39.4% |
| linux-syslog.log | 216,485 | 0.2650 | 0.2449 | 0.2448 | 0.3551 | 0.6172 | -0.0% | +8.2% | +45.0% |
| openssh-auth.log | 225,216 | 0.2318 | 0.2014 | 0.2096 | 0.3737 | 0.5477 | +4.1% | +15.1% | +85.6% |
| enwik8.500k | 512,000 | 0.9805 | 0.9697 | 0.9721 | 1.0256 | 1.6623 | +0.2% | +1.1% | +5.8% |
| dickens.500k | 512,000 | 1.0777 | 1.0620 | 1.0595 | 1.0757 | 1.2304 | -0.2% | +1.5% | +1.3% |
| **5 de industria (bytes totales)** | | 21,895 | 19,850 | 20,050 | 30,637 | 46,747 | +1.0% | +10.3% | +54.3% |

- **4096 es el óptimo práctico.** 8192 queda a ±1.5% en cinco ficheros y pierde un 4% en facturae y openssh;
  sobre los cinco de industria suma un 1.0% más de bytes. 2048 pierde en todos (+10.3% en industria).
- **A partir de 16384 el modelo se degrada**: +21% a +86% en industria con 16384 y hasta 2.7x con 32768. Es el
  colapso con la longitud de Mamba-130M (R-108) visto desde el otro lado: confirma que el reinicio hace falta y
  dónde está el límite.
- En texto la curva es casi plana entre 2048 y 8192 (±1.5%).

El CM amortigua el efecto del periodo: con el backbone solo, pasar de 4096 a 2048 costaba 18-75% en estos
ficheros; con el `.azth` real cuesta 3-15%.

## 3. B — reinicio con solapamiento (estimación, backbone solo)

En cada reinicio, en lugar de arrancar con el estado a cero, se recalienta con los últimos K tokens ya
codificados (el decodificador los conoce) y se sigue. Cuesta (P + K) / P forwards por token. Medido con
Mamba-130M en PyTorch (HF), sin CM; **no hay implementación en Rust**: es una estimación para decidir si merece
el cambio de formato.

| Fichero | P4096 K0 (actual) | K512 | K1024 | K2048 | P8192 K0 | P8192 K1024 | P16384 K0 | mejor vs actual |
|---|---|---|---|---|---|---|---|---|
| facturae-batch.xml | 0.2329 | 0.1769 | 0.1680 | 0.1804 | 0.2579 | 0.2344 | 0.5585 | -27.9% |
| ubl-invoice-batch.xml | 0.4128 | 0.3535 | 0.3049 | 0.2780 | 0.4114 | 0.4436 | 1.1036 | -32.7% |
| apache-access.log | 0.1745 | 0.1365 | 0.1381 | 0.1490 | 0.2057 | 0.2452 | 0.9965 | -21.8% |
| linux-syslog.log | 0.2459 | 0.2074 | 0.2077 | 0.2166 | 0.2686 | 0.2841 | 1.1806 | -15.7% |
| openssh-auth.log | 0.2136 | 0.1550 | 0.1583 | 0.1822 | 0.3141 | 0.4309 | 1.7710 | -27.4% |
| enwik8.500k | 0.9381 | 0.9255 | 0.9250 | 0.9269 | 0.9389 | 0.9380 | 1.0139 | -1.4% |
| forwards por token (media) | 1.000 | 1.121 | 1.242 | 1.484 | 1.000 | 1.114 | 1.000 | |

- **Recalentar con 512-2048 tokens mejora un 16-33% en los cinco ficheros de industria** (bits totales de
  industria: -20.2% con K=512, -22.0% con K=1024) a cambio de un 12-24% más de forwards. En texto, -1.4%.
- Más solapamiento no es mejor: K=2048 ya pierde frente a K=1024 salvo en UBL, y alargar el periodo (8192,
  16384) empeora con o sin solapamiento. El arranque en frío es lo que duele, no la falta de contexto largo.
- **Cuánto de esto llega al producto no está medido.** El CM ya arrastra contexto a través de los reinicios, así
  que parte de la ganancia puede estar capturada: en UBL el backbone solo da 0.413 y el `.azth` real 0.239;
  en apache el margen del CM es pequeño (0.175 vs 0.185) y ahí es donde más cabría esperar que la mejora se
  traslade.

## 4. Decisión

1. **El valor por defecto (4096) no cambia.** Es el óptimo o está a ±1.5% de él en los siete ficheros; ningún
   otro periodo cumple el criterio de aceptación (mejorar en industria sin empeorar texto más de un 1%).
2. **Reinicio con solapamiento: palanca real de ratio, no priorizada.** -16% a -33% de bpb en industria
   (backbone solo) por +12% de cómputo es la mayor mejora de ratio encontrada en la Fase 20. No se implementa
   ahora porque solo actúa en ficheros de más de 4096 tokens (~15 KB), es decir, en el segmento de
   almacenamiento, que no es viable por coste aunque el ratio mejore un 20%; en mensajes cortos (vía 3) el
   reinicio nunca se dispara. Si en algún momento vuelve a interesar el ratio en ficheros grandes, es lo primero
   que hay que hacer: un byte más en la cabecera (K en log2), K forwards extra por periodo en el codec, el
   lockstep y el plugin, y los gates habituales.

## 5. Reproducción

```
cp notebooks/reset-period-sweep.py notebooks/reset-period-sweep/
python -m kaggle kernels push -p notebooks/reset-period-sweep
python -m kaggle kernels output <user>/reset-period-sweep -p <dir> --file-pattern '^reset-sweep-'
```
