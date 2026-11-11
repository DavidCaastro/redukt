# Vía 3 — segmentos donde el byte vale más que el cómputo (2026-11-11)

> R-113 (Fase 20). Precios recogidos el 2026-11-11 (revendedores, no tarifas oficiales) + medida propia de
> mensajes cortos en Kaggle (CPU, 4 vCPU, kernel `short-message-bench` v1, código en e932f38).
> Datos crudos: `data/benchmark/results/short-message-bench/run-1/`. Script: `notebooks/short-message-bench.py`.
> Estado: MEDIDO. Veredicto en §5.

## 1. Pregunta

El producto actual cuesta $33-60 por GB comprimido (CPU spot $56, T4 por la CLI $33-38) y el ahorro de
almacenamiento vale $0.06-0.33/GB: no es viable ahí. ¿Existe algún segmento donde 1 GB ahorrado valga más que
ese coste? Y si existe, ¿Redukt comprime mejor que la alternativa clásica **en el tipo de tráfico de ese
segmento**, que son mensajes cortos y no ficheros?

## 2. Cuánto vale un GB transmitido, por enlace

| Enlace | Precio publicado | $/GB transmitido | Frente a $33-60/GB de coste |
|---|---|---|---|
| Iridium SBD (Short Burst Data) | $0.79-3.15 por 1.000 bytes de exceso; planes $15.75-147/mes | **$790.000-3.150.000** | 4-5 órdenes de magnitud por encima |
| Iridium IMT (Messaging Transport) | $0.30-0.50 por KB de exceso; $17-50/mes | **$300.000-500.000** | 4 órdenes |
| Inmarsat BGAN M2M | hasta $29.99 por MB en planes pequeños | **~$30.000** | 3 órdenes |
| Iridium Certus 100 | $3.50-15 por MB fuera de bolsa | **$3.500-15.000** | 2 órdenes |
| Inmarsat BGAN (IP estándar) | $5.81-6.99 por MB | **$5.800-7.000** | 2 órdenes |
| IoT celular (Hologram de pago por uso) | $0.03-0.21 por MB | $30-210 | mismo orden: sin margen |
| IoT celular (1NCE, tarifa plana 10 años) | $14 por 500 MB | $28 | por debajo |
| Starlink marítimo ("Ocean Mode") | $6 por GB | $6 | por debajo |
| Starlink Global Priority | $250/50 GB a $1.150/1 TB | $1.15-5 | por debajo |
| Egress de nube (AWS/Azure/GCP) | $0.05-0.12 por GB | $0.05-0.12 | 3 órdenes por debajo |

Lectura: **solo el satélite de banda estrecha (Iridium, Inmarsat banda L) vale más que el cómputo**, y lo hace
por un margen enorme. Starlink ha hundido el precio del GB marítimo a $1-6 y deja fuera todo lo que pueda llevar
una antena de banda ancha. El egress de nube y el IoT celular no dan margen.

Los precios son de revendedores y distribuidores (ver Fuentes), con fechas de 2025-2026; varían por contrato y
no son tarifas oficiales de Iridium ni de Inmarsat. Bastan para el orden de magnitud, no para un plan de negocio.

## 3. Medida: mensajes cortos

En estos enlaces el tráfico son mensajes de cientos de bytes a pocos KB. Todas las cifras previas de Redukt eran
ventanas de 10 KB o ficheros enteros. Los compresores clásicos no tienen historia que explotar en 256 bytes; un
modelo preentrenado lleva todo su conocimiento desde el primer byte. Había que medirlo.

**Protocolo.** Mensajes de 128 / 256 / 512 / 1024 / 2048 bytes cortados de datos reales (segunda mitad de cada
fuente, empezando en inicio de línea), 16 mensajes por fuente y tamaño, 480 en total:
Wikipedia (enwik8), prosa inglesa (dickens), tres logs de Loghub (Apache, Linux, OpenSSH) y JSON de eventos de
GitHub (GH Archive). Redukt con la ruta real de CPU (`compress --input-dir`, Mamba-130M Q8 + CM, AZTH v5).
Clásicos con su encuadre mínimo: deflate crudo, LZMA2 crudo, brotli -11 (lleva diccionario propio), zstd -19 sin
magic ni checksum, y **zstd -19 con un diccionario de 110 KB entrenado con la primera mitad de la misma fuente**,
que es la línea base justa para una flota que envía siempre el mismo tipo de mensaje.

Redukt se da dos veces: el `.azth` real (15 B de cabecera + 4 B de CRC) y la carga útil sola (fichero menos
19 B), que es lo que costaría con un encuadre compacto para mensajes. La cabecera es una decisión de formato,
no un límite del modelo.

**Roundtrip**: 24/24 mensajes muestreados descomprimen idénticos. Rendimiento: 480 mensajes (134.667 tokens) en
1.082 s = 124 tok/s agregados en 4 vCPU.

### 3.1 Tamaño medio comprimido en bytes (todas las fuentes y por fuente)

| Fuente | Tamaño | brotli-11 | zstd-19 + dicc. | Redukt `.azth` | Redukt carga útil | `.azth` vs dicc. | carga útil vs dicc. | `.azth` vs brotli |
|---|---|---|---|---|---|---|---|---|
| wikipedia | 128 | 96.6 | 73.8 | 64.1 | 45.1 | -13% | -39% | -34% |
| wikipedia | 256 | 155.4 | 127.1 | 75.8 | 56.8 | -40% | -55% | -51% |
| wikipedia | 512 | 245.4 | 210.2 | 111.0 | 92.0 | -47% | -56% | -55% |
| wikipedia | 1024 | 438.9 | 391.9 | 188.3 | 169.3 | -52% | -57% | -57% |
| wikipedia | 2048 | 794.3 | 747.4 | 412.8 | 393.8 | -45% | -47% | -48% |
| prose | 128 | 93.4 | 68.6 | 55.6 | 36.6 | -19% | -47% | -41% |
| prose | 256 | 160.4 | 120.7 | 79.6 | 60.6 | -34% | -50% | -50% |
| prose | 512 | 275.6 | 211.1 | 123.5 | 104.5 | -42% | -51% | -55% |
| prose | 1024 | 485.9 | 398.2 | 208.9 | 189.9 | -48% | -52% | -57% |
| prose | 2048 | 878.6 | 769.4 | 369.1 | 350.1 | -52% | -54% | -58% |
| log-apache | 128 | 98.9 | 34.8 | 54.8 | 35.8 | +57% | +3% | -45% |
| log-apache | 256 | 125.1 | 44.1 | 69.9 | 50.9 | +59% | +15% | -44% |
| log-apache | 512 | 157.9 | 58.7 | 90.0 | 71.0 | +53% | +21% | -43% |
| log-apache | 1024 | 192.2 | 87.2 | 107.1 | 88.1 | +23% | +1% | -44% |
| log-apache | 2048 | 248.9 | 135.2 | 134.1 | 115.1 | -1% | -15% | -46% |
| log-linux | 128 | 89.7 | 57.2 | 59.3 | 40.3 | +4% | -30% | -34% |
| log-linux | 256 | 108.9 | 62.9 | 68.5 | 49.5 | +9% | -21% | -37% |
| log-linux | 512 | 124.2 | 74.9 | 76.1 | 57.1 | +2% | -24% | -39% |
| log-linux | 1024 | 171.1 | 117.2 | 99.8 | 80.8 | -15% | -31% | -42% |
| log-linux | 2048 | 240.3 | 163.1 | 134.3 | 115.3 | -18% | -29% | -44% |
| log-openssh | 128 | 99.4 | 41.6 | 59.0 | 40.0 | +42% | -4% | -41% |
| log-openssh | 256 | 158.1 | 60.5 | 82.1 | 63.1 | +36% | +4% | -48% |
| log-openssh | 512 | 204.6 | 81.8 | 107.6 | 88.6 | +31% | +8% | -47% |
| log-openssh | 1024 | 245.8 | 111.7 | 127.4 | 108.4 | +14% | -3% | -48% |
| log-openssh | 2048 | 301.6 | 165.5 | 149.1 | 130.1 | -10% | -21% | -51% |
| json-events | 128 | 105.7 | 38.3 | 53.9 | 34.9 | +41% | -9% | -49% |
| json-events | 256 | 162.8 | 65.8 | 68.4 | 49.4 | +4% | -25% | -58% |
| json-events | 512 | 258.2 | 109.3 | 112.2 | 93.2 | +3% | -15% | -57% |
| json-events | 1024 | 382.7 | 194.9 | 169.2 | 150.2 | -13% | -23% | -56% |
| json-events | 2048 | 600.1 | 357.5 | 262.7 | 243.7 | -27% | -32% | -56% |
| ALL | 128 | 97.3 | 52.4 | 57.8 | 38.8 | +10% | -26% | -41% |
| ALL | 256 | 145.1 | 80.2 | 74.0 | 55.0 | -8% | -31% | -49% |
| ALL | 512 | 211.0 | 124.3 | 103.4 | 84.4 | -17% | -32% | -51% |
| ALL | 1024 | 319.4 | 216.8 | 150.1 | 131.1 | -31% | -40% | -53% |
| ALL | 2048 | 510.7 | 389.7 | 243.7 | 224.7 | -37% | -42% | -52% |

### 3.2 Lectura

- **Frente a los clásicos sin diccionario, Redukt gana en las 30 celdas**: el `.azth` real ocupa entre un 41% y
  un 53% menos que brotli -11 de media, y la mitad o menos que gzip/xz/zstd.
- **Frente a zstd con diccionario de dominio (la línea base justa)**:
  - el `.azth` real gana en 16 de 30 celdas: pierde en los mensajes más cortos de logs y JSON (128-512 B), donde
    los 19 B de cabecera pesan hasta un tercio del resultado, y gana en texto a cualquier tamaño (-13% a -52%)
    y en las seis fuentes a 2 KB;
  - la carga útil sola gana en 24 de 30 celdas y de media en todos los tamaños (-26% a 128 B, -42% a 2048 B).
- El diccionario de los logs se entrenó con la primera mitad **del mismo fichero**: es el mejor caso posible
  para zstd (misma máquina, mismo periodo). Redukt no necesita entrenar nada por cliente.
- En texto natural la ventaja es grande y estable: Wikipedia 1 KB, 188 B frente a 392 B de zstd con diccionario
  y 439 B de brotli.

## 4. Economía por segmento

Valor por GB crudo = precio del enlace x (fracción que deja la alternativa - fracción que deja Redukt).
Con la media de todas las fuentes a 1 KB: brotli 31.2%, zstd + diccionario 21.2%, Redukt `.azth` 14.7%,
Redukt carga útil 12.8%. A 256 B: 56.7% / 31.3% / 28.9% / 21.5%.

| Enlace | $/GB | Mensaje típico | Ahorro vs zstd+dicc. (`.azth` / carga útil) | Ahorro vs brotli (`.azth`) | Valor por GB crudo |
|---|---|---|---|---|---|
| Iridium SBD | 790.000-3.150.000 | 256 B | 2.4 / 9.8 puntos | 27.8 puntos | $19.000-76.000 (`.azth`); $77.000-309.000 (carga útil) |
| Iridium IMT | 300.000-500.000 | 1 KB | 6.5 / 8.4 puntos | 16.5 puntos | $19.500-32.500 |
| Certus 100 / BGAN | 3.500-15.000 | 1-2 KB | 6.5 / 8.4 puntos | 16.5 puntos | $230-975 vs diccionario; $580-2.475 vs brotli |
| IoT celular | 28-210 | 256 B-1 KB | 2.4-6.5 puntos | 16-28 puntos | $0.7-14 vs diccionario; hasta $58 vs brotli |
| Starlink marítimo | 1.15-6 | — | — | 16.5 puntos | $0.2-1.0 |
| Egress de nube | 0.05-0.12 | — | — | 16.5 puntos | $0.01-0.02 |

Frente a un coste de cómputo de $33-60 por GB crudo:
- **Banda estrecha por satélite: viable por 1-3 órdenes de magnitud**, incluso contra zstd con diccionario.
- IoT celular, Starlink y egress: no viables.

El coste en la nube ni siquiera es el término relevante en satélite: comprime el terminal emisor, y lo que
cuenta ahí es si puede ejecutar el modelo, no el precio del vCPU.

## 5. Veredicto

**Viable con condiciones.** Es el primer segmento en el que el producto actual, sin cambiar el backbone, vale
más de lo que cuesta. Condiciones, todas sin resolver:

1. **Encuadre compacto para mensajes.** Con 19 B fijos el `.azth` pierde contra zstd con diccionario en los
   mensajes de menos de ~512 B de logs y JSON. Hace falta un modo mensaje (versión + perfil en 1 B, longitud
   en varint, CRC opcional porque el enlace ya trae integridad): 2-4 B en lugar de 19.
2. **El terminal tiene que poder ejecutar el modelo.** Mamba-130M Q8 son ~130 MB de pesos y, en una máquina de
   servidor de 4 vCPU, del orden de 1-4 s por mensaje de 256 B-1 KB (70-110 tok/s en un solo flujo). No hay ninguna medida en ARM (pasarelas de a bordo
   tipo Raspberry Pi) y un microcontrolador queda descartado. Sin medida en el hardware del terminal no hay caso.
3. **Determinismo entre arquitecturas.** El emisor (ARM) y el receptor (x86 en tierra) tienen que producir la
   misma CDF bit a bit. Hoy solo está verificado x86-64 Windows <-> Linux. R-104 F2 (ruta escalar con el orden
   de AVX2) pasa de higiene a requisito de este segmento.
4. **Tamaño del mercado sin verificar.** El volumen por terminal es de MB al mes: el ahorro por terminal son
   decenas de dólares mensuales en Certus/BGAN, más en SBD. Existen productos de compresión de correo para
   Iridium (XGate anuncia "hasta 95%" y "hasta 20x"; PredictMail anuncia cifras mayores), lo que indica que los
   clientes pagan por esto, pero ninguna cifra de esos proveedores es un benchmark independiente y no tenemos
   datos de número de terminales ni de disposición a pagar.
5. **El tráfico tiene que ser texto, JSON o logs.** Mucha telemetría SBD va empaquetada en binario (posición,
   sensores) y ahí un modelo de lenguaje no aporta; no está medido.

Para los otros segmentos (IoT celular, Starlink, egress) el veredicto es **no viable**.

## 6. Siguiente paso si se sigue esta vía

1. Modo mensaje en el formato (encuadre de 2-4 B) y volver a medir las 30 celdas.
2. Compilar y medir en ARM64 (tok/s, memoria, energía por mensaje) y verificar el roundtrip ARM <-> x86.
3. Corpus real del segmento (partes meteorológicos, correo de a bordo, telemetría en JSON) en lugar de
   Wikipedia y logs de servidor.
4. Hablar con un integrador de comunicaciones por satélite antes de escribir más código: el punto 4 del
   veredicto no se resuelve con ingeniería.

## Fuentes

- Iridium SBD: https://www.groundcontrol.com/products/iridium/short-burst-data-range/short-burst-data-plans/ ,
  https://www.satmodo.com/iridium-short-burst-data-service-0-kb-included-2-99-kb-22-99-month ,
  https://canadasatellite.ca/Iridium-Short-Burst-Data-SBD-30kb-Bundle-Plan-p/Iridium-SBD-30kb-Bundle-Plan.htm ,
  https://satellitephonestore.com/service/iridium/sbd
- Iridium IMT: https://groundcontrol.com/products/iridium-messaging-transport-imt-pricing/
- Iridium Certus 100: https://www.groundcontrol.com/products/iridium/iridium-certus-100-range/iridium-certus-100-plans/ ,
  https://www.mailasail.com/Communication/Iridium-Certus-100-Airtime ,
  https://www.osat.com/products/iridium-certus-100-pay-monthly-land-plans
- Inmarsat BGAN: https://www.airsatone.com/bgan-airtime-price-guides ,
  https://www.satmodo.com/inmarsat-bgan-m2m-global-2-mb-24-month
- Starlink marítimo: https://www.rvmobileinternet.com/starlink-introduces-185-month-personal-maritime-plan-for-cruisers-in-small-boats-but-ocean-data-jumps-3x-to-6-gb/ ,
  https://www.shipuniverse.com/tech/starlink-for-ships-2026-guide/
- Egress de nube: https://ecorpit.com/cloud-egress-fees-ai-inference-bill-comparison-2026/ ,
  https://www.cherryservers.com/blog/data-transfer-pricing
- IoT celular: https://1nce.com/en-us/1nce-connect/10-dollars-for-10-years ,
  https://docs.hologram.io/guides/troubleshooting/2023-sim-and-data-pricing , https://www.hologram.io/pricing/
- Mercado de IoT por satélite: https://satellitemarkets.com/market-trends/satellite-iot-2026-two-track-market ,
  https://telecomlead.com/satellite/satellite-iot-subscribers-to-hit-42-million-by-2030-as-market-revenue-tops-e2-billion-127985
- Compresión de correo por satélite (afirmaciones de proveedor): https://panbo.com/isatphone-pro-does-data-xgate-a-must/ ,
  https://predictwind.com/predictmail
- Datos: Loghub (https://github.com/logpai/loghub, CC-BY-4.0), GH Archive (https://www.gharchive.org/),
  enwik8 (http://mattmahoney.net/dc/textdata.html), Silesia dickens
