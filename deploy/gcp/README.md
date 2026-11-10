# Redukt en Google Cloud (referencia de despliegue)

Despliegue de referencia, en región UE, de un compresor lossless batch/archivo con tres modos:

```
                        +-----------------------------+
   Artifact Registry <--| Cloud Build (cloudbuild.yaml)|-- Dockerfile / Dockerfile.cuda
   redukt:cpu-<sha>     +-----------------------------+
   redukt:cuda-<sha>
        |
        |   pesos (NO van en la imagen)            gs://$PROJECT-redukt-weights
        |   rwkv7-0.1b/  mamba-130m/  <gpu>/model.rbin   SHA256SUMS
        v
 (1) Cloud Run  redukt-api      POST /v1/compress | /v1/decompress | GET /v1/status   (cloudrun-service.yaml)
 (2) Cloud Batch  1 tarea = 1 VM grande    redukt compress|decompress --input-dir --output-dir --jobs N   (batch-job.json)
        gs://$PROJECT-redukt-in  --FUSE-->  /mnt/in   ->  /mnt/out  --FUSE-->  gs://$PROJECT-redukt-out
 (3) GCE + GPU (L4 g2-standard-4 / T4 n1-standard-4)   redukt --backbone plugin   (gce-cuda.sh, Dockerfile.cuda)
```

## 0. Lo que el código hace HOY (condiciona todo el despliegue)

Leído de `src/` el 2026-11-09. Son huecos del producto, no del despliegue; el despliegue está diseñado para no
depender de ellos y se actualizará cuando se cierren.

| Hallazgo | Dónde | Consecuencia en este despliegue |
|---|---|---|
| La ruta de pesos es **relativa al CWD** (`weights/mamba-130m` para Mamba, el backbone por defecto; `weights/rwkv7-0.1b` para RWKV, que sí admite `--weights`). No hay variable de entorno. | `model_loader.rs`, `api.rs`, `codec.rs` | La imagen trabaja en `/app` con el symlink `/app/weights -> /weights`; `entrypoint.sh` lo reapunta si `REDUKT_WEIGHTS_DIR` cambia. |
| ~~`redukt api` solo RWKV~~ (obsoleto desde 2026-11-10): ahora carga los backbones de `--backbone` (defecto Mamba, Q8, AZTH v5) con el mismo pipeline que `compress --input`. | `api.rs` | Cloud Run sirve Mamba+CM (~0.96 BPB clase enwik8). `WEIGHTS_SET` se deriva de `REDUKT_BACKBONE`. |
| El servidor API es **monohilo** (`for stream in listener.incoming()`), sin timeouts de lectura, HTTP/1.1 `Connection: close`. | `api.rs` | `containerConcurrency: 1`. Sin livenessProbe (no puede contestar mientras comprime). |
| Endpoint de salud: **existe** `GET /v1/status` (JSON). El modelo se carga **antes** de abrir el socket. | `api.rs` | `startupProbe` HTTP en `/v1/status`; el `HEALTHCHECK` curl del `Dockerfile` raíz no sirve (curl no está instalado allí). |
| El modo batch `--input-dir` exige **`--output-dir`** (no hay otra forma de salida); escribe `<nombre>.azth`, **no recursivo**, salta `*.azth`. | `codec.rs` `batch_compress` | Salida: `gs://OUT/<nombre>.azth`. Directorios planos. |
| ~~El modo batch es solo RWKV, sin preprocesado~~ (obsoleto desde 2026-11-10: batch lockstep con Mamba, preprocesado y `--backbone/--quant`). Sigue sin haber sharding entre procesos (cada proceso trataría **todos** los ficheros). | `lockstep.rs` | `entrypoint.sh batch-compress` / `batch-decompress` genera un directorio de symlinks por tarea (reparto por tamaño, round-robin por `BATCH_TASK_INDEX/COUNT`) y salta lo ya comprimido (reanudable tras preempción Spot). |
| El plugin se busca **junto al ejecutable** (`libredukt_gpu.so`), sin env var ni flag. Se activa con `--backbone plugin`; `--weights` = directorio con `model.rbin`. `--gpu` es un stub que devuelve error. | `loader.rs`, `codec.rs` | `Dockerfile.cuda` instala ambos en `/usr/local/bin`. |
| `PluginTokenizer` es identidad byte-level (vocab 256). Un `model.rbin` convertido desde **mamba-130m (BPE 50280)** no es coherente con él. | `plugin_backend/model.rs`, `plugins/redukt-gpu-cuda/README.md` | El set `gpu` apunta a un `.rbin` byte-level (MambaByte, p.ej. `mambabyte-353m`, +37% BPB según R-088). GPU "útil" depende de entrenar MambaByte (R-084). |
| `.cargo/config.toml` fija `target-cpu=native`. | `.cargo/` | Excluido del contexto Docker (`.dockerignore`) y `RUSTFLAGS` explícito en la imagen. |
| MSRV real: **Rust >= 1.89** (feature `avxvnni`). `Dockerfile` raíz usa `rust:1.82` y **no compila**. | medido: 1.88 falla, 1.89 compila | Imágenes nuevas usan `rust:1.98`. |
| El gobernador de hardware (RAM/batería) es no-op en Linux. | `ram_monitor.rs`, `governor.rs` | No hace falta `--no-governor` en contenedores. |
| `maxperf` usa `panic = "abort"`; el batch hace `h.join()` esperando poder sobrevivir a un panic de hilo. | `Cargo.toml`, `codec.rs` | Un panic en un fichero aborta toda la tarea; los reintentos de Batch + "saltar existentes" lo cubren. |

> **Actualización 2026-11-10.** `--jobs N` ahora significa **N streams lockstep** (cada paso avanza un token en cada
> archivo compartiendo la lectura de pesos), ya no hilos independientes; el batch antiguo (solo RWKV, sin preprocesado)
> se eliminó y existe `decompress --input-dir`. En `c4-highcpu-32` usar **N = 16-32**; con N >= 16 conviene **F32**
> (mismo throughput agregado que Q8, 322 vs 315 tok/s en el portátil de desarrollo, y mejor BPB). `REDUKT_THREADS`
> fija el tamaño del pool de hilos. Las filas de la tabla anterior sobre "batch solo RWKV" y "API solo RWKV", el `Dockerfile` raíz y el
> `PluginTokenizer` byte-level describen el estado del 2026-11-09: el `Dockerfile` raíz se eliminó y el plugin ya
> admite `.rbin` v2 con tokenizer BPE (ver más abajo).

## 1. Archivos

```
.dockerignore                 (raíz) contexto de build mínimo
deploy/gcp/
  Dockerfile                  imagen CPU (maxperf, x86-64-v3)
  Dockerfile.cuda             imagen CPU + libredukt_gpu.so (sm_75 + sm_89)
  entrypoint.sh               bootstrap de pesos, modos api / batch-compress / batch-decompress / fetch-weights / passthrough
  cloudbuild.yaml             Cloud Build (E2_HIGHCPU_32), CPU siempre, CUDA con _BUILD_CUDA=true
  cloudrun-service.yaml       servicio Knative redukt-api (4 vCPU / 2 GiB, Mamba Q8, GCS FUSE en /weights)
  batch-job.json              job de Cloud Batch (plantilla; SPOT con PROVISIONING_MODEL=SPOT)
  gce-cuda.sh                 VM con L4 (g2-standard-4) o T4 (n1-standard-4) + Docker + NVIDIA Container Toolkit
  deploy.sh                   wrapper idempotente: setup | weights | build | run | batch | all
  .gcloudignore               filtro de subida para `gcloud builds submit` (no lee .dockerignore)
```

## 2. Requisitos

- `gcloud` (verificado el formato de flags con SDK 548.0.0), proyecto con facturación, Docker local solo para probar.
- Pesos locales en `./weights` (`scripts/download-weights.sh`, `scripts/download-mamba-weights.sh`).
  `weights/MANIFEST.md` documenta la procedencia (RWKV-7 0.1B Apache-2.0; Mamba-130m Apache-2.0).
- Permisos del usuario que despliega: Cloud Run Developer, Batch Job Editor, Service Account User sobre
  `redukt-runtime`, Cloud Build Editor, Artifact Registry Admin, Storage Admin (roles citados en las páginas de Batch y Cloud Run).

## 3. Pasos

```bash
export PROJECT_ID=mi-proyecto        # y gcloud auth login
export REGION=europe-southwest1      # Madrid (residencia UE); ver sección Variables
deploy/gcp/deploy.sh setup           # APIs, Artifact Registry, 3 buckets, SA redukt-runtime + IAM
deploy/gcp/deploy.sh weights         # gcloud storage rsync ./weights + SHA256SUMS
deploy/gcp/deploy.sh build           # Cloud Build -> redukt:cpu-<sha> y :latest
deploy/gcp/deploy.sh run             # Cloud Run redukt-api

# Compresión masiva (lockstep: 16 streams por VM; F32 recomendado desde N=16)
gcloud storage cp ./corpus/* gs://$PROJECT_ID-redukt-in/
TASK_COUNT=4 PROVISIONING_MODEL=SPOT REDUKT_QUANT=f32 JOBS=16 deploy/gcp/deploy.sh batch
gcloud storage ls gs://$PROJECT_ID-redukt-out/                  # <nombre>.azth

# Descompresión masiva (perfil y backbone salen de la cabecera de cada .azth)
BATCH_MODE=decompress IN_BUCKET=$PROJECT_ID-redukt-out OUT_BUCKET=$PROJECT_ID-redukt-restored deploy/gcp/deploy.sh batch

# GPU (opcional)
BUILD_CUDA=true deploy/gcp/deploy.sh build                      # añade redukt:cuda-<sha> y :cuda-latest
python scripts/convert-weights-gpu.py --input weights/mambabyte-353m/model.safetensors \
                                      --output weights/mambabyte-353m/model.rbin
deploy/gcp/deploy.sh weights
GPU=l4 deploy/gcp/gce-cuda.sh            # SPOT=1 para Spot; GPU=t4 para T4
```

`DRY_RUN=1 deploy/gcp/deploy.sh <paso>` imprime los comandos mutantes sin ejecutarlos.
`deploy/gcp/deploy.sh render batch-job.json` muestra la plantilla con las variables sustituidas.

Prueba local de la imagen (sin GCP):

```bash
docker build -f deploy/gcp/Dockerfile -t redukt:cpu .
docker run --rm redukt:cpu --help
docker run --rm -v "$PWD/weights:/weights:ro" -v "$PWD:/work" --entrypoint redukt-entrypoint.sh redukt:cpu \
  compress --input /work/x.txt --output /work/x.azth
docker run --rm -p 8080:8080 -v "$PWD/weights:/weights:ro" --entrypoint redukt-entrypoint.sh redukt:cpu api
# variantes: -e REDUKT_BACKBONE=mamba,rwkv  -e REDUKT_QUANT=f32
```

## 4. Variables

| Variable | Defecto | Uso |
|---|---|---|
| `PROJECT_ID` | proyecto activo de gcloud | |
| `REGION` | `europe-southwest1` (Madrid) | Alternativas UE: `europe-west1` (Bélgica), `europe-west4` (Países Bajos), `europe-west3` (Fráncfort), `europe-west9` (París), `europe-west8` (Milán), `europe-west6` (Zúrich). Cloud Build y Batch listan `europe-southwest1` como disponible. |
| `BUCKET_LOCATION` | `$REGION` | o multi-región `EU` |
| `WEIGHTS_BUCKET` / `IN_BUCKET` / `OUT_BUCKET` | `$PROJECT_ID-redukt-{weights,in,out}` | `IN_BUCKET` admite `bucket/prefijo` (directorio plano). |
| `BUILD_CUDA` | `false` | construye `Dockerfile.cuda` |
| `MACHINE_TYPE` / `BOOT_DISK_TYPE` | Madrid: `c4-highcpu-32` / `hyperdisk-balanced`; resto: `c3-highcpu-22` / `pd-balanced` | Ver abajo. |
| `TASK_COUNT`, `PARALLELISM` | 1, `TASK_COUNT` | 1 tarea = 1 VM (`taskCountPerNode: 1`) |
| `BATCH_MODE` | `compress` | `compress` o `decompress` (contenedor `batch-<modo>`) |
| `JOBS` | `16` | `--jobs` = **streams lockstep** por tarea (no hilos). 16-32 en c4-highcpu-32 |
| `REDUKT_BACKBONE` | `mamba` | batch: un solo backbone; api: también lista (`mamba,rwkv`) |
| `REDUKT_QUANT` | `auto` | `auto` = Q8 para Mamba (AZTH v5); `f32` (v4) recomendado con N >= 16. Decompress lo ignora |
| `REDUKT_THREADS` | todos los CPUs | hilos GEMV/SSM (no es `--jobs`) |
| `PROVISIONING_MODEL` | `STANDARD` | `SPOT` (reintenta con código 50001) |
| `WEIGHTS_SET` | derivado de `REDUKT_BACKBONE` (`mamba`; `rwkv` solo si se pide; decompress: `mamba,rwkv`) | `rwkv`, `mamba`, `gpu`, o combinados con coma |
| `REDUKT_EXTRA_ARGS` | vacío | flags extra para `redukt` (sin comillas ni espacios dobles; va a JSON). Para quant/backbone usar `REDUKT_QUANT`/`REDUKT_BACKBONE` |
| `GPU`, `ZONE`, `SPOT` (gce-cuda.sh) | `l4`, `europe-west4-a`, `0` | L4/T4 **no existen en Madrid** (ver abajo) |
| En el contenedor | | `REDUKT_WEIGHTS_DIR=/weights`, `WEIGHTS_BUCKET`, `WEIGHTS_PREFIX`, `WEIGHTS_TARBALL_URL`, `WEIGHTS_ACCESS_TOKEN`, `GCS_ENDPOINT`, `VERIFY_WEIGHTS=1`, `IN_DIR`, `OUT_DIR`, `SKIP_EXISTING`, `SHARD_COUNT/INDEX` (los de Batch tienen prioridad) |

Disponibilidad comprobada con `gcloud compute machine-types|accelerator-types list` el 2026-11-09 (puede cambiar):

- Madrid (`europe-southwest1-a/b/c`): hay `c4-highcpu-*` (p.ej. 32 vCPU / 64 GB), **no hay C3** ni G2. `c3-highcpu-22`
  (22 vCPU / 44 GB) sí en europe-west1, west3, west4 (y west2-b).
- L4: europe-west1-b/c, west2-a/b, west3-a/b, west4-a/b/c, west6-b/c. T4: europe-west1-b/c/d, west2-a/b, west3-b,
  west4-a/b/c, central2-b/c. Ninguno en Madrid: el modo GPU sale de España (sigue siendo UE). T4 llega a fin de soporte el 2027-08-01.

## 5. Los tres modos

### 5.1 Cloud Run (`cloudrun-service.yaml`)

- 4 vCPU / **2 GiB** (mínimo documentado para 4 vCPU: 2-16 GiB), CPU siempre asignada (`run.googleapis.com/cpu-throttling: 'false'`),
  `containerConcurrency: 1`, timeout 3600 s (máximo), `minScale 0` (poner `1` para evitar cold start), `startupProbe` HTTP `/v1/status`
  (60 x 10 s = 600 s, el máximo documentado: `failureThreshold x periodSeconds <= 600`; `timeoutSeconds <= periodSeconds`).
- Memoria: Mamba Q8 ~160 MB residentes; el safetensors F32 (~520 MB) se lee durante la carga, pico ~0.7 GB. Con `REDUKT_BACKBONE=mamba,rwkv`
  los modelos se cargan en secuencia. Si en vez de FUSE se descargan los pesos a `/tmp` (memoria), ~0.9 GB más: usar `4Gi`.
- Arranque: `redukt api --bind 0.0.0.0 --port $PORT --backbone $REDUKT_BACKBONE --quant $REDUKT_QUANT`. Flags del producto: `--bind` (alias `--host`,
  defecto 0.0.0.0), `--port` (8080), `--weights` (solo RWKV), `--backbone rwkv|mamba|mambabyte|plugin|auto|lista` (auto = Mamba),
  `--quant f32|q8|auto` (auto = Q8 en Mamba). Los modelos se cargan **antes** de abrir el socket.
- `GET /v1/status` devuelve `{"status":"ok","version","backbone","profile","vocab_size","backbones":[{"backbone","profile","vocab_size","format_write"}],
  "formats_read":["v2","v3","v4","v5"],"threads","max_body"}`; `max_body` es 268435456 (256 MiB, fijo).
- Limitaciones de la API: monohilo (una petición a la vez); solo sirve los backbones/perfiles **cargados** (para aceptar archivos F32 y Q8
  a la vez hacen falta dos servicios/instancias, o `--backbone mamba,rwkv` para routing por contenido); rechaza `Transfer-Encoding: chunked`,
  cuerpos > 256 MiB y archivos con emb-surgery.
- Pesos: volumen GCS FUSE `gcsfuse.run.googleapis.com` solo lectura en `/weights`, `mountOptions: uid=10001,gid=10001`.
  Alternativa sin FUSE: comentar el volumen y definir `WEIGHTS_BUCKET`; el entrypoint descarga con `curl` + token del
  servidor de metadatos (o `gcloud storage cp` si `gcloud` existe en la imagen). Alternativa URL firmada: `WEIGHTS_TARBALL_URL`
  (un `.tar`/`.tar.gz`/`.tgz` con `mamba-130m/...`).
- Límites de plataforma (página de cuotas): HTTP/1 limita petición y respuesta a **32 MiB**, y 60 min por petición. Con Mamba Q8 (~111-127 tok/s, ~3.8 bytes/token)
  una petición de 60 min procesa del orden de unos pocos MB. Cloud Run sirve para tamaños pequeños/demostración; **el volumen va por Cloud Batch**.
- Privado por defecto (requiere identidad IAM). `ingress: all` se puede restringir en el YAML.

### 5.2 Cloud Batch (`batch-job.json`)

- Un `taskGroup`, `taskCountPerNode: 1`: **una VM por tarea y un único proceso `redukt` con `--jobs N` por VM**. `--jobs N` = **N streams lockstep**:
  cada paso avanza un token en cada archivo y los N streams comparten la lectura de pesos (GEMV -> GEMM). La salida por archivo es idéntica byte a byte
  a `compress --input`. Los hilos de cálculo los fija `REDUKT_THREADS` (defecto: todos los CPUs), no `--jobs`. Memoria ~6 MB por stream (Mamba).
  El modelo se carga una vez por VM; no hay motivo para muchas VMs pequeñas.
- N recomendado: **16-32 en c4-highcpu-32** (defecto 16, variable `JOBS`). Con N >= 16 usar `REDUKT_QUANT=f32`: mismo throughput agregado que Q8 y ~1% mejor BPB
  (cabecera AZTH v4); con N pequeño (1-8) Q8 es claramente más rápido (AZTH v5). `decompress` toma backbone y perfil de la cabecera de cada archivo.
- Descompresión batch: `BATCH_MODE=decompress` (contenedor `batch-decompress`, `redukt decompress --input-dir --output-dir --jobs N`): lee `*.azth` de `IN_BUCKET`
  y escribe `<nombre>` en `OUT_BUCKET`, con CRC-32 verificado. Mismo sharding y reanudación (omite `<nombre>` ya presente). Pide por defecto los pesos
  `mamba,rwkv` porque las cabeceras pueden nombrar cualquiera de los dos. No admite archivos v2/token-count ni adaptive-skip (usar `decompress --input`).
- Máquina: `c4-highcpu-32` (Madrid) o `c3-highcpu-22`. Motivo: DDR5 con mucho ancho de banda por VM (el cuello de botella de
  Redukt es DRAM, no cómputo). C3 es "4ª gen Xeon Scalable", C4 "Emerald/Granite Rapids" (documentación GCP). Las Xeon de 4ª-6ª gen tienen
  AVX-512 (dato de Intel; la doc de GCP consultada no lo menciona), pero **Redukt no usa AVX-512** (solo AVX2 + AVX-VNNI con detección en runtime) y el binario va compilado a x86-64-v3, así que
  corre en el **mismo nivel AVX2** que un encoder en N2/E2: no se rompe la compatibilidad de decodificación. C4 solo admite Hyperdisk
  (por eso `hyperdisk-balanced` como disco de arranque).
- Entrada/salida: tres volúmenes GCS (`gcs.remotePath`) montados bajo `/mnt/disks/...` (exigido por Batch) y enlazados al
  contenedor con `container.volumes`. Comando efectivo: `redukt compress --input-dir <shard> --output-dir /mnt/out --jobs N --backbone mamba --quant $REDUKT_QUANT`.
  Los ficheros de salida son `/mnt/out/<nombre>.azth`.
- Sharding: `redukt` no lo implementa; `entrypoint.sh batch-compress|batch-decompress` reparte los ficheros de `/mnt/in` (orden por tamaño desc,
  round-robin por `BATCH_TASK_INDEX` / `BATCH_TASK_COUNT`) a un directorio de symlinks. Reanudable: omite los `<nombre>.azth` ya presentes.
- Spot: `PROVISIONING_MODEL=SPOT` + `maxRetryCount: 3` + `lifecyclePolicies: RETRY_TASK` en código de salida 50001 (preempción, según la página de reintentos).
- El contenedor corre como root (`options: --user=0`) en Batch: no está documentado qué UID/permisos da Batch a los montajes
  GCS, y root evita fallos de escritura en `/mnt/out`. (NO VERIFICADO; alternativa: `mountOptions` de gcsfuse `uid=`/`gid=`/`file-mode`.)

#### Dimensionamiento orientativo (Mamba, lockstep)

Throughput agregado en tokens/s, **medido en un portátil i5-1235U (2P+8E, AVX2, DDR4)**; ~3.8 bytes/token, así que bytes/s ~ 3.8 x tok/s:

| Streams N | Q8 (v5) | F32 (v4) |
|---|---|---|
| 1 | 111 | 58 |
| 8 | 266 | 207 |
| 32 | 315 | 322 |

Regla: tok/s agregado ~ min(N x tok/s de un stream, techo de cómputo/DRAM de la máquina). Hacia N ~ 16-32 domina el techo y Q8 y F32 empatan
(de ahí F32 con N >= 16). Tiempo de una tarea ~ bytes totales / (3.8 x tok/s). **En c4-highcpu-32 (más cores, AVX-512 en hardware pero no usado por Redukt)
se espera más throughput que en el portátil, pero NO está medido**: medir con un lote piloto antes de dimensionar `TASK_COUNT`.

### 5.3 GCE con GPU (`gce-cuda.sh`, `Dockerfile.cuda`)

- G2 (`g2-standard-4`): la L4 se adjunta automáticamente, sin `--accelerator`. N1 + T4: `--accelerator type=nvidia-tesla-t4,count=1`.
  Ambos con `--maintenance-policy TERMINATE` (las VMs con GPU no migran en caliente). Imagen: Deep Learning VM
  `common-cu129-ubuntu-2204-nvidia-580` del proyecto `deeplearning-platform-release` (CUDA 12.9, driver 580; el contenedor CUDA 12.4 funciona con un driver más nuevo).
- El script de arranque (idempotente) instala el driver con `cuda_installer.pyz` si falta, Docker CE (repositorio apt oficial),
  NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`), autentica en Artifact Registry, descarga pesos y deja el wrapper
  `redukt-gpu`:

  ```bash
  redukt-gpu compress --backbone plugin --weights /weights/mambabyte-353m --input /data/f --output /data/f.azth
  redukt-gpu decompress --input /data/f.azth --output /data/f.out --weights /weights/mambabyte-353m
  ```
- Activación del plugin (de `loader.rs`): `libredukt_gpu.so` junto al ejecutable `redukt` (ambos en `/usr/local/bin`), `--backbone plugin`,
  `--weights` = directorio con `model.rbin` (formato "RGPU" v1, ver `plugins/redukt-gpu-cuda/README.md`).
- Compilación: `nvcc -shared -O3 --generate-code=arch=compute_75,code=sm_75 --generate-code=arch=compute_89,code=sm_89
  --generate-code=arch=compute_89,code=compute_89 --compiler-options=-fPIC,-Wall` (T4 = cc 7.5, L4 = cc 8.9). El Makefile del plugin acepta una sola arquitectura, por eso se invoca nvcc directamente.
  `--cudart` por defecto es `static`, así que la imagen puede usar la variante `--build-arg RUNTIME_FLAVOR=base` (362 MB frente a 3.79 GB con `runtime`; construye y arranca, sin GPU no se pudo probar la inferencia).
- Coste de oportunidad: con la proyección del README del plugin (~500 tok/s byte-level en T4, **no medida** en GPU real) 1 GB tardaría ~556 h de GPU
  (L4 spot a $0.17-0.42/h: ~$94-233) frente a ~$47 por 1 GB en un stream de CPU spot a 130 B/s. El GPU solo gana con batching multi-stream o un modelo mejor.

### Pesos para el plugin CUDA (.rbin v2)
python scripts/convert-weights-gpu.py --input weights/mamba-130m/model.safetensors \
    --output weights/mamba-130m/model.rbin        # v2, tokenizer=bpe (auto por vocab 50280)
- La cabecera v2 declara el tokenizer: vocab 256 -> byte-level; 50280 -> BPE GPT-NeoX.
  El conversor copia tokenizer.json junto al .rbin (fallback: weights/mamba-130m).
- Montar ese directorio y usar: compress --backbone plugin --weights /weights/mamba-130m
  (decompress detecta el backbone por la cabecera AZTH, pero también necesita --weights).
- libredukt_gpu.so debe estar junto al binario redukt (/usr/local/bin en la imagen).
- Batch multi-stream (FFI v2, docs/api/PLUGIN-FFI-V2.md): el plugin expone
  redukt_backend_forward_batch; el host lo detecta solo si existe (v1 sin cambios).
- Determinismo: descomprimir con el mismo build del plugin y la misma arquitectura de GPU
  (T4 = sm_75 o L4 = sm_89); GPU y CPU no son intercambiables bit a bit. Sin validar aún en GPU.

## 6. Costes orientativos (us-east, 2026-11; cifras facilitadas, el usuario debe verificarlas en su región)

| Concepto | Cifra |
|---|---|
| vCPU spot | ~$0.022/vCPU-h (referencia AWS c7g/c7a/m7i). Equivalente GCP: c3/c4 spot; **verifica el precio en tu región** (no consultado) |
| GPU T4 spot | ~$0.13-0.25/h |
| GPU L4 spot | ~$0.17-0.42/h |
| Almacenamiento | S3 Standard $0.023/GB-mes; GCS Standard $0.020/GB-mes |
| Egress GCP | $0.085-0.12/GB |

Cálculo de referencia: a **130 B/s por stream/vCPU**, 1 GB = 7.7 x 10^6 s = 2 137 vCPU-h, es decir ~$47 de cómputo spot.
Las optimizaciones (Q8, lockstep multi-stream; ver tabla de dimensionamiento) apuntan a 20-40x: ~$1.2-2.4/GB. El egress entre buckets y VMs de la misma región
no se cobra como egress a Internet (verificar), pero descargar los `.azth` fuera de GCP sí. La salida comprimida reduce el coste de almacenamiento/egress
de forma proporcional a la ratio (p.ej. 0.2-0.4 en texto/logs/XML estructurado).

## 7. Determinismo

La descompresión exige **mismo modelo, mismos pesos y mismo nivel SIMD** que la compresión (CLAUDE.md). Aquí:

- Binario compilado con `-C target-cpu=x86-64-v3` (AVX2 + FMA + BMI2), el mismo nivel que el encoder x86 AVX2 de referencia. **No** `target-cpu=native`
  (ataría el binario a la CPU del worker de Cloud Build; el `.cargo/config.toml` del repo lo fija y se excluye del contexto).
  Un host sin AVX2 falla con SIGILL en lugar de decodificar con otra ruta numérica.
- El código elige rutas con `is_x86_feature_detected!` (AVX2, AVX-VNNI); `tensor.rs` tiene un test que exige igualdad bit a bit VNNI vs escalar en el camino Q8.
  Las rutas F32/exp/softmax entre AVX2 y escalar no están verificadas aquí. Usa **la misma familia de VM** para comprimir y descomprimir hasta
  hacer un canario de ida y vuelta cruzado (p.ej. comprimir en C4, descomprimir en N2).
- **ARM (C4A, Tau T2A) NO soportado** (R-104 verificó solo x86 AVX2).
- Windows (MSVC) vs Linux (glibc 2.36 en `bookworm`, 2.35 en la imagen CUDA): las llamadas a `libm` podrían diferir en el último bit. No verificado.
- Plugin GPU: el flujo depende de logits calculados por kernels CUDA; descomprimir con el **mismo build y la misma arquitectura de GPU** (T4 o L4, no mezclar). No verificado entre sm_75 y sm_89.
- API y batch usan el mismo pipeline que `compress --input` (Mamba por defecto): la cabecera lleva backbone y perfil, y `decompress` carga `weights/mamba-130m` / `weights/rwkv7-0.1b` del CWD (`/app/weights` en la imagen).
  Por eso `batch-decompress` y el modo passthrough piden `WEIGHTS_SET=mamba,rwkv` por defecto.
- Determinismo verificado el 2026-11-10 (R-104): Windows, Linux y Kaggle producen archivos idénticos (x86 AVX2). ARM sigue sin soporte.

## 8. Cumplimiento

- Región UE por defecto (Madrid); buckets creados con `--uniform-bucket-level-access --public-access-prevention`, en la misma región; Cloud Run privado por defecto.
- Sin telemetría: el binario no hace llamadas externas (CLAUDE.md "Data Sovereignty"). El único tráfico saliente del contenedor es la descarga de pesos desde tu bucket (y los metadatos de GCP).
- Los datos no salen del proceso: compresión = transformación matemática. Cuenta de servicio dedicada `redukt-runtime` con mínimos privilegios
  (`objectViewer` en pesos/entrada, `objectUser` en salida, `batch.agentReporter`, `logging.logWriter`, `artifactregistry.reader`).
- A tener en cuenta: `redukt` imprime **nombres de fichero** (batch) e IPs/tamaños (API) en stderr, que llega a Cloud Logging. Si los nombres son datos personales,
  filtra o pseudonimiza antes de subirlos.
- GPU fuera de Madrid pero dentro de la UE (west4 Países Bajos). Para clientes UE actúas como encargado (DPA según CLAUDE.md). No hay cifrado en el producto (sin clasificación de exportación); el cifrado en reposo/CMEK es de GCS.

## 9. Validación realizada (2026-11-09, Windows 11 + Docker Desktop 28.2)

- MSRV medido: `--build-arg RUST_VERSION=1.88` falla (`avxvnni` inestable), `1.89` compila; el `Dockerfile` raíz (rust:1.82) falla igual.
- `docker build -f deploy/gcp/Dockerfile` (maxperf, lto fat, x86-64-v3): compila en ~18 s (build completo con pull ~100 s); imagen 136 MB; `docker run --rm redukt:cpu-test --help` imprime la ayuda del binario.
- Con los pesos locales montados: `compress` (Mamba auto) y `decompress` por el entrypoint con ida y vuelta idéntica; `batch-compress` con 2 tareas simuladas (reparto correcto, `--jobs 2`, reanudación "0 to compress, 2 already done", ida y vuelta de un `.azth` de batch idéntica);
  `api`: `/v1/status`, `POST /v1/compress` y `/v1/decompress` idénticos.
- Descarga de pesos contra un servidor HTTP simulado: ruta bucket (curl + SHA256SUMS correctos), ruta `WEIGHTS_TARBALL_URL` (.tar.gz con query string), fallo con checksum erróneo, error claro sin origen.
- `docker build -f deploy/gcp/Dockerfile.cuda` (nvidia/cuda 12.4.1 devel + rustup 1.98.1 + nvcc, sin GPU): OK en 1 min 23 s con la base `devel` ya descargada (la descarga de la base tardó más de 3 min); imagen 3.79 GB con `runtime`, 362 MB con `RUNTIME_FLAVOR=base`; `--help` funciona en ambas; `cuobjdump` confirma cubin sm_75 y sm_89 y PTX compute_89; `ldd libredukt_gpu.so` solo muestra libc. Sin GPU no se probó el plugin en ejecución.
- `bash -n` en los tres `.sh` (y en el script de arranque embebido); `yaml.safe_load` y `json.load` en las plantillas ya renderizadas; `gcloud run services replace --dry-run` llegó a la API pero el proyecto de prueba no tenía Cloud Run habilitado: **no valida el esquema en servidor**.
- No se creó ningún recurso en GCP.

## 10. Fuentes oficiales (consultadas el 2026-11-09)

Cloud Run
- https://docs.cloud.google.com/run/docs/reference/yaml/v1  (campos del Service, anotaciones: `cpu-throttling`, `minScale/maxScale`, `startup-cpu-boost`, `execution-environment`, `ingress`)
- https://docs.cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts  (`gcsfuse.run.googleapis.com`, `volumeAttributes.bucketName`, `mountOptions`, `uid/gid`, gen2, roles)
- https://docs.cloud.google.com/run/docs/configuring/healthchecks  (startupProbe: initialDelay/timeout/period <= 600 s, failureThreshold x periodSeconds <= 600 s, timeout <= period; reverificado 2026-11-10)
- https://docs.cloud.google.com/run/docs/configuring/cpu-allocation  (`run.googleapis.com/cpu-throttling: 'false'`)
- https://docs.cloud.google.com/run/docs/configuring/concurrency  (`containerConcurrency`, defecto 80)
- https://docs.cloud.google.com/run/quotas  (32 MiB HTTP/1, 60 min)
- https://docs.cloud.google.com/run/docs/configuring/services/memory-limits  (4 vCPU requieren 2-16 GiB; escribir ficheros usa memoria de la instancia)

Cloud Batch
- https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs  (taskGroups, runnables.container, volumes.gcs.remotePath, allocationPolicy, provisioningModel SPOT, lifecyclePolicies)
- https://docs.cloud.google.com/batch/docs/create-run-job-storage  (`remotePath`, `mountPath` bajo `/mnt/disks/`)
- https://docs.cloud.google.com/batch/docs/automate-task-retries  (código 50001, RETRY_TASK)
- https://docs.cloud.google.com/batch/docs/get-started  (APIs y roles: `batch.agentReporter`, `logging.logWriter`)
- https://docs.cloud.google.com/batch/docs/create-run-basic-job  (variables `BATCH_TASK_INDEX/COUNT`, ejemplo con contenedor)
- https://docs.cloud.google.com/batch/docs/locations  (europe-southwest1 disponible)

Cloud Build / Artifact Registry
- https://docs.cloud.google.com/build/docs/build-config-file-schema  (steps, images, substitutions, options)
- https://docs.cloud.google.com/build/docs/configuring-builds/substitute-variable-values  (`SHORT_SHA` vacío en builds manuales, `$$`, dynamicSubstitutions)
- https://docs.cloud.google.com/build/docs/api/reference/rest/v1/projects.builds  (enum `MachineType`: `E2_HIGHCPU_32` válido; `LoggingMode`)
- https://docs.cloud.google.com/build/docs/optimize-builds/increase-vcpu-for-builds
- https://docs.cloud.google.com/build/docs/locations
- https://docs.cloud.google.com/artifact-registry/docs/docker/authentication  (`gcloud auth configure-docker`)

Compute Engine / GPU
- https://docs.cloud.google.com/compute/docs/gpus  (G2 con L4 automática; N1 con T4; T4 fin de soporte 2027-08-01)
- https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series  (`--maintenance-policy=TERMINATE`, `--provisioning-model=SPOT`, DLVM admitido)
- https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose  (`--accelerator type=nvidia-tesla-t4,count=1`)
- https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu  (`cuda_installer.pyz`)
- https://docs.cloud.google.com/compute/docs/general-purpose-machines  (C4 solo Hyperdisk; C3/C4 plataformas)
- https://docs.cloud.google.com/deep-learning-vm/docs/images  (proyecto `deeplearning-platform-release`, familias `common-cu129-ubuntu-2204-nvidia-580`)
- `gcloud ... --help` del SDK 548.0.0 instalado (flags de `run services replace --dry-run`, `builds submit --ignore-file`, `compute instances create`, `storage rsync`, etc.) y `gcloud compute machine-types|accelerator-types list` (disponibilidad por zona).

NVIDIA / Docker / Rust
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/  (`--generate-code`, `--cudart` estático por defecto)
- https://developer.nvidia.com/cuda-gpus  (T4 = 7.5, L4 = 8.9)
- https://hub.docker.com/r/nvidia/cuda  (tags `12.4.1-{devel,runtime,base}-ubuntu22.04`: existencia comprobada con `docker manifest inspect`; también existen 12.6.3 devel/runtime y 12.9.1 devel)
- https://hub.docker.com/_/rust  (tag `1.98-bookworm`: existencia comprobada con `docker pull`, rustc 1.98.1)
- https://docs.docker.com/engine/install/ubuntu/

### NO VERIFICADO en fuente oficial (no inventado; revisar antes de producción)

1. Permisos/UID de los montajes GCS de Batch para contenedores no root, y sintaxis de `mountOptions` de gcsfuse en Batch (la página no la documenta): se usa `--user=0`.
2. Que `cpuMilli = vCPUs x 1000` y `memoryMib` ~ 87% de la RAM encajen en la VM junto al agente de Batch.
3. Que `BATCH_TASK_INDEX/COUNT` lleguen al entorno del proceso del contenedor: el ejemplo oficial los expande con un shell dentro del contenedor; no hay frase explícita. Si faltan, el entrypoint usa `SHARD_COUNT/SHARD_INDEX` (la plantilla renderiza `SHARD_COUNT=$TASK_COUNT`); con `TASK_COUNT > 1` y sin ningún índice **aborta con error** en lugar de duplicar el shard 0.
4. Compatibilidad Deep Learning VM con G2: una consulta resumió que "G2 no puede usar imágenes DLVM como disco de arranque" y la página de creación G-series dice que sí están admitidas; contradictorio, no resuelto. Alternativa en el script: `IMAGE_PROJECT=ubuntu-os-cloud IMAGE_FAMILY=ubuntu-2204-lts` (instala el driver con `cuda_installer.pyz`).
5. Que las imágenes DLVM traigan Docker y NVIDIA Container Toolkit (la página no lo dice): el script los instala si faltan.
6. Spot en G2/N1+T4 y su capacidad (la página `gpus` no lo lista; la de creación G-series sí admite `--provisioning-model=SPOT`).
7. Precios GCP (vCPU spot C3/C4, GPU spot), y que el ID de rol `roles/artifactregistry.reader` sea el que pide la página de autenticación (nombra "Artifact Registry Reader" sin ID).
8. Permisos de la cuenta de Cloud Build para publicar en Artifact Registry en tu proyecto.
9. AVX-512 en C3/C4 no aparece en la documentación de GCP consultada (es irrelevante: Redukt no lo usa).
10. Tiempos reales: montaje GCS FUSE (límite documentado de 30 s para montar), carga del modelo desde FUSE en Cloud Run, memoria por hebra en `--jobs 32`.
11. Que `redukt` + `libredukt_gpu.so` funcionen en una GPU real (build verificado, ejecución no).
12. `batch-job.json` no puede llevar comentarios (JSON): su referencia es esta sección y las cabeceras de `deploy.sh`/`cloudbuild.yaml`.
