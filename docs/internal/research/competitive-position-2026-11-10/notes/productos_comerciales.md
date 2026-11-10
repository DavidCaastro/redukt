# Mercado comercial de compresión LOSSLESS "AI/neural/learned" — inventario de productos y vendors (estado: noviembre 2026)

Notas de investigación para el informe de posición competitiva de Redukt (compresor neural lossless, CPU-first,
binario único, texto/XML/logs). Las citas textuales se mantienen en el idioma original. Las fechas de cada
hecho se marcan explícitamente cuando son anteriores a 2026. Todo precio/ratio es "declarado por el vendor"
salvo que se indique lo contrario; no se ha encontrado NINGÚN benchmark independiente de ningún producto
comercial "AI compression" lossless para datos generales.

---

## P1. ¿Qué vendors reclaman "AI compression" y cuáles son realmente compresores neurales lossless (modelo probabilístico aprendido + codificador entrópico) frente a marketing (dedup, columnar, códecs clásicos seleccionados por ML)?

### Takeaway
A noviembre de 2026 no existe ningún producto comercial que venda compresión neural lossless (modelo
predictivo + codificación aritmética) para datos generales/texto/logs/XML. Todo lo que se comercializa bajo
"AI compression" es (a) codebooks/diccionarios entrenados con ML (AtomBeam), (b) optimización de encodings
columnares Parquet (Granica), (c) pipelines dedup+LZ+"predictive coding" en hardware de backup (Huawei),
(d) compresión de logs estructurada/columnar (Sasquatch, YScope/CLP, Hydrolix) o (e) códecs neurales LOSSY
de vídeo/imagen (Deep Render, WaveOne, The Compression Company en parte). Los únicos compresores neurales
lossless de texto que existen son académicos/experimentales (ts_zip, FineZip, Nacrith, NNCP, cmix).

### Cited Findings — clasificación técnica

| Vendor / producto | Qué hace realmente | ¿Neural lossless (modelo+entropía)? | Lossless | Fuente |
|---|---|---|---|---|
| AtomBeam (Neurpac / "Compaction") | Codebook de codewords de 3-10 bits que sustituyen patrones de 64-200 bits en payloads IoT; el codebook se entrena con ML offline y se instala en ambos extremos; "It is just doing a lookup" | NO (diccionario entrenado, lookup; sin entropy coder adaptativo) | Sí (declarado) | [NextPlatform 2025-01](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/), [Viasat](https://www.viasat.com/enterprise/elevate/find/solutions/compaction/) |
| Granica Crunch | "compression optimizer for columnar files, mainly Apache Parquet", reduce 15-60% "without data loss", salida compatible con lectores Parquet estándar | NO (re-encoding columnar; no se menciona modelo neural ni entropy coder propio) | Sí ("fully lossless" variantes; también lossy para imagen/LiDAR) | [docs.granica.ai FAQ](https://docs.granica.ai/faq/crunch), [docs compat](https://docs.granica.ai/crunch/compatibility), [projectn Crunch](https://projectn-dotcom.netlify.app/crunch) |
| Huawei OceanProtect (HZU/HZBC + tarjeta de compresión) | Pipeline: preprocessing → dedup inline variable-length multicapa → compresión HZBC ("fast nonlinear transformation and lightweight context prediction method") → byte-level compaction; en tarjeta hardware | PARCIAL: "lightweight context prediction" es modelado de contexto clásico tipo CM/PPM, no red neural; no hay evidencia de modelo aprendido | Sí (backup) | [Blocks & Files 2026-05-22](https://www.blocksandfiles.com/data-protection/2026/05/22/huaweis-up-to-901-compression-card/5245077) |
| Sasquatch Labs | "Patent-pending multi-layer compression" para logs/traces/metrics OTLP; agente edge; explícitamente "no ML deciding what your auditor will see" | NO (no describe el algoritmo; niega uso de ML en la ruta de datos) | Sí ("full lossless round-trip", SHA-256) | [benchmarks](https://sasquatch-2og.pages.dev/benchmarks), [markets](https://sasquatch-2og.pages.dev/markets) |
| YScope (CLP, origen Uber) | Parsing de logs en plantillas + variables + diccionarios + zstd; "IR format beats general-purpose compressors like Zstandard" | NO (estructural + zstd) | Sí | [Uber eng blog](https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp), [OSDI'21](https://usenix.org/conference/osdi21/presentation/rodrigues), [docs.yscope.com](https://docs.yscope.com/clp/main) |
| Hydrolix | Data lake de logs columnar "high-density compression", 20-50x declarado | NO (columnar) | Sí | [Hydrolix blog](https://www.hydrolix.io/blog/hydrolix-compression/) |
| Grafana Adaptive Logs / Sazabi / Auditty | ML para DECIDIR qué logs descartar/resumir (reducción de volumen, no compresión) | NO — y además LOSSY (drop/sample) | No | [Grafana 2024-09](https://grafana.com/blog/introducing-adaptive-logs), [SiliconANGLE 2026-04-08](https://siliconangle.com/2026/04/08/startup-sazabi-bets-logs-ai-agents-replace-traditional-observability-stacks/), [AWS Marketplace Auditty](https://aws.amazon.com/marketplace/pp/prodview-5h6p2zjjg74lo) |
| Deep Render (adquirida por InterDigital, 2025-11) | Códec de vídeo "end-to-end AI-based", decodificación en NPU | SÍ neural, pero LOSSY (vídeo) | No | [Pulse2 2025-11-02](https://pulse2.com/interdigital-buying-ai-startup-deep-render-to-advance-ai-native-video-compression), [StreamingLearningCenter 2025-11-03](https://streaminglearningcenter.com/articles/interdigitals-deep-render-bet-ai-native-codec-or-strategic-ip-hedge.html) |
| WaveOne (adquirida por Apple, 2023) | Códec de vídeo "content-aware" neural; "reduce the size of video files by as much as half" | SÍ neural, LOSSY | No | [AppleInsider 2023-03-27](https://appleinsider.com/articles/23/03/27/apple-acquires-ai-powered-video-compression-startup-waveone) |
| The Compression Company (UK, pre-seed 2026) | Códecs neurales para imágenes de observación terrestre a bordo del satélite; "10 to 50 times smaller" con calidad configurable | SÍ neural; MIXTO: "Lossless compression is used for high-value targets, and aggressive (lossy) compression for areas like clouds or open ocean" | Parcial (por regiones) | [Pulse2 2026-07-30](https://pulse2.com/the-compression-company-profile-joe-griffith-interview/), [SatelliteToday 2026-02-12](https://www.satellitetoday.com/imagery-and-sensing/2026/02/12/the-compression-company-announces-3-4m-pre-seed-round-led-by-long-journey/) |
| Bitmovin | Per-title encoding (ladder ABR por convex hull) + códecs HEVC/AV1; "AI" en tooling (MCP server) | NO neural en el códec; LOSSY (vídeo) | No | [SportsVideo IBC 2026-09-10](https://www.sportsvideo.org/2026/09/10/ibc-2026-bitmovin-rolls-out-encoder-version-3/), [Bitmovin per-title](https://bitmovin.com/blog/per-title-encoding-savings/) |
| CompressionX (SISP Technologies, Londres, 2025) | App de escritorio; "smart, adaptive algorithm"; 65% "without any loss"; sin descripción de ML | NO verificable; TechRadar: "claims ... remain unverified" | Sí (declarado) | [TechRadar](https://www.techradar.com/pro/this-compression-algorithm-could-change-everything-heres-what-you-need-to-know-about-compressionx), [TechRadar 2](https://www.techradar.com/pro/british-startup-claims-to-have-developed-tech-that-can-deliver-65-lossless-file-compression-but-you-will-have-to-pay-for-it) |
| Compression AI (incubadora, 2019) | Autoencoders convolucionales para imagen, "cut file sizes by up to 80 percent"; sin noticias posteriores a 2019 | SÍ neural, imagen (lossy por autoencoder) | No | [NVIDIA blog 2019-07-23](https://blogs.nvidia.com/blog/2019/07/23/life-imitating-art-ai-startup-resembles-pied-piper-in-hbos-silicon-valley) |
| ZeroPoint Technologies (Suecia) | IP hardware de compresión de memoria DRAM/caché en nanosegundos, "lossless fast compression with real-time compaction" | NO (hardware, tipo frecuencia/diccionario) | Sí | [Cision 2024-05-23](https://news.cision.com/zeropoint-technologies-ab/r/zeropoint-technologies-closes-funding-round-for-groundbreaking-hardware-accelerated-memory-compressi,c3986225), [TechRadar](https://techradar.com/pro/hardware-startup-wants-to-solve-the-multi-billion-dollar-problem-of-bandwidth-by-using-an-ancient-technique-ai-memory-compression-technique-could-save-google-microsoft-billions-but-nvidia-wont-be-happy) |
| NVIDIA nvCOMP | Librería GPU de códecs clásicos (LZ4, Snappy, zstd, deflate, GDeflate, ANS, Bitcomp, Cascaded) | NO | Sí | [developer.nvidia.com/nvcomp](https://developer.nvidia.com/nvcomp) |
| Intel QAT (QATzip / zstd plugin) | Offload hardware de deflate/zstd | NO | Sí | [Intel community](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-QuickAssist-Technology-Zstandard-Plugin-an-External/post/1509818) |
| IBM zEDC (Integrated Accelerator, z15/z16) | Deflate en el Nest Accelerator Unit del chip | NO | Sí | [IBM](https://www.ibm.com/support/z-content-solutions/compression), [SHARE blog](https://blog.share.org/Article/taking-a-peek-under-the-hood-of-compression-on-z15) |
| ts_zip (Fabrice Bellard, gratuito/experimental) | RWKV 169M v4 cuantizado 8 bits + codificación aritmética, GPU | SÍ neural lossless (texto) — NO comercial | Sí | [bellard.org/ts_zip](https://bellard.org/ts_zip/) |
| Nacrith (preprint, 2026-02) | SmolLM2-135M + predictores online + codificador aritmético 32 bits, GPU; 0.9389 bpb enwik8 | SÍ neural lossless — NO comercial | Sí | [arXiv 2602.19626](https://arxiv.org/abs/2602.19626) |
| FineZip / LLMZip (académico) | Llama-3-8B + codificación aritmética; "LLMs are still not a viable solution for large-scale text compression" | SÍ neural lossless — NO comercial | Sí | [arXiv 2409.17141](https://arxiv.org/abs/2409.17141) |

Hechos adicionales de clasificación:
- AtomBeam contrasta explícitamente su técnica con la compresión: la compresión trabaja "on an individual-only, one-at-a-time basis"; Neurpac "analyzes thousands of these little files" juntos para construir el codebook; funciona "on a 10 cent processor" porque "It is just doing a lookup" — [NextPlatform 2025-01-10](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/).
- SiliconANGLE describe Neurpac como: "finds patterns that repeat within a file and consolidates the duplicates into a single copy ... replaces those copies with placeholder data points called codewords, which are assigned randomly rather than derived from the customer's data" — [SiliconANGLE 2025-02-04](https://siliconangle.com/2025/02/04/atombeam-raises-20m-codeword-based-data-transmission-technology).
- Viasat (partner) describe Compaction: "Machine learning looks for patterns in IoT message payloads ... sends short codewords that stand in for larger patterns"; "rebuilt 100% losslessly"; latencia "microseconds" — [Viasat ELEVATE](https://www.viasat.com/enterprise/elevate/find/solutions/compaction/).
- Granica Crunch: "NOT in the read or write path", lecturas "in the exact same way they would using the vanilla S3/GCS SDK"; reducción "15-60%, depending on the structure and pre-existing compression of your columnar files"; 43% medio en TPC-DS — [docs.granica.ai FAQ](https://docs.granica.ai/faq/crunch), [overview](https://docs.granica.ai/granica-crunch-overview), [how reads work](https://docs.granica.ai/how-reads-work).
- Huawei: la ficha técnica OceanProtect 2022 ya listaba "high-performance predictive coding" entre las técnicas de reducción, junto a compresión y byte-level compaction — [Huawei data sheet (PDF)](https://carrier.huawei.cn/-/media/cnbgv2/download/products/it-new/huawei-oceanprotect-data-sheet.pdf), vía [Blocks & Files 2026-05-22](https://www.blocksandfiles.com/data-protection/2026/05/22/huaweis-up-to-901-compression-card/5245077). Huawei Kunpeng BoostKit incluye "kpszip", algoritmo lossless propio para SDS — [hikunpeng.com](https://www.hikunpeng.com/document/detail/en/kunpengboostkithistory/220RC3/sds/kpszip_hisversion_27.html).
- No se han encontrado productos de compresión neural lossless de Alibaba ni ByteDance para almacenamiento en ninguna búsqueda; sólo trabajos académicos (survey de 17 compresores neurales lossless en 28 datasets, Frontiers of Computer Science, 2025-07) — [journal.hep.com.cn](https://journal.hep.com.cn/fcs/EN/1159675516129173991), [EurekAlert](https://www.eurekalert.org/news-releases/1111971).

### Inferences
- El término "AI compression" en el mercado 2025-2026 significa casi siempre "entrenamiento offline de un diccionario/codebook" (AtomBeam), "ML para decidir qué tirar" (observabilidad) o "códec neural lossy de medios". Redukt sería, de confirmarse, el único producto comercial con modelo neural + codificación aritmética lossless para bytes generales; la categoría está vacía, no saturada.
- La ausencia de competidores directos es coherente con la conclusión económica del paper CIDR 2026 (ver P6): el coste de inferencia hace inviable la categoría a precios actuales, y los actores con capital (Deep Render, WaveOne) eligieron vídeo lossy, donde el ahorro de ancho de banda paga la inferencia.
- Huawei es el actor comercial más cercano a "modelado de contexto" en producto, pero lo vende dentro de un appliance de backup con dedup, donde el ratio 90:1 lo aporta la dedup de copias repetidas, no la compresión.

### Gaps
- Ningún vendor publica la descripción algorítmica suficiente para confirmar o refutar al 100% la ausencia de un entropy coder adaptativo (AtomBeam, Sasquatch, Huawei HZBC son patentados y opacos).
- No se encontró el "Zipline / Zippie" de compresión AI mencionado como candidato: sólo existen Zipline AI (pipelines de datos para IA, seed $7M, 2025-08, [BusinessWire](https://www.businesswire.com/news/home/20250819568349/en/)) y Project Zipline de Microsoft (estándar de compresión hardware OCP 2019, MIT, [ConvergeDigest](https://convergedigest.com/ocp-2019-microsofts-project-zipline/?amp=1)). Ninguno es compresión neural.
- "Compression.ai" como dominio/empresa actual: sólo aparece "Compression AI" (2019, imágenes, incubadora); estado 2026 desconocido.

---

## P2. Verificación de candidatos: AtomBeam, Granica, Compression.ai/CompressionX, Deep Render, Zipline/Zippie, Bitmovin, WaveOne y otros — con precios, ratios, throughput, hardware, modelo de despliegue, financiación, clientes

### Takeaway
De los candidatos, sólo AtomBeam y Granica venden reducción lossless de datos generales/tabulares con
etiqueta "AI"; ninguno publica precio por GB. Deep Render, WaveOne y Bitmovin son vídeo lossy (Deep Render y
WaveOne ya adquiridas). CompressionX es una app de consumo de origen opaco. Zipline/Zippie no existe como
compresor AI.

### Cited Findings

**AtomBeam Technologies (California; productos Neurpac, Neurpac+, Neurcom; "Compaction" es el nombre con que Viasat lo lista)**
- Fundada en 2017 por Charles Yeomans (CEO) y Asghar Riahi (salió en 2023-08); 27 patentes concedidas y 16 pendientes (2025-01) — [NextPlatform 2025-01-10](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/).
- Reducción declarada: "by an average of 75 percent in near real-time" → "average four-times increase in effective bandwidth"; en pruebas SDA Phase II SBIR, Link 16 "up to 89%" y 7-9x de ancho de banda — [NextPlatform](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/), [pulse.bot SDA](https://www.pulse.bot/spacetech/news/sda-expands-atombeam-contract-for-data-transmission-tech-78d5686f-2dd8-4d45-a514-97dc64082584).
- Financiación: equity crowdfunding (StartEngine / Reg A+), no VC: ">$15 million from more than 5,000 investors" (2025-01); ronda Reg A+ de $20M con >6,500 participantes (2025-02) — [NextPlatform](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/), [SiliconANGLE 2025-02-04](https://siliconangle.com/2025/02/04/atombeam-raises-20m-codeword-based-data-transmission-technology).
- Despliegue: Neurpac era cloud de AtomBeam ("Currently, data goes to AtomBeam's cloud"); next-gen Neurpac para nubes privadas prevista "by mid-2025" (Chevron, ExxonMobil, BMW lo pedían) — [NextPlatform](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/).
- Clientes/partners: Saab, U.S. Air Force, CrowdPoint; partners Intel, Cribl, Inmarsat, Ericsson (Cradlepoint/NetCloud) — [NextPlatform](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/); Viasat ELEVATE Solutions Marketplace desde 2025-02 — [Viasat partner page](https://www.viasat.com/enterprise/elevate/find/partners/atombeam/); contrato SDA ampliado y propuesta DARPA "Neurpac-16" con Energration (2026-06) — [TMCnet 2026-06-08](https://www.tmcnet.com/usubmit/2026/06/08/10395727.htm).
- Precio: NO público. G2: "AtomBeam has not provided pricing information for this product or service" — [G2](https://www.g2.com/products/atombeam/pricing).
- Hardware: "it will work on a 10 cent processor"; MQTT gateway; batería IoT +25-50% (Viasat) — [NextPlatform](https://www.nextplatform.com/2025/01/10/a-novel-way-to-compact-data-for-more-efficient-storage-and-transmission/), [Viasat](https://www.viasat.com/enterprise/elevate/find/solutions/compaction/).
- Formato: propietario (codebook en ambos extremos); no abierto. Mercado objetivo: IoT satelital, defensa, telemetría — NO almacenamiento de objetos ni texto/XML.

**Granica (Mountain View; Crunch, Screen, Chronicle, LTMs, Myelin)**
- Serie A $45M anunciada 2023-06-01; inversores NEA, Bain Capital Ventures, Abstract, Frederic Kerrest, Kevin Hartz, Deepak Ahuja — [StartupIntros](https://startupintros.com/news/granica-series-a-2023-06), [Employbl](https://www.employbl.com/companies/Granica/funding-rounds). La web actual dice "more than $60M raised", "Backed by NEA and Bain Capital" — [granica.ai](https://granica.ai/).
- Clientes nombrados en la web: Snap, Grab, Bigabid, Pinterest, ShareChat, Quora, Quantum Metric, 6sense; caso ShareChat "$200K/PB annualized ROI across storage and compute, 4 weeks to verified savings" — [granica.ai](https://granica.ai/).
- Ratio: docs "15-60%"; "up to 60% for at-rest storage and transfer"; TPC-DS "average 43% file-size reduction"; marketing antiguo "up to 80%", FAQ antigua "25-60%", "over $1.3M in annual savings per 10 petabytes" (asumiendo 50%) — [docs FAQ](https://docs.granica.ai/faq/crunch), [docs crunch-faq](https://docs.granica.ai/crunch-faq), [projectn Crunch](https://projectn-dotcom.netlify.app/crunch). Web actual: "Cut data lake costs by 20% to 50%" y vs Databricks Auto Loader "up to 6.0x lower cost per TB, up to 3.8x higher throughput per core" — [granica.ai](https://granica.ai/).
- Throughput (página antigua): "sustained throughput of up to 1.5 GB/s (not Gbps) per node", latencia añadida "~0-100ms" — [projectn Crunch](https://projectn-dotcom.netlify.app/crunch).
- Precio: dos modelos documentados y contradictorios. (a) "outcome-based — you pay a percentage of the savings Crunch generates ... no upfront costs" — [docs FAQ](https://docs.granica.ai/faq/crunch); (b) "licensed and priced based on the volume of compressed data stored per month, measured using uncompressed source bytes ... minimum monthly commitment which allows you to crunch up to 1PB of source data" — [docs crunch-faq](https://docs.granica.ai/crunch-faq). Tarifa $/TB NO pública (contacto sales@granica.ai). Web actual: "Outcome-based pricing tied to the value we create" — [granica.ai](https://granica.ai/).
- Despliegue: "Runs inside your cloud. Granica never needs your bytes"; AWS y GCP; S3-compatible API — [granica.ai](https://granica.ai/), [docs](https://docs.granica.ai/granica-crunch-overview). Gartner Cool Vendor 2023 — [projectn Crunch](https://projectn-dotcom.netlify.app/crunch).
- Formato: Parquet estándar (abierto); no hay formato propietario de salida — [docs compat](https://docs.granica.ai/crunch/compatibility).
- Nota: Granica ha pivotado el mensaje hacia "AI efficiency" / Large Tabular Models / Myelin (estado de agentes); Chronicle (2023-09) era SaaS de visibilidad de accesos S3/GCS con GenAI — [SiliconANGLE 2023-09-14](https://siliconangle.com/?p=630235), [granica.ai](https://granica.ai/).

**Compression.ai / CompressionX**
- "Compression AI" (2019): equipo en incubadora, autoencoders CNN para imagen, "cut file sizes by up to 80 percent", pilotos; sin información posterior — [NVIDIA blog 2019-07-23](https://blogs.nvidia.com/blog/2019/07/23/life-imitating-art-ai-startup-resembles-pied-piper-in-hbos-silicon-valley).
- CompressionX (SISP Technologies, Londres, lanzamiento 2025-07): "reduce file size by 65% without any loss"; gratis hasta 25 GB/mes, plan estándar £3.99/mes; Windows/macOS; TechRadar: "claims of smart algorithms and 65% compression remain unverified" — [TechRadar](https://www.techradar.com/pro/this-compression-algorithm-could-change-everything-heres-what-you-need-to-know-about-compressionx), [TechRadar 2](https://www.techradar.com/pro/british-startup-claims-to-have-developed-tech-that-can-deliver-65-lossless-file-compression-but-you-will-have-to-pay-for-it), [Le Monde Informatique](https://www.lemondeinformatique.fr/actualites/lire-compressionx-entend%C2%A0depoussierer-la-compression-de-donnees%C2%A0-97933.html). Un blog dice "founded in 2012", contradiciendo el lanzamiento 2025 — [continuousdataprotection 2025-09](https://continuousdataprotection.blogspot.com/2025/09/interesting-storage-footprint-reduction.html). Producto de consumo; no es neural ni enterprise.

**Deep Render (Londres) — LOSSY vídeo**
- Serie A $9M (2023-03-29) liderada por IP Group y Pentech; valoración ~$30M; grant EIC $2.7M — [TechCrunch 2023-03-29](https://techcrunch.com/2023/03/29/deep-render-lands-9-million-for-its-ai-powered-video-compression-tech/), [UKTN](https://www.uktech.news/ai/deep-render-funding-20230329). Total ~$13.7-14M — [CB Insights](https://www.cbinsights.com/company/deep-render/financials).
- Adquirida por InterDigital, anuncio 2025-11-02, precio no revelado; ingenieros a InterDigital Video Lab; patentes de AI video coding al portfolio — [Pulse2](https://pulse2.com/interdigital-buying-ai-startup-deep-render-to-advance-ai-native-video-compression).
- Decodificación en NPU ("performs well on Apple NPUs", otras NPUs "unclear"; smart TVs sin NPU "5-7 years"); superó SVT-AV1 y VVenC "in very specific use cases"; el autor consultó para Deep Render — [StreamingLearningCenter 2025-11-03](https://streaminglearningcenter.com/articles/interdigitals-deep-render-bet-ai-native-codec-or-strategic-ip-hedge.html). Sin cifras públicas de bitrate.

**WaveOne — LOSSY vídeo, adquirida por Apple (2023)**
- Revelado por LinkedIn de un ex-ejecutivo (2023-02), Apple no confirmó; web cerrada ~2023-01; códec "content-aware" en aceleradores AI de móviles; "reduce the size of video files by as much as half" — [AppleInsider 2023-03-27](https://appleinsider.com/articles/23/03/27/apple-acquires-ai-powered-video-compression-startup-waveone), [Macworld](https://www.macworld.com/article/1674030/apple-acquires-waveone-ai-video-compression.html).

**Bitmovin — LOSSY vídeo**
- Encoder v3 (IBC 2026-09): HEVC "15.5 per cent lower bitrate with per-title encoding and 17.6 per cent lower on static ladders"; per-title 4K "22.7% bitrate savings"; live per-title 7.21-13.03%; "AI" = MCP server para asistentes — [SportsVideo 2026-09-10](https://www.sportsvideo.org/2026/09/10/ibc-2026-bitmovin-rolls-out-encoder-version-3/), [Bitmovin](https://bitmovin.com/blog/per-title-encoding-savings/). No compite con compresión lossless de datos.

**Zipline / Zippie**
- No existe. Zipline AI = infraestructura de datos para IA (seed $7M, Wing VC, 2025-08-19) — [BusinessWire](https://www.businesswire.com/news/home/20250819568349/en/). Project Zipline = estándar de compresión hardware de Microsoft (OCP 2019, MIT) — [ConvergeDigest](https://convergedigest.com/ocp-2019-microsofts-project-zipline/?amp=1).

**The Compression Company (Imperial College spin-out, UK)**
- Pre-seed $3.4M (≈€2.8M) liderado por Long Journey, 2026-02-12; códecs neurales para EO satelital, "10 to 50 times smaller" vs 2-3x lossless de JPEG2000/CCSDS; lossless sólo en "high-value targets"; primer vuelo en SpaceX Transporter (inicio 2026); partners Tilebox, DPhi Space; sin precio — [SatelliteToday](https://www.satellitetoday.com/imagery-and-sensing/2026/02/12/the-compression-company-announces-3-4m-pre-seed-round-led-by-long-journey/), [EU-Startups 2026-02](https://www.eu-startups.com/2026/02/imperial-college-founders-raise-e2-8-million-for-the-compression-company-a-satellite-data-compression-platform/), [Pulse2 2026-07-30](https://pulse2.com/the-compression-company-profile-joe-griffith-interview/).

**Sasquatch Labs, Inc. (observabilidad, 2026)**
- Benchmarks propios (K8s 50K eventos): logs 100.0x vs gzip 2.1x; traces 100.0x vs 2.4x; metrics 100.0x vs 2.6x; security 150.0x vs 2.3x; en corpus públicos (mongod, spark-event-log, cockroachdb, postgresql) "102.5%" del tamaño de un "leading open-source reference" sin nombrar; "vendor-run results, not an independent certification"; harness "available on request"; sin throughput ni CPU — [benchmarks](https://sasquatch-2og.pages.dev/benchmarks). Otras páginas dicen 18.2x logs / 27.4x traces / 30.1x metrics y "15-30x", cifras inconsistentes entre páginas — [about](https://sasquatch-2og.pages.dev/about), [architecture](https://sasquatch-2og.pages.dev/architecture).
- Precio: no publicado; "80-90%" de reducción de factura; "Pay 5-10% of hot-storage cost" (Tap Out); despliegue agente edge OTLP (Helm, Docker, DEB/RPM, Terraform), BYOC; CEO Micky Multani; sin clientes nombrados — [markets](https://sasquatch-2og.pages.dev/markets).

**YScope (CLP, Toronto)**
- Seed US$3.9M liderado por Two Small Fish Ventures (2026-03-18) — [Private Capital Journal](https://privatecapitaljournal.com/yscope-secures-us-3-9m-seed-financing/). Uber: "169x compression ratio on our log data" (fase 1); pipeline completo 2.16x el ratio de zstd y 2.28x el de gzip en logs Spark — [Uber blog](https://www.uber.com/blog/reducing-logging-cost-by-two-orders-of-magnitude-using-clp); OSDI'21: CLP 32.20 vs gzip 16.38 vs Splunk 2.86 vs Elasticsearch 1.75 — [USENIX](https://usenix.org/conference/osdi21/presentation/rodrigues). Open source (formato abierto), CPU.

**Hydrolix**
- "log data compression rates typically over 90% and up to 99%", "20-50x", "52:1 in benchmarks" — [Hydrolix blog](https://www.hydrolix.io/blog/hydrolix-compression/); sin $/TB público (AWS Marketplace listing) — [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-r3vs5ghcxheje).

**ZeroPoint Technologies (Gotemburgo)**
- Serie A €5M (Matterwave Ventures, 2024-05-23); total ~$14.7M (CB Insights); IP hardware para memoria, "eliminates up to 70% of unnecessary data in microchip memory" — [Cision](https://news.cision.com/zeropoint-technologies-ab/r/zeropoint-technologies-closes-funding-round-for-groundbreaking-hardware-accelerated-memory-compressi,c3986225), [CB Insights](https://www.cbinsights.com/company/zeropoint-technologies/financials), [TechRadar](https://techradar.com/pro/hardware-startup-wants-to-solve-the-multi-billion-dollar-problem-of-bandwidth-by-using-an-ancient-technique-ai-memory-compression-technique-could-save-google-microsoft-billions-but-nvidia-wont-be-happy). No compite en objetos.

**Aceleradores hardware de códecs clásicos (compiten por el mismo presupuesto de "reducir bytes")**
- NVIDIA nvCOMP: v2.0 "Snappy and LZ4 ... up to 100 GB/s", Cascaded "up to 500 GB/s ... up to 80x on numerical data" — [NVIDIA blog v2.0.0](https://developer.nvidia.com/blog/nvcomp-v2-0-0-now-available-with-new-compressors); nvCOMP 4.2 + Blackwell Decompression Engine "up to 600 GB/s decompression" — [developer.nvidia.com/nvcomp](https://developer.nvidia.com/nvcomp); zstd ~16-19 GB/s y ANS 181-190 GB/s en Blackwell — [NVIDIA blog checkpoints](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/). Licencia propietaria gratuita; formatos abiertos (zstd/LZ4/deflate) o propios (GDeflate, Bitcomp, Cascaded, ANS).
- Intel QAT zstd plugin: "up to 3.2x the throughput of zstd" (nivel 5), "3.3x better performance per watt", "core savings of 73%", ratio Silesia QAT-ZSTD L9 2.76 vs zstd L4 2.74; sólo compresión, no descompresión — [Intel community](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-QuickAssist-Technology-Zstandard-Plugin-an-External/post/1509818); Red Hat midió pico 73 Gb/s en Silesia con 4+ procesos en Sapphire Rapids — [Red Hat](https://www.redhat.com/en/blog/red-hat-enterprise-linux-achieves-significant-performance-gains-intels-4th-generation-xeon-scalable-processors); Phoronix advierte que las comparaciones usan niveles distintos — [Phoronix](https://phoronix.com/news/Intel-QAT-Zstd).
- IBM zEDC: z15 con Integrated Accelerator "up to 17 times the total compression throughput of a z14 configured with the maximum number of zEDC Express cards"; "up to 8 times faster application elapsed time with no additional CPU time"; en el NXU de cada chip z15/z16 — [SHARE blog](https://blog.share.org/Article/taking-a-peek-under-the-hood-of-compression-on-z15), [IBM](https://www.ibm.com/support/z-content-solutions/compression), [IntelliMagic](https://www.intellimagic.com/?p=15001). Formato deflate (abierto); hardware mainframe únicamente.

### Inferences
- Los dos "AI compression" lossless enterprise reales (AtomBeam, Granica) venden sin precio público y con modelos "share of savings" o "volumen con mínimo 1 PB": el mercado acepta precios ligados a ahorro, no a $/GB fijo, y exige volúmenes de PB para hablar con ventas.
- Granica demuestra que un 15-60% de reducción adicional sobre Parquet (ya comprimido) es monetizable en data lakes de clientes grandes (Snap, Pinterest, Quora) con ROI "$200K/PB/año". Es el comparable de pricing más útil para Redukt en object storage.
- AtomBeam monetiza el ancho de banda caro (satélite/defensa), no el almacenamiento: coincide con la "vía 3" de la Commercial Target de Redukt (enlaces medidos), y con la mayor disposición a pagar por byte.

### Gaps
- Ningún $/GB o $/TB público para AtomBeam, Granica, Sasquatch, Hydrolix.
- Ningún throughput por core publicado por AtomBeam ni Sasquatch; Granica sólo "1.5 GB/s per node" (página antigua, sin fecha ni tamaño de nodo).
- No se encontró el precio de adquisición de Deep Render ni WaveOne.

---

## P3. ¿Alguien vende compresión neural lossless para TEXTO/XML/LOGS en almacenamiento de objetos caliente a un precio por GB? ¿Precio y ratio?

### Takeaway
No. Nadie vende compresión neural lossless para texto/XML/logs en object storage, ni a precio por GB ni
de otra forma. Lo más cercano son (a) Granica Crunch para Parquet (precio no público, % de ahorro o
volumen con mínimo mensual de 1 PB), (b) compresores estructurales de logs (CLP/YScope open source,
Sasquatch sin precio, Hydrolix sin precio) y (c) herramientas neurales gratuitas/experimentales con GPU
(ts_zip, Nacrith).

### Cited Findings
- Granica: pricing "percentage of the savings" o "volume of compressed data stored per month, measured using uncompressed source bytes" con "minimum monthly commitment which allows you to crunch up to 1PB"; formatos columnares, no texto plano — [docs FAQ](https://docs.granica.ai/faq/crunch), [docs crunch-faq](https://docs.granica.ai/crunch-faq).
- Sasquatch: sin precio; "Pay 5-10% of hot-storage cost and answer like hot storage" para el tier de archivo propio — [markets](https://sasquatch-2og.pages.dev/markets).
- ts_zip (gratuito, experimental): enwik8 1.106 bpb vs xz 1.989; enwik9 1.084 vs xz 1.707; "Up to 1 MB/s compression and decompression on an RTX 4090"; "A GPU is necessary for reasonable speed"; sólo texto; "no backward compatibility should be expected between versions"; ficheros fechados 2024-03-02 — [bellard.org/ts_zip](https://bellard.org/ts_zip/).
- Nacrith (preprint 2026-02, Roberto Tacconelli): enwik8 0.9389 bpb; SmolLM2-135M + codificador aritmético 32 bits; "~500 MB of GGUF weights and ~1.2 GB VRAM per worker, running on consumer GPUs"; backend llama.cpp "~7x faster single-token decoding"; cmix ~1.17, NNCP ~1.19, ts_zip ~1.11, FineZip 1.024 bpb — [arXiv 2602.19626](https://arxiv.org/abs/2602.19626). No comercial.
- FineZip (UC Berkeley/NYU, 2024-09): "54 times faster than LLMZip"; "approximately 4 hours compared to 9.5 days for LLMZip" para 10 MB; "approximately 50% over traditional algorithmic compressors"; "LLMs are still not a viable solution for large-scale text compression" — [arXiv 2409.17141](https://arxiv.org/abs/2409.17141).
- Un paper 2026 sobre predictores neurales ligeros encadenados reporta "34-40 KB/s on NVIDIA GeForce RTX 4060 Ti, which is still too slow for modern needs" — [arXiv 2604.15472](https://arxiv.org/pdf/2604.15472).
- Precio de referencia de almacenamiento usado por académicos para el breakeven: "$20 per TiB per month (Amazon S3)"; "$200k/year for 1 PiB on S3" — [slides CIDR 2026](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf).
- Archivo de e-facturas: retención 10 años en Alemania (§147 AO/GoBD), Francia (comercial), Bélgica; "the archive must retain the original structured file, not merely a visual rendering"; ninguna fuente menciona compresión como característica de producto — [Comarch](https://www.comarch.com/trade-and-services/data-management/news/archive-compliance-how-to-store-e-invoices-for-1030-years/), [VATupdate 2026-06-09](https://www.vatupdate.com/2026/06/09/archive-compliance-how-to-store-e-invoices-for-10-30-years/), [Forvis Mazars](https://www.forvismazars.com/sk/en/insights/publications-and-events/did-you-know-from-a-tax-perspective/archiving-e-invoices-in-xml-from-2027).

### Inferences
- El hueco "neural lossless para texto/XML/logs en object storage" está vacío en el lado comercial; la referencia de calidad que un comprador técnico conocerá es ts_zip (1.106 bpb enwik8, 1 MB/s en RTX 4090) y, si sigue la literatura, Nacrith (0.9389 bpb). Redukt (0.9641 bpb F32 enwik8 10 KB, CPU) se sitúa entre ambos en ratio pero muy por debajo en throughput (1.2 KB/s CPU, 4.2 KB/s T4).
- Los vendors de e-invoicing archive venden integridad/firma/retención, no reducción de bytes: un compresor con CRC, formato documentado (FORMAT-SPEC) y decodificador CPU de referencia encaja en sus requisitos de "legibilidad a 10-20 años", pero el mercado no lo pide hoy explícitamente.

### Gaps
- No existe ningún precio comercial comparable ($/GB) para compresión neural de texto; el breakeven de Redukt sólo puede compararse con costes de almacenamiento (S3 $20/TiB-mes) y con el share-of-savings de Granica (porcentaje no público).

---

## P4. Ofertas nativas de hyperscalers y vendors de almacenamiento que compiten por el mismo presupuesto

### Takeaway
Los hyperscalers no comprimen objetos en el servidor (S3, GCS, R2 almacenan lo que se sube; GCS sólo
descomprime gzip al servir); la reducción de datos "nativa" vive en arrays on-prem (Dell/NetApp/Pure/Huawei)
con garantías contractuales 2:1-6:1 basadas en dedup+LZ, y en appliances de backup (Huawei 72:1-90:1).
Para object storage, la alternativa real a un compresor es "comprimir antes de subir" (zstd) o un proxy
(Granica, MinIO S2).

### Cited Findings
- AWS S3: "Amazon S3 doesn't have native features to compress/decompress data" (AWS re:Post); alternativas: comprimir antes de subir o S3 Object Lambda — [AWS re:Post](https://repost.aws/questions/QUkg2RAWBES2Kvhi-da6P9vg/is-there-a-mechanism-for-automatically-compressing-objects-loaded-to-s3). El "Compress Objects Automatically" pertenece a CloudFront, no a S3 — [Codemia](https://codemia.io/knowledge-hub/path/gzip_compression_on_static_amazon_s3_files). S3 Tables redujo costes de compactación (Iceberg) hasta 90% (2025-07), que es compactación de ficheros, no compresión — [AWS What's New 2025-07](https://aws.amazon.com/es/about-aws/whats-new/2025/07/amazon-s3-tables-reduce-compaction-costs/). Caso de blog: pasar S3 a zstd recortó coste de almacenamiento 30% — [besthub.dev](https://besthub.dev/articles/how-switching-s3-compression-to-zstd-cut-storage-costs-by-30-cc3b2db804f0).
- Google Cloud Storage: "decompressive transcoding" sólo para objetos ya gzip con `Content-Encoding: gzip`; "Charges for downloading the object are based on its decompressed size"; "Cloud Storage does not support decompressive transcoding for Brotli-compressed objects"; "Decompressive transcoding invalidates integrity checking" — [GCS docs](https://docs.cloud.google.com/storage/docs/transcoding).
- Cloudflare R2: documentación de seguridad habla sólo de cifrado AES-256 en reposo; no se encontró ninguna función de compresión server-side — [R2 data security](https://developers.cloudflare.com/r2/reference/data-security/index.md), [How R2 works](https://developers.cloudflare.com/r2/how-r2-works/).
- MinIO: compresión transparente opcional basada en S2 (Snappy), "By default, on-disk compression is disabled"; incompatible con cifrado ("side channel attacks") — [MinIO blog](https://blog.min.io/transparent-data-compression), [MinIO AIStor docs](https://docs.min.io/enterprise/aistor-object-store/administration/objects-and-versioning/data-compression). Proyecto S4 (proxy S3 con zstd 1-22 y códecs GPU) — [docs.rs s4-codec](https://docs.rs/crate/s4-codec/0.8.9).
- Dell: PowerScale "2:1 Data Reduction Guarantee, equivalent to a minimum of 200% capacity efficiency" (2025) — [Dell blog](https://www.dell.com/en-us/blog/dell-powerscale-designed-for-efficiency-under-real-world-ai-constraints/); PowerStore 4:1 (Future-Proof Program, antiguo) — [Dell blog](https://www.dell.com/en-us/blog/dell-technologies-future-proof-program-guarantee-outcomes/); un comparador UK 2026 cita 6:1 para PowerStore Gen 3 — [servnetuk](https://www.servnetuk.com/insights/all-flash-array-shootout-2026-uk-mid-market) (fuente de terceros, sin confirmar).
- NetApp Storage Efficiency Guarantee: 4:1 SAN (FC/NVMe/iSCSI), 1.5:1 NAS, 3:1 VMware/Hyper-V/KVM sobre NAS, en ONTAP all-flash (flyer en alemán) — [NetApp flyer](https://www.netapp.com/de/media/84536-na-937-seg-flyer-aw.pdf); blog "3:1 for mixed workloads" — [NetApp blog](https://www.netapp.com/de/blog/netapp-guarantees-storage-efficiency-aff-systems/); excluye snapshots y thin provisioning del cálculo — [NetApp Epic docs](https://docs.netapp.com/us-en/ontap-apps-dbs/epic/epic-ontap-efficiency.html).
- Pure Storage (ahora "Everpure" en trackers IDC 2026): blog 2017 "average 5:1 data reduction", dedup+compresión "Always-On" — [Pure/Everpure blog 2017-09-07](https://blog.everpuredata.com/?p=37623); Right-Size Guarantee desde 2016, 30 días — [Pure blog](https://blog.purestorage.com/?p=97697). Sin ratio 2025 confirmado.
- Huawei OceanProtect (2026-05-22): tarjeta de compresión en X8100/X9100 "up to 90:1", "about 20% higher than the leading alternative", "industry's only effective capacity guarantee", offload CPU "up to 22%"; generaciones anteriores (X3000/X6000/E8000) "up to 72:1"; HZU "beats Lempel-Ziv ... by about 30%"; Blocks & Files: "too good to be true?" y recomienda probar con datos propios — [Blocks & Files](https://www.blocksandfiles.com/data-protection/2026/05/22/huaweis-up-to-901-compression-card/5245077), [TechRadar](https://techradar.com/pro/huawei-promises-up-to-90-1-compression-technology-by-using-a-proprietary-dynamic-algorithmic-technique-and-a-lot-of-ssds).
- Observabilidad (mismo presupuesto en logs): Grafana Adaptive Logs usa ML para recomendar descartar volumen en ingesta (2024-09) — [Grafana](https://grafana.com/blog/introducing-adaptive-logs); Logz.io añade agentes IA y tiered storage — [Logz.io](https://logz.io/blog/data-compression-techniques-logzio-ai-agents/); Sazabi (stealth 2026-04) decide qué datos "keep, summarize, or move to cheaper storage" — [SiliconANGLE 2026-04-08](https://siliconangle.com/2026/04/08/startup-sazabi-bets-logs-ai-agents-replace-traditional-observability-stacks/); Auditty "80-95% log volume and cost reduction" en K8s — [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-5h6p2zjjg74lo).

### Inferences
- En object storage caliente el "competidor" por defecto de Redukt no es otro producto, sino `zstd -19`/`xz` antes del PUT (coste ≈ 0) o el ahorro del 30% que ya se obtiene cambiando gzip→zstd. El argumento de venta debe ser el delta sobre zstd/brotli (−50% a −88% en logs/XML según benchmarks internos de Redukt), no sobre "sin compresión".
- Las garantías de ratio de arrays (2:1-6:1) se basan en dedup de VMs/bases de datos; no aplican a texto/XML/logs únicos, donde Redukt compite con la compresión pura del array (LZ4/zstd inline).
- En logs, el presupuesto se está yendo a "tirar datos con ML" (Grafana/Sazabi), que es lossy; un posicionamiento "lossless para auditoría/compliance" (igual que Sasquatch) es un ángulo diferenciador explícito.

### Gaps
- No se encontró documentación oficial de Azure Blob ni de Cloudflare R2 que afirme explícitamente "no comprimimos en servidor"; la ausencia de la función se infiere de la ausencia en la documentación.
- No se encontró ratio de reducción actual (2025-2026) publicado por Pure/Everpure.

---

## P5. Startups de compresión neural que cerraron o pivotaron 2023-2026 y por qué

### Takeaway
No se encontró ninguna startup de compresión neural LOSSLESS de datos que haya cerrado o pivotado en
2023-2026 — porque apenas existieron. Los movimientos documentados son adquisiciones de códecs neurales de
vídeo lossy (WaveOne→Apple 2023, Deep Render→InterDigital 2025), un pivot de mensaje de Granica hacia
"AI efficiency"/agentes, y la desaparición silenciosa de Compression AI (2019). La literatura (CIDR 2026,
FineZip) documenta la causa por la que no hay entrantes: coste de inferencia.

### Cited Findings
- WaveOne: web cerrada ~2023-01, equipo absorbido por Apple ML; terms desconocidos — [AppleInsider 2023-03-27](https://appleinsider.com/articles/23/03/27/apple-acquires-ai-powered-video-compression-startup-waveone).
- Deep Render: adquirida por InterDigital (anuncio 2025-11-02) tras ~$13.7-14M levantados; motivación del comprador: "extends InterDigital's leadership in the efficient delivery of high-quality video" y patentes — [Pulse2](https://pulse2.com/interdigital-buying-ai-startup-deep-render-to-advance-ai-native-video-compression), [CB Insights](https://www.cbinsights.com/company/deep-render/financials).
- Compression AI (imagen, incubadora, 2019): sin rastro posterior en las búsquedas — [NVIDIA blog 2019](https://blogs.nvidia.com/blog/2019/07/23/life-imitating-art-ai-startup-resembles-pied-piper-in-hbos-silicon-valley).
- Vectorly pivotó de vídeo a mobile learning en 2019 (fuera de rango) — [MIT News 2019-01](https://news.mit.edu/2019/vectorly-pivot-dotlearn-mobile-learning-0123).
- NeuroPixel.AI (cierre 2026-04) era GenAI para moda, no compresión; irrelevante salvo por el nombre — [Inc42](https://inc42.com/buzz/flipkart-backed-neuropixel-ai-shuts-down/).
- Granica: de "Crunch/Screen/Chronicle" (2023) a "Large Tabular Models" y "Myelin" para agentes (2026), manteniendo Crunch — [SiliconANGLE 2023-09-14](https://siliconangle.com/?p=630235), [granica.ai](https://granica.ai/).
- Causa estructural documentada (no de una empresa concreta): "We conclude that LLM-based compression is not yet ready to replace lightweight, CPU-based compression algorithms for hot or frequently accessed data" — [CIDR 2026 PDF](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf); "LLMs are still not a viable solution for large-scale text compression" — [FineZip](https://arxiv.org/abs/2409.17141).

### Inferences
- El capital riesgo ha financiado compresión neural sólo donde el ancho de banda es el coste dominante y la pérdida es aceptable (vídeo, satélite): ahí el GPU-hour se paga con bits de red. En lossless de datos fríos no ha habido entrantes financiados que puedan haber fracasado.
- Para Redukt, el riesgo no es "un competidor neural lossless mejor financiado" sino que el mercado no perciba la categoría; el comparable de salida (adquisición por un dueño de IP/formatos) es el patrón observado.

### Gaps
- No hay post-mortems públicos de startups de compresión neural lossless; la evidencia de "por qué no" es académica, no empresarial.
- Sin acceso a PitchBook/Crunchbase de pago no se puede descartar una startup pequeña sin cobertura de prensa.

---

## P6. Evidencia económica de la literatura 2024-2026 sobre viabilidad de la compresión LLM/neural lossless (contexto para el breakeven de Redukt)

### Takeaway
El paper CIDR 2026 "Waiting to Decompress" (UTN/TUM) cuantifica el problema: con Qwen3 en una L40S
($0.804/h) frente a c8g.medium ($0.019/h) y S3 a $20/TiB-mes, la compresión LLM gana ~2x de factor sobre
zstd/xz pero cuesta >400x más por byte; recupera su coste de cómputo en ~10 años frente a no comprimir y
~120 años frente a algoritmos CPU. Sólo datos fríos de archivo largo tienen sentido; la tendencia de precios
GPU podría hacerlo viable "within the next decade".

### Cited Findings
- Autores/afiliación: Andreas Kipf, Tobias Schmidt*, Ping-Lin Kuo, Skander Krid, Moritz Rengert, Luca Heller, Andreas Zimmerer, Mihail Stoian, Varun Pandey, Alexander van Renen — University of Technology Nuremberg, *TU Munich; CIDR'26, 18-21 enero 2026 — [PDF](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf), [abstract](https://vldb.org/cidrdb/2026/waiting-to-decompress-the-economics-of-llm-based-compression.html).
- Setup: "NVIDIA L40S GPU, $0.804 per hour (g6e.xlarge)"; "Graviton instance $0.019 per hour (c8g.medium)"; "$20 per TiB per month (Amazon S3)"; precios de instancias reservadas 3 años; 100 MiB de Wikipedia inglesa y 100 MiB de código Python; modelos qwen2.5 0.5B/1.5B/7B, qwen3 0.6B/1.7B/8B, distilgpt2 vs fsst, lz4, zip, gzip, zstd, zstd -19, xz, xz -9e — [slides](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf), [PDF Fig. 4](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf).
- Resultados (slides): "Compression Factor: 2x" sobre los clásicos; "Compression Time: >10x"; "Compression Cost: >400x" — [slides](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf).
- Breakeven (figura del PDF): "LLM compression pays off after 10 years, cost dominated by compression on GPUs"; "higher compression rate in LLMs, break even with CPU algorithms after 120 years" — [PDF](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf).
- Optimizaciones: KV caching "2.5 KiB/s to 57 KiB/s in throughput"; "Decouple AC: Arithmetic coding runs on the CPU (algorithm is bottlenecked by GPU inference)"; batching por chunks; determinismo con temperatura 0 y batch fijo — [slides](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf).
- Conclusión textual: "LLM-based compression is not yet ready to replace lightweight, CPU-based compression algorithms for hot or frequently accessed data ... as inference becomes faster and more affordable, LLM-based methods could become a compelling option for long-term archival of cold datasets"; "Despite, or because of, the increased demand for GPUs ... normalized GPU costs have dropped considerably, especially in the last two years. In contrast ... storage cost pricing for archival storage is dropping at a much slower rate" — [PDF](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf).
- Motivación: "60% of enterprise data is classified as cold when it's created" (IDC Worldwide Global StorageSphere Forecast); "The paradox of using $1M models to save $20/month on storage" — [slides](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf).
- Otras cifras de la literatura: ts_zip 1 MB/s en RTX 4090 con RWKV 169M — [bellard.org](https://bellard.org/ts_zip/); FineZip 4 h por 10 MB con Llama-3-8B — [arXiv 2409.17141](https://arxiv.org/abs/2409.17141); predictores ligeros 34-40 KB/s en RTX 4060 Ti "still too slow" — [arXiv 2604.15472](https://arxiv.org/pdf/2604.15472); modelos de difusión: "severely limited throughput" — [arXiv 2608.11249](https://arxiv.org/pdf/2608.11249). Comentario HN: el encoder y el decoder deben ejecutar el LLM de forma determinista — [HN mirror item 46593802](https://19edd6ee.hackernews-40s.pages.dev/item/46593802).

### Inferences
- El paper fija la vara de medir que usará cualquier comprador técnico: factor ~2x sobre zstd a >400x de coste. Redukt reporta -50% a -88% vs brotli-11 en logs/XML (≈2x-8x) a $9-60/GB; su mejor argumento es que en logs/XML el factor es >2x, lo que acorta el breakeven de 10-120 años del paper, pero sigue necesitando datos que se retengan años — exactamente el segmento regulado (e-factura 10 años, SOX) de la Commercial Target.
- La conclusión del paper ("sólo archivo frío") choca con la tesis de Redukt ("hot object storage, no deep archive"): la resolución está en que el coste de compute por GB de Redukt (CPU $50-60, T4 $9-17) debe bajar hasta ≤$0.06-0.33/GB para que el ahorro en caliente (S3 Standard $20-23/TiB-mes) compense — coincide con el gap 30x-500x ya documentado en CLAUDE.md.
- La propia hoja de ruta del paper (fine-tuning eficiente, early-exit, modelos pequeños) coincide con el "fast tier" R-105 (3-30M params) de Redukt.

### Gaps
- No se extrajeron los factores de compresión numéricos exactos por compresor de la Figura 4 (sólo "2x" de las slides); el PDF contiene las barras pero no los valores en texto.
- El paper no evalúa logs ni XML, sólo Wikipedia y código Python.

---

## P7. ¿Qué dicen los analistas (Gartner, IDC, otros) sobre el tamaño del mercado de data-reduction / storage efficiency 2025-2026?

### Takeaway
Ni Gartner ni IDC publican una cifra pública específica de "data reduction software"; publican el mercado
de sistemas de almacenamiento (IDC: $33.0B externo OEM en 2025, +3.9%; Gartner: $28.5-28.9B ECB storage
2025, ~1.9-2% CAGR) y Gartner avisa de que los precios de memoria/almacenamiento obligarán a presupuestar
250-300% del gasto 2025 para 2027. Las cifras de "deduplication/compression software" provienen de firmas
de informes genéricos y oscilan entre $1.8B y $10.5B según definición: sólo orientativas.

### Cited Findings
- IDC: mercado mundial de External Enterprise Storage Systems 2025 "reached $33.0 billion (+3.9 percent YoY)"; Q4 2025 $9.68B (+5.5%), Dell 23.7%, Huawei 12.6%, NetApp 8.1%, Everpure 7.1%, IBM 6.3% — [IDC press release 2026-03-12](https://www.idc.com/resource-center/press-releases/worldwide-external-enterprise-storage-systems-market-closed-with-a-solid-3-9-year-over-year-growth-in-revenue-according-to-idc/), [Cyprus Mail 2026-03-23](https://cyprus-mail.com/2026/03/23/global-storage-market-hits-record-33bn-as-firms-upgrade-systems); Q3 2025 $8.0B (+2.1%) — [StorageNewsletter 2025-12-24](https://www.storagenewsletter.com/2025/12/24/idc-worldwide-external-enterprise-storage-systems-market-revenue-increased-2-1-during-third-quarter-of-2025/); Q1 2026 ~$9.2B con high-end +60.7% YoY — [Blocks & Files 2026-06-16](https://www.blocksandfiles.com/flash/2026/06/16/idc-ranks-dell-netapp-everpure-huawei-and-hpe-in-external-storage-systems-market/5256070).
- Gartner: ECB storage "$28.5 billion in 2025, with a five-year CAGR of 2.0% ... from 2024 to 2029" — [Gartner doc 6299415](https://www.gartner.com/en/documents/6299415); revisión 2Q25 "$28.9 billion in 2025, and grow 1.9% CAGR by 2029"; 2026 "$30.3 billion ... 0.4% annual growth" — [Gartner docs](https://www.gartner.com/en/documents/6651134); Magic Quadrant Enterprise Storage 2026: "For 2027, heads of I&O must budget for 250 percent to 300 percent of their 2025 storage spend to accommodate surging memory and storage prices" — [Blocks & Files 2026-08-24](https://www.blocksandfiles.com/block/2026/08/24/gartner-weaves-enterprise-storage-magic-spells-again/5291600).
- Gartner nombró a Granica "Cool Vendor" en 2023 (única mención analista a un vendor de compresión "AI") — [projectn Crunch](https://projectn-dotcom.netlify.app/crunch).
- IDC (vía slides CIDR): "60% of enterprise data is classified as cold when it's created" (Global StorageSphere) — [slides](https://www.cidrdb.org/cidr2026/slides/Kipf-115.pdf).
- Firmas de informes (definiciones dispares): dedup tools 2025 ≈ $8.92B (CAGR 8.5-27.8%) — [Datamation 2025-04-17](https://datamation.com/?p=22494); $10.55B (2025) → $24.05B (2032), 12.5% — [Market Research Forecast](https://www.marketresearchforecast.com/reports/data-deduplication-tools-59609); $4.2B (2025) → $9.8B (2034) — [DataHorizzon](https://datahorizzonresearch.com/data-deduplication-tools-market-65315); "data compression software" $1.8B (2025) → $2.8B (2034) — [MarketIntelo](https://marketintelo.com/report/compression-software-market); $5.8B (2024) → $12.4B (2034) — [Emergen](https://www.emergenresearch.com/jp/industry-report/data-compression-software-market); storage management software $9.85B (2024) → $25.85B (2034), 9.9% — [Allied via ContentEngine](https://contentenginellc.com/?p=488521).

### Inferences
- El "presupuesto" real contra el que compite Redukt no es un mercado de software de compresión (inexistente como categoría analista), sino la factura de almacenamiento ($33B sistemas on-prem + object storage cloud) y, según Gartner, un shock de precios 2026-2027 que hace más valiosos los bytes ahorrados — favorable a la tesis de "bytes caros".
- Las cifras de $1.8-10.5B de firmas de informes no deben citarse como tamaño de mercado sin matizar su falta de definición.

### Gaps
- No existe cifra pública Gartner/IDC para "data reduction software" o "storage efficiency software" como mercado separado; harían falta informes de pago (Gartner Market Share, IDC Software Tracker) para confirmarlo.
- No se encontró ningún informe de analista (Gartner, IDC, GigaOm, Futurum) que cubra "neural/AI lossless compression" como categoría emergente a noviembre de 2026.
