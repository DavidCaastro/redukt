# Coste de inferencia secuencial de modelos de 100-350M parámetros (CPU, GPU, aceleradores) — suelo realista 2026

Fecha de investigación: 2026-11-10. Todas las cifras llevan fecha y fuente; se separa MEDIDO de TEÓRICO (roofline).
Convenciones de conversión usadas en todo el documento:
- 1 GB = 1e9 bytes. A 3.8 bytes/token BPE: 263 M tokens/GB. A 1 byte/token (modelo byte-level): 1,000 M tokens/GB.
- $/GB = ($/h) / (tok/s x 3600) x tokens/GB.
- Modelo de referencia: 130M parámetros, 2 FLOP/param/token = 0.26 GFLOP/token. Pesos: Q8 = 130 MB, BF16 = 260 MB, F32 = 520 MB.
- Estado recurrente Mamba-130M (d_model 768, 24 capas, expand 2 -> d_inner 1536, d_state 16): 1536 x 16 x 4 B x 24 = 2.36 MB F32 por stream
  (+ conv state ~18 KB/capa). Lectura+escritura por token ~4.8 MB (coincide con los ~4.9 MB/stream medidos en Redukt, Fase 19).

## Pregunta 1 — Throughput de decode publicado para modelos ~130M (Mamba, Mamba2, RWKV, SmolLM2, GPT-2) por batch y GPU

### Takeaway
No existe ningún benchmark publicado de decode a batch 64/256/1024 para modelos de ~130M en T4/L4/A10G/A100/H100/B200 con vLLM, SGLang o TensorRT-LLM: estos motores publican tablas para 70B-235B. Los únicos datos medidos de decode secuencial para 130M son batch 1: 1,588 tok/s en TPU v6e (bucle compilado) y 210 tok/s en A100 (Mamba-2 130M, 2026), más los datos de compresores neurales (ts_zip hasta 1 MB/s con RWKV 169M en RTX 4090; Nacrith 20-30 tok/s por worker en GTX 1050 Ti).

### Cited Findings
- Mamba-2 130M en TPU v6e, batch 1, bucle `fori_loop` compilado, KV/estado cacheado: 1,588 tok/s (seq 128) y 1,641 tok/s (seq 4096); bucle Python host: 662 tok/s (penalización 2.4x). 370M: 626 tok/s; 780M: 318; 1.3B: 188; 2.7B: 94. Sin caché a 4096: 56 tok/s para 130M — [arXiv 2603.09555v1, Tablas 7/12](https://arxiv.org/html/2603.09555v1)
- Mamba-2 en A100 (mismo paper, Tabla 11/13): 130M cacheado seq 128 = 210 tok/s (batch 1); 370M = 107.9 (128) / 170.2 (4096); 1.3B = 58.6/72.1; 2.7B = 32.0. El paper no publica barrido completo A100 para 130M ni comparación de bucles en A100 — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)
- Utilización de ancho de banda HBM (HBU) en decode TPU v6e: 51.6% para 130M (seq 128) hasta 64.08% para 2.7B (4096). MFU en prefill (batch 1): 6.22-8.23% para 130M, 15.23% máximo para 2.7B. El paper advierte que los bytes están contados sin fusión, así que la HBU es cota superior — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)
- Paper Mamba original: "Mamba achieves 4-5x higher inference throughput than a Transformer of similar size"; benchmark end-to-end en A100 80GB PCIe, Mamba 1.4B y 6.9B (sin entrenar) vs Transformer 1.3B/6.7B (HF transformers), prompt 2048, generación 128, batch 1,2,4,8,16,32,64,128, throughput = batch x 128 / tiempo, 3 repeticiones — [arXiv 2312.00752, Apéndice E.5](https://arxiv.org/html/2312.00752). Los valores absolutos tok/s están solo en figura (no extraíbles como texto).
- TensorRT-LLM: su página oficial de rendimiento solo cubre GPT-OSS 120B, Llama 3.3 70B y Qwen3 235B (p. ej. Llama 3.3 70B FP8 TP2 en H100: 2,209 tok/s/GPU en escenario "max throughput"); ningún modelo Mamba/Mamba2 ni modelo <1B aparece — [TensorRT-LLM perf overview](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html)
- Estimaciones (NO medidas) tipo "ancho de banda / tamaño de modelo": llmrun.dev da ~7,280 tok/s para Mamba 130M en RTX 4090 y ~1,524 tok/s para SmolLM2-135M Q4_K_M en RTX 4090; son fórmulas, no mediciones — [llmrun.dev Mamba-130M](https://llmrun.dev/model/state-spaces-mamba-130m-hf); [llmrun.dev SmolLM2-135M](https://llmrun.dev/model/huggingfacetb-smollm2-135m)
- Nacrith (SmolLM2-135M + codificador aritmético, llama.cpp backend): un worker en GTX 1050 Ti ~50-70 tok/s al inicio del fichero, estabilizándose en ~20-30 tok/s con caché KV de 2,048 tokens; 3 workers (máximo para 4 GB VRAM) ~60-90 tok/s agregados; ~1.2 GB VRAM por worker, máximo 8 workers; decode single-token ~7x más rápido con llama.cpp que PyTorch; "orders of magnitude slower than traditional compressors" — [arXiv 2602.19626v2, secciones 3.12, 5.5, 6.4](https://arxiv.org/html/2602.19626v2)
- ts_zip (Bellard): RWKV 169M v4 cuantizado a 8 bits (evaluado en BF16), velocidad de compresión y descompresión "up to 1 MB/s on a RTX 4090"; requiere GPU para velocidad razonable; salida determinista independiente de GPU/CPU/hilos; enwik8 1.106 bpb, enwik9 1.084 bpb — [bellard.org/ts_zip](https://bellard.org/ts_zip/)
- Hugging Face leaderboard: GPT-2 en A100 a batch 2, seq 256: ~55,351-57,788 tok/s — probablemente throughput de forward completo (prefill), no decode secuencial; no usar como dato de decode — [wizwand/HF benchmark](https://www.wizwand.com/sota/inference-throughput-on-bert-gpt-2-and-opt-bs-2-sl-256)
- Decode de modelos pequeños en GPU a batch 1 está dominado por latencia de lanzamiento de kernels/bucle host: en el paper TPU, el bucle Python cuesta 2.4x a 130M y la diferencia desaparece por encima de 780M — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)

### Inferences
- Los 1,588 tok/s a batch 1 en TPU v6e (1,600 GB/s HBM) equivalen a ~1 GB de tráfico/token según la HBU reportada (51.6% x 1,600 GB/s / 1,588), es decir el pipeline mueve mucho más que los 260 MB de pesos BF16: hay sobrecoste por falta de fusión. El mismo fenómeno (muchos kernels pequeños por capa, 24 capas) explica los 210 tok/s en A100 a batch 1.
- Los 1,117 tok/s de Redukt en T4 a N=128 están por debajo del batch 1 de la referencia TPU compilada (1,588 tok/s): el gap no es de hardware sino de kernels (fusión, bucle en dispositivo, GEMM tensor-core).
- ts_zip es la única evidencia medida de que un modelo 130-170M en GPU de consumo (~$0.3-0.5/h en mercado, precio no verificado aquí) procesa ~1 MB/s en compresión neural real. A ~4 bytes/token RWKV eso son ~250k tok/s, coherente con la cota de cómputo del RTX 4090 (165 TFLOPS FP16 densos / 0.34 GFLOP/token ≈ 485k tok/s) a ~50% de MFU. Es posible porque RWKV-4 tiene un estado minúsculo por stream (~15 KB/capa), lo que hace el batch casi gratis en ancho de banda.

### Gaps
- No se encontró ningún número medido de decode a batch 64/256/1024 para Mamba-130M, Mamba2-130M, RWKV-7 0.1B, SmolLM2-135M o GPT-2 124M en T4/L4/A10G/A100/H100/B200 (vLLM, SGLang, TRT-LLM, llama.cpp, mamba-ssm). Los motores de inferencia no publican modelos <1B; habría que medirlo directamente.
- Los valores absolutos del benchmark del paper Mamba (Fig. 4.5 / E.5) están en imagen; solo se cita la afirmación 4-5x y el protocolo.
- Paper RWKV-7 (arXiv 2503.14456): no se encontró ninguna cifra de tok/s en los resultados; no se fetcheó el paper.
- NNCP (Bellard): la página oficial no publica velocidad (kB/s) ni GPU; solo enwik9 = 0.853 bpb (2023-10-21) — [bellard.org/nncp](https://bellard.org/nncp/).

## Pregunta 2 — Roofline: batch de cruce a cómputo, techo teórico tok/s y $/GB por GPU

### Takeaway
Con pesos Q8 (130 MB) el decode se vuelve "compute-bound por pesos" a batch ~100 en T4, ~200 en L4, ~77 en A100, ~148 en H100, ~141 en B200; pero para Mamba-1 el tráfico del estado recurrente (~4.8 MB leídos+escritos por token, F32) supera al de los pesos a partir de N≈27 streams y fija el techo real: ~65k tok/s en T4, ~62k en L4, ~420k en A100, ~680k en H100, ~1.6M en B200 a N=1024. Convertido a precio spot de nov-2026 y 3.8 B/token, el suelo teórico al 100% de eficiencia es $0.08-0.25/GB (A10G/A100/T4 spot) y el realista (30-50%) $0.3-0.8/GB; con byte-level (1 B/token) multiplicar por 3.8.

### Cited Findings
Especificaciones usadas (densas, sin sparsity):
- A100 80GB: 312 TFLOPS FP16 (624 INT8), ~2 TB/s (2,039 GB/s). H100 SXM: 989 TFLOPS FP16/BF16 densos (1,979 FP8/INT8), 3.35 TB/s. B200: ~2,250 TFLOPS FP16/BF16 densos (4.5 PFLOPS es la cifra sparse), 7.7-8.0 TB/s según fuente. T4: 320 GB/s; la cifra "130 TFLOPS" que circula es INT8, el FP16 denso tensor es 65 TFLOPS. L4: 300 GB/s; "242 TFLOPS" es sparse, denso FP16 = 121 — [gcore GPU comparison](https://gcore.com/blog/nvidia-gpu-comparison); [gpu-markets B200 vs T4](https://gpu-markets.pages.dev/compare/b200-sxm-192gb-vs-t4); [gpu-markets B200 vs L4](https://gpu-markets.pages.dev/compare/b200-sxm-192gb-vs-l4); [hmc-tech B200 vs H100](https://hmc-tech.com/gpus/nvidia-b200-vs-nvidia-h100). Nota: gcore lista A100 "624 TFLOPS" y H100 "1,979" sin asterisco, que son cifras con sparsity; se usan aquí las densas. Las hojas de datos NVIDIA no se fetchearon directamente.
- Paper Mamba-2/XLA: TPU v6e = 918 TFLOPS BF16 y 1,600 GB/s; A100 = 312 TFLOPS BF16 y 1,555 GB/s; saturar cómputo en v6e exige ~574 FLOP/byte, inalcanzable a batch 1 — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)
- Precios spot (snapshot 2026-11-09, mejor precio): T4 $0.16 (Azure; GCP $0.21, AWS $0.26), L4 $0.31 (GCP), A10G $0.15 (Vast; AWS $0.45), A100 40GB $0.47, A100 80GB $0.68 (Azure; GCP $1.76, AWS $2.31), H100 PCIe $1.42, H100 SXM $2.13 (Vast mediana; Azure $2.27, AWS $4.20), H200 $2.37, B200 $3.87 (GCP; AWS $9.33). On-demand mejor: T4 $0.35 (GCP; AWS/Azure $0.53), L4 $0.47 (GCP $0.56, AWS $0.80), A10G $0.23 (AWS $1.01), A100 80GB $1.07 (AWS $3.43, GCP $3.93), H100 SXM $2.33 (AWS $6.88, Azure $12.29), B200 $5.98 (AWS $14.24, GCP $16.11) — [FrenzyCap GPU prices](https://www.frenzycap.com/gpu-prices)
- Precios de GPU dedicadas en proveedores de inferencia: DeepInfra A100 80GB $1.49/h, H100 $2.90/h, B200 $4.89/h; Together dedicated H100 $5.49/h, B200 $8.99/h, clúster H100 $3.99 on-demand / $1.99 preemptible — [DeepInfra pricing](https://deepinfra.com/pricing); [Together pricing](https://www.together.ai/pricing)

Cálculo roofline (TEÓRICO, elaboración propia a partir de las especificaciones anteriores):

| GPU | BW GB/s | FP16 denso TFLOPS | Batch de cruce Q8 / BF16 (FLOPS/BW x bytes/param / 2) | Techo cómputo tok/s (TFLOPS/0.26 GFLOP) | Techo BW batch 1 Q8 (BW/130 MB) | Techo con estado Mamba F32, N=128 [N x BW/(130+4.8N) MB] | Idem N=1024 |
|---|---|---|---|---|---|---|---|
| T4 | 320 | 65 | 102 / 203 | 250k | 2,460 | 55k | 65k |
| L4 | 300 | 121 | 202 / 403 | 465k | 2,300 | 52k | 61k |
| A10G | 600 | 125 (aprox.) | ~104 / 208 | 480k | 4,600 | 103k | 122k |
| A100 80GB | 2,039 | 312 | 77 / 153 | 1.2M | 15,700 | 351k | 414k |
| H100 SXM | 3,350 | 989 | 148 / 295 | 3.8M | 25,800 | 576k | 680k |
| B200 | 8,000 | 2,250 | 141 / 281 | 8.65M | 61,500 | 1.38M | 1.62M |

$/GB a 3.8 B/token, precio spot FrenzyCap, techo N=1024 con estado F32 (100% eficiencia) y a 40% de eficiencia:

| GPU | $/h spot | $/GB @100% | $/GB @40% | $/GB @40%, byte-level (1 B/tok) |
|---|---|---|---|---|
| T4 | 0.16 | 0.18 | 0.45 | 1.71 |
| L4 | 0.31 | 0.37 | 0.93 | 3.5 |
| A10G | 0.15 | 0.09 | 0.22 | 0.85 |
| A100 80GB | 0.68 | 0.12 | 0.30 | 1.14 |
| H100 SXM | 2.13 | 0.23 | 0.57 | 2.2 |
| B200 | 3.87 | 0.17 | 0.44 | 1.66 |

### Inferences
- La métrica que ordena las GPUs para este workload es $/h por GB/s de HBM, no TFLOPS: A10G spot (0.15/600 = 0.00025), A100 80GB spot (0.00033), B200 spot (0.00048), T4 spot (0.0005), H100 SXM spot (0.00064), L4 spot (0.001). Las GPUs de inferencia "baratas" (L4) son las peores por GB/s; A100 spot de segunda mano es la mejor opción fiable.
- Para Mamba-1 con d_state=16 el batch óptimo en ancho de banda está en N≈27 (Q8) o N≈54 (BF16): por encima, el tráfico de estado domina (a N=128 son 614 MB de estado vs 130 MB de pesos) y el throughput se aplana en BW/4.8 MB. Guardar el estado en BF16 duplica el techo; Mamba2 (d_state 128) tiene ~8x más estado por stream (~19 MB) y es mucho peor para decode batch; RWKV-7 0.1B (12 x 64x64 x 12 capas ≈ 2.4 MB) está en el mismo orden que Mamba-1; un Transformer de 135M con ventana KV 2048 (SmolLM2: 30 capas x 3 KV heads x 64 x 2 x 2 B ≈ 23 KB/token -> ~47 MB/stream) mueve ~10x más que Mamba-1 por token.
- Con 3.8 B/token y kernels al 40% del roofline, solo A10G/A100 spot bajan de $0.33/GB (umbral texto en CLAUDE.md); $0.12 (e-factura) exige estado BF16 + >=60% eficiencia + A100/A10G spot; $0.06 (logs) no es alcanzable con un modelo de 130M en decode secuencial a 3.8 B/token en ningún precio de nov-2026 salvo mercado Vast.ai (A100 40GB cheapest $0.13/h, no fiable).
- Byte-level a 130M sale 3.8x más caro por GB que BPE; solo compensa si el modelo byte-level es proporcionalmente más pequeño (p. ej. 3-30M, R-105), porque pesos y estado escalan con el modelo.
- La afirmación "4% del pico de T4" (1,117 tok/s vs 250k de cómputo) sobreestima el margen: el techo relevante a N=128 es ~55k tok/s (ancho de banda con estado F32), luego Redukt está al ~2% de ese techo; el margen realista con kernels al 30-50% es 15-25x (16-28k tok/s).

### Gaps
- Las especificaciones T4/L4/A10G provienen de agregadores (y del brief), no de hojas de datos NVIDIA fetcheadas; A10G se tomó como 600 GB/s / ~125 TFLOPS densos (valor habitual, no verificado aquí).
- No hay medición pública de MFU/HBU de kernels Mamba a batch 128-1024 en GPU; el 40% es una hipótesis de ingeniería (el paper XLA mide 52-64% de HBU en decode batch 1 y 6-15% MFU en prefill para modelos pequeños).

## Pregunta 3 — CPU: mejor throughput publicado con AMX/AVX-512 int8 y coste por vCPU-hora

### Takeaway
No existe un benchmark publicado de decode de modelos ~130M en Xeon AMX/Granite Rapids/Zen 4-5/Graviton 4 en tok/s por core; la evidencia disponible dice que AMX acelera sobre todo prefill y que decode sigue limitado por ancho de banda. Con precios AWS de nov-2026 ($0.015-0.045/vCPU-h) y roofline por socket, el suelo de CPU para 130M a 3.8 B/token es ~$2-6/GB, muy por encima de cualquier umbral del producto.

### Cited Findings
- Precios AWS (vantage.sh, nov-2026): c7i.xlarge (4 vCPU Sapphire Rapids 3.2 GHz, 8 GiB) $0.179/h on-demand, $0.060/h spot -> $0.045 / $0.015 por vCPU-h — [instances.vantage.sh c7i.xlarge](https://instances.vantage.sh/aws/ec2/c7i.xlarge); c8g.xlarge (4 vCPU Graviton4, 8 GiB) $0.160/h OD, $0.096/h spot -> $0.040 / $0.024 por vCPU-h — [instances.vantage.sh c8g.xlarge](https://instances.vantage.sh/aws/ec2/c8g.xlarge)
- Phoronix (Xeon 6 Granite Rapids, llama.cpp): "AMX continued to prove to be a massive benefit for faster prompt processing"; en generación de texto gpt-oss-20b Q8_0 la configuración "AMX Enabled - Default" fue la más rápida, sin que el extracto muestre que decode mejore de forma relevante — [Phoronix Granite Rapids AMX](https://www.phoronix.com/review/intel-xeon-6-granite-rapids-amx/4) (página devolvió 403 al fetch; cita vía snippet de búsqueda)
- OpenMetal: Llama cuantizado hasta 57 tok/s con AMX vs 28 sin AMX ("2x increase in throughput when AMX is enabled"), sin normalizar por núcleos ni indicar tamaño exacto — [OpenMetal AMX](https://openmetal.io/blog/intel-amx-ai-inference-performance)
- Intel SGLang/llama.cpp sobre Xeon 6980P (Granite Rapids) con DeepSeek-R1-671B INT8: TPOT 172 ms (~5.8 tok/s) en llama.cpp mainline, dos sockets; sin paralelismo multi-NUMA — [level1techs thread](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/422)
- Sapphire Rapids expone AMX-INT8 (multiplicación matricial int8 de 8 bits) — [Discoverer+ SPR guide](https://docs.discoverer.bg/sapphirerapids.html)
- Lenovo Press LP2463 "Power-Efficient LLM Inference with Intel Xeon 6 Processors" contiene throughput por usuario para 1-128 usuarios concurrentes y tokens/kWh (modelos 7-8B), pero el PDF no es extraíble como texto — [Lenovo LP2463](https://lenovopress.lenovo.com/lp2463.pdf)
- Dato de referencia del producto (brief, no investigado): ~310 tok/s agregados en 12 vCPU portátil (Q8, 32 streams, 130-190 GFLOPS) = ~26 tok/s/vCPU.

### Inferences
- Roofline CPU (TEÓRICO): un socket Sapphire Rapids de 8 canales DDR5-4800 (~307 GB/s) da techo por estado Mamba F32 de ~64k tok/s por socket (~112 vCPU) = ~570 tok/s/vCPU, 22x el dato actual. Por cómputo AVX-512 F32 (~2 FMA x 16 x 2 x 3 GHz ≈ 190 GFLOPS/core) el techo es ~740 tok/s/core; al 50% ≈ 185 tok/s/vCPU (7x el actual). AMX-INT8 (~2,048 ops/ciclo/core, ~6 TOPS/core) elevaría el techo de cómputo a >20k tok/s/core, así que con AMX el límite vuelve a ser el ancho de banda de estado.
- Coste: 185 tok/s/vCPU a $0.015/vCPU-h spot = $5.9/GB (3.8 B/tok); en el techo de ancho de banda (570 tok/s/vCPU) = $1.9/GB; a 26 tok/s/vCPU actuales = $42/GB (coherente con los $50-60/GB del brief). Graviton4 spot ($0.024/vCPU-h) es peor por vCPU-h spot pero mejor on-demand ($0.040).
- Conclusión: en CPU, ni con AMX perfecto se baja de ~$2/GB para un 130M secuencial a 3.8 B/token; el umbral de $0.33 exige GPU o un modelo 10x más pequeño (fast tier).

### Gaps
- Ningún número medido tok/s/core de modelos 100-350M en AMX/AVX-512 int8 (oneDNN, OpenVINO GenAI, ggml AMX). Los benchmarks CPU públicos son de 7B+; los per-core habría que derivarlos.
- Zen 4/5 (c7a/c8a) y precios de vCPU en GCP/Azure no verificados.

## Pregunta 4 — Precio de APIs para modelos pequeños como suelo de mercado y conversión a $/GB

### Takeaway
El precio de API más bajo encontrado para un modelo abierto pequeño es $0.02 input / $0.04 output por M tokens (Llama 3.1 8B Turbo, DeepInfra, 2026): a 3.8 B/token eso son $5.3-10.5/GB (prefill-output), a 1 B/token $20-40/GB; GPT-4.1 nano y Gemini 2.5 Flash-Lite ($0.10/$0.40) equivalen a $26-105/GB. Ningún proveedor ofrece modelos de 125M-1B por token; el "suelo" de mercado está en 8B y aún así es 15-30x más caro que el roofline GPU de un 130M.

### Cited Findings
- DeepInfra: Meta-Llama-3.1-8B-Instruct-Turbo $0.02 in / $0.04 out por M tokens; gemma-3-4b-it $0.05/$0.10; Qwen3.5-9B $0.10/$0.15; no lista Llama 3.2 1B/3B ni Qwen 0.5B/1.5B — [DeepInfra pricing](https://deepinfra.com/pricing)
- Together: Tev1 4B Experimental $0.04 input / output gratis; Qwen2.5 7B Turbo $0.30/$0.30; Qwen3.5 9B $0.17/$0.25; ningún modelo <1B listado — [Together pricing](https://www.together.ai/pricing)
- GPT-4.1 nano ~$0.10/M input según agregadores; GPT-5 nano $0.05 in / $0.40 out; gpt-oss-120b $0.04 in / $0.18 out en proveedores económicos; Mistral Nemo $0.02/$0.02 (abril 2026) — [DataLLMLab cheapest APIs 2026](https://www.datallmlab.com/blog/cheapest-llm-api-2026.html); [dev.to cheapest AI APIs 2026](https://dev.to/leolionel221/top-10-cheapest-ai-apis-in-2026-ranked-by-real-cost-2f98); [BenchLM pricing 2026](https://www.benchlm.ai/blog/posts/llm-pricing-2026). La página oficial de OpenAI devolvió 403 al fetch.
- Gemini 2.5 Flash-Lite: $0.10/M input, $0.40/M output (texto), caché $0.01/M; Gemini 3.1 Flash-Lite aparece en $0.10/$0.40 (abril 2026) y $0.25/$1.50 (julio 2026) según fuente — [pricepertoken Gemini 2.5 Flash-Lite](https://pricepertoken.com/pricing-page/model/google-gemini-2.5-flash-lite); [morphllm Gemini pricing Oct 2026](https://www.morphllm.com/gemini-api-pricing)
- a16z (nov 2024): Llama 3.2 3B en Together a $0.06/M tokens — [a16z LLMflation](https://a16z.com/llmflation-llm-inference-cost/)

Conversión a $/GB (elaboración propia; precio de output = decode secuencial; precio de input = prefill paralelo):

| Oferta | $/M tok (in / out) | $/GB @3.8 B/tok (in / out) | $/GB @1 B/tok (in / out) |
|---|---|---|---|
| Llama 3.1 8B Turbo (DeepInfra) | 0.02 / 0.04 | 5.3 / 10.5 | 20 / 40 |
| Tev1 4B (Together) | 0.04 / 0 | 10.5 / — | 40 / — |
| gemma-3-4b (DeepInfra) | 0.05 / 0.10 | 13 / 26 | 50 / 100 |
| GPT-4.1 nano / Gemini 2.5 Flash-Lite | 0.10 / 0.40 | 26 / 105 | 100 / 400 |

### Inferences
- Los precios de API incluyen margen y modelos 30-60x mayores que 130M. Escalando linealmente el precio de output de Llama 8B ($0.04/M) por el ratio de parámetros (~60x) se obtiene ~$0.0007/M tokens -> ~$0.17/GB a 3.8 B/tok, consistente con el roofline GPU de la Pregunta 2 (0.1-0.3 $/GB al 40-100%). Es decir, el mercado confirma el orden de magnitud: un 130M en decode secuencial cuesta como mínimo $0.1-0.3/GB en GPU bien aprovechada.
- Para compresión, el precio de input (prefill) es la referencia correcta del lado de compresión si el backbone es congelado: sus logits para todo el fichero pueden calcularse en un forward paralelo (teacher forcing) y solo la descompresión exige decode secuencial. Esto no se ha investigado en fuentes, pero el ratio input/output de 2-4x de las tablas lo refleja parcialmente (el ratio real de coste prefill/decode en GPU es mucho mayor que 2-4x).

### Gaps
- Ninguna API pública sirve modelos de 100-350M por token; el "suelo" medible es 4-8B.
- Precios oficiales de OpenAI y Google no verificados directamente (403 / no fetcheado); cifras de agregadores de 2026 con discrepancias entre versiones (Gemini 3.1 Flash-Lite).

## Pregunta 5 — Curva de descenso del coste de inferencia 2023-2026 y proyección 2027-2028

### Takeaway
A capacidad fija, el precio de inferencia ha caído 10x/año (a16z, nov 2024), 9-900x/año con titular 40x/año (Epoch, mar 2025) y ~47%/trimestre ≈ 13x/año desde 2023 (Epoch, sep 2026, vía prensa). Pero esas curvas se deben sobre todo a modelos más pequeños/mejores y a competencia de proveedores, no a hardware; para un modelo FIJO de 130M el coste cae solo con $/h de GPU y eficiencia de kernels, y en 2026 los precios de alquiler de GPU subieron (RunPod, Nebius, Lightning AI).

### Cited Findings
- a16z (12 nov 2024): "For an LLM of equivalent performance, the cost is decreasing by 10x every year"; GPT-3 (MMLU 42) a $60/M en nov 2021 vs Llama 3.2 3B a $0.06/M en nov 2024 (1,000x en 3 años); nivel MMLU-83: 62x desde marzo 2023; drivers: GPU precio/rendimiento, cuantización 16->4 bits, software, modelos más pequeños sobre-entrenados, instruction tuning, open source — [a16z LLMflation](https://a16z.com/llmflation-llm-inference-cost/)
- Epoch AI (12 mar 2025): el precio para alcanzar un rendimiento fijo cayó entre 9x y 900x por año según benchmark; ejemplo GPQA Diamond nivel GPT-4: 40x/año; MMLU 86-88: GPT-4-0314 $37.50/M (2023-03-14) vs Gemini 2.0 Flash $0.18/M (2025-02-05); "the fastest price drops in that range have occurred in the past year, so it's less clear that those will persist" — [Epoch LLM inference price trends](https://epoch.ai/data-insights/llm-inference-price-trends)
- Epoch AI "AI and Compute" (actualizado 5 feb 2026): precios de inferencia LLM caen ~40x/año (2 órdenes de magnitud/año) a nivel de rendimiento fijo — [Epoch AI and Compute](https://epoch.ai/research/ai-and-compute)
- Epoch AI, informe "The plunging price of thought" (22 sep 2026, citado por prensa; la URL primaria probada devolvió 404): coste a rendimiento fijo cae ~47% por trimestre desde 2023 ≈ 13x/año; tareas matemáticas 50-52%/trimestre, puzzles 39-43%/trimestre; ejemplo de caída 725x en <18 meses ($0.0004 por pregunta) — [fourweekmba](https://fourweekmba.com/ai-epoch-ai-the-cost-of-ai-performance-falls-13x-every-year/); [pasqualepillitteri](https://pasqualepillitteri.it/en/news/21044/epoch-ai-ai-prices-fall-47-percent-quarter)
- Paper "The Price of Progress" (MIT, nov 2025): coste a rendimiento fijo cae 5-10x/año para modelos frontera; declives por pasada de 24.5x/año (MATH 500) y 3.23x/año (AIME 2024) — [arXiv 2511.23455](https://arxiv.org/html/2511.23455v1)
- Precios de alquiler GPU en 2026 subieron: Nebius H100 on-demand $3.85 -> $4.50 (1 oct 2026), B200 $7.15 -> $8.50; RunPod H100 SXM $2.99 -> $3.29, B200 $5.89 -> $6.79, A100 SXM $1.49 -> $1.59, L4 $0.39 -> $0.49; Lightning AI H100 +76% a $3.50/h; "in 2026 the trend reversed" — [Hostinger cloud GPU pricing 2026](https://www.hostinger.com/tutorials/?p=158956); [pricingsaas RunPod 2026W34](https://pricingsaas.com/companies/runpod/diffs/2026W34); [neurohive cloud GPU pricing](https://neurohive.io/en/guides/cloud-gpu-pricing/)
- Mercado L4: $0.44-0.80/h, mediana -7% en el último año — [Hostinger](https://www.hostinger.com/tutorials/?p=158956)

### Inferences
- Las tasas de 10-40x/año no son aplicables a Redukt mientras el modelo y su BPB sean fijos: la compresión no puede sustituir el modelo por uno "más barato con la misma calidad" sin cambiar el formato. El componente trasladable es el hardware ($/GB/s de HBM) y la eficiencia de kernels; el primero se estancó o empeoró en 2026 (subidas de precio), el segundo depende de ingeniería propia (15-25x de margen según Pregunta 2).
- Proyección 2027-2028 (propia, no sourceada): para un 130M fijo, esperar a lo sumo 1.5-2x/año de mejora en $/GB por generación de hardware (B200/B300 a precio spot maduro, HBM3e) si los precios spot vuelven a bajar; el grueso del recorte debe venir de kernels (10-25x) y de reducir bytes de estado/pesos por token (BF16 state, fast tier).

### Gaps
- No se obtuvo el informe primario de Epoch de sep 2026 ni ninguna fuente SemiAnalysis con cifras de $/token de hardware; las cifras 2026 son de prensa secundaria.
- No se encontró una serie temporal de $/h spot de T4/A100 2023-2026 que permita medir la tasa de descenso del hardware per se.

## Pregunta 6 — Compresión neural sin pérdida en GPU: throughput publicado y cuello de botella del codificador aritmético

### Takeaway
Las cifras publicadas van de ~20-30 tok/s (Nacrith, SmolLM2-135M, GTX 1050 Ti, 1 worker) a ~1 MB/s (ts_zip, RWKV 169M Q8, RTX 4090), pasando por CMIX 0.5-5 KB/s en CPU. La codificación de entropía en GPU no es el cuello de botella: nvCOMP gANS alcanza ~530 GB/s compresión / ~265 GB/s descompresión en Blackwell, DietGPU 365/593 GB/s en H100 y Recoil rANS 90+ GB/s en RTX 2080 Ti; el límite es la inferencia secuencial del modelo, 4-6 órdenes de magnitud más lenta.

### Cited Findings
- ts_zip: hasta 1 MB/s compresión y descompresión en RTX 4090 con RWKV 169M (pesos 8 bits, BF16 eval); enwik8 1.106 bpb, enwik9 1.084, alice29 1.142, book1 1.431, linux-1.2.13.tar 1.021; determinista — [bellard.org/ts_zip](https://bellard.org/ts_zip/)
- Nacrith: 0.9389 bpb enwik8, 0.918 alice29; ~20-30 tok/s estable por worker (GTX 1050 Ti), 60-90 tok/s con 3 workers, hasta 8 workers multi-GPU, ~1.2 GB VRAM/worker; "LLM inference is the dominant computational cost"; skip de LLM por confianza 30-70% en texto muy comprimible; CMIX 0.5-5 KB/s con 16-64 GB RAM — [arXiv 2602.19626v2](https://arxiv.org/html/2602.19626v2)
- NNCP: enwik9 106,632,363 B (0.853 bpb), enwik8 14,915,298 B (1.19 bpb), 2023-10-21; v3.3 (2024-06-05) con CUDA; sin velocidad publicada en la página — [bellard.org/nncp](https://bellard.org/nncp/)
- nvCOMP (NVIDIA blog, 9 abr 2026): ANS/gANS ~530 GB/s compresión en BF16 (~30x ZSTD), ~265 GB/s descompresión en SMs; ZSTD ~16 GB/s; motor de descompresión Blackwell LZ4/Snappy/Deflate hasta 280 GB/s — [NVIDIA nvCOMP checkpoint blog](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/)
- Medición independiente en H100: nvcomp-ANS 480 GB/s end-to-end; DietGPU ANS encode 364.9 GB/s, decode 592.5 GB/s (256 MB, bit-perfect) — [arXiv 2602.08190](https://arxiv.org/pdf/2602.08190)
- Recoil (rANS paralelo, 2023): >90 GB/s de decode (solo kernel) en RTX 2080 Ti con 32-way interleaving por warp y 2,176 splits; la codificación no es paralelizable en su esquema; razones de la dificultad: códigos de longitud variable con dependencias internas, estado del decoder dependiente del anterior, búsqueda de símbolo memory-bound, sincronización entre núcleos cara — [ar5iv 2306.12141](https://ar5iv.labs.arxiv.org/html/2306.12141)

### Inferences
- Con N streams independientes, el codificador aritmético adaptativo (CDF de 50k símbolos por token) se paraleliza trivialmente por stream en CPU; a 1 MB/s total son ~250k símbolos/s, cuatro órdenes de magnitud por debajo de lo que un solo core CPU hace en rANS (y seis por debajo de GPU ANS). El cuello de botella es exclusivamente el modelo.
- La diferencia ts_zip (1 MB/s) vs Nacrith (~0.1 KB/s por worker a ~4 B/tok) con modelos de tamaño parecido muestra que arquitectura de estado (RWKV ~0.2 MB/stream vs KV 2048 tokens) + batch grande + bucle en C deciden 3-4 órdenes de magnitud.

### Gaps
- Ningún trabajo publica bytes/s de compresión neural en T4/L4/A100/H100 con batch explícito; ts_zip no documenta el batch ni el precio de la GPU.
- No se encontró "Nacrith-GPU" ni cifras de NNCP CUDA en kB/s.

## Pregunta 7 — Precios de aceleradores no-NVIDIA (TPU v5e/v6e, Inferentia2, Groq, Cerebras) para modelos de 130M

### Takeaway
TPU v6e cuesta ~$2.70/chip-h on-demand ($1.22-1.89 con compromiso) y es el único acelerador con un dato medido de 130M (1,588 tok/s batch 1); inf2.xlarge cuesta $0.758/h OD / $0.170 spot sin ningún benchmark de modelos pequeños; Groq y Cerebras no publican ni sirven modelos de 130M.

### Cited Findings
- TPU v6e (Trillium): $2.70/chip-h on-demand (figura más repetida), $1.89 (1 año), $1.22 (3 años); una fuente da $1.375 OD y $0.39 con compromiso máximo (conflicto) — [Qovery TPU vs GPU 2026](https://www.qovery.com/blog/tpu-vs-gpu-cost-per-million-tokens-2026); [gpuperhour TPU vs GPU](https://gpuperhour.com/blog/tpu-vs-gpu); [AI wiki TPU pricing](https://artificial-intelligence-wiki.com/ai-development/ai-hardware-infrastructure/google-cloud-tpu-pricing/)
- TPU v6e: 918 TFLOPS BF16, 1,600 GB/s; Mamba-2 130M decode batch 1: 1,588 tok/s compilado, HBU 51.6% — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)
- inf2.xlarge (Inferentia2): $0.758/h on-demand, $0.170/h spot, 4 vCPU — [instances.vantage.sh inf2.xlarge](https://instances.vantage.sh/aws/ec2/inf2.xlarge); $0.76/h citado también en Qovery y HF endpoints $0.75 — [HF Inference Endpoints pricing](https://huggingface.co/docs/inference-endpoints/support/pricing)
- GroqRack vs A100 (Llama2-7B): batch 1 328-336 tok/s en Groq vs 73-84 en GPU; GPU alcanza 2,144 tok/s a B=32 con secuencias cortas — [arXiv 2606.17104](https://arxiv.org/pdf/2606.17104)

### Inferences
- TPU v6e a $2.70/h y 1,588 tok/s (batch 1) = $0.124/GB-h... es decir $2.70/(1,588 x 3600) x 263e6 = $124/GB a batch 1; solo con batch de cientos de streams (techo por estado F32: 1,600/4.8 MB ≈ 330k tok/s) bajaría a ~$0.6/GB al 100%; peor que A100 spot por $/GB/s.
- Inferentia2 spot ($0.17/h) es atractivo en precio pero sin kernels Mamba/SSM públicos ni datos; riesgo de ingeniería alto.

### Gaps
- Sin cifras de Groq/Cerebras para modelos <1B; sus precios por token son para 8B-70B y no se investigaron.
- Precios oficiales de Google Cloud TPU no verificados (fuentes secundarias en conflicto).

## Pregunta 8 — Estimación explícita para el modelo del producto con eficiencia cuBLAS/tensor-core en T4/L4/H100 y $/GB implícito

### Takeaway
Partiendo de 1,117 tok/s en T4 a N=128 (brief) y del roofline de la Pregunta 2, kernels al 30-50% del techo de ancho de banda darían 16-28k tok/s en T4 (15-25x), 15-26k en L4, 170-290k en H100 SXM (N=1024) y 100-170k en A100 80GB; en precios spot nov-2026 y 3.8 B/token eso es $0.36-0.63/GB (T4), $0.74-1.2 (L4), $0.46-0.76 (H100), $0.24-0.40 (A100), $0.17-0.29 (A10G). Solo A100/A10G spot con estado BF16 y >=50% de eficiencia bajan a ~$0.1-0.15/GB; el umbral de logs ($0.06) sigue fuera de alcance con decode secuencial de 130M.

### Cited Findings
- Dato actual del producto (brief, no investigado): 1,117 tok/s en T4 a N=128 (~4% del pico de cómputo), 310 tok/s en 12 vCPU, 1 token ≈ 3.8 bytes, $9-17/GB en T4, $50-60/GB en CPU spot.
- Techos y precios: ver tablas de la Pregunta 2 — [FrenzyCap](https://www.frenzycap.com/gpu-prices); [gcore](https://gcore.com/blog/nvidia-gpu-comparison); [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)
- Referencia empírica de lo alcanzable: ts_zip 1 MB/s (RWKV 169M Q8, RTX 4090, 1,008 GB/s) — [bellard.org/ts_zip](https://bellard.org/ts_zip/); Mamba-2 130M 1,588 tok/s a batch 1 en TPU v6e con bucle compilado — [arXiv 2603.09555v1](https://arxiv.org/html/2603.09555v1)

Estimación (TEÓRICA, elaboración propia; estado F32 4.8 MB rw/token, pesos Q8 130 MB; eficiencia = fracción del techo de ancho de banda):

| GPU, batch | Techo tok/s | 30% | 50% | $/h spot | $/GB @30% | $/GB @50% | $/GB @50%, estado BF16 |
|---|---|---|---|---|---|---|---|
| T4, N=128 | 55k | 16.5k | 27.5k | 0.16 | 0.71 | 0.43 | 0.23 |
| T4, N=1024 | 65k | 19.5k | 32.5k | 0.16 | 0.60 | 0.36 | 0.19 |
| L4, N=1024 | 61k | 18k | 30k | 0.31 | 1.24 | 0.74 | 0.39 |
| A10G, N=1024 | 122k | 37k | 61k | 0.15 | 0.30 | 0.18 | 0.10 |
| A100 80GB, N=1024 | 414k | 124k | 207k | 0.68 | 0.40 | 0.24 | 0.13 |
| H100 SXM, N=1024 | 680k | 204k | 340k | 2.13 | 0.76 | 0.46 | 0.24 |
| B200, N=1024 | 1.62M | 486k | 810k | 3.87 | 0.58 | 0.35 | 0.19 |

Mismas cifras on-demand (FrenzyCap mejor OD): T4 $0.35 -> x2.2; A100 $1.07 -> x1.6; H100 $2.33 -> x1.1; B200 $5.98 -> x1.5.

### Inferences
- El margen real sobre los 1,117 tok/s de T4 es 15-30x (kernels fusionados, bucle en dispositivo, GEMM tensor-core, estado en BF16), no 60x: el techo de T4 a N=128 lo pone el tráfico de estado (614 MB/paso) y no los 65 TFLOPS.
- La palanca más barata no es cambiar de GPU sino reducir bytes por token: estado BF16 (-50% tráfico) y mantener Q8 en pesos; y elegir A100 80GB/A10G spot (mejor $/GB/s). Con eso, texto ($0.33) es alcanzable con margen y e-factura ($0.12) queda al borde; logs ($0.06) exigiría además pasar a ~3.8x menos bytes/token de estado (modelo más pequeño o d_state menor) o encode paralelo.
- Asimetría compresión/descompresión (no investigada en fuentes, inferencia propia): si el backbone es congelado, los logits de compresión pueden calcularse en prefill paralelo (MFU 30-50% sobre 0.26 GFLOP/token -> ~1-2M tok/s en H100, <$0.002/GB), dejando el coste secuencial solo en descompresión; para retención regulada donde se descomprime raramente, el $/GB efectivo cae uno o dos órdenes de magnitud. El mezclador CM adaptativo seguiría en CPU, pero su coste no depende del backbone.
- En CPU, incluso con AMX y 50% del roofline, el suelo es ~$2-6/GB (Pregunta 3): la CPU queda como implementación de referencia/decodificador universal, no como tier de coste.

### Gaps
- Todas las cifras de esta sección son roofline; no hay ninguna medición pública de kernels Mamba a batch >=128 en T4/L4/A100/H100 con la que calibrar el 30-50% asumido (la única calibración empírica cercana es ts_zip ~1 MB/s en 4090 con RWKV-4, cuyo estado es ~13x menor que el de Mamba-1).
- Precio de RTX 4090 en mercado (Vast/RunPod) no verificado; por eso no se convierte ts_zip a $/GB.
