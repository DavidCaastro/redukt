# Research Transfer Plan — Redukt → Redukt

Plan para trasladar investigación relevante al repo de producto sin exponer
ni modificar el repositorio original.

Fecha: 2026-10-21

---

## Principios de Transferencia

1. **Solo COPIAR, nunca mover** — Redukt no se toca
2. **Sanitizar antes de copiar** — eliminar toda referencia interna
3. **Renombrar**: `Redukt` → `Redukt`, `Redukt` → `Redukt`
4. **Eliminar**: números R-series (R01, R102...), commits SHA, rutas internas, sesiones de debug
5. **Mantener**: datos técnicos, métricas, decisiones de diseño, benchmarks
6. **Formato destino**: documentación de producto, no notas de investigación

---

## Reglas de Sanitización Global

### Términos a reemplazar

| Original | Reemplazo |
|---|---|
| `Redukt` | `Redukt` |
| `Redukt` | `Redukt` |
| `Redukt-7` | `Redukt` |
| Referencias a `R01`, `R102`, etc. | Descripción funcional ("Phase 1: Coder Design") |
| Commits SHA (`7e19322`, `a3d1c87`...) | Eliminar |
| Rutas internas (`src/domain/crg.rs`, etc.) | Rutas del nuevo repo |
| `hybrid_eval`, `cm_eval`, `probe_cmd` | Solo lo que aplique a compresión |
| Chat flags (`--jrt`, `--freeze-cm`, `--rewrite`) | Eliminar |
| `CLAUDE.md`, `.claude/rules/` | No transferir (Redukt tiene su propio) |

### Secciones a eliminar siempre

- Todo sobre chat, generación, factual recall, steering
- JRT (Joint Retrieval Test), CM freeze para chat, state-offset
- Probes de activación (forward_probed, state analysis para chat)
- Referencias a otros repos (analytic-lm, edge-lm, nacrith) — excepto como "prior art"
- Notas de sesión, timestamps de debug, `now_barcelona()`
- GCP deployment internals

---

## Inventario Completo: docs/research/

### TRANSFERIR — 15 documentos (compresión core)

| # | Archivo original | Destino en Redukt | Síntesis |
|---|---|---|---|
| 1 | `r01-rwkv7-due-diligence.md` | `docs/research/backbone-evaluation.md` | Evaluación del backbone RWKV-7: arquitectura, seguridad, rendimiento. Eliminar comparativas con otros LLMs para chat |
| 2 | `r02-rwkv7-forward-pass.md` | `docs/research/rwkv7-forward-pass.md` | Especificación completa del forward pass en Rust. Transferir íntegro, solo renombrar proyecto |
| 3 | `r03-hardware-optimization.md` | `docs/research/hardware-optimization.md` | Análisis CPU bottleneck (memory bandwidth vs compute), Q8. Eliminar referencias a fases R-series |
| 4 | `r04-landscape-analysis.md` | `docs/research/competitive-landscape.md` | Ecosistema competitivo (cmix, NNCP, ts_zip, Nacrith). Muy valioso para posicionamiento. Renombrar solo |
| 5 | `r05-scaling-analysis.md` | `docs/research/scaling-analysis.md` | Scaling laws: modelo 0.1B vs 1.5B para compresión. Eliminar secciones de chat scaling |
| 6 | `r11-q8-quantization.md` | `docs/research/q8-quantization.md` | Kernel VNNI, BPB improvements. Transferir íntegro |
| 7 | `r12-q4-quantization.md` | `docs/research/q4-quantization-killed.md` | Por qué Q4 falla en modelos pequeños. Valioso como decisión documentada |
| 8 | `r13-mixed-precision-quantization.md` | `docs/research/mixed-precision.md` | Per-matrix quantization, GGUF standards. Transferir íntegro |
| 9 | `r14-alternative-number-systems.md` | `docs/research/number-systems-survey.md` | Survey de LNS, RNS, ternario, BFP. Justifica IEEE 754. Transferir íntegro |
| 10 | `r15-arithmetic-coder.md` | `docs/research/arithmetic-coder.md` | CDF-24 range coder, overhead analysis. **Documento fundacional del codec** |
| 11 | `r16-byte-level-cm.md` | `docs/research/context-mixing.md` | 19 modelos CM, logistic mixer, 4-way hash. **Core del sistema** |
| 12 | `r17-rwkv-byte-bridge.md` | `docs/research/token-byte-bridge.md` | Marginalization via trie, bit decomposition MSB. **Clave arquitectural** |
| 13 | `r19-lstm-mixer.md` | `docs/research/lstm-mixer.md` | LSTM dynamic mixing, BPTT=64. **Mayor ganancia arquitectural** |
| 14 | `r21-match-model.md` | `docs/research/match-model.md` | Longest-match con hash tables, recency decay. Transferir íntegro |
| 15 | `r24-benchmark-selection.md` | `docs/research/benchmark-methodology.md` | Silesia + Weissman Score. **Base del protocolo de evaluación** |

### SINTETIZAR PARCIALMENTE — 5 documentos (extraer solo lo de compresión)

| # | Archivo original | Destino en Redukt | Qué extraer | Qué eliminar |
|---|---|---|---|---|
| 16 | `r25-silesia-evaluation.md` | `docs/research/silesia-baseline.md` | Resultados Silesia 12 files, BPB por dominio | Referencias a fases chat |
| 17 | `r50-t1b-t2b-t3-baseline-eval.md` | `docs/research/cross-domain-baseline.md` | Composite BPB multi-tier, throughput | Menciones a R-series, commit hashes |
| 18 | `r61-ait-dcc-backbone-coverage.md` | `docs/research/domain-coverage-analysis.md` | Gap analysis por dominio, cluster taxonomy | Referencias internas de fase |
| 19 | `r100-cm-relevance-gate.md` | `docs/research/noise-gate.md` | Diseño del CRG como middleware de calidad | Todo lo de chat recall, JRT |
| 20 | `r14-alternative-number-systems.md` | (ya incluido arriba) | — | — |

### NO TRANSFERIR — 60+ documentos (chat/generación/steering)

| Rango | Temática | Motivo de exclusión |
|---|---|---|
| `r06-r10` | Experimentos deprecated | Superseded por r11+ |
| `r26-r34` | Domain modeling, preprocessing | Específico de pipeline byte-level legacy |
| `r35-r49` | Architectural experiments | Caminos muertos documentados en heritage |
| `r52-r56` | Adaptive preprocessing phases | KILLED (R52: catastrophic con RWKV pretrained) |
| `r57-r74` | Domain analysis, MoE | Chat-oriented domain specialization |
| `r75-r82` | Strategic pivots, chat roadmap | Chat product planning, no compresión |
| `r83-r91` | Kernel smoke tests, sweep, reranking | Chat factual recall optimization |
| `r101` | Format gap mechanistic analysis | Chat prompt format, no compresión |
| `r102-r104` | JRT, steering, CM freeze, state-offset | Chat-only backbone adaptations |
| `r105-r112` | State delta, accuracy ceiling, SFT | Chat/instruct tuning |
| `domain-*.md` | Domain-specific chat analysis | Chat domain modeling |

---

## Inventario Completo: docs/results/

### TRANSFERIR — 7 archivos/directorios

| # | Archivo original | Destino en Redukt | Síntesis |
|---|---|---|---|
| 1 | `INDEX.md` (líneas 1-640) | `docs/results/INDEX.md` | Registro de milestones de compresión (R00-R75). Cortar a línea 640, eliminar R102+ |
| 2 | `domain-coverage.md` | `docs/results/domain-coverage.md` | Análisis de cobertura por dominio. Transferir íntegro, solo renombrar proyecto |
| 3 | `phase5/RESULTS.md` | `docs/results/cross-domain-validation.md` | 25 archivos, composite BPB, throughput. **Dato principal de marketing** |
| 4 | `phase5/phase5-baseline.csv` | `docs/results/baseline-25files.csv` | Tabla maestra: BPB + throughput por archivo. Transferir íntegro |
| 5 | `ablation-t1.csv` | `docs/results/ablation-10kb.csv` | Ablación de modelos CM en 10KB. Justifica ensemble de 14 modelos |
| 6 | `ablation-100k.csv` | `docs/results/ablation-100kb.csv` | Ablación a 100KB con significancia estadística (p<0.05) |
| 7 | `phase5/intrinsic/` | `docs/results/intrinsic/` (3-5 samples) | Per-byte cost breakdown. Seleccionar: enwik8, dickens, xml |

### TRANSFERIR OPCIONAL — 3 archivos (telemetría)

| # | Archivo original | Destino | Notas |
|---|---|---|---|
| 8 | `t2-final/*.jsonl` (1-2 files) | `docs/results/telemetry-examples/` | Ejemplo de telemetría L2 para documentación de API |
| 9 | `scale-test-200k.jsonl` | `docs/results/scale-convergence.jsonl` | Convergencia BPB a 200KB |
| 10 | `c1-enwik8-1mb.log` | `docs/results/enwik8-1mb-validation.log` | Prueba de escala 1MB |

### NO TRANSFERIR — 5 directorios

| Directorio | Motivo |
|---|---|
| `b8/` (800 KB) | Chat generation sweep, 35 preguntas factual recall |
| `r102/` (5.9 KB) | JRT factual recall CSV |
| `r79/`, `r80/` logs | Debug de crashes, no producto |
| `phase5/states/` (383 MB) | CM state binaries — demasiado grandes, no públicos |
| `shard-0/1/2/` | Particiones de eval internas |

---

## Inventario Completo: docs/ raíz

### TRANSFERIR CON SANITIZACIÓN — 3 archivos

| # | Archivo original | Destino en Redukt | Sanitización requerida |
|---|---|---|---|
| 1 | `ARCHITECTURE.md` (1,280 líneas) | `docs/ARCHITECTURE.md` | **Pesado**: eliminar §8 (chat features, líneas 164-184), limpiar commits SHA, reemplazar "Redukt", eliminar menciones a `probe_cmd`/`chat_cmd`/`generate`/`steer_sweep`/`offset_sweep`. Mantener: pipeline compresión, quantization, forward pass, state management |
| 2 | `BENCHMARKS.md` (365 líneas) | `docs/BENCHMARKS.md` | Eliminar: fases R-series, tiers de steering (T0-diag), JRT/state-offset (líneas 222-253). Mantener: metodología evaluación, tiers T1-T4, métricas composite, adversarial tests |
| 3 | `weights/MANIFEST.md` | `weights/MANIFEST.md` | Eliminar: números R-series, weights de chat (1.5B, 2.9B solo si no aplican). Mantener: 0.1B provenance, SHA256, restauración, licencia Apache 2.0 |

### NO TRANSFERIR — 4 archivos

| Archivo | Motivo |
|---|---|
| `CHANGELOG.md` | Historial de investigación R-series. Redukt empieza con su propio changelog |
| `ROADMAP.md` | Roadmap de investigación (Phase A-D chat). Redukt tiene su propio roadmap |
| `GCP-DEPLOY.md` | Infraestructura interna de ablation campaigns |
| `telemetry-manifest.md` | Metadata de research telemetry |

---

## Inventario Completo: docs/business-case/

### TRANSFERIR ÍNTEGRO — 13 archivos (ya documentados)

Todos los archivos de `docs/business-case/` se transfieren al nuevo repo
como referencia interna de negocio. La mayoría ya usa "Redukt"
que se renombra a "Redukt".

| Archivo | Edición requerida |
|---|---|
| `README.md` | Renombrar Redukt → Redukt |
| `MARKET-ANALYSIS.md` | Renombrar |
| `BENCHMARK-RESULTS.md` | Renombrar |
| `BUSINESS-CASES.md` | Renombrar |
| `PRODUCT-ARCHITECTURE.md` | Renombrar + actualizar rutas de ficheros |
| `LIMITATIONS.md` | Renombrar |
| `MIGRATION-GUIDE.md` | Renombrar + eliminar rutas internas de Redukt |
| `LEGAL-ANALYSIS.md` | Renombrar |
| `CLAUDE-TEMPLATE.md` | Se convierte en `CLAUDE.md` del nuevo repo |
| `FORMAT-SPEC.md` | Renombrar |
| `LICENSE-TEMPLATE` | Se convierte en `LICENSE` (rellenar copyright) |
| `NOTICE-TEMPLATE` | Se convierte en `NOTICE` (rellenar nombre) |
| `REPO-SETUP-CHECKLIST.md` | Renombrar, se convierte en checklist interno |
| `RESEARCH-TRANSFER-PLAN.md` | Este documento — referencia de migración |

---

## Resumen Cuantitativo

| Categoría | Documentos | Acción |
|---|---|---|
| Research transferir íntegro | 15 | Copiar + renombrar proyecto |
| Research sintetizar parcial | 4 | Extraer secciones de compresión |
| Results transferir | 7 + 3 opcionales | Copiar + sanitizar |
| Docs raíz sanitizar | 3 | Edición pesada (ARCHITECTURE, BENCHMARKS, MANIFEST) |
| Business-case íntegro | 14 | Copiar + find-replace nombre |
| **Total a transferir** | **43-46** | — |
| Research NO transferir | 60+ | Chat/generation/steering |
| Results NO transferir | 5 dirs | Chat sweep/states |
| Docs NO transferir | 4 | Research changelog/roadmap |

---

## Procedimiento de Transferencia Seguro

### Paso 1: Crear repo Redukt vacío
```bash
gh repo create DavidCaastro/redukt --private --clone
cd redukt
```

### Paso 2: Copiar archivos (NUNCA mover)
```bash
# Desde Redukt, COPIAR a redukt — no git mv, no git rm
cp -r ../Redukt/docs/business-case/ docs/business-case/
cp ../Redukt/docs/ARCHITECTURE.md docs/ARCHITECTURE.md
# etc.
```

### Paso 3: Sanitizar en el NUEVO repo
```bash
# En redukt/, no en Redukt/
cd redukt/
# Find-replace global
find docs/ -name "*.md" -exec sed -i 's/Redukt/Redukt/g' {} +
find docs/ -name "*.md" -exec sed -i 's/Redukt/Redukt/g' {} +
# Luego revisión manual de cada archivo
```

### Paso 4: Verificar que Redukt no cambió
```bash
cd ../Redukt/
git status  # debe mostrar "nothing to commit, working tree clean"
git diff    # debe estar vacío
```

### Paso 5: Commit en redukt
```bash
cd ../redukt/
git add -A
git commit -m "docs: initial research transfer from development history"
git push
```

---

## Estructura Final en Redukt

```
redukt/
├── LICENSE                          (Apache 2.0)
├── NOTICE                           (atribuciones)
├── README.md                        (producto)
├── CLAUDE.md                        (instrucciones AI)
├── Cargo.toml
├── .cargo/config.toml
├── src/                             (25 ficheros portados)
├── weights/
│   ├── MANIFEST.md                  (provenance 0.1B)
│   └── rwkv7-0.1b/                  (pesos)
├── docs/
│   ├── ARCHITECTURE.md              (sanitizado)
│   ├── BENCHMARKS.md                (sanitizado)
│   ├── CHANGELOG.md                 (nuevo, producto)
│   ├── research/                    (20 docs transferidos)
│   │   ├── backbone-evaluation.md
│   │   ├── rwkv7-forward-pass.md
│   │   ├── hardware-optimization.md
│   │   ├── competitive-landscape.md
│   │   ├── scaling-analysis.md
│   │   ├── q8-quantization.md
│   │   ├── q4-quantization-killed.md
│   │   ├── mixed-precision.md
│   │   ├── number-systems-survey.md
│   │   ├── arithmetic-coder.md
│   │   ├── context-mixing.md
│   │   ├── token-byte-bridge.md
│   │   ├── lstm-mixer.md
│   │   ├── match-model.md
│   │   ├── benchmark-methodology.md
│   │   ├── silesia-baseline.md
│   │   ├── cross-domain-baseline.md
│   │   ├── domain-coverage-analysis.md
│   │   ├── noise-gate.md
│   │   └── README.md                (índice)
│   ├── results/
│   │   ├── INDEX.md                 (compresión only)
│   │   ├── domain-coverage.md
│   │   ├── cross-domain-validation.md
│   │   ├── baseline-25files.csv
│   │   ├── ablation-10kb.csv
│   │   ├── ablation-100kb.csv
│   │   └── intrinsic/              (3-5 samples)
│   ├── api/
│   │   ├── FORMAT-SPEC.md
│   │   ├── API-REFERENCE.md         (nuevo)
│   │   └── INTEGRATION-GUIDE.md     (nuevo)
│   ├── legal/                       (Fase 0-1)
│   ├── security/                    (Fase 1-2)
│   ├── compliance/                  (Fase 2-3)
│   └── internal/                    (no público)
│       └── business-case/           (los 14 docs originales)
```

---

## Métricas Clave para Trasladar (Product Messaging)

Extraídas de los documentos de results, listas para README y marketing:

| Métrica | Valor | Fuente |
|---|---|---|
| enwik8 BPB (warm-start) | 1.11 | INDEX.md |
| Silesia 12-file composite | 1.8186 BPB | phase5-baseline.csv |
| 25-file global composite | 2.2823 BPB | phase5/RESULTS.md |
| XML/markup vs brotli-11 | -44.7% | BENCHMARK-RESULTS.md |
| Text vs brotli-11 | -37.2% | BENCHMARK-RESULTS.md |
| Throughput (text) | 27-72 B/s | phase5-baseline.csv |
| CM models in ensemble | 19 (14 activos) | ablation CSVs |
| LSTM mixer gain | +0.22 BPB over linear | r19-lstm-mixer.md |
| Match model gain | -0.0095 BPB | r21-match-model.md |
| Embedding surgery gain | -0.0282 BPB | INDEX.md |

---

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|---|---|
| Dejar referencia a Redukt en archivo copiado | `grep -r "Redukt" docs/` post-copia, CI check |
| Copiar archivo de chat por error | Checklist explícito: solo los 46 archivos listados |
| Modificar accidentalmente Redukt | `git status` en Redukt después de cada paso |
| Exponer commit hashes internos | `grep -r "[0-9a-f]\{7\}" docs/` post-sanitización |
| Exponer rutas internas del repo original | `grep -r "Redukt/" docs/` + `grep -r "src/domain/crg" docs/` |
| Phase5 states (383 MB) copiados por error | .gitignore: `docs/results/states/` |
