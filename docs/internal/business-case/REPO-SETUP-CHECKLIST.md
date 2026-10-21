# Redukt — Repo Setup & Documentation Checklist

Checklist completo para crear el repositorio `redukt` y toda la documentación
requerida para comercialización global (EU, USA, y resto del mundo).

Fecha: 2026-10-21

---

## 1. Creación del Repo

- [ ] Nombre: `redukt` (o `redukt-compress` — pendiente confirmación)
- [ ] Visibilidad: pendiente confirmación (público / privado)
- [ ] GitHub user: `DavidCaastro` (pendiente confirmación)
- [ ] Inicializar con README.md + LICENSE + .gitignore (Rust)
- [ ] Sincronizar con local
- [ ] Configurar branch protection en `main`

---

## 2. Nombre Comercial — Verificación de Marca

Nombre elegido: **Redukt**

### Búsqueda realizada (2026-10-21)

| Base de datos | Resultado | Riesgo |
|---|---|---|
| USPTO (USA) | No encontrado en Clase 9/42 | Libre |
| EUIPO (EU) | No encontrado en Clase 9/42 | Libre |
| OEPM (España) | No encontrado en Clase 9/42 | Libre |
| Trademarkia / Justia | No encontrado en software | Libre |
| Conflicto identificado | REDUKT Tiny Houses (redukt.eu) — Clase 19/20 (construcción, Polonia) | Sin conflicto de clase |

### Registro recomendado

| Oficina | Costo aprox. | Prioridad | Clases |
|---|---|---|---|
| OEPM (España) | ~EUR 150 | Alta | 9 (software), 42 (SaaS) |
| EUIPO (EU-wide) | ~EUR 850 | Media | 9, 42 |
| USPTO (USA) | ~$250-350 | Si entra en mercado USA | 9, 42 |
| Dominio redukt.com | ~EUR 12/año | Inmediata | N/A |

### Nombres descartados

| Nombre | Motivo de descarte |
|---|---|
| Compakt | Compakt Automatisering B.V. (compakt.nl) — software holandés desde 1983 |
| DeepSqueeze | Paper académico "DeepSqueeze: Deep Semantic Compression" (Brown University, SIGMOD 2020) — mismo dominio técnico |

---

## 3. Portabilidad del Código

Seguir: `docs/business-case/MIGRATION-GUIDE.md`

### Archivos a portar (25 ficheros)

**Domain layer (13):**
- `src/domain/mod.rs` (reducido: solo módulos de compresión)
- `src/domain/error.rs`
- `src/domain/tensor.rs`
- `src/domain/ngram.rs` (TokenNgram)
- `src/domain/bias_head.rs` (BiasHead)
- `src/domain/mixer.rs` (AdaptiveMixer)
- `src/domain/coder.rs` (RangeEncoder/Decoder)
- `src/domain/quant.rs`
- `src/domain/ram_monitor.rs`
- `src/domain/spectral.rs` (WHT)
- `src/domain/intrinsic.rs`
- `src/domain/backbone.rs` (trait NeuralBackbone)
- `src/domain/preprocess.rs`

**Application layer (4):**
- `src/application/mod.rs` (reducido)
- `src/application/codec.rs` (compress/decompress)
- `src/application/model_loader.rs`
- `src/application/telemetry.rs`

**Infrastructure layer (5):**
- `src/infrastructure/mod.rs`
- `src/infrastructure/rwkv7/mod.rs`
- `src/infrastructure/rwkv7/model.rs`
- `src/infrastructure/rwkv7/gguf.rs`
- `src/infrastructure/rwkv7/safetensors.rs`

**Root (3):**
- `src/main.rs` (nuevo, solo compress/decompress CLI)
- `Cargo.toml` (nuevo, renombrado a redukt)
- `.cargo/config.toml`

---

## 4. Documentación — Fases de Ejecución

### Fase 0 — Día 1 (antes de publicar código)

| Archivo | Descripción | Base existente |
|---|---|---|
| `LICENSE` | Apache 2.0 completo | `LICENSE-TEMPLATE` — rellenar copyright holder |
| `NOTICE` | Atribuciones: RWKV-7, PAQ, Schindler, SafeTensors, GGUF | `NOTICE-TEMPLATE` — rellenar nombre |
| `README.md` | Pitch comercial + quickstart + badges | Nuevo |
| `CLAUDE.md` | Instrucciones para Claude Code (legal compliance matrix) | `CLAUDE-TEMPLATE.md` |
| `docs/legal/PRIVACY-POLICY.md` | Qué datos procesamos, base legal (GDPR Art. 6), retención, derechos del usuario, contacto DPO | Nuevo |
| `docs/legal/TERMS-OF-SERVICE.md` | Contrato SaaS: licencia de uso, limitación responsabilidad, IP ownership, ley aplicable, arbitraje | Nuevo |
| `.gitignore` | Rust + weights/ + target/ + *.azth | Adaptar del actual |

### Fase 1 — Antes de primer cliente

| Archivo | Descripción | Base existente |
|---|---|---|
| `docs/legal/DPA-TEMPLATE.md` | Data Processing Agreement (GDPR Art. 28): roles, subprocesadores, transferencias internacionales, auditoría, notificación breach 72h | Nuevo |
| `docs/legal/ACCEPTABLE-USE.md` | Usos prohibidos: CSAM, sanciones OFAC/EU, violación derechos terceros, reverse engineering | Nuevo |
| `docs/legal/SLA.md` | Uptime (99.5%), latencia máxima, créditos por incumplimiento, exclusiones (fuerza mayor), ventanas mantenimiento | Nuevo |
| `docs/legal/EXPORT-CONTROL.md` | Declaración EAR/ITAR: compresión sin cifrado = ECCN EAR99, no restringida. EU Dual-Use Regulation: no aplica | `LEGAL-ANALYSIS.md` sección 4 |
| `docs/security/SECURITY-POLICY.md` | Responsible disclosure, contacto seguridad, PGP key, scope, safe harbor, tiempo respuesta | Nuevo |
| `docs/api/API-REFERENCE.md` | Endpoints REST: POST /compress, POST /decompress, GET /status. Auth (API key), rate limits, ejemplos curl | Nuevo |
| `docs/api/FORMAT-SPEC.md` | Especificación binaria AZTH v1 | `FORMAT-SPEC.md` — copiar directamente |
| `docs/api/INTEGRATION-GUIDE.md` | SDKs (Python, JS, Rust), ejemplos, webhooks, retry strategy, error codes | Nuevo |
| `docs/legal/COOKIE-POLICY.md` | Cookies técnicas vs analíticas, consentimiento (ePrivacy Directive), opt-out | Nuevo (si hay dashboard web) |
| `docs/legal/SUBPROCESSORS.md` | Lista de subprocesadores: hosting (Hetzner/GCP), CDN, payments (Stripe), monitoring. Proceso de notificación de cambios | Nuevo |

### Fase 2 — Antes de primer enterprise

| Archivo | Descripción | Base existente |
|---|---|---|
| `docs/legal/IP-INDEMNIFICATION.md` | Cláusula de indemnización IP: cobertura, exclusiones, cap, procedimiento de defensa, training data risk disclosure | `LEGAL-ANALYSIS.md` sección 3 |
| `docs/security/SECURITY-WHITEPAPER.md` | Arquitectura seguridad: TLS 1.3, cifrado en reposo (AES-256), gestión claves, network isolation, access control, logging | Nuevo |
| `docs/security/INCIDENT-RESPONSE.md` | Plan IRP: detección, clasificación, contención, erradicación, recuperación, notificación (72h GDPR, state breach laws USA), post-mortem | Nuevo |
| `docs/legal/DATA-RESIDENCY.md` | Regiones disponibles: EU (Hetzner Falkenstein), USA (GCP us-central1). Schrems II compliance: SCCs, no transferencia fuera de región elegida | Nuevo |
| `docs/compliance/GDPR-ROPA.md` | Record of Processing Activities (Art. 30): categorías datos, finalidad, base legal, retención, transferencias, medidas seguridad | Nuevo |

### Fase 3 — Certificación y expansión

| Archivo | Descripción | Base existente |
|---|---|---|
| `docs/compliance/SOC2-READINESS.md` | Checklist SOC2 Type II: Trust Service Criteria (seguridad, disponibilidad, confidencialidad), controles implementados, gaps, plan remediación | Nuevo |
| `docs/compliance/ISO27001-SCOPE.md` | Alcance ISMS, Statement of Applicability (SoA), controles Annex A seleccionados, justificación exclusiones | Nuevo |
| `docs/compliance/CCPA-DISCLOSURE.md` | California Consumer Privacy Act: categorías datos recopilados, propósito, derechos (know, delete, opt-out), "Do Not Sell" | Nuevo |
| `docs/compliance/ENS-READINESS.md` | Esquema Nacional de Seguridad: categorización sistema, medidas marco organizativo/operacional/protección, nivel aplicable | Nuevo |
| `docs/compliance/AI-TRANSPARENCY.md` | Declaración EU AI Act Art. 50: por qué no aplica (compresión, no generativo), clasificación riesgo, no high-risk Annex III | `LEGAL-ANALYSIS.md` sección 5 |
| `docs/compliance/HIPAA-READINESS.md` | Si medical imaging: BAA template, PHI handling, encryption requirements, audit logging | Nuevo (solo si vertical PACS) |

### Fase 4 — Expansión jurisdiccional

| Archivo | Jurisdicción | Descripción |
|---|---|---|
| `docs/compliance/UK-GDPR.md` | UK | UK GDPR + Data Protection Act 2018: ICO registration, international transfers post-Brexit (UK adequacy) |
| `docs/compliance/LGPD-BRAZIL.md` | Brasil | LGPD: DPO obligatorio, base legal, ANPD notification, transferencia internacional |
| `docs/compliance/APPI-JAPAN.md` | Japón | APPI amended 2022: consentimiento, transferencia cross-border, PPC notification |
| `docs/compliance/DPDP-INDIA.md` | India | DPDP Act 2023: consent manager, data localization (sensitive data), Data Protection Board |
| `docs/compliance/PIPL-CHINA.md` | China | PIPL + DSL + Algorithm Regulation: local deployment review, algorithm registration, CAC approval |
| `docs/compliance/PIPEDA-CANADA.md` | Canadá | PIPEDA + AIDA (cuando entre en vigor): consentimiento, accountability, breach reporting OPC |

---

## 5. Documentos Internos (no público — .gitignore o repo privado)

| Archivo | Descripción |
|---|---|
| `docs/internal/RISK-REGISTER.md` | Registro riesgos: training data (Books3/The Pile), patentes futuras, competidores, regulación |
| `docs/internal/VENDOR-ASSESSMENT.md` | Due diligence proveedores: Hetzner (GDPR), Stripe (PCI-DSS), Cloudflare, monitoring |
| `docs/internal/DATA-FLOW-MAP.md` | Diagrama flujo: input usuario → API gateway → compresión → storage → output. Retención y borrado |
| `docs/internal/PENETRATION-TEST.md` | Template para resultados pentest (completar cuando se realice) |
| `docs/internal/BUSINESS-CONTINUITY.md` | Plan continuidad: backups, failover, RTO/RPO, key person risk |

---

## 6. Archivos de Configuración del Repo

| Archivo | Descripción |
|---|---|
| `.github/SECURITY.md` | Política de seguridad pública (GitHub lo detecta automáticamente) |
| `.github/CONTRIBUTING.md` | Guía de contribución (si open source) |
| `.github/CODE_OF_CONDUCT.md` | Código de conducta (Contributor Covenant) |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Template para bug reports |
| `.github/ISSUE_TEMPLATE/feature_request.md` | Template para feature requests |
| `.github/PULL_REQUEST_TEMPLATE.md` | Template para PRs |
| `.github/workflows/ci.yml` | GitHub Actions: build, test, clippy, format check |
| `.github/FUNDING.yml` | Sponsors/funding info (opcional) |

---

## 7. Resumen de Volumen

| Fase | Documentos nuevos | Documentos reutilizados | Total |
|---|---|---|---|
| Fase 0 | 4 | 3 (LICENSE, NOTICE, CLAUDE.md) | 7 |
| Fase 1 | 8 | 2 (FORMAT-SPEC, EXPORT-CONTROL base) | 10 |
| Fase 2 | 5 | 1 (IP-INDEMNIFICATION base) | 6 |
| Fase 3 | 5 | 1 (AI-TRANSPARENCY base) | 6 |
| Fase 4 | 6 | 0 | 6 |
| Internos | 5 | 0 | 5 |
| GitHub config | 7 | 0 | 7 |
| **Total** | **40** | **7** | **47** |

---

## 8. Preguntas Pendientes (antes de ejecutar)

1. **Nombre exacto del repo**: `redukt` o `redukt-compress`?
2. **Visibilidad**: público o privado?
3. **GitHub user**: `DavidCaastro` — ¿confirmar?
4. **Copyright holder**: nombre personal o nombre de empresa?
5. **Dominio**: ¿registrar redukt.com ahora?
6. **Hosting elegido**: Hetzner (EUR 45/mo) o GCP? (afecta subprocessors y data residency docs)

---

## Referencias

- `docs/business-case/MIGRATION-GUIDE.md` — mapa de dependencias y pasos de portabilidad
- `docs/business-case/LEGAL-ANALYSIS.md` — análisis legal completo (base para múltiples docs)
- `docs/business-case/PRODUCT-ARCHITECTURE.md` — arquitectura de producto y deployment
- `docs/business-case/LIMITATIONS.md` — throughput constraints (base para SLA)
- `docs/business-case/BENCHMARK-RESULTS.md` — datos para README y marketing
- `docs/business-case/BUSINESS-CASES.md` — casos de uso con ROI (base para sales docs)
- `docs/business-case/FORMAT-SPEC.md` — spec AZTH v1 (copiar directo)
- `docs/business-case/CLAUDE-TEMPLATE.md` — CLAUDE.md para nuevo repo
- `docs/business-case/LICENSE-TEMPLATE` — Apache 2.0 (rellenar copyright)
- `docs/business-case/NOTICE-TEMPLATE` — atribuciones (rellenar nombre)
