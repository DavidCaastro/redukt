# Product Architecture — Redukt

## Delivery Models Evaluated

### Model A: Raw API (compress/decompress endpoints)
- Client sends data, receives compressed blob
- Simple but exposes AZTH proprietary format
- Client must call us for every decompression
- **Rejected**: vendor lock-in perception, latency concerns

### Model B: SDK/Library (embedded in client stack)
- Rust library with C FFI, Python/Java bindings
- Client compresses/decompresses locally
- **Rejected**: requires shipping weights (~400MB), complex deployment

### Model C: Transparent Proxy (RECOMMENDED)
- Client configures S3/GCS bucket + credentials
- We compress on write, decompress on read — transparently
- Client never sees AZTH format
- Works as a middleware layer between application and object storage
- **Selected**: zero friction, invisible to client applications

## Technical Components

### Compression-Only Codebase

From Redukt (15,093 LOC total), the compression product requires:

| Layer | Keep | LOC | Remove | LOC |
|---|---|---|---|---|
| Application | codec, baseline, cm_eval, hybrid_eval, model_loader | 1,273 | generate, chat_cmd, probe_cmd, steer_sweep, offset_sweep | 2,154 |
| Domain | cm, coder, context_mixing, hash, lstm, predictor, sse, quant, probes, telemetry | 6,848 | chat, crg, jrt (dead code) | 1,549 |
| Infrastructure | rwkv7 (model, gguf, safetensors, bridge, backbone) | 3,269 | — | — |

**Compression-only binary: ~11,390 LOC** (75% of current codebase).

### API Design

```
POST /v1/compress
  Body: raw bytes
  Headers: Content-Type, X-Original-Size
  Response: compressed bytes + metadata

POST /v1/decompress
  Body: compressed bytes (AZTH format)
  Response: original bytes

GET /v1/status
  Response: health, version, throughput stats
```

### AZTH Format

```
MAGIC:       8 bytes  "AZTH\x01\x00\x00\x00"
TOTAL_BYTES: 4 bytes  LE (original size)
TOKEN_COUNT: 4 bytes  LE
STREAM:      variable (arithmetic coded probability stream)
```

Decompression requires identical model + weights + hyperparameters (deterministic).
This is inherent to ALL neural compressors (cmix, NNCP, etc).

## Deployment

### Recommended: Hetzner AX41 (EUR 45/month)

- AMD Ryzen 5 3600, 6c/12t, 64 GB RAM
- Dedicated server, no noisy neighbors
- Sufficient for batch workloads up to ~5 GB/day at current throughput
- Scale horizontally: add nodes behind load balancer

### Alternative: GCP GCE c3-highcpu-8 (EUR 130-274/month)

- 8 vCPUs, 16 GB RAM
- Cloud Run not viable (cold starts + always-on cost for stateful model)
- Only justified if client requires cloud compliance (ISO 27001, SOC 2)

### Why Not Serverless

- Model requires ~48 MB RAM persistent state
- Sequential processing (not parallelizable per-request)
- Cold start would reload 400 MB weights
- Always-on requirement makes serverless cost-ineffective

## Pricing Strategy

### Usage-Based (Stripe Metered Billing + OpenMeter)

| Tier | Volume/month | Price per GB |
|---|---|---|
| Starter | < 100 GB | EUR 0.50/GB |
| Growth | 100 GB - 1 TB | EUR 0.30/GB |
| Enterprise | > 1 TB | EUR 0.15/GB |
| Custom | > 10 TB | Negotiated |

### Alternative: Per-TB-Saved (PetaGene model)

Charge based on storage savings achieved, not data processed.
More aligned with client value but harder to meter.

## Integration Points

| Client Stack | Integration |
|---|---|
| S3/GCS | Transparent proxy (Lambda/Cloud Function trigger) |
| Kafka/Splunk | Post-ingestion batch job |
| ELK/OpenSearch | Index lifecycle policy hook |
| PACS/DICOM | HL7 FHIR gateway middleware |
| E-invoicing | XML archive stage |

## Encryption Scope

**Out of scope for v1.** Compression MUST happen before encryption (encrypted data is incompressible). Client's cloud provider handles encryption at rest. TLS handles encryption in transit.

Order: Raw data -> Redukt compress -> cloud encrypt at rest -> store.
