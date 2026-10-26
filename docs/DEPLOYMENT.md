# Deployment Guide — Redukt API on GCP

## Overview

Redukt runs as a single-binary HTTP API server on a GCE instance.
The model loads once at startup (~130 MB Q8 in RAM), then serves
compress/decompress requests sequentially on port 8080.

```
Client -> POST /v1/compress -> [redukt api] -> AZTH bytes
Client -> POST /v1/decompress -> [redukt api] -> original bytes
Client -> GET  /v1/status -> {"status":"ok", ...}
```

## Architecture

```
                    ┌──────────────────────────┐
                    │  GCP europe-west1        │
                    │  ┌────────────────────┐  │
  Internet :8080 ──>│  │  GCE c3-highcpu-8  │  │
                    │  │  COS + Docker       │  │
                    │  │  redukt api          │  │
                    │  │  weights: /mnt/...   │  │
                    │  └────────────────────┘  │
                    │  VPC: redukt-vpc          │
                    │  Subnet: 10.0.1.0/24     │
                    └──────────────────────────┘
```

- **Machine**: c3-highcpu-8 (8 vCPU, 16 GB RAM) — sufficient for 0.1B model
- **OS**: Container-Optimized OS (COS) — auto-updates, minimal surface
- **SSH**: IAP tunnel only (no public SSH port)
- **Firewall**: port 8080 open, health checks from GCP ranges

## Prerequisites

1. GCP project with billing enabled
2. `gcloud` CLI authenticated (`gcloud auth login`)
3. Terraform >= 1.5
4. Docker
5. APIs enabled:
   ```bash
   gcloud services enable compute.googleapis.com \
       artifactregistry.googleapis.com \
       iap.googleapis.com
   ```

## Deploy Step by Step

### 1. Build and Push Container Image

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Build image
docker build -t redukt .

# Tag for Artifact Registry
docker tag redukt \
  europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/redukt/redukt:latest

# Push (Terraform creates the repo first — run step 2 with -target first)
docker push \
  europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/redukt/redukt:latest
```

### 2. Provision Infrastructure

```bash
cd infra/terraform

# Initialize
terraform init

# Create Artifact Registry first (needed for image push)
terraform apply -target=google_artifact_registry_repository.redukt \
  -var="project_id=YOUR_PROJECT_ID"

# After pushing the image (step 1), apply everything
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

Terraform creates:
- VPC + subnet
- Firewall rules (API, SSH via IAP, health checks)
- Static external IP
- Service account (Artifact Registry reader + logging + metrics)
- Artifact Registry repository
- GCE instance with COS + auto-start container

### 3. Verify

```bash
# Get the API URL from Terraform output
terraform output api_url

# Health check
curl http://EXTERNAL_IP:8080/v1/status

# Compress a file
curl -X POST --data-binary @myfile.txt \
  http://EXTERNAL_IP:8080/v1/compress -o myfile.azth

# Decompress
curl -X POST --data-binary @myfile.azth \
  http://EXTERNAL_IP:8080/v1/decompress -o restored.txt

# Verify roundtrip
diff myfile.txt restored.txt
```

### 4. SSH (debugging)

```bash
# Via IAP tunnel (no public SSH key needed)
gcloud compute ssh redukt-api --zone=europe-west1-b --tunnel-through-iap

# Check container logs
docker logs $(docker ps -q)
```

## API Reference

### POST /v1/compress

Compress raw bytes to AZTH format.

| | |
|---|---|
| **Content-Type** | `application/octet-stream` |
| **Body** | Raw bytes (max 256 MB) |
| **Response** | AZTH v2 bytes |
| **Status** | 200 OK, 400 Bad Request |

### POST /v1/decompress

Decompress AZTH bytes to original.

| | |
|---|---|
| **Content-Type** | `application/octet-stream` |
| **Body** | AZTH v2 bytes (min 28 bytes) |
| **Response** | Original raw bytes |
| **Status** | 200 OK, 400 Bad Request, 422 Unprocessable Entity |

### GET /v1/status

Health check endpoint.

```json
{"status":"ok","version":"0.1.0","vocab_size":65536,"max_body":268435456}
```

## Configuration

### Terraform Variables

| Variable | Default | Description |
|---|---|---|
| `project_id` | (required) | GCP project ID |
| `region` | `europe-west1` | GCP region |
| `zone` | `europe-west1-b` | GCP zone |
| `machine_type` | `c3-highcpu-8` | Instance type |
| `disk_size_gb` | `30` | Boot disk size |
| `image_tag` | `latest` | Container image tag |

### API Server Flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Listen port |
| `--weights` | `weights/rwkv7-0.1b` | Weights directory |

## Cost Estimate

| Resource | Monthly (EUR) |
|---|---|
| c3-highcpu-8 (on-demand) | ~274 |
| c3-highcpu-8 (1yr commit) | ~175 |
| 30 GB SSD | ~5 |
| Static IP | ~3 |
| Egress (100 GB) | ~9 |
| **Total (on-demand)** | **~291** |
| **Total (committed)** | **~192** |

## Weights on the Instance

Weights are downloaded automatically on first boot via the startup script.
On subsequent reboots, cached weights on `/mnt/stateful_partition/` persist.

To pre-bake weights into the Docker image (faster cold start, larger image):
```dockerfile
COPY weights/rwkv7-0.1b/ /app/weights/rwkv7-0.1b/
```

## Scaling

The server is single-threaded by design (compression is CPU-bound and
sequential). For higher throughput:

1. **Vertical**: use a larger machine (c3-highcpu-22)
2. **Horizontal**: run multiple instances behind a TCP load balancer
   (each instance handles one request at a time)

## Security Notes

- SSH restricted to IAP tunnel (no public key exposure)
- Service account has minimal permissions (AR read + logging)
- No secrets in Terraform state (weights are public Apache-2.0)
- TLS termination: add a Cloud Load Balancer with managed certificate
  for production HTTPS (out of scope for initial deploy)
