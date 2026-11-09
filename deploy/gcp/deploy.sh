#!/usr/bin/env bash
# Idempotent deployment wrapper for Redukt on Google Cloud.
#
#   ./deploy.sh [all|setup|weights|build|run|batch]          (default: all = setup weights build run)
#
#   setup    enable APIs, Artifact Registry repo, buckets (weights/in/out), runtime service account + IAM
#   weights  upload ./weights (rwkv7-0.1b, mamba-130m, optional <GPU_SUBDIR>/model.rbin) + SHA256SUMS
#   build    Cloud Build (deploy/gcp/cloudbuild.yaml): CPU image, CUDA image if BUILD_CUDA=true
#   run      deploy the Cloud Run service `redukt-api` (deploy/gcp/cloudrun-service.yaml)
#   batch    submit a Cloud Batch job (deploy/gcp/batch-job.json) that compresses gs://$IN_BUCKET -> gs://$OUT_BUCKET
#   render F print cloudrun-service.yaml / batch-job.json with the current variables substituted (no GCP calls)
#
# Every step is safe to re-run.  DRY_RUN=1 prints the mutating gcloud commands instead of executing them
# (read-only describe/list calls still run).
#
# References (consulted 2026-11-09): see deploy/gcp/README.md, section "Fuentes oficiales".
set -euo pipefail

# ---- variables ---------------------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
# EU data residency.  europe-southwest1 = Madrid.  Alternatives: europe-west1 (Belgium), europe-west4
# (Netherlands), europe-west3 (Frankfurt), europe-west9 (Paris), europe-west8 (Milan), europe-west6 (Zurich).
# Check that the machine types / GPUs you need exist there (see README: Madrid has C4 but no C3, no L4/T4).
REGION="${REGION:-europe-southwest1}"
BUCKET_LOCATION="${BUCKET_LOCATION:-$REGION}"          # or the multi-region `EU`
REPO="${REPO:-redukt}"                                 # Artifact Registry repository
SERVICE="${SERVICE:-redukt-api}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-redukt-runtime}"
WEIGHTS_BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-redukt-weights}"
IN_BUCKET="${IN_BUCKET:-${PROJECT_ID}-redukt-in}"      # may be `bucket/prefix` for batch (flat directory, non-recursive)
OUT_BUCKET="${OUT_BUCKET:-${PROJECT_ID}-redukt-out}"
LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-./weights}"
GPU_WEIGHTS_SUBDIR="${GPU_WEIGHTS_SUBDIR:-mambabyte-353m}"
BUILD_CUDA="${BUILD_CUDA:-false}"
IMAGE_TAG_SUFFIX="${IMAGE_TAG_SUFFIX:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE_TAG="${IMAGE_TAG:-cpu-${IMAGE_TAG_SUFFIX}}"      # tag deployed to Cloud Run / Batch

# Batch defaults.  c3-highcpu-22 is NOT offered in Madrid (checked 2026-11-09 with
# `gcloud compute machine-types list`); c4-highcpu-* is, and C4 only supports Hyperdisk (boot disk).
if [ "$REGION" = europe-southwest1 ]; then
  : "${MACHINE_TYPE:=c4-highcpu-32}" "${BOOT_DISK_TYPE:=hyperdisk-balanced}" "${VCPUS:=32}" "${MEMORY_MIB:=56000}"
else
  : "${MACHINE_TYPE:=c3-highcpu-22}" "${BOOT_DISK_TYPE:=pd-balanced}" "${VCPUS:=22}" "${MEMORY_MIB:=40000}"
fi
TASK_COUNT="${TASK_COUNT:-1}"                          # 1 task = 1 VM (taskCountPerNode=1)
PARALLELISM="${PARALLELISM:-$TASK_COUNT}"
JOBS="${JOBS:-$VCPUS}"                                 # redukt --jobs per task
CPU_MILLI="${CPU_MILLI:-$((VCPUS * 1000))}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"   # STANDARD | SPOT
MAX_RUN_DURATION="${MAX_RUN_DURATION:-86400s}"
WEIGHTS_SET="${WEIGHTS_SET:-rwkv}"                     # batch mode is RWKV-only today (codec.rs batch_compress)
REDUKT_EXTRA_ARGS="${REDUKT_EXTRA_ARGS:-}"

RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# ---- helpers -----------------------------------------------------------------------------------------
log() { printf '\033[1m[deploy]\033[0m %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then { printf '[dry-run]'; printf ' %q' "$@"; echo; } >&2; else "$@"; fi; }

# render FILE: replace ${VAR} placeholders (only the listed variables) -- no envsubst dependency
RENDER_VARS="PROJECT_ID REGION REPO IMAGE_TAG WEIGHTS_BUCKET IN_BUCKET OUT_BUCKET RUNTIME_SA TASK_COUNT PARALLELISM \
JOBS MACHINE_TYPE BOOT_DISK_TYPE PROVISIONING_MODEL CPU_MILLI MEMORY_MIB MAX_RUN_DURATION WEIGHTS_SET REDUKT_EXTRA_ARGS"
render() {
  local out v; out="$(cat "$1")"
  for v in $RENDER_VARS; do out="${out//\$\{$v\}/${!v}}"; done
  printf '%s\n' "$out"
}

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"; }
need gcloud
[ -n "$PROJECT_ID" ] || die "PROJECT_ID not set and no default gcloud project"

# ---- steps -------------------------------------------------------------------------------------------
step_setup() {
  log "project=$PROJECT_ID region=$REGION"
  run gcloud services enable --project "$PROJECT_ID" \
    run.googleapis.com batch.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com compute.googleapis.com logging.googleapis.com

  if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT_ID" >/dev/null 2>&1; then
    run gcloud artifacts repositories create "$REPO" --repository-format=docker \
      --location "$REGION" --project "$PROJECT_ID" --description "Redukt images"
  fi

  local b
  for b in "$WEIGHTS_BUCKET" "${IN_BUCKET%%/*}" "${OUT_BUCKET%%/*}"; do
    if ! gcloud storage buckets describe "gs://$b" --project "$PROJECT_ID" >/dev/null 2>&1; then
      run gcloud storage buckets create "gs://$b" --project "$PROJECT_ID" --location "$BUCKET_LOCATION" \
        --uniform-bucket-level-access --public-access-prevention
    fi
  done

  if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project "$PROJECT_ID" >/dev/null 2>&1; then
    run gcloud iam service-accounts create "$RUNTIME_SA_NAME" --project "$PROJECT_ID" \
      --display-name "Redukt runtime (Cloud Run / Batch / GCE GPU)"
  fi
  local role
  # Batch job SA roles per Batch "get started" doc: batch.agentReporter + logging.logWriter.
  # artifactregistry.reader: pull the image (not stated in the Batch doc; standard role for pulls).
  for role in roles/batch.agentReporter roles/logging.logWriter roles/artifactregistry.reader; do
    run gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None \
      --member "serviceAccount:$RUNTIME_SA" --role "$role" >/dev/null
  done
  run gcloud storage buckets add-iam-policy-binding "gs://$WEIGHTS_BUCKET" \
    --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectViewer >/dev/null
  run gcloud storage buckets add-iam-policy-binding "gs://${IN_BUCKET%%/*}" \
    --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectViewer >/dev/null
  run gcloud storage buckets add-iam-policy-binding "gs://${OUT_BUCKET%%/*}" \
    --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectUser >/dev/null
}

step_weights() {
  [ -d "$LOCAL_WEIGHTS" ] || die "$LOCAL_WEIGHTS not found (run scripts/download-weights.sh and scripts/download-mamba-weights.sh)"
  local sums; sums="$(mktemp)"; : >"$sums"
  local f d
  # Files the entrypoint expects (see deploy/gcp/entrypoint.sh required_files)
  for f in rwkv7-0.1b/model.safetensors rwkv7-0.1b/rwkv_vocab_v20230424.txt \
           mamba-130m/model.safetensors mamba-130m/tokenizer.json "$GPU_WEIGHTS_SUBDIR/model.rbin"; do
    if [ -f "$LOCAL_WEIGHTS/$f" ]; then
      (cd "$LOCAL_WEIGHTS" && sha256 "$f") >>"$sums"
    else
      log "skip (not found locally): $f"
    fi
  done
  [ -s "$sums" ] || die "no weight files found under $LOCAL_WEIGHTS"
  log "SHA256SUMS:"; cat "$sums" >&2

  for d in rwkv7-0.1b mamba-130m; do
    [ -d "$LOCAL_WEIGHTS/$d" ] || continue
    # exclude leftovers that are not read by redukt (.pth/.bin/.md/.cache)
    run gcloud storage rsync --recursive --project "$PROJECT_ID" \
      --exclude='.*\.(pth|bin|md|gitattributes)$|.*/\.cache/.*' \
      "$LOCAL_WEIGHTS/$d" "gs://$WEIGHTS_BUCKET/$d"
  done
  if [ -f "$LOCAL_WEIGHTS/$GPU_WEIGHTS_SUBDIR/model.rbin" ]; then
    run gcloud storage cp --project "$PROJECT_ID" \
      "$LOCAL_WEIGHTS/$GPU_WEIGHTS_SUBDIR/model.rbin" "gs://$WEIGHTS_BUCKET/$GPU_WEIGHTS_SUBDIR/model.rbin"
  fi
  run gcloud storage cp --project "$PROJECT_ID" "$sums" "gs://$WEIGHTS_BUCKET/SHA256SUMS"
  rm -f "$sums"
}

step_build() {
  log "Cloud Build: tag=$IMAGE_TAG_SUFFIX cuda=$BUILD_CUDA"
  # gcloud builds submit ignores .dockerignore -> explicit ignore file
  run gcloud builds submit "$REPO_ROOT" --project "$PROJECT_ID" --region "$REGION" \
    --config "$HERE/cloudbuild.yaml" --ignore-file "$HERE/.gcloudignore" \
    --substitutions "_REGION=$REGION,_REPO=$REPO,_TAG=$IMAGE_TAG_SUFFIX,_BUILD_CUDA=$BUILD_CUDA"
}

step_run() {
  local tmp; tmp="$(mktemp)"
  render "$HERE/cloudrun-service.yaml" >"$tmp"
  local extra=(); if [ "${DRY_RUN:-0}" = 1 ]; then extra=(--dry-run); fi
  log "deploying Cloud Run service $SERVICE (image tag $IMAGE_TAG)"
  gcloud run services replace "$tmp" --project "$PROJECT_ID" --region "$REGION" "${extra[@]+"${extra[@]}"}"
  rm -f "$tmp"
  gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" \
    --format='value(status.url)' 2>/dev/null || true
  log "the service is private by default; call it with: curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" <url>/v1/status"
}

step_batch() {
  local tmp name; tmp="$(mktemp)"
  name="redukt-$(date +%Y%m%d-%H%M%S)"
  render "$HERE/batch-job.json" >"$tmp"
  log "submitting Batch job $name: $TASK_COUNT task(s) on $MACHINE_TYPE ($PROVISIONING_MODEL), $JOBS jobs/task"
  run gcloud batch jobs submit "$name" --project "$PROJECT_ID" --location "$REGION" --config "$tmp"
  rm -f "$tmp"
  log "follow with: gcloud batch jobs describe $name --location $REGION --project $PROJECT_ID"
}

case "${1:-all}" in
  all)     step_setup; step_weights; step_build; step_run ;;
  setup)   step_setup ;;
  weights) step_weights ;;
  build)   step_build ;;
  run)     step_run ;;
  batch)   step_batch ;;
  render)  render "${2:?usage: $0 render <cloudrun-service.yaml|batch-job.json>}"; exit 0 ;;   # debugging aid
  *) die "usage: $0 [all|setup|weights|build|run|batch|render FILE]" ;;
esac
log "done"
