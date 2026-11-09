#!/usr/bin/env bash
# Create (or delete / ssh into) a single-GPU Compute Engine VM that runs the Redukt CUDA image.
#
#   ./gce-cuda.sh [create|delete|ssh|status]            (default: create)
#   GPU=t4 ./gce-cuda.sh                                 # T4 on n1-standard-4 instead of L4 on g2-standard-4
#   SPOT=1 ./gce-cuda.sh                                 # Spot VM
#
# References (consulted 2026-11-09):
#   https://docs.cloud.google.com/compute/docs/gpus                           (G2 = L4 attached automatically; N1 + T4)
#   https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series    (G2 create command, --maintenance-policy=TERMINATE)
#   https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose (N1: --accelerator type=nvidia-tesla-t4,count=1)
#   https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu       (cuda_installer.pyz)
#   https://docs.cloud.google.com/deep-learning-vm/docs/images                (project deeplearning-platform-release, family names)
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#   https://docs.docker.com/engine/install/ubuntu/
#
# NOT EXECUTED against a real project when this file was written: the gcloud flags were checked
# against `gcloud ... --help` (SDK 548) and the docs above; the startup script was not run on a GPU VM.
set -euo pipefail

ACTION="${1:-create}"

# ---- configuration (env overridable) -----------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
GPU="${GPU:-l4}"                                   # l4 | t4
SPOT="${SPOT:-0}"                                  # 1 -> --provisioning-model=SPOT
AR_REGION="${AR_REGION:-europe-southwest1}"        # region of the Artifact Registry repo (deploy.sh REGION)
REPO="${REPO:-redukt}"
IMAGE_TAG="${IMAGE_TAG:-cuda-latest}"
IMAGE_URI="${IMAGE_URI:-${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/redukt:${IMAGE_TAG}}"
WEIGHTS_BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-redukt-weights}"
GPU_WEIGHTS_SUBDIR="${GPU_WEIGHTS_SUBDIR:-mambabyte-353m}"   # must contain model.rbin (byte-level vocab=256)
RUNTIME_SA="${RUNTIME_SA:-redukt-runtime@${PROJECT_ID}.iam.gserviceaccount.com}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"          # docs: at least 40 GB for GPU VMs
# Deep Learning VM image (driver preinstalled).  Family taken from the DLVM image table, 2026-11-09.
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
# Alternative without DLVM: IMAGE_PROJECT=ubuntu-os-cloud IMAGE_FAMILY=ubuntu-2204-lts  (the startup
# script then installs the NVIDIA driver with cuda_installer.pyz and reboots once if needed).

# L4 / T4 are NOT offered in europe-southwest1 (Madrid): checked with
# `gcloud compute accelerator-types list` on 2026-11-09.  EU zones that had them:
#   L4 : europe-west1-b/c, europe-west2-a/b, europe-west3-a/b, europe-west4-a/b/c, europe-west6-b/c
#   T4 : europe-west1-b/c/d, europe-west2-a/b, europe-west3-b, europe-west4-a/b/c, europe-central2-b/c
GPU_FLAGS=()
case "$GPU" in
  l4)
    ZONE="${ZONE:-europe-west4-a}"
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"  # G2: the L4 is attached automatically, no --accelerator
    ;;
  t4)
    ZONE="${ZONE:-europe-west4-b}"
    MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
    GPU_FLAGS=(--accelerator "type=nvidia-tesla-t4,count=1")
    ;;
  *) echo "GPU must be l4 or t4" >&2; exit 2 ;;
esac
INSTANCE="${INSTANCE:-redukt-gpu-${GPU}}"

[ -n "$PROJECT_ID" ] || { echo "PROJECT_ID not set (and no gcloud default project)" >&2; exit 2; }

# ---- startup script (runs as root on every boot; idempotent) -----------------------------------
# Configuration comes from instance metadata attributes, so this script has no baked-in values.
STARTUP="$(mktemp)"
trap 'rm -f "$STARTUP"' EXIT
cat >"$STARTUP" <<'STARTUP_EOF'
#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/redukt-startup.log) 2>&1
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1"; }
IMAGE_URI="$(md redukt-image)"; WEIGHTS_BUCKET="$(md redukt-weights-bucket)"
GPU_SUBDIR="$(md redukt-gpu-weights-subdir || echo mambabyte-353m)"
REGISTRY_HOST="${IMAGE_URI%%/*}"
echo "[startup] $(date -Is) image=$IMAGE_URI"

# 1. NVIDIA driver (preinstalled on Deep Learning VM images; otherwise install per the GCP docs)
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[startup] no working NVIDIA driver, installing with cuda_installer.pyz"
  apt-get update -y && apt-get install -y python3 curl
  curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    --output /root/cuda_installer.pyz
  python3 /root/cuda_installer.pyz install_driver --installation-branch=lts || true
  # docs: if the instance restarts during installation, run the same command again (this script re-runs at boot)
  nvidia-smi >/dev/null 2>&1 || { echo "[startup] driver installed, rebooting"; reboot; exit 0; }
fi
nvidia-smi

# 2. Docker Engine (official apt repository, https://docs.docker.com/engine/install/ubuntu/)
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat >/etc/apt/sources.list.d/docker.sources <<DOCKER_SRC
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
DOCKER_SRC
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# 3. NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl gnupg2
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
fi
if ! grep -q nvidia /etc/docker/daemon.json 2>/dev/null; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# 4. Artifact Registry auth (gcloud auth configure-docker, or an OAuth token if gcloud is absent)
if command -v gcloud >/dev/null 2>&1; then
  gcloud auth configure-docker "$REGISTRY_HOST" --quiet
else
  curl -sf -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
    | sed -n 's/.*"access_token" *: *"\([^"]*\)".*/\1/p' \
    | docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY_HOST"
fi
docker pull "$IMAGE_URI"

# 5. Weights (GPU set = <subdir>/model.rbin) cached on the VM disk, owned by the image user (uid 10001)
install -d -o 10001 -g 10001 /var/lib/redukt/weights /var/lib/redukt/data
docker run --rm -v /var/lib/redukt/weights:/weights \
  -e WEIGHTS_BUCKET="$WEIGHTS_BUCKET" -e WEIGHTS_SET=gpu -e GPU_WEIGHTS_SUBDIR="$GPU_SUBDIR" \
  --entrypoint redukt-entrypoint.sh "$IMAGE_URI" fetch-weights \
  || echo "[startup] WARNING: weights not fetched (is $GPU_SUBDIR/model.rbin in gs://$WEIGHTS_BUCKET?)"

# 6. Wrapper:  redukt-gpu compress --backbone plugin --weights /weights/<subdir> --input /data/f --output /data/f.azth
cat >/usr/local/bin/redukt-gpu <<WRAPPER
#!/usr/bin/env bash
exec docker run --rm --gpus all -v /var/lib/redukt/weights:/weights:ro -v /var/lib/redukt/data:/data \\
  "$IMAGE_URI" "\$@"
WRAPPER
chmod 0755 /usr/local/bin/redukt-gpu

# 7. Smoke test: GPU visible from the container
docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE_URI" -L && touch /var/lib/redukt/ready
echo "[startup] done"
STARTUP_EOF

# ---- actions -------------------------------------------------------------------------------------
instance_exists() {
  gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --project "$PROJECT_ID" >/dev/null 2>&1
}

case "$ACTION" in
  create)
    if instance_exists; then
      echo "instance $INSTANCE already exists in $ZONE (idempotent no-op); '$0 delete' first to recreate"
      exit 0
    fi
    SPOT_FLAGS=()
    if [ "$SPOT" = 1 ]; then SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP); fi
    gcloud compute instances create "$INSTANCE" \
      --project "$PROJECT_ID" --zone "$ZONE" \
      --machine-type "$MACHINE_TYPE" \
      "${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"}" \
      --maintenance-policy TERMINATE --restart-on-failure \
      --boot-disk-size "$BOOT_DISK_SIZE" \
      --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
      --service-account "$RUNTIME_SA" --scopes cloud-platform \
      --metadata "redukt-image=${IMAGE_URI},redukt-weights-bucket=${WEIGHTS_BUCKET},redukt-gpu-weights-subdir=${GPU_WEIGHTS_SUBDIR}" \
      --metadata-from-file "startup-script=${STARTUP}" \
      "${SPOT_FLAGS[@]+"${SPOT_FLAGS[@]}"}"
    echo "created $INSTANCE ($MACHINE_TYPE, GPU=$GPU, zone $ZONE). Startup log: /var/log/redukt-startup.log;"
    echo "ready when /var/lib/redukt/ready exists.  Use:  $0 ssh   then   redukt-gpu --help"
    ;;
  delete)
    gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --project "$PROJECT_ID" --quiet
    ;;
  ssh)
    gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT_ID"
    ;;
  status)
    gcloud compute instances describe "$INSTANCE" --zone "$ZONE" --project "$PROJECT_ID" \
      --format='value(name,status,machineType.basename(),guestAccelerators)'
    ;;
  *) echo "usage: $0 [create|delete|ssh|status]" >&2; exit 2 ;;
esac
