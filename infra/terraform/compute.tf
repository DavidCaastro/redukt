# --- Service Account ---

resource "google_service_account" "redukt" {
  account_id   = "redukt-api"
  display_name = "Redukt API Server"
}

resource "google_project_iam_member" "redukt_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.redukt.email}"
}

resource "google_project_iam_member" "redukt_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.redukt.email}"
}

resource "google_project_iam_member" "redukt_metrics" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.redukt.email}"
}

# --- GCE Instance ---

locals {
  image_uri = "${var.region}-docker.pkg.dev/${var.project_id}/redukt/redukt:${var.image_tag}"
}

resource "google_compute_instance" "redukt" {
  name         = "redukt-api"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["redukt-api"]

  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.redukt.id
    access_config {
      nat_ip = google_compute_address.redukt.address
    }
  }

  service_account {
    email  = google_service_account.redukt.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    # Container-Optimized OS runs containers via cloud-init
    gce-container-declaration = yamlencode({
      spec = {
        containers = [{
          name  = "redukt"
          image = local.image_uri
          env   = []
          volumeMounts = [{
            name      = "weights"
            mountPath = "/app/weights/rwkv7-0.1b"
            readOnly  = true
          }]
        }]
        volumes = [{
          name = "weights"
          hostPath = {
            path = "/mnt/stateful_partition/weights/rwkv7-0.1b"
          }
        }]
        restartPolicy = "Always"
      }
    })
  }

  # Startup script: download weights if not cached on persistent disk
  metadata_startup_script = <<-SCRIPT
    #!/bin/bash
    WEIGHTS_DIR="/mnt/stateful_partition/weights/rwkv7-0.1b"
    if [ ! -f "$WEIGHTS_DIR/model.safetensors" ]; then
      echo "Downloading weights..."
      mkdir -p "$WEIGHTS_DIR"
      # Install Python in toolbox for weight conversion
      toolbox --no-tty run apt-get update
      toolbox --no-tty run apt-get install -y python3 python3-pip
      toolbox --no-tty run pip3 install torch safetensors huggingface_hub
      toolbox --no-tty run python3 -c "
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
pth = hf_hub_download(repo_id='BlinkDL/rwkv-7-world',
    filename='RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth')
sd = torch.load(pth, map_location='cpu', weights_only=True)
save_file(sd, '/media/root/weights/rwkv7-0.1b/model.safetensors')
"
      # Download tokenizer
      curl -sL "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/tokenizer/rwkv_vocab_v20230424.txt" \
        -o "$WEIGHTS_DIR/rwkv_vocab_v20230424.txt"
      echo "Weights ready."
    else
      echo "Weights already present."
    fi
  SCRIPT

  allow_stopping_for_update = true
}
