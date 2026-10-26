terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state — uncomment and configure for team use:
  # backend "gcs" {
  #   bucket = "redukt-tfstate"
  #   prefix = "prod"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Artifact Registry (container images) ---

resource "google_artifact_registry_repository" "redukt" {
  location      = var.region
  repository_id = "redukt"
  format        = "DOCKER"
  description   = "Redukt container images"
}
