# --- VPC ---

resource "google_compute_network" "redukt" {
  name                    = "redukt-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "redukt" {
  name          = "redukt-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.redukt.id
}

# --- Firewall ---

resource "google_compute_firewall" "allow_api" {
  name    = "redukt-allow-api"
  network = google_compute_network.redukt.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["redukt-api"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "redukt-allow-ssh"
  network = google_compute_network.redukt.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # IAP tunnel range — restrict SSH to Google Identity-Aware Proxy only
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["redukt-api"]
}

resource "google_compute_firewall" "allow_health" {
  name    = "redukt-allow-health"
  network = google_compute_network.redukt.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  # GCP health check ranges
  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["redukt-api"]
}

# --- Static IP ---

resource "google_compute_address" "redukt" {
  name   = "redukt-api-ip"
  region = var.region
}
