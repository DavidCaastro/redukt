variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "europe-west1-b"
}

variable "machine_type" {
  description = "GCE instance machine type"
  type        = string
  default     = "c3-highcpu-8"
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 30
}

variable "image_tag" {
  description = "Container image tag in Artifact Registry"
  type        = string
  default     = "latest"
}
