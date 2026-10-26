output "api_url" {
  description = "Redukt API endpoint"
  value       = "http://${google_compute_address.redukt.address}:8080"
}

output "instance_name" {
  description = "GCE instance name"
  value       = google_compute_instance.redukt.name
}

output "ssh_command" {
  description = "SSH via IAP tunnel"
  value       = "gcloud compute ssh ${google_compute_instance.redukt.name} --zone=${var.zone} --tunnel-through-iap"
}

output "image_push_command" {
  description = "Push container image to Artifact Registry"
  value       = "docker push ${local.image_uri}"
}
