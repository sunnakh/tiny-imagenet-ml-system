terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6, < 8"
    }
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type        = string
  description = "Artifact Registry image URL pinned to a digest."
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image))
    error_message = "Set an image URL with a sha256 digest."
  }
}

variable "runtime_service_account" {
  type = string
}

resource "google_cloud_run_v2_service" "classifier" {
  name                = "tiny-imagenet"
  project             = var.project_id
  location            = var.region
  deletion_protection = false

  template {
    service_account                  = var.runtime_service_account
    max_instance_request_concurrency = 1
    timeout                          = "15s"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      resources {
        cpu_idle = true
        limits   = { cpu = "2", memory = "2Gi" }
      }
    }
  }
}

output "service_url" {
  value = google_cloud_run_v2_service.classifier.uri
}
