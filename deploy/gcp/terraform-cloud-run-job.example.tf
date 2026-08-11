variable "project_id" {
  type    = string
  default = "example-project"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "image" {
  type    = string
  default = "us-docker.pkg.dev/example-project/example-repo/ragtune-governance:latest"
}

resource "google_cloud_run_v2_job" "ragtune" {
  name     = "ragtune-governance-job"
  location = var.region
  project  = var.project_id

  template {
    template {
      containers {
        image = var.image
        args = [
          "run-governance-job",
          "--config",
          "configs/jobs/public_mini_governance_job.yaml",
          "--output-root",
          "/outputs",
          "--decision-out",
          "/outputs/promotion_decision.json",
        ]
      }
      max_retries = 0
    }
  }
}
