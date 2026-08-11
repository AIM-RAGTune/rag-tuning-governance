#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_GCP_REGION:?set RAGTUNE_GCP_REGION}"
gcloud run jobs replace deploy/gcp/cloud-run-job.yaml --region "$RAGTUNE_GCP_REGION"
