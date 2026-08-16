#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_GCP_REGION:?set RAGTUNE_GCP_REGION}"
image_ref="$(deploy/load-image-reference.sh)"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT
sed "s#ghcr.io/aim-ragtune/rag-tuning-governance@sha256:PENDING_FIRST_WORKFLOW_RUN#$image_ref#g" \
  deploy/gcp/cloud-run-job.yaml > "$tmp_file"
gcloud run jobs replace "$tmp_file" --region "$RAGTUNE_GCP_REGION"
