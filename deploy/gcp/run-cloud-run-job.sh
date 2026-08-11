#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_GCP_REGION:?set RAGTUNE_GCP_REGION}"
gcloud run jobs execute ragtune-governance-job --region "$RAGTUNE_GCP_REGION" --wait
