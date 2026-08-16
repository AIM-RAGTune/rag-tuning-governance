#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../load-image-reference.sh" >/dev/null
: "${RAGTUNE_AWS_BATCH_QUEUE:?set RAGTUNE_AWS_BATCH_QUEUE}"
aws batch submit-job \
  --job-name ragtune-governance-job \
  --job-queue "$RAGTUNE_AWS_BATCH_QUEUE" \
  --job-definition ragtune-governance-batch
