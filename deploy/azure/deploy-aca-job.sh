#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_AZURE_RESOURCE_GROUP:?set RAGTUNE_AZURE_RESOURCE_GROUP}"
image_ref="$(deploy/load-image-reference.sh)"
az deployment group create \
  --resource-group "$RAGTUNE_AZURE_RESOURCE_GROUP" \
  --template-file deploy/azure/container-apps-job.bicep \
  --parameters image="$image_ref"
