#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_AZURE_RESOURCE_GROUP:?set RAGTUNE_AZURE_RESOURCE_GROUP}"
az deployment group create \
  --resource-group "$RAGTUNE_AZURE_RESOURCE_GROUP" \
  --template-file deploy/azure/container-apps-job.bicep \
  --parameters @deploy/azure/container-apps-job.parameters.example.json
