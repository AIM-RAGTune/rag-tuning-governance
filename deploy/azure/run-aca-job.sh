#!/usr/bin/env bash
set -euo pipefail
: "${RAGTUNE_AZURE_RESOURCE_GROUP:?set RAGTUNE_AZURE_RESOURCE_GROUP}"
: "${RAGTUNE_AZURE_JOB_NAME:=ragtune-governance-job}"
az containerapp job start --resource-group "$RAGTUNE_AZURE_RESOURCE_GROUP" --name "$RAGTUNE_AZURE_JOB_NAME"
