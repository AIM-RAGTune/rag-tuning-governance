#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../load-image-reference.sh" >/dev/null
: "${RAGTUNE_AWS_CLUSTER:?set RAGTUNE_AWS_CLUSTER}"
: "${RAGTUNE_AWS_SUBNET:?set RAGTUNE_AWS_SUBNET}"
: "${RAGTUNE_AWS_SECURITY_GROUP:?set RAGTUNE_AWS_SECURITY_GROUP}"
aws ecs run-task \
  --cluster "$RAGTUNE_AWS_CLUSTER" \
  --launch-type FARGATE \
  --task-definition ragtune-governance-task \
  --network-configuration "awsvpcConfiguration={subnets=[$RAGTUNE_AWS_SUBNET],securityGroups=[$RAGTUNE_AWS_SECURITY_GROUP],assignPublicIp=DISABLED}"
