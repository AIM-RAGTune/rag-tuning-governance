#!/usr/bin/env bash
set -euo pipefail
image_ref="$(deploy/load-image-reference.sh)"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT
sed "s#ghcr.io/aim-ragtune/rag-tuning-governance@sha256:PENDING_FIRST_WORKFLOW_RUN#$image_ref#g" \
  deploy/aws/ecs-fargate-task.json > "$tmp_file"
aws ecs register-task-definition --cli-input-json "file://$tmp_file"
