#!/usr/bin/env bash
set -euo pipefail
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs-fargate-task.json
