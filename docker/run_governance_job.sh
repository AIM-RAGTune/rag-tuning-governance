#!/usr/bin/env bash
set -euo pipefail
ragtune run-governance-job \
  --config "${RAGTUNE_JOB_CONFIG:-configs/jobs/public_mini_governance_job.yaml}" \
  --output-root "${RAGTUNE_OUTPUT_ROOT:-/outputs}" \
  --decision-out "${RAGTUNE_DECISION_OUT:-/outputs/promotion_decision.json}"
