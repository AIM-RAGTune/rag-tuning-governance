#!/usr/bin/env bash
set -euo pipefail
ragtune run-public-mini --output-root "${RAGTUNE_OUTPUT_ROOT:-/outputs/public_mini_reproduction}" --force
