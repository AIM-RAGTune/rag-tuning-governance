#!/usr/bin/env bash
set -euo pipefail
ragtune inspect-environment >/tmp/ragtune_environment.json
ragtune validate-bundle
