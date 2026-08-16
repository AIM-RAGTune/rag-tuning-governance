#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
digest_file="${RAGTUNE_IMAGE_DIGEST_FILE:-$repo_root/deploy/IMAGE_DIGEST}"

if [[ ! -f "$digest_file" ]]; then
  echo "Missing image digest file: $digest_file" >&2
  exit 2
fi

reference="$(awk -F= '$1 == "REFERENCE" {print $2}' "$digest_file")"
digest="$(awk -F= '$1 == "DIGEST" {print $2}' "$digest_file")"

if [[ -z "$reference" || "$reference" == "PENDING_FIRST_WORKFLOW_RUN" || "$digest" == "PENDING_FIRST_WORKFLOW_RUN" ]]; then
  echo "Image digest is pending; run the GHCR publish workflow and update deploy/IMAGE_DIGEST before deployment." >&2
  exit 2
fi

if [[ "$reference" != *@sha256:* ]]; then
  echo "Image reference must be digest-pinned: $reference" >&2
  exit 2
fi

printf '%s\n' "$reference"

