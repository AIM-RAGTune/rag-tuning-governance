#!/usr/bin/env bash
set -euo pipefail

python scripts/validate_publication_bundle.py
bash scripts/reproduce_dataset_matrix.sh
bash scripts/reproduce_multihop_confirmatory.sh
bash scripts/reproduce_crag_mock_api.sh
