# Mounted Governance-Job I/O Contract

RAGTune cloud jobs are finite batch jobs. They read sanitized configuration from a mounted input directory, write sanitized artifacts to a mounted output directory, emit `promotion_decision.json`, and exit with a machine-readable status.

## Runtime mounts

- Input mount: `/inputs`
- Output mount: `/outputs`
- Optional config override: `--config <path>` or `RAGTUNE_INPUT_DIR/<path>`
- Optional output override: `--output-root <path>` or `RAGTUNE_OUTPUT_DIR`
- Optional decision override: `--decision-out <path>`

The default runtime image sets:

```text
RAGTUNE_CONTAINER=1
RAGTUNE_REPO_ROOT=/app
RAGTUNE_INPUT_DIR=/inputs
RAGTUNE_OUTPUT_DIR=/outputs
RAGTUNE_OUTPUT_ROOT=/outputs
```

## Exit codes

- `0`: job completed and publication validator passed.
- `2`: config or required input was missing.
- `3`: publication validator or governance gate failed.
- `4`: runtime execution failure.

## Required outputs

Every mounted governance job writes:

- `promotion_decision.json`
- `run_manifest.json`
- `validation_report.json`
- `validation_report.md`

The decision file includes the result class, selected policy, baseline policy, deltas when available, claim-boundary flags, and artifact URIs. Public artifacts must never include raw CRAG questions, raw HotpotQA questions, raw source text, raw prompts, raw generated answers, secrets, private local paths, or cloud account identifiers.

## Storage staging

The publication repository includes local staging and object-storage adapters for S3-compatible storage, Azure Blob, and GCS-compatible storage. Local staging is executable for smoke tests. Object-storage modes require optional SDKs and endpoint configuration outside the base runtime image.

`scripts/run_storage_emulator_tests.sh` runs the shared emulator-backed validation path against MinIO, Azurite, and fake-gcs-server using pinned image digests. The same script is used by the `storage-staging-validation` GitHub Actions workflow. Emulator credentials are local, non-secret test credentials. Passing emulator validation does not claim live AWS, Azure, GCP, platform-native benchmark, or production evidence.

No real cloud deployment is performed by the repository tests or runbooks. Cloud templates are digest-pinned examples that require operator review and externally supplied cloud configuration.

## Verification

Run the contract check locally:

```bash
python3 scripts/check_mounted_job_contract.py --output-root /tmp/ragtune-contract-check
```

Run the contract check inside the container:

```bash
docker run --rm \
  -v "$PWD/configs/jobs:/inputs:ro" \
  -v "$PWD/.local_outputs:/outputs" \
  ragtune:wp1 run-governance-job
```

The container command writes sanitized outputs under the mounted `/outputs` directory.
