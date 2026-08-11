# Docker Hardening Post-Validation Report

Repository path: `<public-ragtune-repository>`

Branch: `codex/docker-runtime-hardening`

Starting commit: `f9a67ac89bfe75b0dcd286d5b9bc22c839ee317f`

## Runtime Diagnostic

- Result class: `CONTAINER_RUNTIME_DOCKER_READY`
- Docker CLI present: yes
- Docker daemon available: yes
- Docker Compose plugin available: yes
- Legacy `docker-compose` available: yes
- Buildx available: yes
- Podman ready: no
- Colima ready: no

The local runtime path completed after Docker Desktop was started and the Docker socket became readable from the validation shell.

## Static Docker Validation

- Result class: `DOCKER_STATIC_VALIDATION_PASSED`
- Checks passed: 21 / 21
- Checks failed: 0

Static checks cover the Dockerfile, `.dockerignore`, Docker Compose public-mini path, helper scripts, Makefile Docker targets, public-mini job config, promotion decision schema, and Docker documentation boundaries.

## Container Smoke Test

- Result class: `DOCKER_RUNTIME_VALIDATED_PUBLIC_MINI`
- Skip reason: none
- Docker build: passed
- `ragtune --help` in container: passed
- `ragtune validate-bundle` in container: passed
- Public-mini governance job in container: passed
- Docker Compose public-mini: passed

## Optional Security Scans

- Result class: `CONTAINER_SECURITY_SCANS_SKIPPED_TOOLS_UNAVAILABLE`
- Critical findings: 0

Optional scanners were not installed locally. Their absence is recorded as skipped and does not upgrade or weaken the runtime validation claim.

## Deployment Readiness

- Result class: `DEPLOYMENT_READINESS_SUPPORTED_WITH_BOUNDARIES`
- Live cloud execution: not run, no credentials
- Official platform benchmarking claimed: no
- Production readiness claimed: no

## Validation

- `PYTHONPATH=src python3 scripts/validate_publication_bundle.py`: passed
- `PYTHONPATH=src pytest -q tests/publication`: 169 passed
- `PYTHONPATH=src make validate-publication`: passed
- `PYTHONPATH=src make test`: 169 passed
- `PYTHONPATH=src python3 -m compileall src scripts`: passed
- `git diff --check`: passed

## Hygiene

- Tracked large-file scan: passed
- Broad large-file scan: only an ignored `.local_data` HotpotQA cache file was present locally
- Raw text scan: passed with expected sanitized field names, scanner patterns, and false flags
- Secret scan: passed with expected environment-variable names, scanner patterns, and `local-not-secret` placeholder only
- Private path scan: passed with expected negative-test string only
- Overclaim scan: passed with explicit unsupported-claim statements and scanner definitions

## Claim Boundaries

This hardening pass does not claim live cloud validation, official platform benchmarking, production operation, human validation, RAG Compass superiority, hallucination elimination, or stable generative cost/latency superiority.
