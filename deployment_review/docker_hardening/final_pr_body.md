## Summary

Hardens the local Docker/container runtime path for the public RAGTune repository. This adds safe runtime diagnostics, static Docker validation, a public-mini container smoke-test runner, optional security-scan reporting, Docker Compose documentation, and deployment-readiness integration.

The local Docker CLI is present, but no usable Docker daemon or alternate container machine was available. The PR therefore records a bounded runtime skip after static Docker/package/deployment checks passed.

## Runtime diagnostics

- Runtime diagnostic result: `CONTAINER_RUNTIME_CLI_PRESENT_DAEMON_UNAVAILABLE`
- Docker CLI present: yes
- Docker daemon available: no
- Docker Compose plugin available: yes
- Legacy `docker-compose` available: yes
- Buildx available: yes
- Podman ready: no
- Colima ready: no

## Docker static validation

- Result class: `DOCKER_STATIC_VALIDATION_PASSED`
- Checks passed: 17 / 17
- Dockerfile: hardened with container contract, non-root runtime, `/outputs` support, and healthcheck
- `.dockerignore`: hardened for local data, env files, secrets, caches, model/data binaries, and raw artifact patterns
- Compose path: `docker/compose.public-mini.yml` plus root `docker-compose.yml`
- Helper scripts/docs/tests: added or updated

## Container smoke test

- Result class: `CONTAINER_RUNTIME_VALIDATION_SKIPPED_DAEMON_UNAVAILABLE`
- Skip reason: container CLI present but usable daemon/machine unavailable
- Docker build: not run
- `ragtune --help` in container: not run
- `ragtune validate-bundle` in container: not run
- Public-mini governance job in container: not run
- Compose public-mini: not run

## Optional security scans

- Result class: `CONTAINER_SECURITY_SCANS_SKIPPED_TOOLS_UNAVAILABLE`
- Critical findings: 0
- Scanner absence is recorded as skipped, not treated as a hard failure.

## Deployment readiness

- Result class: `DEPLOYMENT_READINESS_SUPPORTED_WITH_BOUNDARIES`
- Live cloud execution: not run, no credentials
- Official platform benchmarking claimed: no
- Production readiness claimed: no

## Validation

- publication validator: passed
- pytest: `169 passed`
- make validate-publication: passed
- make test: `169 passed`
- compile: passed
- git diff check: passed
- tracked large-file scan: passed
- raw data/generated-answer scan: passed with expected sanitized field names and scanner references
- secret scan: passed with expected env-var names and scanner patterns only
- private path scan: passed with expected negative-test string only

## Claim boundaries

This PR does not claim live cloud validation, official platform benchmarking, production readiness, production operation, human validation, hallucination elimination, RAG Compass superiority, or stable generative cost/latency superiority.
