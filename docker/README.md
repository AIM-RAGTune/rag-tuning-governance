# Docker

RAGTune can run as a finite governance job in a container. The container starts, evaluates or imports sanitized RAG policy metrics, writes audit artifacts, emits `promotion_decision.json`, validates claim boundaries, and exits.

Default command:

```bash
docker build -t ragtune-governance:local .
docker run --rm -v "$(pwd)/docker_outputs:/outputs" ragtune-governance:local run-governance-job --config configs/jobs/public_mini_governance_job.yaml --output-root /outputs --decision-out /outputs/promotion_decision.json
```

The image does not include raw CRAG or HotpotQA datasets, local caches, credentials, prompts, or generated answers. Mount approved local data at `/data` only for local experiments that require it.
