# Operator Workflow

1. Choose a governance job config, such as `configs/jobs/public_mini_governance_job.yaml`.
2. Mount sanitized inputs at `/inputs` if the job needs external data.
3. Mount configs at `/configs` or use the repository configs already present in the image.
4. Mount an output directory at `/outputs`.
5. Run `ragtune run-governance-job`.
6. Inspect `/outputs/promotion_decision.json`.
7. Preserve `/outputs/validation_report.json` and related audit artifacts.
8. Promote, block, reject, or continue review according to the decision and local governance process.

The default public mini job needs no raw dataset, generator, cloud credential, or private data path.

Example:

```bash
ragtune run-governance-job \
  --config configs/jobs/public_mini_governance_job.yaml \
  --output-root artifacts/public_mini_governance_job \
  --decision-out artifacts/public_mini_governance_job/promotion_decision.json
```

Operators should not commit mounted raw datasets, local evaluator inputs, prompts, generated answers, API responses, secrets, or private paths.
