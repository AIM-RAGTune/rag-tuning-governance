# RAGTune Governance

RAGTune is a governance framework for RAG policy promotion. RAG Compass is a candidate optimizer within the framework. Current evidence supports RAGTune governance value more strongly than RAG Compass optimizer superiority.

## Current Strongest Result

The strongest current result is a CRAG mock-API validation run:

- Run ID: `ragtune_crag_mock_api_validation_v1_20260809-165415-92d8c0edd4`
- Result: `MOCK_API_VALIDATION_GOVERNANCE_SUPERIOR`
- Governed winner: `top_k_low`
- Quality-only winner: `greedy_regression_aware_search`
- RAG Compass rank: 5th
- Confirmatory rows: 571 / 571
- API calls: 14,172
- Failure rate: 0.0
- Governance delta: +0.0010025405
- Bootstrap CI: [0.0010022708, 0.0010028250]
- Win/tie/loss: 571 / 0 / 0
- Sensitivity: governance superior in 14 / 15 cost-latency settings

The hardening package reports that this result was decision-reproduced in Docker, explained by cost/latency ablation, illustrated with cases, and replicated under frozen-observation resplits.

## Behaviorally Distinct Policy Experiment

A follow-up experiment reanalyzes the sanitized CRAG mock-API frozen observations with behaviorally distinct candidate policies: low retrieval, expanded retrieval, adaptive routing, cost-aware selection, latency-aware selection, quality-only selection, constrained optimization, and Pareto selection.

Primary result: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`. Governed selection chose `low_retrieval_single_endpoint`; quality-only selection chose `optuna_tpe` under the proxy-plus-evidence quality metric. Governed selection was within the predeclared 0.01 quality noninferiority margin while reducing measured cost and latency. This result is not a small weighted-utility-only claim, but it remains derived from sanitized frozen observations and uses `QUALITY_MEASURE_PROXY_PLUS_EVIDENCE`, not human-calibrated or generative LLM validation.

See `results/behavioral_governance/paper_ready_summary.md` and `docs/behaviorally_distinct_policy_experiment_plan.md`.

## Fresh Live CRAG And HotpotQA Behaviorally Distinct Governance Experiment

The next validation phase adds a harness for fresh live CRAG mock-API collection and HotpotQA alternate-corpus validation. The goal is to test whether the behaviorally distinct governance result holds beyond frozen observations and whether HotpotQA answer labels/supporting facts support a stronger answer-quality endpoint.

Current execution status in this public repository environment: approved local CRAG root/data paths and the local evaluator were configured, but fresh CRAG remains blocked as `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE` because the mock-API KG data were not readable by the runtime. HotpotQA ran from a local Hugging Face cache on a sanitized 1,000-example validation sample using answer-label and supporting-fact evidence metrics. The HotpotQA result was `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`: governance selected a lower-cost policy, but the confirmatory quality loss exceeded the predeclared 0.01 noninferiority margin, so this does not replicate the prior frozen-observation claim.

Raw CRAG data are not redistributed. CRAG remains noncommercial research-only. HotpotQA raw data are not committed to this repository. See `docs/fresh_live_crag_hotpotqa_behavioral_governance_plan.md`, `docs/dataset_acquisition.md`, and `results/multi_dataset_behavioral_governance/paper_ready_summary.md`.

## What This Repository Contains

- `src/ragtune/`: RAGTune implementation code.
- `configs/`: experiment, dataset, optimizer, and policy-space configuration.
- `tests/`: unit, integration, and reproducibility tests from the validation harness.
- `artifacts/selected_run_summaries/`: selected small manifests and run outputs needed for review.
- `results/`: processed summary tables and claim-status records.
- `data/`: dataset availability, license notes, checksums, and fixtures. Raw licensed datasets and raw CRAG question text are not redistributed.
- `reproduction/`: Docker and command documentation for reproducing allowed runs.
- `paper/`: a paper scaffold with explicit limitations.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart

```bash
make validate-publication
make test
```

Full CRAG reproduction requires externally obtained CRAG data mounted according to `data/DATA_AVAILABILITY.md` and `reproduction/docker/README.md`. CRAG query wording is redacted from publication artifacts; reviewers with approved CRAG access can match local data through query IDs and `query_text_hash` values.

## Key Results

See `results/run_index.csv`, `results/evidence_summary.json`, and `results/claim_status/claim_status_table.csv`.

## Historical Validation Record

The historical ledger is intentionally included to reduce cherry-picking and preserve negative, blocked, refused, and inconclusive evidence. Reviewers should start with:

- `results/historical/historical_evidence_timeline.md`
- `results/historical/historical_run_index.csv`
- `results/historical/historical_negative_results.md`
- `results/historical/historical_blocked_results.md`
- `docs/historical_validation_summary.md`

The ledger contains sanitized summaries only. It does not include raw datasets, raw CRAG question text, raw source documents, or raw API responses.

## Claim Boundaries

This repository does not claim:

- RAG Compass superiority.
- Generative LLM validation.
- Human-evaluation validation.
- Official LangSmith, Ragas, DeepEval, or RAGChecker benchmarking.
- Production readiness.
- Hallucination elimination.

## Dataset Availability

This repository does not include raw CRAG datasets, raw CRAG question text, raw source documents, or raw API responses. It includes processed metrics, IDs, hashes, configuration files, and sanitized summaries. See `docs/crag_query_text_sanitization.md`.

## Testing Status

Latest packaged status from the source repository:

- Focused CRAG tests: 151 passed
- Full repository tests: 530 passed, 1 skipped, 4 warnings
- Lint: passed
- Strict Git provenance: passed

## Docker Status

Docker decision reproduction supported the parent CRAG mock-API validation decision. It was not an exact numeric match; it preserved the decision, winners, rows, failure rate, and win/tie/loss while some floating metrics differed. See `docs/docker_reproduction.md` and `docs/crag_mock_api_validation.md`.

## Publication-Governance Gate

This bundle was created locally because external GitHub upload was blocked by tenant governance controls. No bypass was attempted. The repository has no remote configured in the approval package.

## Public Repository Status

This public repository was created as a fresh one-commit sanitized tree with no prior private Git history. See `docs/public_repository_note.md`.

## Citation

See `CITATION.cff`.

## License

Code in this repository is released under Apache-2.0 unless otherwise noted. Dataset licenses remain governed by their original providers.

## Maintainer

Maintainer placeholder: AIM-RAGTune.
