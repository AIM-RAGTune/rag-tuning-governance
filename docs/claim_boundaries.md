# Claim Boundaries

RAGTune is a governance framework for RAG policy promotion. RAG Compass is one optimizer candidate inside RAGTune and keeps the machine-readable legacy id `ragtune_no_fork`.

Supported or partially supported:

- Append-only provenance and strict Git checks were exercised in the source validation repo.
- MultiHop-RAG produced a public confirmatory governance noninferiority signal, not superiority.
- RAGBench HotpotQA was enabled as context-retrieval evidence, not full corpus-backed retrieval.
- CRAG web-document evaluation produced governance noninferiority, not superiority.
- CRAG mock-API validation produced the strongest governance result: governed selection chose `top_k_low` over quality-only `greedy_regression_aware_search` on the full held-out confirmatory split.
- A behaviorally distinct follow-up over sanitized frozen CRAG mock-API observations produced `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`: governed selection chose `low_retrieval_single_endpoint` while quality-only selected `optuna_tpe` under a proxy-plus-evidence quality metric and a predeclared 0.01 noninferiority margin.
- The fresh live CRAG + HotpotQA phase currently provides an acquisition and validation harness, a blocked fresh-CRAG artifact, and a real sanitized HotpotQA alternate-corpus run. Fresh CRAG is blocked as `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE` because the approved local CRAG root/data paths were configured but the mock-API KG data were not readable by the runtime. HotpotQA produced `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`, so it does not replicate the frozen-observation result.
- CRAG publication artifacts preserve metrics and hashes but redact raw CRAG question text, source passages, and mock-API responses.

Unsupported:

- RAG Compass superiority.
- Broad universal governance superiority across all RAG settings.
- Generative LLM validation.
- Human-evaluation validation.
- Official external platform benchmarking.
- Production readiness.
- Hallucination elimination.
- SQUARE hardware or quantum advantage is unsupported and not claimed.

Evidence classes must not be relabeled. Context-retrieval evidence is weaker than full corpus-backed evidence. Workflow simulation is not an official platform benchmark.

This repository does not grant CRAG redistribution rights. The CRAG mock-API result and behaviorally distinct follow-up support source/retrieval governance evidence under bounded operating constraints; they are not generative LLM validation and are not human-calibrated answer-quality validation. Fresh CRAG replication remains blocked because the configured local mock-API runtime could not read the required KG data. HotpotQA provides stronger answer-label/supporting-fact evidence, but the current result is a quality-loss negative finding rather than a governance success claim.
