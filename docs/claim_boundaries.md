# Claim Boundaries

RAGTune is a governance framework for RAG policy promotion. RAG Compass is one optimizer candidate inside RAGTune and keeps the machine-readable legacy id `ragtune_no_fork`.

Supported or partially supported:

- Append-only provenance and strict Git checks were exercised in the source validation repo.
- MultiHop-RAG produced a public confirmatory governance noninferiority signal, not superiority.
- RAGBench HotpotQA was enabled as context-retrieval evidence, not full corpus-backed retrieval.
- CRAG web-document evaluation produced governance noninferiority, not superiority.
- CRAG mock-API validation produced the strongest governance result: governed selection chose `top_k_low` over quality-only `greedy_regression_aware_search` on the full held-out confirmatory split.
- A behaviorally distinct follow-up over sanitized frozen CRAG mock-API observations produced `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`: governed selection chose `low_retrieval_single_endpoint` while quality-only selected `optuna_tpe` under a proxy-plus-evidence quality metric and a predeclared 0.01 noninferiority margin.
- The fresh live CRAG + HotpotQA phase currently provides an acquisition and validation harness, a sanitized fresh-CRAG live sample, and a real sanitized HotpotQA alternate-corpus run. Fresh CRAG is blocked as `FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY`: approved local CRAG data and the mock-API runtime were restored, but the live sample did not produce a usable answer/evidence quality signal. HotpotQA produced `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`, so it does not replicate the frozen-observation result.
- CRAG publication artifacts preserve metrics and hashes but redact raw CRAG question text, source passages, and mock-API responses.

Unsupported:

- RAG Compass superiority.
- Broad universal governance superiority across all RAG settings.
- Broad generative LLM governance validation beyond the current mixed bounded local CRAG/HotpotQA evidence.
- Human-evaluation validation.
- Official external platform benchmarking.
- Production readiness.
- Hallucination elimination.
- SQUARE hardware or quantum advantage is unsupported and not claimed.

Evidence classes must not be relabeled. Context-retrieval evidence is weaker than full corpus-backed evidence. Workflow simulation is not an official platform benchmark.

This repository does not grant CRAG redistribution rights. The CRAG mock-API result and behaviorally distinct follow-up support source/retrieval governance evidence under bounded operating constraints; they are not human-calibrated answer-quality validation. Fresh CRAG replication remains blocked because the current live sample lacks a usable answer/evidence quality signal. HotpotQA provides stronger answer-label/supporting-fact evidence, but the non-generative behavioral result is a quality-loss negative finding rather than a governance success claim.

## Generative LLM Boundary

`ragtune_generative_llm_validation_v1.1` adds a HotpotQA quality-signal audit and CRAG generator/evaluator repair using local Ollama `qwen3:8b`. After increasing the bounded CRAG sample to 12 examples, the primary CRAG slice produced `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG`; three independent deterministic CRAG repeats at offsets 24, 36, and 60 returned `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`. A second pinned local model, Ollama `gpt-oss:20b`, was run on offsets 0, 24, 36, and 60 and produced `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS` with no positive cost-result slices. A faster non-thinking instruct model, Ollama `llama3.2:3b`, repaired answer emission on four 16-example fixed-offset CRAG slices with 0 / 704 parse failures, but the cost-at-equivalent-generated-quality result remained `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`. HotpotQA remains `GEN_LLM_GOVERNANCE_INCONCLUSIVE`. The synthesis is `GEN_LLM_SYNTHESIS_MIXED`. This is not official platform benchmarking, human validation, production validation, broad governance superiority, or RAG Compass superiority.
