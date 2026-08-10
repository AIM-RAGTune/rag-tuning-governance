# Historical Validation Summary

RAGTune is a governance-first RAG policy promotion and validation framework. The historical validation record shows an evidence arc from synthetic simulations and refusal discipline through public end-to-end development, public confirmatory noninferiority, CRAG full corpus-backed acquisition, and CRAG mock-API governance superiority.

RAGTune governance is the main evidence-backed contribution. It evaluates candidate policies and optimizers under declared quality, cost, latency, regression, safety, provenance, reproducibility, and statistical constraints.

RAG Compass is a candidate optimizer inside RAGTune. Its legacy machine-readable ID is `ragtune_no_fork`. Historical evidence does not support RAG Compass optimizer superiority. In the strongest CRAG mock-API validation run, RAG Compass ranked 5th.

Negative, blocked, refused, and inconclusive results are intentionally preserved in `results/historical/`. They reduce cherry-picking and explain why the current claim boundary is narrow.

The strongest current result is CRAG mock-API governance superiority under a source/retrieval governance evaluation path: governed selection chose `top_k_low` over quality-only `greedy_regression_aware_search` on held-out confirmatory rows under the configured utility.

A later behaviorally distinct follow-up over the sanitized frozen CRAG mock-API observations produced `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`: governed selection chose `low_retrieval_single_endpoint` while quality-only selected `optuna_tpe` under a proxy-plus-evidence quality metric. This strengthens the operating-constraint interpretation but remains frozen-observation evidence, not independent human/generative validation.

The fresh live CRAG + HotpotQA phase adds the next validation harness. Fresh CRAG now has a restored local data/mock-API environment and a sanitized 50-example live sample, but remains blocked as `FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY` because the sample did not produce a usable answer/evidence quality signal. HotpotQA ran from local cached public data with sanitized outputs and returned `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`, meaning operating cost improved but quality noninferiority was not established. This negative/mixed status is preserved to avoid upgrading frozen-observation evidence into a replication claim.

Generative LLM Validation v1 adds bounded local HotpotQA generated-answer evidence with Ollama `qwen3:8b`, producing `GEN_LLM_SYNTHESIS_DIRECTIONAL`. CRAG generative validation remains blocked or incomplete, so this is not replicated generative evidence across datasets.

Unsupported areas remain: RAG Compass superiority, broad generative LLM validation beyond the bounded local HotpotQA run, human-evaluation validation, official external-platform benchmarking, production readiness, broad governance superiority across many public datasets, and SQUARE hardware/quantum advantage.

No raw datasets, raw CRAG query wording, raw source documents, raw API responses, or private paths are included in the historical ledger.
