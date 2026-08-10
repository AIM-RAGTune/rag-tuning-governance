# Claim Boundaries

RAGTune is a governance framework for RAG policy promotion. RAG Compass is one optimizer candidate inside RAGTune and keeps the machine-readable legacy id `ragtune_no_fork`.

Supported or partially supported:

- Append-only provenance and strict Git checks were exercised in the source validation repo.
- MultiHop-RAG produced a public confirmatory governance noninferiority signal, not superiority.
- RAGBench HotpotQA was enabled as context-retrieval evidence, not full corpus-backed retrieval.
- CRAG web-document evaluation produced governance noninferiority, not superiority.
- CRAG mock-API validation produced the strongest governance result: governed selection chose `top_k_low` over quality-only `greedy_regression_aware_search` on the full held-out confirmatory split.
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

This repository does not grant CRAG redistribution rights. The CRAG mock-API result supports source/retrieval governance evidence under the configured utility; it is not generative LLM validation.
