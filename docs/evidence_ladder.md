# Evidence Ladder

The RAGTune validation program separates evidence by strength and claim boundary.

## Level 0 — Simulated / Synthetic Exploration

Synthetic and simulation-only results exercise mechanisms, controls, refusal behavior, and candidate ideas. They cannot establish external RAG performance.

## Level 1 — Engineering Smoke And Fixture Validation

Fixture, smoke, and implementation tests validate code paths and guardrails. They are necessary but not performance evidence.

## Level 2 — External-Transfer Development

External-transfer runs can produce candidate signals across imported datasets or scenarios. They remain development evidence unless frozen confirmatory designs are satisfied.

## Level 3 — Public End-To-End Development

Public end-to-end development runs evaluate policies on public data but may still be exploratory, non-confirmatory, or limited by context-only evidence classes.

## Level 4 — Public Confirmatory Noninferiority

Frozen held-out public confirmatory runs can support noninferiority when governance or optimizer choices match or avoid regressions, but do not necessarily show superiority.

## Level 5 — Multi-Corpus Mixed Evidence

Multi-corpus evidence improves breadth but must be stratified by evidence class. Context-retrieval evidence must not be relabeled as full corpus-backed evidence.

## Level 6 — Beneficial Natural Governance Divergence

Governance reaches this level when it changes a natural public promotion decision and evidence supports the governed choice as beneficial under the declared utility. RAGTune governance has reached Level 6 in a bounded CRAG mock-API source/retrieval governance setting.

The behaviorally distinct follow-up strengthens this level by reframing the strongest CRAG mock-API evidence around a predeclared quality noninferiority margin and measured operating outcomes. It does not move the program to Level 7 because it uses frozen observations and a proxy-plus-evidence quality metric rather than independent human, generative, or official platform validation.

## Level 7 — Docker-Reproduced And Human/Generative/Platform Validated Evidence

This level would require stronger independent reproduction and validation layers, including human evaluation, pinned generative LLM paths, and official platform integrations where claimed.

## Current Position

- RAGTune governance: Level 6 in a bounded CRAG mock-API source/retrieval governance setting.
- RAGTune governance has not reached Level 7.
- Fresh live CRAG replication remains blocked in the current public-repository execution environment. HotpotQA alternate-corpus validation ran with sanitized answer-label/supporting-fact metrics and returned `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`, a negative/mixed result rather than a Level 7 replication.
- Generative LLM Validation v1.1 adds bounded local HotpotQA and CRAG quality-signal audits with Ollama `qwen3:8b` and sanitized metrics. The larger bounded CRAG generative sample supports cost reduction at equivalent generated-answer quality, while HotpotQA remains inconclusive; the synthesis is directional and does not move the program to Level 7.
- RAG Compass has not reached optimizer-superiority support.
- Human, official platform, production, broad-generalization, and hardware/quantum claims remain unsupported.
