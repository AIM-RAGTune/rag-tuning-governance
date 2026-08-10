# Generative LLM Validation v1.1 Post-Validation Report

- Publication validator: passed
- `pytest -q tests/publication`: 70 passed
- `make validate-publication`: passed
- `make test`: 70 passed
- `python3 -m compileall src scripts`: passed
- Large-file scan: passed for tracked/public tree with `.local_data` excluded
- Raw prompt/generated-answer scan: passed with expected sanitizer, test, field-name, and sanitized-artifact references only
- Raw dataset text scan: passed with expected scanner/test/field-name references only
- Secret scan: passed with expected scanner/config variable-name references only
- Private-path scan: passed with no committed hits
- Overclaim scan: passed with expected explicit unsupported-claim/boundary references only

## HotpotQA Audit

- Result class: `HOTPOTQA_GEN_LLM_QUALITY_SIGNAL_CONFIRMED`
- Primary generated-governance result: `GEN_LLM_GOVERNANCE_INCONCLUSIVE`
- Sample size: 12 examples
- Generation rows: 96
- Unique answer hashes: 2
- Non-empty generated answers: 2
- Quality variance: 0.024150548562885805
- Governed winner: `expanded_retrieval_multi_context`
- Quality-only winner: `expanded_retrieval_multi_context`
- RAG Compass rank: 2

## CRAG Repair

- Generator access diagnosis: passed
- CRAG root/data configured at runtime: yes
- Mock API available: yes
- Local evaluator available: yes
- qwen3 answer-emission repair: passed by sending Ollama `think: false`
- Evaluator mapping result class: `CRAG_GENERATED_QUALITY_LOCAL_EVALUATOR_ACTIVE`
- CRAG generative validation result: `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG`
- Sample size: 12 examples / 132 policy-generation rows
- Non-empty generated answers: 132
- Unique answer hashes: 48
- Governed winner: `pareto_frontier_selector`
- Quality-only winner: `expanded_retrieval_multi_endpoint`
- RAG Compass rank: 8

## Synthesis

- CRAG repeat comparison: `CRAG_GEN_LLM_COST_RESULT_NOT_REPLICATED`
- Result class: `GEN_LLM_SYNTHESIS_MIXED`
- Interpretation: Superseded by `results/generative_llm_validation/crag_stability_comparison.json`; the primary CRAG slice produced generative support, but independent deterministic CRAG repeats did not reproduce the cost result and HotpotQA remained inconclusive.

No raw prompts, raw generated answers, raw CRAG text, raw HotpotQA questions/context, raw API responses, secrets, or private paths are committed.
