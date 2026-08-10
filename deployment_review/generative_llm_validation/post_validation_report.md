# Generative LLM Validation Post-Validation Report

- Publication validator: passed
- `pytest -q tests/publication`: 51 passed
- `make validate-publication`: passed
- `make test`: 51 passed
- `python3 -m compileall src scripts`: passed
- Large-file scan: passed for public tree; only untracked `.local_data` cache exceeds 50 MB
- Raw prompt/generated-answer scan: passed with expected field-name, sanitizer, test, and sanitized-artifact references only
- Secret scan: passed with expected scanner/config variable-name references only
- Private-path scan: passed
- Overclaim scan: passed with expected unsupported-claim statements only

## Generator Status

- Local generator provider: `ollama`
- Model: `qwen3:8b`
- Hosted generator: not used
- Raw prompts committed: no
- Raw generated answers committed: no

## Results

- CRAG generative result: superseded by v1.1 as `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG`
- HotpotQA generative result: superseded by v1.1 as `GEN_LLM_GOVERNANCE_INCONCLUSIVE`
- Synthesis result: superseded by v1.1 as `GEN_LLM_SYNTHESIS_DIRECTIONAL`

The earlier bounded result is superseded by the HotpotQA quality-signal audit and CRAG generator/evaluator repair. These are not official platform benchmarking, human validation, production validation, broad governance superiority, or RAG Compass superiority.
