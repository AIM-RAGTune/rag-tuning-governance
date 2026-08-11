# Generative LLM Answer-Emission Repair Validation Report

## Experiment

- Branch: `codex/gpt-oss-answer-emission-repair`
- Starting commit: `8a2c27a3a3d0c95660daac9156c619fc2d432ae8`
- Initial repair target: Ollama `gpt-oss:20b`
- Faster non-thinking instruct model added: Ollama `llama3.2:3b`
- Fixed CRAG offsets: `0`, `24`, `36`, `60`
- Larger bounded sample: 16 CRAG examples per offset
- Candidate generation rows: 704

## Result

- gpt-oss repair smoke: improved but still high parse failures, 20 / 44 failures.
- llama3.2 smoke: 0 / 44 parse failures.
- llama3.2 larger fixed-offset comparison: 0 / 704 parse failures.
- Answer-emission comparison: `CRAG_GEN_LLM_ANSWER_EMISSION_REPAIRED_NO_COST_RESULT`
- llama3.2 stability result: `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`

## Publication Hygiene

- Raw CRAG questions committed: no
- Raw CRAG evidence/source text committed: no
- Raw CRAG API responses committed: no
- Raw prompts committed: no
- Raw generated answers committed: no
- Secrets committed: no
- Private paths committed: no

## Validation

- publication validator: passed
- pytest: `76 passed`
- make validate-publication: passed
- make test: `76 passed`
- compile: passed
- large-file scan: passed; no public tracked files over threshold
- raw prompt/generated-answer scan: passed with expected sanitizer, test, field-name, and sanitized artifact references only
- raw dataset text scan: passed with expected sanitizer, test, field-name, and sanitized artifact references only
- secret scan: passed with expected scanner/config variable-name references only
- private path scan: passed; no committed private paths found
- overclaim scan: passed; hits were explicit unsupported-claim boundary language only

## Claim Boundary

This experiment repairs answer emission for the local CRAG generative path but does not support broad generative governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
