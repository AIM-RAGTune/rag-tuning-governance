# Open-Source arXiv Readiness Post-Validation

Repository path: `<repo>`

Branch: `codex/open-source-arxiv-readiness`

Validator: passed.

Publication tests: `124 passed`.

Make gates: `make validate-publication` passed; `make test` passed with `124 passed`.

Compile: passed.

Diff check: passed.

Large-file scan: only ignored `.local_data` cache files exceeded 50 MB; no tracked public artifact exceeded the threshold.

Raw text, prompt, generated-answer, secret, private-path, and overclaim scans were manually inspected. Hits were expected scanner definitions, sanitizer field names, hash fields, explicit false flags, placeholders, and unsupported-claim statements.

Results:

- Public mini reproduction: `PUBLIC_MINI_REPRODUCTION_FAIL_CLOSED`
- HotpotQA quality-signal audit: `HOTPOTQA_GEN_LLM_QUALITY_SIGNAL_CONFIRMED`
- CRAG evaluator mapping: `CRAG_GENERATED_QUALITY_LOCAL_EVALUATOR_ACTIVE`
- External evaluator adapters: `EXTERNAL_EVALUATOR_ADAPTER_PROMOTION_DECISION_GENERATED`
- Selector ablation matrix: `SELECTOR_ABLATION_GOVERNANCE_BLOCKS_UNSAFE_SELECTORS`
- AIM hardware characterization: `AIM_HARDWARE_CHARACTERIZATION_COMPLETED`
- Open-source/arXiv readiness synthesis: `OPEN_SOURCE_ARXIV_READINESS_SUPPORTED_WITH_BOUNDARIES`

Claims still unsupported: RAG Compass superiority, stable generative cost or latency superiority, broad generative governance superiority, human validation, official platform benchmarking, production readiness, and hallucination elimination.
