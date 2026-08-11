# RC1 Reproducibility And arXiv Post-Validation

- Fresh clone reproducibility: `FRESH_CLONE_REPRODUCTION_PASSED_GIT_CLONE`
- Docker runtime/static validation: `DOCKER_RUNTIME_VALIDATED_PUBLIC_MINI`
- Release candidate: `RELEASE_CANDIDATE_READY`
- CRAG evaluator mapping v2: `CRAG_EVALUATOR_MAPPING_V2_ACTIVE_NONCONSTANT_SIGNAL`
- HotpotQA quality-signal audit v2: `HOTPOTQA_GEN_QUALITY_AUDIT_V2_CONFIRMED_NONCONSTANT_SIGNAL`
- Selector ablation stress v2: `SELECTOR_ABLATION_STRESS_V2_GOVERNANCE_BLOCKS_UNSAFE_SELECTORS`
- Verify-run: `VERIFY_RUN_PASSED`
- External evaluator adapters v2: `EXTERNAL_EVALUATOR_ADAPTER_V2_PROMOTION_DECISION_GENERATED`
- AIM hardware matrix: `AIM_HARDWARE_MATRIX_COMPLETED`
- RC1/arXiv synthesis: `RC1_ARXIV_READINESS_SUPPORTED_WITH_BOUNDARIES`

Validation:

- Publication validator: passed
- Deployment readiness validator: `DEPLOYMENT_READINESS_SUPPORTED_WITH_BOUNDARIES`
- `pytest -q tests/publication`: `222 passed`
- `make validate-publication`: passed
- `make test`: `222 passed`
- Compile: passed
- Diff check: passed
- Large-file scan: passed; only ignored local HotpotQA cache exceeded threshold
- Raw text scan: passed with expected scanner and sanitized field-name references
- Secret scan: passed with expected environment-variable names and scanner definitions only
- Private-path scan: passed with expected negative-test strings only
- Overclaim scan: passed with explicit unsupported-claim statements only

Claims still unsupported: RAG Compass superiority, stable generative cost/latency superiority, broad generative governance superiority, human validation, official platform benchmarking, production readiness, and hallucination elimination.
