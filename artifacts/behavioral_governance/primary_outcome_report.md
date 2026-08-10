# Behavioral Governance Primary Outcome v1

- Result: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`
- Evidence class: `public_full_corpus_mock_api_validation_derived_frozen_observation`
- Governed winner: `low_retrieval_single_endpoint`
- Quality-only winner: `optuna_tpe`
- Constrained optimizer winner: `low_retrieval_single_endpoint`
- Pareto frontier: `low_retrieval_single_endpoint, optuna_tpe`
- RAG Compass rank: `6`
- Final quality delta: `-0.0051537115` CI [-0.0051537115, -0.0051537115]
- Measured cost delta: `-2.3683887916` CI [-2.3683887916, -2.3683887916]
- Measured latency delta: `-86.1388368144` ms CI [-86.1388368144, -86.1388368144]

Primary interpretation: governed selection used a predeclared quality floor and measured operating constraints, not a small weighted-utility tie-break. It selected the lower-cost policy at equivalent proxy-plus-evidence quality. This is still frozen-observation source/retrieval evidence, not human-eval or generative LLM validation.
