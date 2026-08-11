# CRAG Learned Quality-Risk Predictor Comparison

Result class: `CRAG_GEN_LLM_LEARNED_RISK_PREDICTOR_LATENCY_MIXED_CONFIRMATORY_QUALITY_LOSS`

Interpretation: Validation-trained deployable predictors reduced expansion rates on all fixed offsets, but confirmatory quality loss appeared on offsets 36 and 60; the result is mixed and does not support increasing sample size yet.

Runs: `4`  
Predictor validation gates passed: `4`  
Positive latency slices: `1`  
Quality-loss slices: `2`  
Inconclusive slices: `1`

The predictor was trained on validation generated-quality outcomes and used only deployable retrieval metadata/count features at selection time. Public artifacts do not include raw CRAG questions, raw evidence/source text, raw prompts, raw generated answers, or raw API responses.
