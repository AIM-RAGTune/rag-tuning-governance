# CRAG Answer-Emission Model Comparison

Result class: `CRAG_GEN_LLM_ANSWER_EMISSION_REPAIRED_NO_COST_RESULT`

Baseline model: `gpt-oss:20b` parse failures `362` / `528`.
Candidate model: `llama3.2:3b` parse failures `0` / `704`.

The faster non-thinking instruct model repaired answer emission but did not recover a stable cost-at-equivalent-generated-quality result.

Raw CRAG questions, evidence text, prompts, and generated answers are not committed.
