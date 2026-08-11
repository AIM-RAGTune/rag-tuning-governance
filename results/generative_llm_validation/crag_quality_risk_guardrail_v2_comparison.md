# CRAG Generative Quality-Risk Guardrail v2

Result class: `CRAG_GEN_LLM_QUALITY_RISK_GUARDRAIL_V2_BLOCKED_HELDOUT_QUALITY_LOSS`

Interpretation: The pooled cross-offset guardrail is not promoted because at least one held-out offset breached the strict generated-quality loss guardrail.

This v2 guardrail trains candidate expansion rules on pooled validation evidence from other deterministic offsets and evaluates only on held-out offsets. Features are deployable retrieval metrics only. Strict quality-loss blocking prevents promotion if any held-out offset breaches generated-quality noninferiority or leaves quality-risk examples unexpanded.

Raw CRAG questions, raw evidence, raw API responses, raw prompts, raw generated answers, secrets, and private paths are not exported.
