# Quality Measurement v1

Result: `QUALITY_MEASURE_PROXY_PLUS_EVIDENCE`.

The publication bundle does not include answer text, source passages, raw API responses, human annotations, or a pinned judge model. The stronger quality path therefore decomposes the frozen parent score into an answer-correctness proxy, evidence-support proxy from successful calls and result counts, and abstention handling. This is stronger than endpoint success alone, but it is not human-calibrated answer-quality evidence.
