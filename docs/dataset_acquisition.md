# Dataset Acquisition

This repository does not redistribute raw CRAG or HotpotQA data.

## CRAG

CRAG fresh live mock-API runs require approved noncommercial research-only access. Configure:

```bash
export RAGTUNE_CRAG_APPROVED_NONCOMMERCIAL_RESEARCH_ONLY=true
export RAGTUNE_CRAG_ROOT=<approved-local-path>/CRAG
export RAGTUNE_CRAG_DATA=<approved-local-path>/crag-data
```

Then run:

```bash
python3 scripts/acquire_crag_live_mock_api_inputs.py --dry-run
python3 scripts/run_fresh_live_crag_behavioral_governance.py
```

Raw CRAG query text, source documents, and API responses must remain outside this repository.

## HotpotQA

HotpotQA should be acquired into `.local_data/hotpotqa` or another approved local data root:

```bash
pip install datasets
python3 scripts/acquire_hotpotqa_public_corpus.py --source huggingface --config distractor --output-root "${RAGTUNE_DATA_ROOT:-.local_data}/hotpotqa"
python3 scripts/run_hotpotqa_behavioral_governance.py
```

Raw HotpotQA questions, context paragraphs, and supporting-fact sentences must not be committed.
