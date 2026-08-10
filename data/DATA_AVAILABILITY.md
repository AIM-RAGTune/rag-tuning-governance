# Data Availability

Raw datasets are not redistributed in this repository unless their license and repository policy explicitly permit redistribution. This repository does not include raw CRAG datasets, raw CRAG question text, raw source documents, or raw API responses.

## Included

- Dataset manifests
- Checksum expectations
- Processed aggregate metrics
- Small fixtures where safe
- Reproduction scripts

## Excluded

- Raw CRAG data
- Raw CRAG question text
- Raw CRAG source-document passages
- Raw CRAG mock-API responses
- Raw licensed dataset dumps
- Model weights
- Private human-eval answer keys
- Credentials or environment files

## CRAG

CRAG is treated as noncommercial-research-only for this evidence package. Reviewers must obtain CRAG from the original provider and place it under a local mount such as `/data/crag/raw`. The acquisition manifest should verify the expected raw SHA-256 from the source run.

This repository does not grant CRAG raw-data or query-text redistribution rights. Commercial use requires separate license and legal review. Publication artifacts preserve query IDs, domains, question types, static/dynamic labels, metrics, and `query_text_hash` values instead of raw CRAG wording.

Known processed facts from the source validation:

- Rows read: 2,706
- Web documents: 9,848
- Confirmatory rows: 571
- Cross-split leakage: 0
- Raw SHA-256: matched expected CRAG LFS hash

## MultiHop-RAG

Used as a public corpus-backed confirmatory anchor. Consult the source provider and local manifests for exact revision and license details.

## RAGBench HotpotQA

Used as context-retrieval evidence. Raw RAGBench data are not redistributed here.

## Fresh Live CRAG And HotpotQA Phase

Fresh live CRAG requires approved noncommercial research-only local access to CRAG and its mock API. This repository does not include raw CRAG queries, source documents, or API responses.

The latest fresh-live attempt configured approved local CRAG root/data paths and found the local evaluator, but the mock-API KG data were not readable by the runtime. The result remains `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`; reviewers should restore a complete, readable CRAG mock-API KG/data installation before rerunning.

HotpotQA is the preferred alternate public corpus for stronger answer-label and supporting-fact evaluation. This repository does not include raw HotpotQA questions, context paragraphs, or supporting-fact sentences. Reviewers should acquire HotpotQA from the original approved source or Hugging Face into a local data cache, then run the scripts documented in `docs/dataset_acquisition.md`.
