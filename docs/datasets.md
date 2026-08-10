# Datasets

This repository includes dataset manifests, checksums, and processed evidence summaries. It does not redistribute raw licensed datasets, raw CRAG question text, raw source documents, or raw API responses.

## MultiHop-RAG

Used as a public corpus-backed confirmatory anchor. The MultiHop-RAG confirmatory run produced `GOVERNANCE_NONINFERIOR_NOT_SUPERIOR`.

## RAGBench HotpotQA

RAGBench HotpotQA is included as `END_TO_END_CONTEXT_RETRIEVAL_ELIGIBLE`. It supports policy-dependent retrieval over reconstructed context units but is not treated as full source-document corpus-backed evidence.

## CRAG

CRAG is used under a noncommercial restriction and manual approval policy. Raw CRAG data and raw CRAG question wording are not included. Reviewers must obtain the raw data from the original provider and verify expected hashes.

CRAG all-row streaming acquisition read 2,706 rows, produced 9,848 web documents, and created a 571-row confirmatory split with zero cross-split leakage.

This repository does not grant CRAG raw-data or query-text redistribution rights. Commercial use requires separate license and legal review. Sanitized artifacts retain query IDs, domains, question types, static/dynamic labels, metrics, and hashes.
