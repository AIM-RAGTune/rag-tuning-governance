# Fresh Live CRAG + HotpotQA Behavioral Governance Summary

## Why frozen-observation evidence was insufficient

The prior behaviorally distinct result used sanitized frozen CRAG mock-API observations. It reduced measured cost at equivalent proxy-plus-evidence quality, but it was not a fresh live collection and did not use a second corpus with stronger labels.

## Why fresh CRAG was attempted

Fresh live CRAG would test whether the policy behavior and operating-cost result persist when the mock API is called again under approved noncommercial constraints.

## Why HotpotQA was selected

HotpotQA provides answer labels, multi-hop structure, bridge/comparison types, difficulty levels, and supporting-fact labels for stronger answer correctness and evidence-support scoring.

## Dataset acquisition status

Fresh CRAG: `FRESH_CRAG_BLOCKED_NO_APPROVED_DATA`. HotpotQA: `HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE`.

## Policy suite

The planned suite includes low retrieval, expanded retrieval, adaptive routing, BM25/reranking for HotpotQA, quality-only, constrained optimizer, Pareto selector, and governed selection.

## Quality metrics

CRAG would use proxy-plus-evidence plus any available local evaluator. HotpotQA would use exact match, F1, supporting-fact title recall, supporting-fact sentence recall, evidence efficiency, and abstention correctness.

## Primary endpoint

Equivalent quality with lower measured cost/latency, or improved quality under a fixed deployment budget.

## Fresh CRAG result

`FRESH_CRAG_BLOCKED_NO_APPROVED_DATA`.

## HotpotQA result

`HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE`.

## Multi-dataset synthesis

`MULTI_DATASET_BEHAVIORAL_GOVERNANCE_BLOCKED`.

## Negative findings

This run did not move beyond frozen-observation evidence because approved local CRAG and HotpotQA data were unavailable.

## Claim boundaries

No human validation, generative validation, official platform benchmark, production readiness, broad governance superiority, or RAG Compass superiority is claimed.

## Reproduction instructions

Configure approved CRAG and/or HotpotQA local data roots, then run the acquisition and governance scripts documented in this repository.

## Recommended next experiment

Run HotpotQA from an approved local dataset cache and repeat fresh live CRAG after configuring the CRAG mock API runtime.
