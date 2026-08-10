from __future__ import annotations

import numpy as np
import pandas as pd

from square_sim.utils.hashing import stable_hash


def generate_rag_proxy(rows: int, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sources = np.array(["ragtruth", "hagrid", "expertqa", "ragbench"])
    source = rng.choice(sources, size=rows, p=[0.48, 0.18, 0.14, 0.20])
    uncertainty = rng.beta(2.2, 3.0, size=rows)
    context_recall = np.clip(rng.normal(0.64, 0.18, rows), 0, 1)
    faithfulness = np.clip(0.76 - 0.55 * uncertainty + 0.22 * context_recall + rng.normal(0, 0.06, rows), 0, 1)
    hallucination = np.clip(1.0 - faithfulness + rng.normal(0, 0.08, rows), 0, 1)
    return pd.DataFrame(
        {
            "row_id": [f"rag-{seed}-{i}" for i in range(rows)],
            "source_dataset": source,
            "track": "rag",
            "input_text": [f"Question {i}: retrieve and answer from evidence." for i in range(rows)],
            "context_text": [f"Evidence bundle {stable_hash({'seed': seed, 'i': i}, 8)}" for i in range(rows)],
            "reference_answer": [f"Reference answer {i}" for i in range(rows)],
            "candidate_response": [f"Candidate answer {i}" for i in range(rows)],
            "uncertainty": uncertainty,
            "context_recall": context_recall,
            "context_precision": np.clip(context_recall - hallucination * 0.2, 0, 1),
            "faithfulness": faithfulness,
            "hallucination_risk": hallucination,
            "latency_proxy": np.clip(0.18 + context_recall * 0.45 + rng.normal(0, 0.04, rows), 0, 1),
            "policy_risk": np.clip(hallucination * 0.7 + uncertainty * 0.3, 0, 1),
        }
    )
