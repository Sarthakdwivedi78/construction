"""Select hybrid fusion weights on reviewed validation judgments, then score held-out queries."""
import argparse
import json
from pathlib import Path

from evaluate import score_retrieval
from retrieval import Index
from score_predictions import read_jsonl

def load_queries(path, index):
    rows = read_jsonl(path)
    known = {row["hazard_id"] for row in index.rows}
    queries = []
    for key, row in rows.items():
        if row.get("reviewed") is not True:
            raise ValueError(f"Review required for {key}")
        ids = set(row["relevant_ids"])
        if not ids or not ids <= known:
            raise ValueError(f"Missing or unknown relevant provisions for {key}")
        queries.append((row["query"], row["category"], row.get("authority", "Unknown"), ids,
                        row.get("state", "")))
    if not queries:
        raise ValueError(f"No reviewed queries in {path}")
    return rows, queries

def tune(index, validation, test, k):
    validation_ids, validation_queries = load_queries(validation, index)
    test_ids, test_queries = load_queries(test, index)
    if validation_ids.keys() & test_ids.keys():
        raise ValueError("Validation and test case IDs overlap.")
    normalize = lambda value: " ".join(value.casefold().split())
    if {normalize(q[0]) for q in validation_queries} & {normalize(q[0]) for q in test_queries}:
        raise ValueError("Validation and test contain identical normalized queries.")
    trials = []
    for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
        index.dense_weight = weight
        trials.append({"dense_weight": weight, **score_retrieval(index, validation_queries, k)})
    best = max(trials, key=lambda row: (row["ndcg"], row["mrr"], -abs(row["dense_weight"]-0.5)))
    index.dense_weight = best["dense_weight"]
    return {"selected_dense_weight": index.dense_weight, "validation_trials": trials,
            "heldout": score_retrieval(index, test_queries, k)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("-k", type=int, choices=range(1, 21), default=3)
    args = parser.parse_args()
    print(json.dumps(tune(Index(hybrid=True), args.validation, args.test, args.k), indent=2))
