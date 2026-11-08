"""Score one prediction per independently reviewed case, keyed by case_id."""
import argparse
import json
from pathlib import Path
from statistics import mean

STATUSES = ("VIOLATION", "NO_VIOLATION", "INSUFFICIENT_EVIDENCE")

def read_jsonl(path):
    rows = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["case_id"]
        if key in rows:
            raise ValueError(f"Duplicate case ID in {path.name}: {key}")
        rows[key] = row
    return rows

def score(references, predictions):
    if not references or references.keys() != predictions.keys():
        raise ValueError("Predictions must cover the same nonempty set of cases as references.")
    confusion = {label: {p: 0 for p in STATUSES} for label in STATUSES}
    provision_hits = []
    for key, truth in references.items():
        candidate = predictions[key]
        if truth["status"] not in STATUSES or candidate["status"] not in STATUSES:
            raise ValueError(f"Invalid status in case {key}")
        confusion[truth["status"]][candidate["status"]] += 1
        if truth.get("acceptable_provision_ids"):
            provision_hits.append(candidate.get("provision_id") in truth["acceptable_provision_ids"])
    by_status = {}
    for label in STATUSES:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(row[label] for row in confusion.values())
        by_status[label] = {"support": support,
                            "recall": true_positive/support if support else None,
                            "f1": 2*true_positive/(support+predicted) if support+predicted else None}
    available_f1 = [row["f1"] for row in by_status.values() if row["f1"] is not None]
    return {"cases": len(references),
            "accuracy": sum(confusion[label][label] for label in STATUSES)/len(references),
            "macro_f1": mean(available_f1), "by_status": by_status, "confusion": confusion,
            "mapping_cases": len(provision_hits),
            "mapping_accuracy": mean(provision_hits) if provision_hits else None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(read_jsonl(args.references), read_jsonl(args.predictions)), indent=2))
