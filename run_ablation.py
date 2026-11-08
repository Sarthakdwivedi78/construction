"""Run paired, hazard-targeted ablations on reviewed image cases."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from detection import detect, load_detector
from local_vision import LocalVision
from measurements import parse_measurements
from reasoning import CUES, Vision, inspect
from retrieval import Index, ROOT
from score_predictions import STATUSES, read_jsonl, score

VARIANTS = ("yolo", "general-vision", "unranked-corpus", "bm25", "dense", "hybrid", "complete")

class UnrankedCorpus:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, category, authority, k=4, state="", inspection_date=""):
        eligible = {"India", authority}
        if authority == "State":
            eligible.add(state)
        return [row for row in self.rows if row["jurisdiction"] in eligible
                and (not inspection_date or row.get("effective_from", "") <= inspection_date)][:k]

def choose_prediction(case_id, category, findings, reviewed_policy, warnings):
    matches = [f for f in findings if f["category"] == category]
    finding = next((f for f in matches if f["model_assessment"]), matches[0] if matches else None)
    if not finding:
        return {"case_id": case_id, "status": "INSUFFICIENT_EVIDENCE", "warnings": warnings,
                "hazard_identified": False}
    suggestion = finding["model_assessment"] or {}
    status = (suggestion.get("status", "Insufficient Evidence").upper().replace(" ", "_")
              if not reviewed_policy else finding["decision_code"])
    return {"case_id": case_id, "status": status,
            "provision_id": (finding["selected_provision_id"] if reviewed_policy else suggestion.get("cited_id")),
            "hazard_identified": True, "warnings": warnings, "finding": finding}

def load_references(path, taxonomy):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Use independently reviewed evaluation cases.")
    for key, row in rows.items():
        if row.get("reviewed") is not True:
            raise ValueError(f"Review required for {key}.")
        answer = row.get("answer", {})
        status = row.get("status", answer.get("violation_status"))
        if status not in STATUSES or row.get("hazard_category") not in taxonomy:
            raise ValueError(f"Invalid reference status or category for {key}.")
        if answer.get("violation_status", status) != status:
            raise ValueError(f"Conflicting reference statuses for {key}.")
        row["status"] = status
        row.setdefault("acceptable_provision_ids", row.get("retrieved_regulations", []))
    return rows

def prediction_context(case):
    context = {key: case.get(key, "") for key in ("state", "activity", "inspection_date")}
    context["authority"] = case.get("authority", "Unknown")
    if "site_measurements" in case:
        context["site_measurements"] = parse_measurements(json.dumps(case["site_measurements"]))
    return context

def run(args):
    import cv2

    taxonomy = json.loads((ROOT/"taxonomy.json").read_text(encoding="utf-8-sig"))
    references = load_references(args.cases, taxonomy)
    variants = args.variants
    vision = None
    if any(v != "yolo" for v in variants):
        vision = LocalVision() if args.local_vision else Vision(os.environ["SAFETY_VISION_MODEL"])
    baseline = Index()
    indexes = {"unranked-corpus": UnrankedCorpus(baseline.rows), "bm25": baseline}
    if "dense" in variants:
        indexes["dense"] = Index(dense_only=True)
    if {"hybrid", "complete"}.intersection(variants):
        indexes["complete"] = indexes["hybrid"] = Index(hybrid=True)
    detector = load_detector(args.weights)
    records = {variant: {} for variant in variants}
    for case_id, case in references.items():
        category = case["hazard_category"]
        if category not in taxonomy:
            raise ValueError(f"Unknown target hazard category for {case_id}")
        image = (args.media_root / case["image"]).resolve()
        if not image.is_relative_to(args.media_root.resolve()):
            raise ValueError(f"Image outside media root for {case_id}")
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError(f"Cannot decode image for {case_id}")
        start = perf_counter()
        objects, _ = detect(detector, frame)
        detection_seconds = perf_counter() - start
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError(f"Cannot encode image for {case_id}")
        jpeg = encoded.tobytes()
        for variant in variants:
            start = perf_counter()
            mode = "complete" if variant == "complete" else "fixed"
            if variant == "yolo":
                row = {"case_id": case_id, "status": "INSUFFICIENT_EVIDENCE",
                       "hazard_identified": any(CUES.get(o["label"], (None,))[0] == category for o in objects)}
            else:
                # Keep adjudicated ground truth and reviewer decisions out of every prediction.
                context = prediction_context(case)
                findings, warnings = inspect(objects, indexes.get(variant), context, taxonomy, jpeg, vision, mode)
                if warnings:
                    raise RuntimeError(f"Assessment failed for {case_id} in {variant}: {warnings}")
                row = choose_prediction(case_id, category, findings, variant == "complete", warnings)
            row["detection_seconds"] = detection_seconds
            row["assessment_seconds"] = perf_counter() - start
            records[variant][case_id] = row
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant, rows in records.items():
        (args.output/f"{variant}.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows.values()), encoding="utf-8")
        results[variant] = score(references, rows)
    (args.output/"scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("media_root", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT/"models/yolo/yolo26n.pt")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/ablation")
    parser.add_argument("--local-vision", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
