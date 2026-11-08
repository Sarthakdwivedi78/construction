import argparse
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from retrieval import Index, ROOT

FIXTURES = [
    ("excavated spoil setback 0.65 m", "excavation", "Unknown", {"BOCW-spoil"}),
    ("trench falling 4 metres engineered protection", "excavation", "Unknown", {"BOCW-fall"}),
    ("helmet head protection", "ppe", "Unknown", {"OSH6-ppe"}),
    ("toxic oxygen-deficient atmosphere entry", "confined", "Central", {"CR2026-31(iii)"}),
    ("dust fume vapour emissions", "dust", "Central", {"CR2026-32(i)"}),
    ("blocked exits fire", "fire", "Unknown", {"OSH6-fire"}),
    ("construction access lighting portable emergency lamps", "access", "Central", {"CR2026-46-lighting"}),
    ("quarterly construction emergency preparedness mock drill", "emergency", "Central", {"CR2026-59-emergency-drills"}),
]

def retrieval(hybrid, k, dense_only=False, dense_weight=0.5, reranker_model=None):
    index = Index(hybrid, dense_only=dense_only, dense_weight=dense_weight, reranker_model=reranker_model)
    queries = [(*fixture, "") for fixture in FIXTURES]
    return score_retrieval(index, queries, k)

def score_retrieval(index, queries, k):
    scores = []
    for query, category, authority, truth, state in queries:
        start = perf_counter()
        rows = index.search(query, category, authority, k, state=state)
        latency = (perf_counter() - start) * 1000
        hits = [int(r["hazard_id"] in truth) for r in rows]
        dcg = sum(hit/math.log2(i+2) for i, hit in enumerate(hits))
        ideal = sum(1/math.log2(i+2) for i in range(min(k, len(truth))))
        scores.append({"recall": sum(hits)/len(truth), "precision": sum(hits)/k,
                       "mrr": next((1/(i+1) for i, hit in enumerate(hits) if hit), 0),
                       "ndcg": dcg/ideal, "latency_ms": latency})
    return {"queries": len(scores), "k": k, "mode": "reranked" if index.reranker else "dense" if index.dense_only else "hybrid" if index.encoder else "BM25",
            **{key: mean(s[key] for s in scores) for key in scores[0]}}

def detection(weights, data, split):
    from detection import load_detector
    metrics = load_detector(weights).val(data=data, split=split, imgsz=640)
    p, r = metrics.box.mp, metrics.box.mr
    return {"precision": p, "recall": r, "f1_from_macro_precision_recall": 2*p*r/(p+r) if p+r else 0,
            "map50": metrics.box.map50, "map50_95": metrics.box.map, "split": split}

def smoke(output):
    from pypdf import PdfReader
    from reasoning import inspect, report
    from reporting import pdf
    from io import BytesIO
    taxonomy = json.loads((ROOT/"taxonomy.json").read_text(encoding="utf-8-sig"))
    index = Index()
    objects = [{"label": "NO-Hardhat", "confidence": 0.83, "xyxy": [15, 25, 45, 61]}]
    meta = {"site_type": "Synthetic test fixture", "authority": "Unknown"}
    start = perf_counter()
    hazards, warnings = inspect(objects, index, meta, taxonomy)
    result = report(meta, [{"timestamp_seconds": 0.0, "objects": objects, "hazards": hazards,
                            "warnings": warnings}], perf_counter() - start)
    assert hazards[0]["violation_status"] == "Insufficient Evidence"
    assert hazards[0]["source_url"].startswith("https://www.indiacode.nic.in/")
    assert not index.search("dust", "operations", "State")[0]["jurisdiction"] == "Central"
    assert inspect([], index, meta, taxonomy)[0] == []
    binary = pdf(result)
    assert "Insufficient Evidence" in PdfReader(BytesIO(binary)).pages[0].extract_text()
    output.mkdir(parents=True, exist_ok=True)
    (output/"synthetic-inspection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output/"synthetic-inspection.pdf").write_bytes(binary)
    return {"smoke_test": "passed", "pdf": "text verified", "retrieval_fixture": retrieval(False, 3)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("smoke")
    demo.add_argument("--output", type=Path, default=Path("reports"))
    search = sub.add_parser("retrieval")
    mode = search.add_mutually_exclusive_group()
    mode.add_argument("--hybrid", action="store_true")
    mode.add_argument("--dense", action="store_true")
    search.add_argument("--reranker-model", help="Local or authorized cross-encoder model")
    search.add_argument("--dense-weight", type=float, default=0.5)
    search.add_argument("-k", type=int, choices=range(1, 21), default=3)
    yolo = sub.add_parser("detection")
    yolo.add_argument("--weights", required=True)
    yolo.add_argument("--data", required=True)
    yolo.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    if args.command == "smoke":
        result = smoke(args.output)
    elif args.command == "retrieval":
        if not 0 < args.dense_weight < 1:
            parser.error("--dense-weight must be between zero and one")
        result = retrieval(args.hybrid, args.k, args.dense, args.dense_weight, args.reranker_model)
    else:
        result = detection(args.weights, args.data, args.split)
    print(json.dumps(result, indent=2))
