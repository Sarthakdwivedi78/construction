"""Evaluate visible missing-PPE screening using the publisher's held-out validation split."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median, mean, quantiles
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from detection import detect, load_detector
from reasoning import inspect
from retrieval import Index

CLASSES = {0: "Helmet", 1: "NoHelmet", 2: "NoVest", 3: "Vest"}
CUES = {1: "NO-Hardhat", 2: "NO-Safety Vest"}
DATASET = "https://data.mendeley.com/datasets/zkzghjvpn2/6"
DATA_HASH = "7a22e5cbc0327971f56358e14f1fce88fec1d6ab30cfac642de0746202ae1f0d"
THRESHOLD = .25


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_boxes(raw, width, height):
    result = []
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        category, x, y, w, h = map(float, line.split())
        if category not in CLASSES or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid YOLO annotation: {line}")
        result.append((int(category), [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]))
    return result


def image_rows(archive):
    paths = [PurePosixPath(n.replace("\\", "/")) for n in archive.namelist()]
    names = {PurePosixPath(n.replace("\\", "/")): n for n in archive.namelist()}
    configs = [p for p in paths if p.name == "data.yaml"]
    if not configs:
        raise ValueError("Publisher dataset YAML missing.")
    names_config = yaml.safe_load(archive.read(names[configs[0]]))["names"]
    class_names = dict(enumerate(names_config)) if isinstance(names_config, list) else names_config
    canonical = lambda s: "".join(c for c in str(s).casefold() if c.isalnum())
    assert {int(k): canonical(v) for k,v in class_names.items()} == {k:canonical(v) for k,v in CLASSES.items()}, class_names
    test, train = [], []
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or "images" not in path.parts:
            continue
        if "valid" in path.parts:
            test.append(path)
        elif any(part in path.parts for part in ("train",)):
            train.append(path)
    if not test:
        raise ValueError("No explicit publisher validation split found.")
    other_hashes = {sha256(archive.read(names[p])) for p in train}
    stats = Counter()
    seen = set()
    seen_pixels = set()
    rows = []
    for path in sorted(test):
        stats["publisher_validation_images"] += 1
        label_path = PurePosixPath(*["labels" if s == "images" else s for s in path.parts]).with_suffix(".txt")
        if label_path not in names:
            stats["missing_annotation"] += 1
            continue
        data = archive.read(names[path])
        digest = sha256(data)
        if digest in other_hashes:
            stats["exact_cross_split_duplicates"] += 1
            continue
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Undecodable image: {path}")
        pixel_digest = sha256(str(frame.shape).encode()+frame.tobytes())
        if digest in seen or pixel_digest in seen_pixels:
            stats["validation_duplicates"] += 1
            continue
        height, width = frame.shape[:2]
        boxes = parse_boxes(archive.read(names[label_path]),width,height)
        labels = {kind for kind,_ in boxes}
        if not labels:
            stats["empty_annotation"] += 1
            continue
        seen.add(digest)
        seen_pixels.add(pixel_digest)
        rows.append((path.as_posix(), digest, boxes))
    stats["included"] = len(rows)
    return rows, stats, names


def ratio(a,b):
    return a/b if b else None


def wilson(success,total):
    if not total:
        return None
    p=success/total
    z=1.95996398454
    denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0,center-radius),min(1,center+radius)]


def metrics(truth, scores, threshold=THRESHOLD):
    pred=[v>=threshold for v in scores]
    tp=sum(t and p for t,p in zip(truth,pred))
    fp=sum(not t and p for t,p in zip(truth,pred))
    tn=sum(not t and not p for t,p in zip(truth,pred))
    fn=sum(t and not p for t,p in zip(truth,pred))
    recall=ratio(tp,tp+fn)
    specificity=ratio(tn,tn+fp)
    both=bool(tp+fn and tn+fp)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=ratio(tp+tn,len(truth)),
                precision=ratio(tp,tp+fp),recall=recall,specificity=specificity,
                f1=ratio(2*tp,2*tp+fp+fn),negative_predictive_value=ratio(tn,tn+fn),
                balanced_accuracy=(recall+specificity)/2 if both else None,
                recall_wilson95=wilson(tp,tp+fn),specificity_wilson95=wilson(tn,tn+fp),
                roc_auc=float(roc_auc_score(truth,scores)) if both else None,
                average_precision=float(average_precision_score(truth,scores)) if both else None)


def evaluate(args):
    archive_hash=hashlib.file_digest(args.dataset.open("rb"),"sha256").hexdigest()
    if archive_hash != DATA_HASH:
        raise ValueError("Publisher SHA-256 verification failed.")
    weights=args.project/"models/yolo/yolo26n.pt"
    taxonomy=json.loads((args.project/"taxonomy.json").read_text(encoding="utf-8-sig"))
    index=Index()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    cv2.setNumThreads(1)
    with ZipFile(args.dataset) as archive:
        test, excluded, names=image_rows(archive)
        if len(test)<100:
            raise ValueError(f"Only {len(test)} eligible labeled validation images.")
        model=load_detector(weights)
        model.to("cpu")
        frame=np.zeros((640,640,3),np.uint8)
        detect(model,frame)
        predictions=[]
        print(f"Evaluating {len(test)} labeled validation images.",flush=True)
        for i,(name,digest,boxes) in enumerate(test,1):
            frame=cv2.imdecode(np.frombuffer(archive.read(names[PurePosixPath(name)]),dtype=np.uint8),cv2.IMREAD_COLOR)
            start=perf_counter()
            outputs,_=detect(model,frame,.01)
            inference_ms=(perf_counter()-start)*1000
            # Apply the deployed default threshold. Lower-score boxes support curve metrics.
            deployed=[o for o in outputs if o["confidence"]>=THRESHOLD]
            start=perf_counter()
            findings,errors=inspect(deployed,index,{"authority":"Unknown","site_type":"External benchmark"},taxonomy)
            reasoning_ms=(perf_counter()-start)*1000
            conf={cue:max((o["confidence"] for o in outputs if o["label"]==cue),default=0) for cue in CUES.values()}
            kinds={k for k,_ in boxes}
            predicates={cue:k in kinds for k,cue in CUES.items()}
            expected=any(predicates.values())
            predicted=max(conf.values())>=THRESHOLD
            record=dict(image=name,sha256=digest,truth_missing_ppe=expected,predicted_missing_ppe=predicted,
                        result=("TP" if expected else "FP") if predicted else ("FN" if expected else "TN"),
                        annotation_classes=sorted(CLASSES[k] for k in kinds),truth_by_cue=predicates,
                        score_by_cue=conf,score=max(conf.values()),ground_truth_boxes=boxes,detections=deployed,
                        inference_ms=inference_ms,reasoning_ms=reasoning_ms,
                        decisions=dict(Counter(f["violation_status"] for f in findings)),
                        selected_provisions=sorted({f["selected_provision_id"] for f in findings if f.get("selected_provision_id")}),
                        reasoning_errors=errors)
            predictions.append(record)
            if i%25==0 or i==len(test):
                print(f"{i}/{len(test)}",flush=True)
    truth=[r["truth_missing_ppe"] for r in predictions]
    scores=[r["score"] for r in predictions]
    assert any(truth) and not all(truth)
    misses=[r["image"] for r in predictions if r["result"]=="FN"]
    false_alarms=[r["image"] for r in predictions if r["result"]=="FP"]
    inference=[r["inference_ms"] for r in predictions]
    reason=[r["reasoning_ms"] for r in predictions]
    all_decisions=Counter()
    for row in predictions:
        all_decisions.update(row["decisions"])
    summary=dict(dataset_url=DATASET,dataset_doi="10.17632/zkzghjvpn2.6",dataset_sha256=archive_hash,
                 checkpoint=weights.name,checkpoint_sha256=sha256(weights.read_bytes()),images=len(truth),
                 validation_selection=dict(excluded),threshold=THRESHOLD,score_floor=.01,device="CPU",imgsz=640,
                 reference="At least one published NoHelmet or NoVest annotation. Negative means no such annotation on an image with PPE annotations.",
                 scope="Publisher validation split of mixed PPE imagery. Geography, independent site count, and upstream checkpoint training overlap are unverified. No Indian legal compliance ground truth.",
                 positive_images=sum(truth),negative_images=len(truth)-sum(truth),metrics=metrics(truth,scores),
                 by_cue={cue:metrics([r["truth_by_cue"][cue] for r in predictions],[r["score_by_cue"][cue] for r in predictions]) for cue in CUES.values()},
                 descriptive_thresholds={str(t):metrics(truth,scores,t) for t in (.15,.25,.40,.60)},
                 latency_ms=dict(mean_detector=mean(inference),median_detector=median(inference),p95_detector=quantiles(inference,n=100)[94],mean_bm25_and_gate=mean(reason)),
                 evidence_gate=dict(finding_status_counts=dict(all_decisions),
                                    images_with_cited_findings=sum(bool(r["selected_provisions"]) for r in predictions),
                                    images_without_findings=sum(not r["decisions"] for r in predictions),
                                    images_with_errors=sum(bool(r["reasoning_errors"]) for r in predictions),
                                    vision_provider="disabled; this run tests YOLO and deterministic BM25/evidence gate"),
                 missed_images=misses,false_alarm_images=false_alarms)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (args.output/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions),encoding="utf-8")
    with (args.output/"predictions.csv").open("w",encoding="utf-8",newline="") as stream:
        fields=["image","sha256","truth_missing_ppe","predicted_missing_ppe","score","result","inference_ms","reasoning_ms"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({k:summary[k] for k in ("images","positive_images","negative_images","metrics","evidence_gate")},indent=2),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset",type=Path)
    cli.add_argument("project",type=Path)
    cli.add_argument("output",type=Path)
    evaluate(cli.parse_args())
