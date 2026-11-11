"""Fine-tune a separate YOLO checkpoint on the India-relevant VOC dataset."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO
import ultralytics

CLASSES = ["NOTSAFE", "NoHardhat", "NoJacket", "SAFE"]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/indian-ppe-yolo-ready/indian_ppe.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolo/yolo26n.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/indian-ppe"))
    parser.add_argument("--name", default="")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or not args.data.is_file() or not args.weights.is_file():
        parser.error("epochs/batch must be positive and both data YAML and starting weights must exist")

    run_name = args.name or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-utc")
    model = YOLO(str(args.weights))
    starting_names = [model.names[index] for index in sorted(model.names)]
    conversion_summary_path = args.data.with_name("conversion_summary.json")
    conversion_summary = json.loads(conversion_summary_path.read_text(encoding="utf-8"))
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "starting_weights": str(args.weights.resolve()),
        "starting_weights_sha256": digest(args.weights),
        "dataset_yaml": str(args.data.resolve()),
        "dataset_conversion_summary": str(conversion_summary_path.resolve()),
        "dataset_conversion_counts": {key: conversion_summary[key] for key in ("train", "val")},
        "initial_classes": starting_names,
        "trained_classes": CLASSES,
        "ultralytics_version": ultralytics.__version__,
        "settings": {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                     "device": args.device, "workers": args.workers, "seed": 42,
                     "patience": 5, "amp": False, "pretrained": True},
        "dataset_source": "Sankarlalr23/Detection-of-PPE-Compliance-on-Construction-Site-Using-CV (Data/train and Data/val VOC XML)",
        "dataset_source_git_commit": "b5eddccfae85c2c606c9773c8cdb1f92b98890d0",
        "dataset_notice": "Author repository has no license file; keep raw images and derived weights local unless usage/redistribution rights are confirmed.",
    }
    run_dir = args.project / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "training_manifest.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    model.train(data=str(args.data.resolve()), device=args.device, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, workers=args.workers, amp=False,
                patience=5, seed=42, project=str(args.project.resolve()), name=run_name,
                plots=True, exist_ok=True, pretrained=True)

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["best_weights"] = str(best.resolve()) if best.is_file() else None
    manifest["best_weights_sha256"] = digest(best) if best.is_file() else None
    manifest["last_weights"] = str(last.resolve()) if last.is_file() else None
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
