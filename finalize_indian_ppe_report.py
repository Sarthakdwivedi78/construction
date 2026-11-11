"""Build a concise, reproducible summary for the completed Indian PPE run."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "indian-ppe"
run = max((p for p in RUNS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
manifest = json.loads((run / "training_manifest.json").read_text(encoding="utf-8"))
conversion = json.loads(Path(manifest["dataset_conversion_summary"]).read_text(encoding="utf-8"))
validation = json.loads((run / "best_validation.json").read_text(encoding="utf-8"))
with (run / "results.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))

train, val = conversion["train"], conversion["val"]
manifest["dataset_source_git_commit"] = "b5eddccfae85c2c606c9773c8cdb1f92b98890d0"
manifest["dataset_conversion_counts"] = {"train": train, "val": val}
manifest["completed_epochs"] = len(rows)
manifest["best_epoch_by_validation_map50_95"] = int(best["epoch"])
(run / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
metrics = {
    "run": run.name,
    "status": "completed_training_and_validation",
    "dataset": {
        "source": manifest["dataset_source"],
        "source_commit": "b5eddccfae85c2c606c9773c8cdb1f92b98890d0",
        "classes": manifest["trained_classes"],
        "train": train,
        "validation": val,
        "total_images": train["images"] + val["images"],
        "annotation_text_xml_discrepancies": conversion.get("source_text_vs_xml_mismatches", []),
        "stale_xml_filename_fields": len(conversion.get("xml_filename_mismatches", [])),
    },
    "training": {
        "best_epoch_by_validation_map50_95": int(best["epoch"]),
        "epochs_completed": len(rows),
        "settings": manifest["settings"],
        "starting_weights": manifest["starting_weights"],
        "starting_weights_sha256": manifest["starting_weights_sha256"],
        "best_weights": manifest["best_weights"],
        "best_weights_sha256": manifest["best_weights_sha256"],
        "ultralytics_version": manifest["ultralytics_version"],
    },
    "validation_of_selected_best_checkpoint": validation,
    "caveats": [
        "Validation data is not an independent test set; it was used during checkpoint selection.",
        "No separate public test split was present in the author's repository.",
        "Validation contains only 18 NoHardhat boxes, so its class score is unstable.",
        "The paper describes a mix of Indian construction-site and web images; this is Indian-relevant, not exclusively India-origin imagery.",
        "The dataset repository has no license file; keep raw images and derived weights local until redistribution rights are confirmed.",
        "Images were not established to be site-disjoint, and possible train/validation overlap cannot be ruled out.",
        "The model is a screening aid, not proof that a site is safe or unsafe.",
    ],
}
(run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

classes = metrics["validation_of_selected_best_checkpoint"]["classes"]
overall = metrics["validation_of_selected_best_checkpoint"]["overall"]
class_rows = []
for name in manifest["trained_classes"]:
    n = val["classes"].get(name, 0)
    m = classes[name]
    class_rows.append(
        f"| {name} | {n} | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{m['map50']:.3f} | {m['map50_95']:.3f} |"
    )
report = f"""# Indian PPE YOLO fine-tuning report

## Result

Fine-tuned the project's YOLO26n starting checkpoint for **20 epochs** on the authors' India-relevant construction PPE dataset. The selected checkpoint is `weights/best.pt` (best validation mAP@0.5:0.95 epoch: {int(best['epoch'])}). Training completed at {manifest['finished_at_utc']} UTC. The baseline checkpoint was preserved.

The best checkpoint was re-evaluated on the supplied validation split: precision **{overall['precision']:.3f}**, recall **{overall['recall']:.3f}**, mAP@0.5 **{overall['map50']:.3f}**, mAP@0.5:0.95 **{overall['map50_95']:.3f}**. These are validation results, not an independent test score.

## Dataset and labels

Source: [authors' project repository](https://github.com/Sankarlalr23/Detection-of-PPE-Compliance-on-Construction-Site-Using-CV), commit `b5eddccfae85c2c606c9773c8cdb1f92b98890d0`. The associated [paper](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2020.00136/full) describes a mix of Indian construction-site photos and web imagery. The public repository subset used here contains {train['images']:,} training images / {train['boxes']:,} VOC bounding boxes and {val['images']:,} validation images / {val['boxes']:,} boxes ({train['images'] + val['images']:,} total). It does not contain a separate test split.

The conversion used paired Pascal VOC XML boxes as the ground truth, not the text annotation lists. Three image records differed between the text lists and XML. There are {len(conversion.get('xml_filename_mismatches', []))} stale filename fields inside XML; image/XML pairs were matched by same-basename file and image dimensions checked. The discrepancy details are preserved in `conversion_summary.json`. The training run reported zero corrupt validation images.

| Class | Validation boxes | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}
| **Overall** | **{val['boxes']}** | **{overall['precision']:.3f}** | **{overall['recall']:.3f}** | **{overall['map50']:.3f}** | **{overall['map50_95']:.3f}** |

The `NoHardhat` validation result is based on only {val['classes'].get('NoHardhat', 0)} boxes. Its class score is therefore especially uncertain.

## Training and inference settings

- Starting weights: `models/yolo/yolo26n.pt` (SHA-256 `{manifest['starting_weights_sha256']}`).
- Training: {manifest['settings']['epochs']} epochs maximum, image size {manifest['settings']['imgsz']}, batch {manifest['settings']['batch']}, CPU, seed {manifest['settings']['seed']}, pretrained fine-tune, Ultralytics {manifest['ultralytics_version']}.
- Labels: `{', '.join(manifest['trained_classes'])}`. The original baseline uses a different 11-class mapping and remains available separately.
- App inference settings: confidence 0.25 and image size 640 (`detection.py`).
- Best checkpoint SHA-256: `{manifest['best_weights_sha256']}`.

## Limitations and rights

- The validation split was used for checkpoint selection and is not an independent test set. The reported performance may be optimistic.
- The paper's data is mixed-source, rather than exclusively Indian-origin imagery. The available public subset is smaller than the 2,509 images described by the paper.
- No site-disjoint split or duplicate audit was available, so generalization to unseen Indian sites is not established.
- The dataset repository has no license file. Raw photos and derived weights remain local until redistribution rights are confirmed; public deployment/publishing of weights is not cleared by this report.
- Results are screening output only. A qualified person must inspect the scene and applicable requirements; detections do not certify site safety.

## Artifacts

- `weights/best.pt` — selected India-relevant fine-tuned model.
- `weights/last.pt` — final epoch checkpoint.
- `results.csv`, `results.png` — epoch metrics and plots.
- `confusion_matrix.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` — validation plots.
- `val_batch*_labels.jpg`, `val_batch*_pred.jpg` — validation labels and predictions.
- `training_manifest.json`, `best_validation.json`, `metrics.json`, `report.md` — reproducibility and summary.
- Dataset conversion: `data/raw/indian-ppe-yolo-ready/conversion_summary.json`.
"""
(run / "report.md").write_text(report, encoding="utf-8")
print(f"Wrote {(run / 'metrics.json')} and {(run / 'report.md')}")
