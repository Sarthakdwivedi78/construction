import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from time import perf_counter

from detection import detect, frames, load_detector
from local_vision import LocalVision
from measurements import parse_measurements
from reasoning import inspect, report
from reporting import pdf
from retrieval import Index, ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local image or video inspection.")
    parser.add_argument("media", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT/"models/yolo/yolo26n.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"reports"/"live")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--export-index", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--local-vision", action="store_true")
    parser.add_argument("--measurements", type=Path, help="Measured site evidence JSON")
    parser.add_argument("--inspection-date", type=date.fromisoformat)
    parser.add_argument("--state", default="")
    parser.add_argument("--authority", choices=["Unknown", "Central", "State"], default="Unknown")
    args = parser.parse_args()
    import cv2
    taxonomy = json.loads((ROOT/"taxonomy.json").read_text(encoding="utf-8-sig"))
    model = load_detector(args.weights)
    index = Index(args.hybrid)
    vision = LocalVision() if args.local_vision else None
    if args.export_index:
        index.save_dense(args.export_index)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    meta = {"site_type": "Construction scene", "authority": args.authority, "state": args.state}
    if args.inspection_date:
        meta["inspection_date"] = args.inspection_date.isoformat()
    if args.measurements:
        meta["site_measurements"] = parse_measurements(args.measurements.read_bytes())
    start = perf_counter()
    for i, (timestamp, frame) in enumerate(frames(args.media, args.video)):
        tick = perf_counter()
        objects, annotated = detect(model, frame)
        detection_ms = (perf_counter()-tick)*1000
        jpeg = None
        if vision:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Cannot encode frame for local vision.")
            jpeg = encoded.tobytes()
        hazards, warnings = inspect(objects, index, meta, taxonomy, jpeg, vision)
        records.append(dict(timestamp_seconds=timestamp, objects=objects, hazards=hazards,
                            warnings=warnings, detection_ms=detection_ms))
        if not cv2.imwrite(str(args.output/f"frame-{i:03d}.jpg"), annotated):
            raise OSError("Could not write annotated frame")
    output = report(meta, records, perf_counter()-start)
    with args.media.open("rb") as stream:
        output["media_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    output["vision_provider"] = "Local Ollama" if vision else "Off"
    output["vision_model"] = vision.model if vision else None
    output["detection_ms"] = [r["detection_ms"] for r in records]
    (args.output/"inspection.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (args.output/"inspection.pdf").write_bytes(pdf(output))
    print(json.dumps({"frames": len(records), "detections": len(output["detected_objects"]),
                      "hazards": len(output["identified_hazards"]), "detection_ms": output["detection_ms"],
                      "processing_seconds": output["processing_seconds"]}, indent=2))
