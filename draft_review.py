"""Create unlabeled reviewer worklists from attributed media and optional detections."""
import argparse
import hashlib
import json
from pathlib import Path

def draft(manifest, report_dir=None):
    rows = json.loads(manifest.read_text(encoding="utf-8-sig"))
    seen = set()
    for row in rows:
        key = row["case_id"]
        if key in seen or not key or not key.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid or repeated case ID: {key}")
        seen.add(key)
        path = (manifest.parent / row["image"]).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError(f"Image outside manifest directory: {key}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Source checksum mismatch: {key}")
        detections = []
        if report_dir:
            report = report_dir / f"{key}.json"
            if report.is_file():
                evidence = json.loads(report.read_text(encoding="utf-8-sig"))
                if evidence.get("media_sha256") != digest:
                    raise ValueError(f"Report does not match the source image: {key}")
                detections = evidence["detected_objects"]
        yield {**row, "reviewed": False, "reviewer": "", "question": "",
               "hazard_category": "", "detector_cues": detections, "retrieved_regulations": [],
               "answer": {"violation_status": "", "reasoning": "", "corrective_action": ""}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    rows = list(draft(args.manifest, args.report_dir))
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(f"Drafted {len(rows)} unlabeled cases.")
