"""Run a reproducible detector check on the upstream repository's labeled fixtures."""
import argparse
import json
from pathlib import Path
import shutil

import yaml
from detection import LABELS, load_detector

def run(upstream, weights, output):
    fixture = upstream / "tests" / "cv_dataset"
    source_names = yaml.safe_load((fixture/"data.yaml").read_text(encoding="utf-8"))["names"]
    remap = {i: LABELS.index(name) for i, name in enumerate(source_names)}
    output.mkdir(parents=True, exist_ok=False)
    images = output/"images"
    labels = output/"labels"
    images.mkdir()
    labels.mkdir()
    count = 0
    for image in sorted((fixture/"images").glob("*.jpg")):
        original = fixture/"labels"/(image.stem+".txt")
        if not original.is_file():
            raise FileNotFoundError(original)
        mapped = []
        for line in original.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, *coordinates = line.split()
                if len(coordinates) != 4:
                    raise ValueError(f"Expected a detection box in {original}")
                mapped.append(" ".join([str(remap[int(cls)]), *coordinates]))
        (labels/original.name).write_text("\n".join(mapped)+"\n", encoding="utf-8")
        shutil.copyfile(image, images/image.name)
        count += 1
    if not count:
        raise ValueError("No upstream fixture images.")
    data = output/"fixture.yaml"
    data.write_text(yaml.safe_dump({"path": output.resolve().as_posix(), "train": "images",
                                   "val": "images", "names": list(LABELS)}), encoding="utf-8")
    metrics = load_detector(weights).val(data=str(data.resolve()), device="cpu", imgsz=640,
                                         batch=2, workers=0, plots=False,
                                         project=str(output.resolve()), name="metrics")
    p, r = float(metrics.box.mp), float(metrics.box.mr)
    result = {"scope": "Upstream repository fixtures; training overlap unknown",
              "images": count, "precision": p, "recall": r,
              "f1_from_macro_precision_recall": 2*p*r/(p+r) if p+r else 0,
              "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
    (output/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "weights", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.upstream, args.weights, args.output), indent=2))
