"""Convert the authors' India-relevant PPE text annotations to YOLO format."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2

CLASSES = ("NOTSAFE", "NoHardhat", "NoJacket", "SAFE")


def convert(source_root: Path, output: Path):
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")

    totals = {split: {"images": 0, "boxes": 0, "classes": Counter()} for split in ("train", "val")}
    annotation_mismatches = []
    xml_filename_mismatches = []
    for split in ("train", "val"):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        list_path = source_root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(list_path)

        # XML is the paper's stated ground-truth format. The text lists are
        # retained only for discrepancy auditing; they contain repeated rows.
        listed = {}
        for line_no, line in enumerate(list_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            tokens = line.strip().split()
            if not tokens:
                continue
            rel = Path(tokens[0].replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe image path at {list_path}:{line_no}")
            key = rel.name.casefold()
            listed.setdefault(key, [])
            for token in tokens[1:]:
                fields = token.split(",")
                if len(fields) != 5:
                    raise ValueError(f"Malformed text annotation at {list_path}:{line_no}: {token}")
                listed[key].append((fields[4], *(float(value) for value in fields[:4])))

        source_split_dir = source_root / split
        source_images = sorted(path for path in source_split_dir.iterdir()
                               if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not source_images:
            raise ValueError(f"No images found in {source_split_dir}")
        for source_image in source_images:
            frame = cv2.imread(str(source_image))
            if frame is None:
                raise ValueError(f"Unreadable image: {source_image}")
            height, width = frame.shape[:2]
            if width < 1 or height < 1:
                raise ValueError(f"Invalid image size: {source_image}")
            xml_path = source_image.with_suffix(".xml")
            if not xml_path.is_file():
                raise FileNotFoundError(f"Missing VOC annotation: {xml_path}")
            annotation = ET.parse(xml_path).getroot()
            filename = annotation.findtext("filename")
            if filename and filename.casefold() != source_image.name.casefold():
                xml_filename_mismatches.append({"split": split, "image": source_image.name,
                                                 "xml_filename": filename})
            size = annotation.find("size")
            if size is None or (int(size.findtext("width", "0")), int(size.findtext("height", "0"))) != (width, height):
                raise ValueError(f"VOC dimensions differ from image at {xml_path}")

            yolo_rows = []
            xml_boxes = []
            for obj in annotation.findall("object"):
                name = (obj.findtext("name") or "").strip()
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError(f"Missing bounding box in {xml_path}")
                x1, y1, x2, y2 = (float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
                xml_boxes.append((name, x1, y1, x2, y2))
                if name not in CLASSES:
                    raise ValueError(f"Unknown class {name!r} in {xml_path}")
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Out-of-bounds box in {xml_path}: {(x1, y1, x2, y2)} in {width}x{height}")
                class_id = CLASSES.index(name)
                cx, cy = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                bw, bh = (x2 - x1) / width, (y2 - y1) / height
                yolo_rows.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
                totals[split]["boxes"] += 1
                totals[split]["classes"][name] += 1

            text_boxes = listed.pop(source_image.name.casefold(), [])
            if sorted(xml_boxes) != sorted(text_boxes):
                annotation_mismatches.append({"split": split, "image": source_image.name,
                                              "xml_box_count": len(xml_boxes), "text_box_count": len(text_boxes)})

            destination_image = image_dir / source_image.name
            if destination_image.exists():
                raise ValueError(f"Duplicate basename within {split}: {source_image.name}")
            shutil.copy2(source_image, destination_image)
            (label_dir / f"{source_image.stem}.txt").write_text("\n".join(yolo_rows) + "\n", encoding="utf-8")
            totals[split]["images"] += 1
        if listed:
            raise ValueError(f"Text list references images absent from {source_split_dir}: {sorted(listed)[:10]}")

    yaml_path = output / "indian_ppe.yaml"
    yaml_path.write_text(
        "path: " + str(output).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: {json.dumps(list(CLASSES))}\n",
        encoding="utf-8",
    )
    summary = {split: {**row, "classes": dict(row["classes"])} for split, row in totals.items()}
    summary["source_text_vs_xml_mismatches"] = annotation_mismatches
    summary["xml_filename_mismatches"] = xml_filename_mismatches
    (output / "conversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Author checkout's Data directory")
    parser.add_argument("--output", type=Path, default=Path("data/raw/indian-ppe-yolo"))
    args = parser.parse_args()
    summary, yaml_file = convert(args.source_root, args.output)
    print(json.dumps({"dataset_yaml": str(yaml_file), "splits": summary}, indent=2))
