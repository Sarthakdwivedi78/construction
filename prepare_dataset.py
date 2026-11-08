"""Validate reviewed records and assign deterministic site-disjoint splits."""
import argparse
import hashlib
import json
from pathlib import Path

from retrieval import Index

STATUSES = {"VIOLATION", "NO_VIOLATION", "INSUFFICIENT_EVIDENCE"}

def validate_answer(row, known):
    answer = row.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("Answer must be an object.")
    if not isinstance(answer.get("violation_status"), str) or answer["violation_status"] not in STATUSES:
        raise ValueError("Invalid decision status.")
    for field in ("hazard", "reasoning", "corrective_action"):
        if not isinstance(answer.get(field), str) or not answer[field].strip():
            raise ValueError(f"Answer needs nonempty text: {field}")
    citations = row.get("retrieved_regulations")
    if not isinstance(citations, list) or any(not isinstance(value, str) for value in citations):
        raise ValueError("Retrieved regulations must be a list of provision IDs.")
    if not set(citations) <= known or (answer["violation_status"] != "INSUFFICIENT_EVIDENCE" and not citations):
        raise ValueError("Unresolved source citation.")

def build(manifest, media_root, destination):
    targets = [destination / f"{name}.jsonl" for name in ("train", "validation", "test")]
    if any(path.exists() for path in targets):
        raise ValueError("Split files already exist. Choose a fresh output directory to preserve reviewed data.")
    known = {row["hazard_id"] for row in Index().rows}
    splits = {name: [] for name in ("train", "validation", "test")}
    fingerprints = {}
    site_groups = {}
    case_ids = set()
    records_by_group = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("image", "site_id", "source_url", "usage_permission", "reviewer", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Required field missing: {field}")
        if row.get("reviewed") is not True:
            raise ValueError("Only reviewed examples can enter supervised splits.")
        image = (media_root / row["image"]).resolve()
        if not image.is_relative_to(media_root.resolve()) or not image.is_file():
            raise ValueError(f"Image missing or outside media root: {row['image']}")
        validate_answer(row, known)
        site = row["site_id"].strip().casefold()
        raw_group = row.get("site_group", site)
        if not isinstance(raw_group, str):
            raise ValueError("Site group must be nonempty text.")
        group = raw_group.strip().casefold()
        if not group or site_groups.setdefault(site, group) != group:
            raise ValueError("All records from a site must share one nonempty site group.")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("Case IDs must be unique nonempty strings.")
        case_ids.add(case_id)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if row.get("sha256") and row["sha256"] != digest:
            raise ValueError("Image checksum differs from the reviewed record.")
        if fingerprints.setdefault(digest, site) != site:
            raise ValueError("Identical image is assigned to multiple sites.")
        row["sha256"] = digest
        records_by_group.setdefault(group, []).append(row)
    groups = sorted(records_by_group, key=lambda group: (hashlib.sha256(group.encode()).digest(), group))
    if len(groups) < 3:
        raise ValueError("Every split needs at least one reviewed site group. Add more independent sites.")
    # Allocate by group count. Hash order keeps assignment independent of manifest order.
    holdout_count = max(1, round(len(groups) * 0.15))
    for split, partition in (
        ("train", groups[:-2 * holdout_count]),
        ("validation", groups[-2 * holdout_count:-holdout_count]),
        ("test", groups[-holdout_count:]),
    ):
        for group in partition:
            splits[split].extend(records_by_group[group])
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (destination / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            stream.write("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
    return {name: len(rows) for name, rows in splits.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("media_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/instruction_dataset"))
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.media_root, args.output), indent=2))
