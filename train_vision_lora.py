"""Train a SmolVLM2 LoRA adapter from independently reviewed examples."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from retrieval import Index
from prepare_dataset import validate_answer
from score_predictions import STATUSES

def load_splits(root, media):
    media = media.resolve()
    ids, groups, sites, fingerprints = set(), {}, {}, {}
    provisions = {r["hazard_id"]: r for r in Index().rows}
    splits = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{split} contains no reviewed examples.")
        for row in rows:
            key = row["case_id"]
            if row.get("reviewed") is not True or not row["reviewer"].strip() or key in ids:
                raise ValueError(f"Missing independent review or duplicate case ID: {key}")
            ids.add(key)
            site = row["site_id"].strip().casefold()
            group = row.get("site_group", site).strip().casefold()
            image = (media / row["image"]).resolve()
            if not site or not group or not image.is_relative_to(media) or not image.is_file():
                raise ValueError(f"Invalid site group or source image: {key}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if row.get("sha256", digest) != digest:
                raise ValueError(f"Image checksum mismatch: {key}")
            if (groups.setdefault(group, split) != split or sites.setdefault(site, split) != split
                or fingerprints.setdefault(digest, split) != split):
                raise ValueError(f"Cross-split site or image overlap: {key}")
            validate_answer(row, set(provisions))
            answer = row["answer"]
            if answer["violation_status"] not in STATUSES or not all(
                isinstance(answer.get(field), str) and answer[field].strip()
                for field in ("reasoning", "corrective_action")
            ) or not row["question"].strip():
                raise ValueError(f"Incomplete instruction target: {key}")
            cited = row["retrieved_regulations"]
            if any(source not in provisions for source in cited):
                raise ValueError(f"Unknown provision for {key}")
            if answer["violation_status"] != "INSUFFICIENT_EVIDENCE" and not cited:
                raise ValueError(f"Conclusive label needs a provision: {key}")
            row["image_path"] = str(image)
            row["sources"] = [provisions[source] for source in cited]
        splits[split] = rows
    return splits

def train(args, splits):
    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from transformers import (AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback,
                              Trainer, TrainingArguments)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"]))
    model.config.use_cache = False

    def collate(rows):
        # Single-image microbatches avoid padding a variable image-token prefix.
        row, = rows
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        image.thumbnail((768, 768))
        context = json.dumps({"question": row["question"], "sources": row["sources"],
                              "detections": row.get("detector_cues", [])}, ensure_ascii=False)
        user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": context}]}]
        assistant = {"role": "assistant", "content": [{"type": "text",
                     "text": json.dumps(row["answer"], ensure_ascii=False)}]}
        prefix = processor.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        conversation = processor.apply_chat_template(user + [assistant], tokenize=False)
        prefix_ids = processor(text=prefix, images=[image], return_tensors="pt")["input_ids"]
        batch = processor(text=conversation, images=[image], return_tensors="pt")
        labels = batch["input_ids"].clone()
        length = prefix_ids.shape[-1]
        if length >= labels.shape[-1] or not torch.equal(labels[:, :length], prefix_ids):
            raise ValueError(f"Cannot locate assistant target in {row['case_id']}")
        labels[:, :length] = -100
        batch["labels"] = labels
        return batch

    # Check template compatibility before creating training checkpoints.
    for rows in splits.values():
        collate(rows[0])
    cfg = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
                            per_device_train_batch_size=1, per_device_eval_batch_size=1,
                            gradient_accumulation_steps=4, learning_rate=1e-4,
                            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                            load_best_model_at_end=True, metric_for_best_model="eval_loss",
                            greater_is_better=False, remove_unused_columns=False,
                            prediction_loss_only=True, dataloader_num_workers=0,
                            use_cpu=args.device == "cpu", report_to=[], seed=42)
    trainer = Trainer(model=model, args=cfg, train_dataset=splits["train"],
                      eval_dataset=splits["validation"], data_collator=collate,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    result = trainer.train()
    heldout = trainer.evaluate(splits["test"], metric_key_prefix="test")
    trainer.save_model(str(args.output / "adapter"))
    processor.save_pretrained(args.output / "adapter")
    metrics = {"model": args.model, "device": args.device, "seconds": perf_counter()-start,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "training": result.metrics, "heldout": heldout, "history": trainer.state.log_history}
    (args.output/"training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", type=Path)
    parser.add_argument("media_root", type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/smolvlm2-lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    try:
        splits = load_splits(args.splits, args.media_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({split: len(rows) for split, rows in splits.items()}))
    if not args.check_only:
        if args.output.exists():
            parser.error("Choose a new output directory to preserve existing checkpoints.")
        train(args, splits)
