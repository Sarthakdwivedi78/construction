import argparse
from pathlib import Path

from detection import load_detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune only after evaluating the upstream checkpoint.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    model = load_detector("models/yolo/yolo26n.pt")
    model.train(data=str(args.dataset.resolve()), device=args.device, epochs=args.epochs,
                imgsz=640, batch=2, workers=0, amp=args.device != "cpu", patience=5,
                seed=42, project="models/checkpoints", name="indian-yolo", plots=True)
