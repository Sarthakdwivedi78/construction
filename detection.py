from pathlib import Path

LABELS = ("Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person",
          "Safety Cone", "Safety Vest", "machinery", "utility pole", "vehicle")

def load_detector(path):
    from ultralytics import YOLO
    if not Path(path).is_file():
        raise FileNotFoundError("Supply a trusted upstream construction checkpoint.")
    model = YOLO(str(path))
    if tuple(model.names[i] for i in sorted(model.names)) != LABELS:
        raise ValueError("Checkpoint label map differs from the documented baseline.")
    return model

def detect(model, frame, confidence=0.25):
    result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
    objects = [{"label": model.names[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(value, 2) for value in box.xyxy[0].tolist()]} for box in result.boxes]
    return objects, result.plot()

def frames(path, video, interval=2, limit=12):
    import cv2
    if not video:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Cannot decode image.")
        yield 0.0, frame
        return
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not fps > 0:
            raise ValueError("Cannot decode video or determine frame rate.")
        stride = max(1, round(fps * interval))
        for i in range(limit):
            frame_number = i * stride
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                if i == 0:
                    raise ValueError("Cannot decode first video frame.")
                break
            yield frame_number / fps, frame
    finally:
        capture.release()
