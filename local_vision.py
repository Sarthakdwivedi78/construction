"""Local Ollama vision adapter. Requests stay on the Windows loopback interface."""
import base64
import json
import os
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, ProxyHandler, build_opener

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None

class LocalVision:
    def __init__(self, model=None):
        self.model = model or os.environ.get("SAFETY_LOCAL_VISION_MODEL", "moondream")
        self.category_only = True
        if self.model.lower().endswith((":cloud", "-cloud")):
            raise ValueError("Configure an installed local vision model.")
        port = int(os.environ.get("SAFETY_OLLAMA_PORT", "11434"))
        if not 1 <= port <= 65535:
            raise ValueError("Invalid Ollama port.")
        self.url = f"http://127.0.0.1:{port}/api/chat"
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def ask(self, instructions, data, jpeg):
        if not jpeg:
            raise ValueError("Vision assessment requires image bytes.")
        if "categories" in data:
            try:
                import cv2
                import numpy as np
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None and max(frame.shape[:2]) > 768:
                    scale = 768 / max(frame.shape[:2])
                    frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        jpeg = encoded.tobytes()
            except ImportError:
                pass
        schema = "json"
        if "categories" in data:
            fields = {
                "category": {"type": "string", "enum": list(data["categories"])},
                "hazard": {"type": "string"},
                "visual_evidence": {"type": "string"},
            }
            schema = {"type": "object", "required": ["hazards"], "properties": {
                "hazards": {"type": "array", "maxItems": 3, "items": {
                    "type": "object", "properties": fields, "required": list(fields),
                    "additionalProperties": False}}}, "additionalProperties": False}
        elif "cue" in data:
            ids = [row["hazard_id"] for row in data.get("sources", [])]
            schema = {"type": "object", "properties": {
                "status": {"enum": ["Violation", "No Violation", "Insufficient Evidence"]},
                "reasoning": {"type": "string", "maxLength": 240}, "cited_id": {"enum": [None, *ids]}},
                "required": ["status", "reasoning", "cited_id"], "additionalProperties": False}
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": json.dumps(data),
                          "images": [base64.b64encode(jpeg).decode("ascii")]}],
            "options": {"temperature": 0,
                        "num_ctx": 2048 if "categories" in data else 4096,
                        "num_batch": 16,
                        "num_predict": 192 if "categories" in data else 256},
        }
        request = Request(self.url, json.dumps(payload).encode("utf-8"),
                          {"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=180) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail[:300]}") from exc
        answer = json.loads(result["message"]["content"])
        if not isinstance(answer, dict):
            raise ValueError("Expected a JSON object.")
        return answer
