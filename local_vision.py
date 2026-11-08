"""Local Ollama vision adapter. Requests stay on the Windows loopback interface."""
import base64
import json
import os
from urllib.request import HTTPRedirectHandler, Request, ProxyHandler, build_opener

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None

class LocalVision:
    def __init__(self, model=None):
        self.model = model or os.environ.get("SAFETY_LOCAL_VISION_MODEL", "qwen2.5vl:3b")
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
        schema = "json"
        if "categories" in data:
            fields = {key: {"type": "string"} for key in ("category", "hazard", "visual_evidence", "query")}
            schema = {"type": "object", "required": ["hazards"], "properties": {
                "hazards": {"type": "array", "maxItems": 6, "items": {
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
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        }
        request = Request(self.url, json.dumps(payload).encode("utf-8"),
                          {"Content-Type": "application/json"})
        with self.opener.open(request, timeout=180) as response:
            result = json.load(response)
        answer = json.loads(result["message"]["content"])
        if not isinstance(answer, dict):
            raise ValueError("Expected a JSON object.")
        return answer
