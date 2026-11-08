"""Validate externally collected site measurements before forwarding them as context."""
import json
import math

def parse_measurements(payload):
    rows = json.loads(payload)
    if not isinstance(rows, list) or len(rows) > 30:
        raise ValueError("Measurements must be a JSON array with at most 30 entries.")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each measurement must be a JSON object.")
        fields = {}
        for key in ("name", "unit", "method", "source_reference"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 300:
                raise ValueError(f"Measurement requires a short nonempty {key}.")
            fields[key] = value.strip()
        value = row.get("value")
        if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
            raise ValueError("Measurement value must be a finite number.")
        result.append({**fields, "value": value})
    return result
