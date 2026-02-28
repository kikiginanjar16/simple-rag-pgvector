import json

def validate_guarded_output(model_text, hits):
    data = json.loads(model_text)
    allowed = {(h["source"], h["chunk_id"]): h for h in hits}

    for c in data.get("citations", []):
        key = (c["source"], c["chunk_id"])
        if key not in allowed:
            raise ValueError("Invalid citation reference")
        if c["quote"].lower() not in allowed[key]["content"].lower():
            raise ValueError("Quote not found in cited content")

    return data
