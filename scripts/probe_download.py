"""Read only a small weight prefix to diagnose official download connectivity."""

import json
import time
from pathlib import Path
from urllib.parse import urlsplit
import requests

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads(
    (ROOT / "reports/models-v1/manifest.json").read_text(encoding="utf-8")
)
model = manifest["models"][0]
item = next(row for row in model["files"] if row["path"].endswith(".safetensors"))
base = (
    "https://huggingface.co/"
    + model["repository"]
    + "/resolve/"
    + model["revision"]
    + "/"
    + item["path"]
)
results = []
for cache_bust in [False, True]:
    url = base + ("?download=true&probe=" + str(time.time_ns()) if cache_bust else "")
    started = time.monotonic()
    try:
        with requests.get(
            url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=(10, 15)
        ) as response:
            sample = next(response.iter_content(1024), b"") if response.ok else b""
            record = {
                "cache_bust": cache_bust,
                "status": response.status_code,
                "host": urlsplit(response.url).hostname,
                "content_range": response.headers.get("Content-Range"),
                "bytes_read": len(sample),
                "seconds": time.monotonic() - started,
            }
    except Exception as error:
        record = {
            "cache_bust": cache_bust,
            "error_type": type(error).__name__,
            "seconds": time.monotonic() - started,
        }
    results.append(record)
    print(json.dumps(record), flush=True)
(ROOT / "reports/models-v1/connectivity.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
