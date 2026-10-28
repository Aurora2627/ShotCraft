"""Bounded HTTP Range download from pinned official Hugging Face revisions.

Retains partial bytes, validates every range and checks LFS SHA-256 before
publishing a weight file. No third-party mirror or certificate bypass is used.
"""

import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("D:/career-projects/models/ShotCraft")
RUN = Path("D:/career-projects/runs/ShotCraft/setup-20261028")
CHUNK = 4 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def announce(*values):
    with PRINT_LOCK:
        print(*values, flush=True)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch(model, item):
    folder = MODEL_ROOT / model["directory"]
    target = folder / item["path"]
    if not target.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Unexpected file path in model manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item["bytes"]
    if target.is_file() and target.stat().st_size == expected:
        actual = digest(target)
        if not item["sha256"] or actual == item["sha256"]:
            announce("VERIFIED_EXISTING", model["repository"], item["path"])
            return dict(
                repository=model["repository"], path=item["path"], sha256=actual
            )
    if target.exists():
        target.rename(
            target.with_name(target.name + ".unverified-" + str(time.time_ns()))
        )
    part = target.with_name(target.name + ".part")
    if not part.exists() and item["sha256"]:
        legacy = folder / ".cache/huggingface/download" / Path(item["path"]).parent
        candidates = list(
            legacy.glob(Path(item["path"]).name + "." + item["sha256"] + ".incomplete")
        )
        if len(candidates) == 1 and candidates[0].stat().st_size <= expected:
            candidates[0].rename(part)
            announce("RESUMED_EXISTING_PREFIX", item["path"], part.stat().st_size)
    if part.exists() and part.stat().st_size > expected:
        raise RuntimeError("Partial file exceeds manifest size")
    session = requests.Session()
    base = (
        "https://huggingface.co/"
        + model["repository"]
        + "/resolve/"
        + model["revision"]
        + "/"
        + quote(item["path"], safe="/")
    )
    position = part.stat().st_size if part.exists() else 0
    last_report = time.monotonic()
    while position < expected:
        end = min(expected - 1, position + CHUNK - 1)
        for attempt in range(12):
            try:
                url = (
                    base
                    + "?download=true&range_start="
                    + str(position)
                    + "&range_end="
                    + str(end)
                    + "&attempt="
                    + str(attempt)
                )
                with session.get(
                    url,
                    headers={"Range": f"bytes={position}-{end}"},
                    stream=True,
                    timeout=(20, 60),
                ) as response:
                    response.raise_for_status()
                    wanted = f"bytes {position}-{end}/{expected}"
                    if response.status_code == 206:
                        if response.headers.get("Content-Range") != wanted:
                            raise RuntimeError(
                                "The server returned a mismatched byte range"
                            )
                    elif not (
                        response.status_code == 200
                        and position == 0
                        and expected <= CHUNK
                    ):
                        raise RuntimeError("The server did not honor the bounded range")
                    data = bytearray()
                    started = time.monotonic()
                    for block in response.iter_content(64 * 1024):
                        data.extend(block)
                        if len(data) > end - position + 1:
                            raise RuntimeError("Received more bytes than requested")
                        if time.monotonic() - started > 90:
                            raise TimeoutError(
                                "Range transfer exceeded the 90-second budget"
                            )
                    if len(data) != end - position + 1:
                        raise RuntimeError("Incomplete HTTP range")
                with part.open("ab") as output:
                    output.write(data)
                position += len(data)
                if time.monotonic() - last_report > 20 or position == expected:
                    announce(
                        "PROGRESS",
                        model["repository"],
                        item["path"],
                        position,
                        expected,
                    )
                    last_report = time.monotonic()
                break
            except Exception as error:
                announce(
                    "RANGE_RETRY",
                    item["path"],
                    position,
                    "attempt",
                    attempt + 1,
                    type(error).__name__,
                )
                if attempt == 11:
                    raise
                time.sleep(min(2**attempt, 30))
    actual = digest(part)
    if item["sha256"] and actual != item["sha256"]:
        raise RuntimeError("LFS SHA-256 mismatch: " + item["path"])
    os.replace(part, target)
    announce("VERIFIED", model["repository"], item["path"])
    return dict(repository=model["repository"], path=item["path"], sha256=actual)


def main():
    manifest = json.loads(
        (ROOT / "reports/models-v1/manifest.json").read_text(encoding="utf-8")
    )
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    total = sum(
        item["bytes"] for model in manifest["models"] for item in model["files"]
    )
    existing = sum(p.stat().st_size for p in MODEL_ROOT.rglob("*") if p.is_file())
    if shutil.disk_usage(MODEL_ROOT).free < max(0, total - existing) + 5 * 2**30:
        raise RuntimeError("Insufficient disk space for remaining weights plus reserve")
    tasks = [(model, item) for model in manifest["models"] for item in model["files"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda pair: fetch(*pair), tasks))
    (ROOT / "reports/models-v1/verified.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    announce("MODELS_VERIFIED", len(records))


if __name__ == "__main__":
    main()
