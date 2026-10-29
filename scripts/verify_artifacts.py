"""Check real video artifacts and archive small, auditable experiment records."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    b = args.experiment
    report = args.report
    result = json.loads((b / "result.json").read_text())
    config = json.loads((b / "config.json").read_text())
    assert result["status"] == "succeeded"
    with np.load(b / "raw-candidate.npz", allow_pickle=False) as data:
        frames = data["frames"]
    assert list(frames.shape) == config["raw_shape"] and np.isfinite(frames).all()
    reader = imageio.get_reader(b / "preview.mp4")
    try:
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    assert count == len(frames)
    report.mkdir(exist_ok=True)
    for name in ["result.json", "config.json", "memory.jsonl"]:
        shutil.copy2(b / name, report / ("native9-" + name))
    artifacts = {
        p.name: {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in [b / "raw-candidate.npz", b / "preview.mp4"]
    }
    artifacts.update(
        shape=list(frames.shape),
        dtype=str(frames.dtype),
        finite=True,
        value_range=[float(frames.min()), float(frames.max())],
        video_frames=count,
    )
    (report / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifacts), flush=True)


if __name__ == "__main__":
    main()
