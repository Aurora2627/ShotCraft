"""Run actual core tests and record the environment; no generation quality claims."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from shotcraft.mask_refiner import TemporalMaskRefiner

    output = ROOT / "reports/core-v1"
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "trainable_parameters": sum(
            p.numel() for p in TemporalMaskRefiner().parameters() if p.requires_grad
        ),
        "scope": "Core contracts and untrained module forward/backward, not a VideoPainter generation or training result",
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "tests.txt").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT,
            env=dict(
                os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8"
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(json.dumps(environment), flush=True)
    print((output / "tests.txt").read_text(encoding="utf-8"), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
