"""Recover verified installer downloads and resume setup in an independent process."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wheelhouse = Path("D:/career-projects/cache/ShotCraft-wheels")
wheelhouse.mkdir(parents=True, exist_ok=True)
for path in Path(tempfile.gettempdir()).glob("pip-unpack*/*.whl"):
    if path.name.startswith(("torch-2.4.0+cu124-", "torchvision-0.19.0+cu124-")):
        destination = wheelhouse / path.name
        if (
            not destination.exists()
            or destination.stat().st_size != path.stat().st_size
        ):
            shutil.copy2(path, destination)
        print("RECOVERED", path.name, destination.stat().st_size, flush=True)
result = subprocess.run(
    [
        "D:/career-projects/runtimes/python311/python.exe",
        "-u",
        str(ROOT / "scripts/setup_windows.py"),
    ],
    cwd=ROOT,
)
raise SystemExit(result.returncode)
