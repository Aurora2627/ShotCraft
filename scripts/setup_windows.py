"""Create an isolated CUDA environment and pin the official generation backend.

Run with the existing Windows Python runtime. All installation output and runtime
state are outside Git. This does not download model weights or execute generation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = Path("D:/career-projects/envs/shotcraft")
RUN = Path("D:/career-projects/runs/ShotCraft/setup-20261028")
UPSTREAM = "https://github.com/TencentARC/VideoPainter.git"
COMMIT = "bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a"


def run(arguments, **kwargs):
    print("RUN", arguments, flush=True)
    subprocess.run(arguments, check=True, cwd=ROOT, **kwargs)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if not (ENV / "Scripts/python.exe").is_file():
        run(
            [
                "D:/career-projects/runtimes/python311/python.exe",
                "-m",
                "venv",
                str(ENV),
                "--prompt",
                "shotcraft-cuda",
            ]
        )
    python = str(ENV / "Scripts/python.exe")
    environment = dict(os.environ, PIP_NO_CACHE_DIR="1", PYTHONIOENCODING="utf-8")
    wheelhouse = Path("D:/career-projects/cache/ShotCraft-wheels")
    cached_torch = sorted(wheelhouse.glob("torch-2.4.0+cu124-*.whl"))
    cached_vision = sorted(wheelhouse.glob("torchvision-0.19.0+cu124-*.whl"))
    torch_requirement = str(cached_torch[0]) if cached_torch else "torch==2.4.0"
    vision_requirement = (
        str(cached_vision[0]) if cached_vision else "torchvision==0.19.0"
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--find-links",
            str(wheelhouse),
            "--index-url",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu124",
            torch_requirement,
            vision_requirement,
            "numpy==1.26.0",
        ],
        env=environment,
    )
    if not (ROOT / ".git").exists():
        run(["git", "init", "-b", "main"])
    backend = ROOT / "third_party/VideoPainter"
    if not backend.exists():
        run(
            [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                "submodule",
                "add",
                "--depth",
                "1",
                UPSTREAM,
                "third_party/VideoPainter",
            ],
            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"),
        )
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=backend,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != COMMIT:
        run(["git", "-C", str(backend), "fetch", "origin", COMMIT, "--depth", "1"])
        run(["git", "-C", str(backend), "checkout", "--detach", COMMIT])
    lock = {
        "repository": UPSTREAM,
        "commit": COMMIT,
        "role": "Third-party generation backend",
        "license": "Upstream research/education restrictions apply",
        "weights_downloaded": False,
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs/upstream-lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [python, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (RUN / "environment.txt").write_text(result.stdout, encoding="utf-8")
    run([python, "-m", "pip", "check"])
    print(
        "SETUP_CORE_COMPLETE. Full generation dependencies and weights are a separate validated step.",
        flush=True,
    )


if __name__ == "__main__":
    main()
