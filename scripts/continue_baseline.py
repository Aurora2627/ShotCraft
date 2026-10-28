"""Complete the current download -> verify -> smoke experiment pipeline once.

This is a bounded, single-run local job, not a recurring scheduler. Logs and
process state live outside the repository. Never starts a second downloader.
"""

import json
import msvcrt
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("D:/career-projects/runs/ShotCraft/setup-20261028")
TARGET = Path("D:/career-projects/runs/ShotCraft/20261028-baseline-smoke-v1")


def main():
    lock = (RUN / "baseline-pipeline.lock").open("a+b")
    lock.seek(0)
    if lock.read(1) == b"":
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError("A baseline pipeline is already active")
    state = {
        "pid": os.getpid(),
        "phase": "waiting_for_verified_weights",
        "started": time.time(),
    }

    def save():
        (RUN / "baseline-pipeline.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        deadline = time.monotonic() + 12 * 3600
        while not (ROOT / "reports/models-v1/verified.json").is_file():
            metadata = json.loads(
                (RUN / "download-process.json").read_text(encoding="utf-8")
            )
            try:
                process = psutil.Process(metadata["pid"])
                running = any(
                    ("models.py" in part or "download_ranges.py" in part)
                    for part in process.cmdline()
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                running = False
            if not running:
                raise RuntimeError(
                    "The weight download stopped before verification; inspect "
                    + metadata.get("log", "download-ranges.log")
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Download wait budget exceeded; partial files remain resumable"
                )
            state["checked"] = time.time()
            save()
            time.sleep(15)
        if TARGET.exists():
            raise FileExistsError(
                "An experiment already exists at the target; inspect it instead of overwriting"
            )
        state["phase"] = "running_real_baseline"
        save()
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/run_baseline.py"),
            "--input",
            "D:/career-projects/data/ShotCraft/smoke-car-roundabout-v1/clip.npz",
            "--output",
            str(TARGET),
            "--prompt",
            "An empty curved road and surrounding buildings, no car on the road.",
            "--steps",
            "8",
            "--seed",
            "42",
        ]
        with (RUN / "real-baseline.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            state["child_pid"] = child.pid
            save()
            try:
                code = child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True,
                )
                raise TimeoutError("Generation exceeded its one-hour smoke budget")
        if code:
            raise RuntimeError(
                "Real baseline failed; inspect real-baseline.log and experiment result.json"
            )
        state.update(phase="completed", finished=time.time())
        save()
        print("REAL_BASELINE_COMPLETED", TARGET, flush=True)
    except Exception as error:
        state.update(phase="failed", error=str(error), finished=time.time())
        save()
        print("PIPELINE_STOPPED", str(error), flush=True)
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()
