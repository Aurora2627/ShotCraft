"""Install the reviewed inference subset into the ShotCraft environment only."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = Path("D:/career-projects/envs/shotcraft/Scripts/python.exe")
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError("Run this script with the dedicated ShotCraft interpreter")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "-r",
            str(ROOT / "requirements-inference.txt"),
        ],
        cwd=ROOT,
        env=dict(os.environ, PIP_NO_CACHE_DIR="1"),
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    sys.path.insert(0, str(ROOT / "third_party/VideoPainter/diffusers/src"))
    from diffusers import CogvideoXBranchModel, CogVideoXI2VDualInpaintAnyLPipeline

    print(
        "INFERENCE_IMPORT_OK",
        CogvideoXBranchModel.__name__,
        CogVideoXI2VDualInpaintAnyLPipeline.__name__,
        flush=True,
    )


if __name__ == "__main__":
    main()
