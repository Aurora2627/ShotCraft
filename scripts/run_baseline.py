"""Generate and preserve a raw baseline candidate from a prepared local clip."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotcraft.videopainter import (
    VideoPainterBackend,
    validate_clip,
    validate_model_geometry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ with frames uint8 THWC and masks uint8 THW",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--model-root", type=Path, default=Path("D:/career-projects/models/ShotCraft")
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="8 is a smoke profile; quality experiments require a fixed larger budget",
    )
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument(
        "--attention-backend", choices=["auto", "efficient"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if (
        not 1 <= args.guidance_scale <= 20
        or not 1 <= args.steps <= 100
        or not args.prompt.strip()
        or not 0 <= args.seed < 2**32
    ):
        raise ValueError("Invalid steps, prompt or seed")
    with np.load(args.input, allow_pickle=False) as data:
        frames = data["frames"]
        masks = data["masks"]
        anchor = data["reference"] if "reference" in data else None
    validate_clip(frames, masks)
    args.output.mkdir(parents=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        first_frame=(
            "provided_reference" if anchor is not None else "classical_Telea_inpaint"
        ),
        profile="smoke" if args.steps < 30 else "fixed_budget_candidate",
        torch=torch.__version__,
        raw_shape=list(frames.shape),
    )
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    result = {"status": "loading", "started": time.time()}

    def record_memory(stage, step=None):
        snapshot = {"stage": stage, "step": step, "time": time.time()}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snapshot.update(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                free_bytes=free,
                total_bytes=total,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        with (args.output / "memory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + chr(10))
        print("MEMORY", json.dumps(snapshot), flush=True)

    def on_step(pipeline, step, timestep, callback_kwargs):
        record_memory("denoise_step", step)
        return callback_kwargs

    try:
        backend = VideoPainterBackend(
            args.model_root / "CogVideoX-5b-I2V",
            args.model_root / "VideoPainter/VideoPainter/checkpoints/branch",
        )
        validate_model_geometry(frames.shape, backend.base_model, backend.branch)
        started = time.monotonic()
        record_memory("before_load")
        backend.load()
        record_memory("after_load")
        result["load_seconds"] = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        from torch.nn.attention import sdpa_kernel, SDPBackend

        attention_context = (
            sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
            if args.attention_backend == "efficient"
            else nullcontext()
        )
        with attention_context:
            candidate = backend.generate(
                frames,
                masks,
                args.prompt,
                seed=args.seed,
                steps=args.steps,
                clean_first_frame=anchor,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=on_step,
            )
        torch.cuda.synchronize()
        result.update(
            status="succeeded",
            inference_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        np.savez_compressed(args.output / "raw-candidate.npz", frames=candidate)
        import imageio.v2 as imageio

        imageio.mimwrite(
            args.output / "preview.mp4",
            (candidate.clip(0, 1) * 255).round().astype(np.uint8),
            fps=8,
        )
        result["scope"] = (
            "Real backend output; low-step smoke is not a quality benchmark"
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        try:
            record_memory("failed")
        except Exception as diagnostic_error:
            result["diagnostic_error"] = str(diagnostic_error)
        raise
    finally:
        result["finished"] = time.time()
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
