"""Isolate nine-frame codec and pixel/mask preprocessing from denoising."""

import hashlib
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/VideoPainter/diffusers/src"))
from diffusers import AutoencoderKLCogVideoX
from diffusers.video_processor import VideoProcessor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("D:/career-projects/runs/ShotCraft/20261029-video-codec-audit-v1"),
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(exist_ok=False)
    input_path = Path(
        "D:/career-projects/data/ShotCraft/smoke-car-roundabout-native9-v1/clip.npz"
    )
    with np.load(input_path, allow_pickle=False) as data:
        source = data["frames"].copy()
        masks = data["masks"].copy()
        valid = torch.from_numpy(data["valid_region"] > 0)
    frames = [Image.fromarray(x) for x in source]
    mask_images = [Image.fromarray(x).convert("RGB") for x in masks]
    pixels = VideoProcessor(vae_scale_factor=8).preprocess_video(
        frames, height=480, width=720
    )
    conditions = VideoProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    ).preprocess_video(mask_images, height=480, width=720)
    expected = torch.from_numpy(source).permute(3, 0, 1, 2)[None].float() / 127.5 - 1
    expected_mask = torch.from_numpy(masks > 0)[None, None]
    report = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "pixel_shape": list(pixels.shape),
        "mask_shape": list(conditions.shape),
        "max_rgb_normalization_error": float((pixels - expected).abs().max()),
        "mask_mismatch_count": int(((conditions > 0.5) != expected_mask).sum()),
        "scope": "Single nine-frame reconstruction and preprocessing audit. Does not test text/branch conditioning or denoising.",
    }
    assert torch.allclose(pixels, expected) and report["mask_mismatch_count"] == 0
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        print("LOAD_VAE", flush=True)
        vae = (
            AutoencoderKLCogVideoX.from_pretrained(
                "D:/career-projects/models/ShotCraft/CogVideoX-5b-I2V/vae",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
        vae.enable_slicing()
        vae.enable_tiling()
        report["sequential_offload"] = args.offload
        if args.offload:
            from accelerate import cpu_offload

            vae.to("cpu")
            torch.cuda.empty_cache()
            cpu_offload(vae, execution_device=torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        print("ENCODE_9_FRAMES", flush=True)
        latent = vae.encode(pixels.to("cuda", dtype=torch.bfloat16)).latent_dist.mode()
        report["latent_shape"] = list(latent.shape)
        print("DECODE", report["latent_shape"], flush=True)
        decoded = vae.decode(latent).sample
        candidate = ((decoded[0].float().permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1).cpu()
        target = torch.from_numpy(source).float() / 255
        assert candidate.shape == target.shape and torch.isfinite(candidate).all()
        mse = (candidate - target).square().mean(-1)
        report.update(
            status="succeeded",
            seconds=time.monotonic() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            per_frame_psnr=[
                float(-10 * torch.log10(frame[valid].mean())) for frame in mse
            ],
            per_frame_mae=[
                float(frame[valid].mean())
                for frame in (candidate - target).abs().mean(-1)
            ],
        )
        np.savez_compressed(output / "reconstruction.npz", frames=candidate.numpy())
        sheet = Image.new("RGB", (840, 420), "white")
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(["source", "reconstruction"]):
            values = (
                source
                if row == 0
                else (candidate.numpy() * 255).round().astype(np.uint8)
            )
            for column, index in enumerate([0, 4, 8]):
                sheet.paste(
                    Image.fromarray(values[index]).resize((280, 186)),
                    (column * 280, row * 210 + 24),
                )
                draw.text(
                    (column * 280 + 4, row * 210 + 4),
                    name + " frame " + str(index),
                    fill="black",
                )
        sheet.save(output / "comparison.png")
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
