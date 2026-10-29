"""Isolate VAE reconstruction from text conditioning and denoising."""

import json
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/VideoPainter/diffusers/src"))
from diffusers import AutoencoderKLCogVideoX


@torch.inference_mode()
def main():
    output = Path("D:/career-projects/runs/ShotCraft/20261029-vae-diagnostic-v1")
    output.mkdir(exist_ok=False)
    with np.load(
        "D:/career-projects/data/ShotCraft/smoke-car-roundabout-native9-v1/clip.npz",
        allow_pickle=False,
    ) as data:
        source = data["frames"][:1].copy()
    video = (
        torch.from_numpy(source)
        .permute(3, 0, 1, 2)[None]
        .to(device="cuda", dtype=torch.bfloat16)
        / 127.5
        - 1
    )
    vae = (
        AutoencoderKLCogVideoX.from_pretrained(
            "D:/career-projects/models/ShotCraft/CogVideoX-5b-I2V/vae",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    report = {
        "scope": "Single-frame VAE reconstruction diagnostic; no edit or temporal quality claim.",
        "modes": {},
    }
    Image.fromarray(source[0]).save(output / "source.png")
    for mode in ["tiled", "untiled"]:
        try:
            if mode == "tiled":
                vae.enable_tiling()
            else:
                vae.disable_tiling()
            torch.cuda.reset_peak_memory_stats()
            latent = vae.encode(video).latent_dist.mode()
            decoded = vae.decode(latent).sample
            frame = ((decoded[0, :, 0].float().permute(1, 2, 0) + 1) / 2).clamp(0, 1)
            target = torch.from_numpy(source[0]).to("cuda").float() / 255
            mse = (frame - target).square().mean()
            report["modes"][mode] = {
                "status": "succeeded",
                "psnr": float(-10 * torch.log10(mse)),
                "mae": float((frame - target).abs().mean()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            Image.fromarray((frame.cpu().numpy() * 255).round().astype(np.uint8)).save(
                output / (mode + ".png")
            )
            del latent, decoded, frame, target
        except Exception:
            report["modes"][mode] = {
                "status": "failed",
                "error": traceback.format_exc(),
            }
        (output / "result.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
