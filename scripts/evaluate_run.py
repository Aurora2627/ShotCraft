"""Evaluate a real candidate against input-preservation and classical controls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotcraft.evaluation import composite, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new evaluation directory")
    config = json.loads((args.experiment / "config.json").read_text())
    result = json.loads((args.experiment / "result.json").read_text())
    if result["status"] != "succeeded":
        raise ValueError("Cannot evaluate failed inference")
    input_path = Path(config["input"])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != config["input_sha256"]:
        raise ValueError("Input changed since inference")
    with np.load(input_path, allow_pickle=False) as data:
        source_bytes = data["frames"].copy()
        masks = data["masks"].copy()
        valid = data["valid_region"].copy()
    with np.load(args.experiment / "raw-candidate.npz", allow_pickle=False) as data:
        raw = torch.from_numpy(data["frames"].copy())
    source = torch.from_numpy(source_bytes).float() / 255
    masks_t = torch.from_numpy(masks > 0)
    valid_t = torch.from_numpy(valid > 0)
    traditional = np.stack(
        [
            cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            for frame, mask in zip(source_bytes, masks)
        ]
    )
    candidates = {
        "input_copy_control": source,
        "telea_per_frame": torch.from_numpy(traditional).float() / 255,
        "raw": raw,
        "hard_composite": composite(source, raw, masks_t, valid_t),
        "inner_feather_3": composite(source, raw, masks_t, valid_t, 3),
    }
    report = {
        "scope": "Single low-step development clip. Reference-deviation diagnostics only; no ground-truth edited video or semantic success claim. Composited background preservation is enforced by construction. Delta error is not flow compensated and is not a temporal-quality score.",
        "input_sha256": config["input_sha256"],
        "raw_sha256": hashlib.sha256(
            (args.experiment / "raw-candidate.npz").read_bytes()
        ).hexdigest(),
        "shape": list(source.shape),
        "metrics": {
            name: diagnostics(source, value, masks_t, valid_t)
            for name, value in candidates.items()
        },
    }
    args.output.mkdir(parents=True)
    for name in ["hard_composite", "inner_feather_3"]:
        value = candidates[name].numpy()
        np.savez_compressed(args.output / (name + ".npz"), frames=value)
        imageio.mimwrite(
            args.output / (name + ".mp4"),
            (value.clip(0, 1) * 255).round().astype(np.uint8),
            fps=8,
        )
    indices = sorted(set([0, len(source) // 2, len(source) - 1]))
    rows = [
        "source_mask",
        "raw",
        "hard_composite",
        "inner_feather_3",
        "telea_per_frame",
    ]
    sheet = Image.new("RGB", (len(indices) * 280, len(rows) * 210), "white")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(rows):
        for col, index in enumerate(indices):
            if name == "source_mask":
                frame = source_bytes[index].copy()
                selected = masks[index] > 0
                frame[selected] = (
                    frame[selected].astype(float) * 0.5 + np.array([255, 0, 0]) * 0.5
                ).astype(np.uint8)
            else:
                frame = (
                    (candidates[name][index].numpy().clip(0, 1) * 255)
                    .round()
                    .astype(np.uint8)
                )
            sheet.paste(
                Image.fromarray(frame).resize((280, 186)), (col * 280, row * 210 + 24)
            )
            draw.text(
                (col * 280 + 5, row * 210 + 5),
                name + " frame " + str(index),
                fill="black",
            )
    sheet.save(args.output / "comparison.png")
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
