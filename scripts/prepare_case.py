"""Pack existing, attributed DAVIS frames into an immutable backend smoke input."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--object-id", type=int, default=1)
    args = parser.parse_args()
    if (
        args.count < 5
        or (args.count - 1) % 4
        or min(args.width, args.height) < 16
        or args.width % 16
        or args.height % 16
    ):
        raise ValueError("Use 4k+1 frames and image dimensions divisible by 16")
    if args.output.exists():
        raise FileExistsError("Use a new input directory")
    images = sorted((args.raw_dir / "JPEGImages").rglob("*.jpg"))
    masks = {path.stem: path for path in (args.raw_dir / "Annotations").rglob("*.png")}
    if len(images) < args.count:
        raise ValueError("Not enough source frames")
    frames = []
    labels = []
    records = []
    valid = None
    for path in images[: args.count]:
        mask_path = masks[path.stem]
        with Image.open(path) as image, Image.open(mask_path) as label:
            image = ImageOps.exif_transpose(image).convert("RGB")
            mask = Image.fromarray(
                (np.asarray(label) == args.object_id).astype(np.uint8) * 255
            )
            if mask.size != image.size:
                raise ValueError("Image and mask shapes differ")
            scale = min(args.width / image.width, args.height / image.height)
            size = (round(image.width * scale), round(image.height * scale))
            left = (args.width - size[0]) // 2
            top = (args.height - size[1]) // 2
            canvas = Image.new("RGB", (args.width, args.height))
            canvas.paste(image.resize(size, Image.Resampling.BICUBIC), (left, top))
            mask_canvas = Image.new("L", canvas.size)
            mask_canvas.paste(mask.resize(size, Image.Resampling.NEAREST), (left, top))
            frames.append(np.asarray(canvas))
            labels.append(np.asarray(mask_canvas))
            roi = np.zeros((args.height, args.width), dtype=np.uint8)
            roi[top : top + size[1], left : left + size[0]] = 1
            if valid is not None and not np.array_equal(valid, roi):
                raise ValueError("Source geometry changes within clip")
            valid = roi
        records.append(
            {
                "image": str(path),
                "mask": str(mask_path),
                "image_sha256": digest(path),
                "mask_sha256": digest(mask_path),
            }
        )
    if not np.any(labels):
        raise ValueError("Chosen object ID does not occur in this clip")
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "clip.npz",
        frames=np.stack(frames),
        masks=np.stack(labels),
        valid_region=valid,
    )
    provenance = args.raw_dir.parent / "provenance.json"
    source = (
        json.loads(provenance.read_text(encoding="utf-8"))
        if provenance.exists()
        else {}
    )
    report = {
        "source_dataset": source.get("dataset", "DAVIS"),
        "source_page": source.get("official_page"),
        "sequence": source.get("sequence", args.raw_dir.parent.name),
        "object_id": args.object_id,
        "count": args.count,
        "shape": [args.count, args.height, args.width, 3],
        "records": records,
        "clip_sha256": digest(args.output / "clip.npz"),
        "scope": "Frame-only development smoke. Nominal 8 fps playback; original timing/audio not established. Letterbox excluded by valid_region.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("PREPARED", args.output, report["shape"], flush=True)


if __name__ == "__main__":
    main()
