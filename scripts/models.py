"""Inspect, pin and optionally download the two required official model snapshots.

No image generation model, external API key, or unrelated training data is fetched.
Use --download only after inspecting the actual byte budget.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = [
    (
        "THUDM/CogVideoX-5b-I2V",
        "CogVideoX-5b-I2V",
        ("transformer/", "text_encoder/", "tokenizer/", "scheduler/", "vae/"),
    ),
    ("TencentARC/VideoPainter", "VideoPainter", ("VideoPainter/checkpoints/branch/",)),
]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--model-root", type=Path, default=Path("D:/career-projects/models/ShotCraft")
    )
    args = parser.parse_args()
    output = ROOT / "reports/models-v1"
    output.mkdir(parents=True, exist_ok=True)
    lock_file = output / "manifest.json"
    if lock_file.exists():
        manifest = json.loads(lock_file.read_text(encoding="utf-8"))
    else:
        manifest = {
            "models": [],
            "scope": "Pinned official inference weights; no generation result",
        }
        api = HfApi()
        for repo, directory, prefixes in REPOSITORIES:
            info = api.model_info(repo, files_metadata=True)
            if getattr(info, "gated", False):
                raise RuntimeError("The model needs user access approval: " + repo)
            files = []
            for sibling in info.siblings:
                name = sibling.rfilename
                selected = (
                    name == "model_index.json"
                    or name.startswith(prefixes)
                    or name in ("LICENSE", "LICENSE.txt")
                )
                if not selected or name.endswith(
                    (".bin", ".pt", ".pth", ".msgpack", ".h5")
                ):
                    continue
                lfs = getattr(sibling, "lfs", None)
                size = getattr(sibling, "size", None) or (
                    getattr(lfs, "size", None) if lfs else None
                )
                if size is None:
                    raise RuntimeError("Missing file size metadata: " + name)
                files.append(
                    {
                        "path": name,
                        "bytes": size,
                        "sha256": getattr(lfs, "sha256", None) if lfs else None,
                    }
                )
            if not any(item["path"].endswith(".safetensors") for item in files):
                raise RuntimeError("Required weight selection is empty: " + repo)
            manifest["models"].append(
                {
                    "repository": repo,
                    "revision": info.sha,
                    "directory": directory,
                    "files": files,
                }
            )
        lock_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    required = sum(
        item["bytes"] for model in manifest["models"] for item in model["files"]
    )
    print("WEIGHT_BUDGET_GIB", round(required / 2**30, 3), flush=True)
    if not args.download:
        print("INSPECT_ONLY", lock_file, flush=True)
        return
    args.model_root.mkdir(parents=True, exist_ok=True)
    missing = sum(
        item["bytes"]
        for model in manifest["models"]
        for item in model["files"]
        if not (args.model_root / model["directory"] / item["path"]).is_file()
    )
    if shutil.disk_usage(args.model_root).free < missing + 5 * 2**30:
        raise RuntimeError(
            "Insufficient free disk for missing weights plus 5 GiB reserve"
        )
    verification = []
    for model in manifest["models"]:
        folder = args.model_root / model["directory"]
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            allow_patterns=[item["path"] for item in model["files"]],
            local_dir=folder,
            max_workers=2,
            resume_download=True,
        )
        for item in model["files"]:
            path = folder / item["path"]
            if path.stat().st_size != item["bytes"]:
                raise RuntimeError("Downloaded file size mismatch: " + str(path))
            actual = sha256(path)
            if item["sha256"] and actual != item["sha256"]:
                raise RuntimeError("Downloaded file checksum mismatch: " + str(path))
            verification.append(
                {
                    "repository": model["repository"],
                    "path": item["path"],
                    "sha256": actual,
                }
            )
    (output / "verified.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("MODELS_VERIFIED", len(verification), flush=True)


if __name__ == "__main__":
    main()
