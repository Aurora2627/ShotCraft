"""Direct, replaceable adapter for the pinned VideoPainter pipeline.

This module intentionally does not import the upstream demonstration script:
that script assumes a benchmark layout and constructs an external API client.
Actual generation remains unverified until recorded in an experiment report.
"""

from pathlib import Path
import sys
import json
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/VideoPainter"


def validate_model_geometry(shape, base_model, branch):
    """Reject unsupported learned-position geometry before loading model weights."""
    base_model, branch = Path(base_model), Path(branch)
    vae = json.loads((base_model / "vae/config.json").read_text())
    scale = 2 ** (len(vae["block_out_channels"]) - 1)
    for path in [base_model / "transformer/config.json", branch / "config.json"]:
        config = json.loads(path.read_text())
        if config.get("use_learned_positional_embeddings", False):
            expected = (config["sample_height"] * scale, config["sample_width"] * scale)
            if tuple(shape[1:3]) != expected:
                raise ValueError(
                    f"Model requires height/width {expected}, got {tuple(shape[1:3])}; prepare a new native-resolution input"
                )


def validate_clip(frames, masks):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Frames must be uint8 T,H,W,3 RGB")
    if masks.dtype != np.uint8 or masks.shape != frames.shape[:3]:
        raise ValueError("Masks must be uint8 T,H,W")
    if len(frames) < 5 or (len(frames) - 1) % 4:
        raise ValueError("The backend requires 4k+1 frames, at least 5")
    if not np.isin(masks, [0, 255]).all():
        raise ValueError("Masks must contain only 0 and 255; white means edit")
    if min(frames.shape[1:3]) < 16 or any(size % 16 for size in frames.shape[1:3]):
        raise ValueError("Height and width must be positive multiples of 16")


class VideoPainterBackend:
    def __init__(self, base_model, branch, memory_mode="sequential"):
        self.base_model = Path(base_model)
        self.branch = Path(branch)
        if memory_mode not in ("sequential", "resident"):
            raise ValueError("Choose sequential CPU offload or resident loading")
        self.memory_mode = memory_mode
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("The generation backend requires an NVIDIA GPU")
        for path in [self.base_model / "model_index.json", self.branch / "config.json"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        sys.path.insert(0, str(UPSTREAM / "diffusers/src"))
        import diffusers

        if not Path(diffusers.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
            raise RuntimeError("An incompatible external diffusers package is loaded")
        from diffusers import (
            CogvideoXBranchModel,
            CogVideoXI2VDualInpaintAnyLPipeline,
            CogVideoXDPMScheduler,
        )

        dtype = torch.bfloat16
        context = CogvideoXBranchModel.from_pretrained(
            str(self.branch), torch_dtype=dtype, local_files_only=True
        )
        pipeline = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            str(self.base_model),
            branch=context,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline.scheduler = CogVideoXDPMScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        for module in [
            pipeline.vae,
            pipeline.text_encoder,
            pipeline.transformer,
            pipeline.branch,
        ]:
            module.requires_grad_(False)
            module.eval()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        if self.memory_mode == "sequential":
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.to("cuda")
        self.pipe = pipeline

    @torch.inference_mode()
    def generate(
        self, frames, masks, prompt, seed=42, steps=30, clean_first_frame=None
    ):
        validate_clip(frames, masks)
        if not isinstance(prompt, str) or not prompt.strip() or not 1 <= steps <= 100:
            raise ValueError("Provide a prompt and 1-100 denoising steps")
        if clean_first_frame is None:
            # An explicit, deterministic classical first-frame baseline.
            # This is not a ground-truth edited frame or a learned image editor.
            import cv2

            anchor = cv2.inpaint(frames[0], masks[0], 3, cv2.INPAINT_TELEA)
        else:
            anchor = np.asarray(clean_first_frame)
            if anchor.shape != frames[0].shape or anchor.dtype != np.uint8:
                raise ValueError("Reference frame must match the source RGB frame")
        context = np.where(masks[:, :, :, None] > 0, 0, frames).copy()
        edit_masks = masks.copy()
        context[0] = anchor
        edit_masks[0] = 0
        images = [Image.fromarray(frame) for frame in context]
        masks_pil = [Image.fromarray(mask).convert("RGB") for mask in edit_masks]
        self.load()
        result = self.pipe(
            prompt=prompt,
            image=Image.fromarray(anchor),
            video=images,
            masks=masks_pil,
            num_frames=len(frames),
            height=frames.shape[1],
            width=frames.shape[2],
            num_inference_steps=steps,
            num_videos_per_prompt=1,
            use_dynamic_cfg=True,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
            strength=1.0,
            replace_gt=False,
            mask_add=True,
            stride=len(frames),
            prev_clip_weight=0.0,
            id_pool_resample_learnable=False,
            output_type="np",
        ).frames[0]
        result = np.asarray(result)
        if result.shape != frames.shape or not np.isfinite(result).all():
            raise RuntimeError("Backend returned an invalid or unexpected video tensor")
        return result
