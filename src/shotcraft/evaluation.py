"""Region-constrained compositing and reference-only diagnostics (not quality scores)."""

import torch
import torch.nn.functional as F


def validate_inputs(source, candidate, masks, valid):
    if source.ndim != 4 or source.shape[-1] != 3 or source.shape != candidate.shape:
        raise ValueError("Expected matching T,H,W,3 videos")
    if masks.shape != source.shape[:3] or valid.shape != source.shape[1:3]:
        raise ValueError("Mask or valid-region geometry mismatch")
    if masks.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError("Masks and valid regions must be boolean")
    for tensor in [source, candidate]:
        if (
            not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
            or (tensor < 0).any()
            or (tensor > 1).any()
        ):
            raise ValueError("Videos must be finite floats in [0,1]")
    if not valid.any():
        raise ValueError("Empty valid region")


@torch.inference_mode()
def composite(source, candidate, masks, valid, blend_width=1):
    """Blend strictly inside the edit mask; background is preserved by construction."""
    validate_inputs(source, candidate, masks, valid)
    if not isinstance(blend_width, int) or not 1 <= blend_width <= 32:
        raise ValueError("Blend width must be an integer in [1,32]")
    support = masks & valid[None]
    current = support[:, None].to(source.dtype)
    alpha = current.clone()
    for _ in range(blend_width - 1):
        current = -F.max_pool2d(F.pad(-current, (1, 1, 1, 1), value=0), 3, stride=1)
        alpha = alpha + current
    alpha = (alpha / blend_width).permute(0, 2, 3, 1)
    mixed = source * (1 - alpha) + candidate * alpha
    return torch.where(support[..., None], mixed, source)


def masked_mean(error, region):
    count = int(region.sum())
    return float(error[region].mean()) if count else None


@torch.inference_mode()
def diagnostics(source, candidate, masks, valid):
    """Measure input deviation. Low values do not prove removal or realism."""
    validate_inputs(source, candidate, masks, valid)
    edit = masks & valid[None]
    background = (~masks) & valid[None]
    error = (candidate - source).abs().mean(-1)
    delta = (
        ((candidate[1:] - candidate[:-1]) - (source[1:] - source[:-1])).abs().mean(-1)
    )
    stable_background = background[1:] & background[:-1]
    seam_errors = []
    for dimension in [1, 2]:
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[dimension] = slice(None, -1)
        right[dimension] = slice(1, None)
        left, right = tuple(left), tuple(right)
        seam = (
            (masks[left] != masks[right])
            & valid[None].expand_as(masks)[left]
            & valid[None].expand_as(masks)[right]
        )
        gradient = (
            ((candidate[left] - candidate[right]) - (source[left] - source[right]))
            .abs()
            .mean(-1)
        )
        seam_errors.append(gradient[seam])
    seams = torch.cat(seam_errors)
    return {
        "background_pixels": int(background.sum()),
        "edited_pixels": int(edit.sum()),
        "background_mae": masked_mean(error, background),
        "edited_area_change_mae": masked_mean(error, edit),
        "background_delta_error": masked_mean(delta, stable_background),
        "boundary_gradient_error": float(seams.mean()) if seams.numel() else None,
    }
