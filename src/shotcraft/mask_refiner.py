"""Trainable temporal mask residual module; untrained initialization is a baseline.

Input B,T,8,H,W: RGB(3), propagated mask(1), normalized flow(2),
forward/backward propagation error(1), protected region(1).
Flow normalization and error construction belong to the data pipeline.
"""

import torch
from torch import nn


class TemporalMaskRefiner(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(8, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.residual = nn.Conv3d(hidden, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        if features.ndim != 5 or features.shape[2] != 8:
            raise ValueError("Expected B,T,8,H,W features")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain nonfinite values")
        if features.shape[1] < 1 or min(features.shape[-2:]) < 2:
            raise ValueError("Empty time axis or invalid image dimensions")
        mask = features[:, :, 3:4]
        protected = features[:, :, 7:8]
        if torch.any((mask < 0) | (mask > 1) | (protected < 0) | (protected > 1)):
            raise ValueError("Mask and protection values must lie in [0,1]")
        residual = self.residual(self.encoder(features.permute(0, 2, 1, 3, 4)))
        logits = (mask * 2 - 1) * 2 + residual.permute(0, 2, 1, 3, 4)
        probabilities = torch.sigmoid(logits) * (1 - protected)
        return {"logits": logits, "probabilities": probabilities}


def segmentation_loss(logits, target):
    """Supervised spatial objective. No unaligned frame-difference temporal loss."""
    if logits.shape != target.shape or target.ndim != 5:
        raise ValueError("Target must match B,T,1,H,W logits")
    if not torch.isfinite(target).all() or torch.any((target < 0) | (target > 1)):
        raise ValueError("Target must contain finite mask values in [0,1]")
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    axes = (2, 3, 4)
    dice = (2 * (probabilities * target).sum(axes) + 1) / (
        probabilities.sum(axes) + target.sum(axes) + 1
    )
    return bce + (1 - dice).mean()
