"""Validated edit intent, independent of the generation backend."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EditRequest:
    video: Path
    mask: Path
    prompt: str
    start_frame: int
    end_frame: int
    operation: str = "remove"
    reference_image: Path | None = None
    seed: int = 42

    def __post_init__(self):
        if self.operation not in ("remove", "replace"):
            raise ValueError("operation must be remove or replace")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Describe the desired edited scene")
        if type(self.start_frame) is not int or type(self.end_frame) is not int:
            raise ValueError("Use integer frame boundaries")
        if not 0 <= self.start_frame < self.end_frame:
            raise ValueError("Use a nonempty half-open frame interval")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        if self.operation == "replace" and self.reference_image is None:
            raise ValueError("Replacement requires a reference image in this protocol")

    def validate_assets(self):
        """Called at execution, not while constructing a portable request."""
        for path in (self.video, self.mask, self.reference_image):
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(path)
