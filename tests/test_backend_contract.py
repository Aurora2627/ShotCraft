"""Adapter contracts only. Test doubles here are not generation experiments."""

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotcraft.videopainter import VideoPainterBackend, validate_clip


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.full((5, 16, 16, 3), 100, dtype=np.uint8)
        self.masks = np.zeros((5, 16, 16), dtype=np.uint8)
        self.masks[:, 4:8, 4:8] = 255

    def test_valid_clip(self):
        validate_clip(self.frames, self.masks)

    def test_nonbinary_masks_rejected(self):
        self.masks[:, 0, 0] = 1
        with self.assertRaises(ValueError):
            validate_clip(self.frames, self.masks)

    def test_mask_size_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate_clip(self.frames, self.masks[:4])

    def test_invalid_temporal_length_rejected(self):
        with self.assertRaises(ValueError):
            validate_clip(self.frames[:4], self.masks[:4])

    def test_float_source_rejected(self):
        with self.assertRaises(ValueError):
            validate_clip(self.frames.astype(np.float32), self.masks)

    def test_adapter_keeps_input_and_returns_raw_frames(self):
        from types import SimpleNamespace

        frames = self.frames.copy()
        masks = self.masks.copy()
        calls = []

        class PipelineDouble:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(frames=[np.zeros_like(frames, dtype=np.float32)])

        backend = VideoPainterBackend("not-loaded", "not-loaded")
        backend.pipe = PipelineDouble()
        result = backend.generate(
            frames, masks, "an empty table", clean_first_frame=frames[0]
        )
        self.assertEqual(result.shape, frames.shape)
        self.assertEqual(calls[0]["num_frames"], 5)
        self.assertTrue(np.array_equal(frames, self.frames))
        self.assertTrue(np.array_equal(masks, self.masks))
        self.assertEqual(np.asarray(calls[0]["masks"][0]).sum(), 0)

    def test_invalid_reference_rejected_before_loading(self):
        backend = VideoPainterBackend("missing", "missing")
        with self.assertRaises(ValueError):
            backend.generate(
                self.frames,
                self.masks,
                "empty table",
                clean_first_frame=np.zeros((8, 8, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
