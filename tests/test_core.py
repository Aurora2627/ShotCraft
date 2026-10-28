"""Behavioral contracts for planning and the untrained PyTorch module."""

import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotcraft.contracts import EditRequest
from shotcraft.planning import plan_windows
from shotcraft.mask_refiner import TemporalMaskRefiner, segmentation_loss


class PlanningTests(unittest.TestCase):
    def test_every_frame_is_covered(self):
        for count in [1, 4, 49, 50, 97, 150]:
            with self.subTest(count=count):
                windows = plan_windows(count)
                coverage = set()
                for window in windows:
                    coverage.update(range(window.start, window.stop))
                    self.assertLessEqual(window.valid_frames, 49)
                    self.assertEqual(window.padded_frames, 49)
                self.assertEqual(coverage, set(range(count)))

    def test_windows_do_not_cross_shots(self):
        for window in plan_windows(160, cuts=(25, 90)):
            self.assertFalse(any(window.start < c < window.stop for c in (25, 90)))

    def test_overlap_has_expected_size(self):
        windows = plan_windows(100)
        for first, second in zip(windows, windows[1:]):
            self.assertEqual(first.stop - second.start, 9)

    def test_invalid_cut_order_and_duplicates(self):
        for cuts in [(30, 10), (10, 10), (0,), (100,), (1.5,)]:
            with self.assertRaises(ValueError):
                plan_windows(100, cuts)

    def test_invalid_dimensions(self):
        for arguments in [
            dict(frame_count=0),
            dict(frame_count=3, capacity=48),
            dict(frame_count=3, overlap=49),
        ]:
            with self.assertRaises(ValueError):
                plan_windows(**arguments)


class RequestTests(unittest.TestCase):
    def test_remove_request(self):
        request = EditRequest(
            Path("video.mp4"), Path("mask.npz"), "an empty table", 0, 49
        )
        self.assertEqual(request.operation, "remove")

    def test_replacement_requires_reference(self):
        with self.assertRaises(ValueError):
            EditRequest(Path("v"), Path("m"), "a vase", 0, 49, operation="replace")

    def test_empty_frame_interval_rejected(self):
        with self.assertRaises(ValueError):
            EditRequest(Path("v"), Path("m"), "table", 10, 10)

    def test_missing_asset_rejected_at_execution(self):
        request = EditRequest(Path("not-existing-video.mp4"), Path("m"), "table", 0, 1)
        with self.assertRaises(FileNotFoundError):
            request.validate_assets()


class MaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TemporalMaskRefiner().to(self.device)
        self.features = torch.rand(1, 5, 8, 12, 16, device=self.device)
        self.features[:, :, 7] = 0

    def test_zero_initialized_residual_preserves_thresholded_mask(self):
        output = self.model(self.features)
        self.assertTrue(
            torch.equal(output["probabilities"] >= 0.5, self.features[:, :, 3:4] >= 0.5)
        )

    def test_protected_region_never_becomes_editable(self):
        self.features[:, :, 7] = 1
        self.assertEqual(
            self.model(self.features)["probabilities"].count_nonzero().item(), 0
        )

    def test_loss_backward_reaches_trainable_residual(self):
        output = self.model(self.features)
        target = torch.zeros_like(output["logits"])
        loss = segmentation_loss(output["logits"], target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(self.model.residual.weight.grad.abs().sum().item(), 0)

    def test_invalid_feature_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.features[:, :, :7])

    def test_nonfinite_features_rejected(self):
        self.features[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(self.features)

    def test_invalid_target_rejected(self):
        logits = self.model(self.features)["logits"]
        with self.assertRaises(ValueError):
            segmentation_loss(logits, torch.full_like(logits, 2))


if __name__ == "__main__":
    unittest.main()
