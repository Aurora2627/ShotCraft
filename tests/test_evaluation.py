import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shotcraft.evaluation import composite, diagnostics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.zeros(3, 8, 8, 3)
        self.candidate = torch.ones_like(self.source)
        self.mask = torch.zeros(3, 8, 8, dtype=torch.bool)
        self.mask[:, 2:6, 2:6] = True
        self.valid = torch.ones(8, 8, dtype=torch.bool)

    def test_composites_preserve_background_and_inputs(self):
        for width in [1, 3]:
            result = composite(
                self.source, self.candidate, self.mask, self.valid, width
            )
            self.assertTrue(torch.equal(result[~self.mask], self.source[~self.mask]))
            self.assertGreater(float(result[self.mask].mean()), 0)
        self.assertEqual(float(self.source.sum()), 0)
        self.assertEqual(float(self.candidate.mean()), 1)

    def test_letterbox_excluded(self):
        self.valid[:2] = False
        changed = self.source.clone()
        changed[:, :2] = 1
        report = diagnostics(self.source, changed, self.mask, self.valid)
        self.assertEqual(report["background_mae"], 0)

    def test_empty_regions_are_not_perfect_scores(self):
        self.mask[:] = True
        report = diagnostics(self.source, self.candidate, self.mask, self.valid)
        self.assertIsNone(report["background_mae"])
        self.assertIsNone(report["background_delta_error"])
        self.assertIsNone(report["boundary_gradient_error"])

    def test_input_copy_does_not_demonstrate_edit_success(self):
        report = diagnostics(self.source, self.source, self.mask, self.valid)
        self.assertEqual(report["edited_area_change_mae"], 0)
        self.assertEqual(report["background_mae"], 0)

    def test_nonfinite_and_wrong_masks_rejected(self):
        for candidate, mask in [
            (self.candidate * float("nan"), self.mask),
            (self.candidate, self.mask.float()),
        ]:
            with self.assertRaises(ValueError):
                diagnostics(self.source, candidate, mask, self.valid)
