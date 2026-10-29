import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shotcraft.videopainter import validate_model_geometry


class ModelGeometryTests(unittest.TestCase):
    def test_native_accepted_and_small_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ["vae", "transformer", "branch"]:
                (root / sub).mkdir()
            (root / "vae/config.json").write_text(
                json.dumps({"block_out_channels": [128, 256, 256, 512]})
            )
            config = {
                "use_learned_positional_embeddings": True,
                "sample_height": 60,
                "sample_width": 90,
            }
            for sub in ["transformer", "branch"]:
                (root / sub / "config.json").write_text(json.dumps(config))
            validate_model_geometry((17, 480, 720, 3), root, root / "branch")
            with self.assertRaisesRegex(ValueError, "480, 720"):
                validate_model_geometry((17, 288, 432, 3), root, root / "branch")
            config["sample_height"] = 40
            (root / "branch/config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                validate_model_geometry((17, 480, 720, 3), root, root / "branch")
