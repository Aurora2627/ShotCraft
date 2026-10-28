"""Adversarial transport contracts; no network and no generation results."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "range_download", ROOT / "scripts/download_ranges.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, data, status, content_range):
        self.data = data
        self.status_code = status
        self.headers = {"Content-Range": content_range}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.data


class RangeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = {
            "repository": "test/model",
            "directory": "model",
            "revision": "a" * 40,
        }
        self.data = b"abcdefgh"
        self.item = {
            "path": "weights.safetensors",
            "bytes": 8,
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }
        self.patches = [
            patch.object(module, "MODEL_ROOT", self.root),
            patch.object(module, "CHUNK", 4),
            patch.object(module.time, "sleep", lambda _: None),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_resumes_prefix_and_publishes_only_matching_hash(self):
        folder = self.root / "model"
        folder.mkdir()
        (folder / "weights.safetensors.part").write_bytes(b"abcd")
        ranges = []

        class Session:
            def get(inner, url, headers, **kwargs):
                ranges.append(headers["Range"])
                return Response(b"efgh", 206, "bytes 4-7/8")

        with patch.object(module.requests, "Session", Session):
            record = module.fetch(self.model, self.item)
        self.assertEqual(ranges, ["bytes=4-7"])
        self.assertEqual((folder / "weights.safetensors").read_bytes(), self.data)
        self.assertEqual(record["sha256"], self.item["sha256"])

    def test_server_ignoring_range_cannot_publish_a_file(self):
        class Session:
            def get(inner, *args, **kwargs):
                return Response(b"abcdefgh", 200, "")

        with patch.object(module.requests, "Session", Session):
            with self.assertRaises(RuntimeError):
                module.fetch(self.model, self.item)
        self.assertFalse((self.root / "model/weights.safetensors").exists())

    def test_wrong_hash_is_not_published(self):
        folder = self.root / "model"
        folder.mkdir()
        (folder / "weights.safetensors.part").write_bytes(b"badbytes")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            module.fetch(self.model, self.item)
        self.assertFalse((folder / "weights.safetensors").exists())
        self.assertTrue((folder / "weights.safetensors.part").exists())


if __name__ == "__main__":
    unittest.main()
