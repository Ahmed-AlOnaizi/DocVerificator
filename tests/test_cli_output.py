from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docverificator.cli import save_result_json


class CliOutputTests(unittest.TestCase):
    def test_save_result_json_writes_file(self) -> None:
        payload = {"hello": "world"}
        with tempfile.TemporaryDirectory() as tmp:
            out_path = save_result_json(payload, input_file="sample_data/Civil-ID.pdf", output_dir=tmp)
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["hello"], "world")
            self.assertTrue(out_path.name.startswith("Civil-ID_"))
            self.assertEqual(out_path.suffix, ".json")


if __name__ == "__main__":
    unittest.main()

