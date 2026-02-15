from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_RUN = os.getenv("RUN_OCR_INTEGRATION", "0") == "1"
_SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "Civil-ID.pdf"


@unittest.skipUnless(_RUN and _SAMPLE.exists(), "Set RUN_OCR_INTEGRATION=1 and provide sample_data/Civil-ID.pdf")
class SampleCivilIDIntegrationTests(unittest.TestCase):
    def test_extracts_expected_civil_id_and_dob(self) -> None:
        from docverificator.config import Settings
        from docverificator.pipeline import process_document

        settings = Settings(
            ocr_engine="paddle",
            pdf_dpi=300,
            retry_missing_key_fields=True,
            try_rotations_on_missing_fields=True,
            max_deskew_angle=12.0,
        )
        result = process_document(
            file_path=str(_SAMPLE),
            expected_dob="2003-09-16",
            settings=settings,
        )

        self.assertEqual(result.extracted.civil_id, "303091600084")
        self.assertEqual(result.extracted.birth_date, "16/09/2003")


if __name__ == "__main__":
    unittest.main()
