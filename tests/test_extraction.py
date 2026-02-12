from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docverificator.extraction import extract_fields
from docverificator.models import OCRLine, OCRResult


class ExtractionTests(unittest.TestCase):
    def test_extract_fields_basic(self) -> None:
        ocr = OCRResult(
            lines=[
                OCRLine(text="Civil ID: 299041212345", confidence=0.99),
                OCRLine(text="Name: John Doe", confidence=0.97),
                OCRLine(text="DOB: 12/04/1999", confidence=0.95),
            ],
            full_text="",
            mean_confidence=0.97,
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields.civil_id, "299041212345")
        self.assertEqual(fields.name, "John Doe")
        self.assertEqual(fields.date_of_birth, "1999-04-12")

    def test_extract_arabic_digits(self) -> None:
        ocr = OCRResult(
            lines=[OCRLine(text="\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0645\u062f\u0646\u064a \u0663\u0660\u0660\u0660\u0661\u0660\u0661\u0662\u0661\u0662\u0663\u0664", confidence=0.9)],
            full_text="",
            mean_confidence=0.9,
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields.civil_id, "300010121234")

    def test_extract_known_id_and_dob(self) -> None:
        ocr = OCRResult(
            lines=[
                OCRLine(text="Civil ID 303091600084", confidence=0.92),
                OCRLine(text="DOB: 16/09/2003", confidence=0.9),
            ],
            full_text="",
            mean_confidence=0.91,
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields.civil_id, "303091600084")
        self.assertEqual(fields.date_of_birth, "2003-09-16")

    def test_extract_civil_id_split_across_lines(self) -> None:
        ocr = OCRResult(
            lines=[
                OCRLine(text="3030916", confidence=0.88),
                OCRLine(text="00084", confidence=0.87),
            ],
            full_text="",
            mean_confidence=0.875,
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields.civil_id, "303091600084")

    def test_extract_prefers_valid_window_from_noisy_serial(self) -> None:
        ocr = OCRResult(
            lines=[
                OCRLine(text="I DKWTA315822348303091600084", confidence=0.86),
                OCRLine(text="D309161M2209162KhT118102527", confidence=0.8),
                OCRLine(text="AlonAIzIAhmEd", confidence=0.84),
            ],
            full_text="",
            mean_confidence=0.833,
        )
        fields = extract_fields(ocr)
        self.assertEqual(fields.civil_id, "303091600084")
        self.assertEqual(fields.date_of_birth, "2003-09-16")


if __name__ == "__main__":
    unittest.main()
