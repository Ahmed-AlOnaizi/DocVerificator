from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docverificator.config import Settings
from docverificator.models import ExtractedFields
from docverificator.validation import (
    compute_civil_id_check_digit,
    extract_dob_from_civil_id,
    name_similarity,
    validate_fields,
    validate_civil_id_checksum,
)


def _build_valid_id(year: int, month: int, day: int) -> str:
    century = "2" if year < 2000 else "3"
    yy = year % 100
    prefix = f"{century}{yy:02d}{month:02d}{day:02d}"
    for serial in range(0, 10000):
        first11 = prefix + f"{serial:04d}"
        check = compute_civil_id_check_digit(first11)
        if check is not None:
            return first11 + str(check)
    raise RuntimeError("Could not generate a valid Civil ID test vector.")


class ValidationTests(unittest.TestCase):
    def test_checksum_accepts_generated_vector(self) -> None:
        cid = _build_valid_id(1996, 4, 12)
        self.assertTrue(validate_civil_id_checksum(cid))

    def test_checksum_rejects_mutated_digit(self) -> None:
        cid = _build_valid_id(2001, 11, 8)
        invalid = cid[:-1] + str((int(cid[-1]) + 1) % 10)
        self.assertFalse(validate_civil_id_checksum(invalid))

    def test_extract_dob_from_civil_id(self) -> None:
        cid = _build_valid_id(2004, 9, 17)
        dob = extract_dob_from_civil_id(cid)
        self.assertIsNotNone(dob)
        self.assertEqual(dob.isoformat(), "2004-09-17")

    def test_name_similarity(self) -> None:
        score = name_similarity("Mohammed Al Ahmad", "Mohamad Al-Ahmad")
        self.assertGreaterEqual(score, 0.7)

    def test_validation_fails_when_no_key_fields_extracted(self) -> None:
        fields = ExtractedFields(civil_id=None, date_of_birth=None, name=None)
        result = validate_fields(fields, Settings())
        self.assertFalse(result.overall_pass)
        self.assertTrue(any("No key fields" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
