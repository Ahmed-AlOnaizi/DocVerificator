from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher

from .config import Settings
from .models import ExtractedFields, ValidationResult

_CIVIL_RE = re.compile(r"^\d{12}$")
_CHECKSUM_WEIGHTS = (2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def _normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).strip().casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_similarity(expected_name: str, extracted_name: str) -> float:
    left = _normalize_name(expected_name)
    right = _normalize_name(extracted_name)
    if not left or not right:
        return 0.0

    try:
        from rapidfuzz import fuzz

        score = max(
            fuzz.ratio(left, right),
            fuzz.token_set_ratio(left, right),
            fuzz.partial_ratio(left, right),
        )
        return round(float(score) / 100.0, 4)
    except ImportError:
        return round(SequenceMatcher(None, left, right).ratio(), 4)


def compute_civil_id_check_digit(first_11_digits: str) -> int | None:
    if not re.fullmatch(r"\d{11}", first_11_digits):
        return None
    total = sum(int(first_11_digits[i]) * _CHECKSUM_WEIGHTS[i] for i in range(11))
    expected = 11 - (total % 11)
    # Community-sourced logic commonly used in open validators.
    if expected < 0 or expected > 9:
        return None
    return expected


def validate_civil_id_checksum(civil_id: str) -> bool:
    if not _CIVIL_RE.fullmatch(civil_id):
        return False
    expected = compute_civil_id_check_digit(civil_id[:11])
    if expected is None:
        return False
    return expected == int(civil_id[-1])


def extract_dob_from_civil_id(civil_id: str) -> date | None:
    if not _CIVIL_RE.fullmatch(civil_id):
        return None

    century_digit = civil_id[0]
    yy = int(civil_id[1:3])
    mm = int(civil_id[3:5])
    dd = int(civil_id[5:7])

    if century_digit == "2":
        year = 1900 + yy
    elif century_digit == "3":
        year = 2000 + yy
    else:
        return None

    try:
        dt = date(year, mm, dd)
    except ValueError:
        return None

    if dt > date.today():
        return None
    return dt


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def is_plausible_dob(value: date | None, min_age: int, max_age: int) -> bool | None:
    if value is None:
        return None
    today = date.today()
    if value > today:
        return False
    age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
    return min_age <= age <= max_age


def validate_fields(
    fields: ExtractedFields,
    settings: Settings,
    expected_name: str | None = None,
    expected_dob: str | None = None,
) -> ValidationResult:
    out = ValidationResult()
    no_key_fields = not (fields.civil_id or fields.birth_date or fields.name)
    if no_key_fields:
        out.warnings.append("No key fields were extracted from OCR output.")

    if fields.civil_id:
        out.civil_id_format_valid = bool(_CIVIL_RE.fullmatch(fields.civil_id))
        if out.civil_id_format_valid:
            out.civil_id_checksum_valid = validate_civil_id_checksum(fields.civil_id)
            if not out.civil_id_checksum_valid:
                out.warnings.append("Civil ID checksum failed (community algorithm).")

            cid_dob = extract_dob_from_civil_id(fields.civil_id)
            extracted_dob = parse_iso_date(fields.birth_date)
            expected_dob_parsed = parse_iso_date(expected_dob)

            if extracted_dob and cid_dob:
                out.civil_id_dob_consistent = extracted_dob == cid_dob
                if not out.civil_id_dob_consistent:
                    out.warnings.append("DOB from OCR does not match DOB encoded in Civil ID.")

            if expected_dob_parsed and cid_dob and expected_dob_parsed != cid_dob:
                out.warnings.append("Expected DOB does not match DOB encoded in Civil ID.")
        else:
            out.civil_id_checksum_valid = False
            out.warnings.append("Civil ID format is invalid (must be 12 digits).")

    dob_for_plausibility = parse_iso_date(fields.birth_date) or parse_iso_date(expected_dob)
    out.dob_plausible = is_plausible_dob(dob_for_plausibility, settings.min_age, settings.max_age)
    if out.dob_plausible is False:
        out.warnings.append("DOB failed plausibility checks.")

    if expected_name and fields.name:
        score = name_similarity(expected_name, fields.name)
        out.name_similarity_score = score
        out.name_match = score >= 0.84
        if not out.name_match:
            out.warnings.append(f"Name similarity is low ({score:.2f}).")

    hard_failures = [
        no_key_fields,
        out.civil_id_format_valid is False,
        out.civil_id_checksum_valid is False if fields.civil_id else False,
        out.civil_id_dob_consistent is False,
        out.dob_plausible is False,
        out.name_match is False if expected_name and fields.name else False,
    ]
    out.overall_pass = not any(hard_failures)
    return out
