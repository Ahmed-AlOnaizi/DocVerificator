from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float
    bbox: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class OCRResult:
    lines: list[OCRLine]
    full_text: str
    mean_confidence: float


@dataclass(slots=True)
class ExtractedFields:
    civil_id: str | None = None
    birth_date: str | None = None
    expiry_date: str | None = None
    name: str | None = None
    doc_type_hint: str | None = None
    candidate_names: list[str] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationResult:
    civil_id_format_valid: bool | None = None
    civil_id_checksum_valid: bool | None = None
    civil_id_dob_consistent: bool | None = None
    dob_plausible: bool | None = None
    name_similarity_score: float | None = None
    name_match: bool | None = None
    overall_pass: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    extracted: ExtractedFields
    validation: ValidationResult
    ocr_engine: str
    ocr_source: str
    ocr_mean_confidence: float
    preprocess_angle: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
