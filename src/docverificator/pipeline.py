from __future__ import annotations

import cv2

from .config import Settings, load_settings
from .extraction import extract_fields
from .input_loader import load_document
from .models import ExtractedFields, OCRResult
from .models import PipelineResult
from .ocr_backends import build_ocr_backend
from .preprocessing import preprocess_for_ocr
from .validation import validate_fields


def _field_score(fields: ExtractedFields) -> int:
    score = 0
    if fields.civil_id:
        score += 6
    if fields.date_of_birth:
        score += 3
    if fields.name:
        score += 1
    if fields.doc_type_hint == "civil_id":
        score += 1
    return score


def _is_better_candidate(
    candidate_ocr: OCRResult,
    candidate_fields: ExtractedFields,
    best_ocr: OCRResult,
    best_fields: ExtractedFields,
) -> bool:
    candidate_rank = (_field_score(candidate_fields), candidate_ocr.mean_confidence)
    best_rank = (_field_score(best_fields), best_ocr.mean_confidence)
    return candidate_rank > best_rank


def process_document(
    file_path: str,
    expected_name: str | None = None,
    expected_dob: str | None = None,
    settings: Settings | None = None,
) -> PipelineResult:
    cfg = settings or load_settings()

    image = load_document(file_path, max_file_size_mb=cfg.max_file_size_mb, pdf_dpi=cfg.pdf_dpi)
    prep = preprocess_for_ocr(image, max_deskew_angle=cfg.max_deskew_angle)

    backend = build_ocr_backend(cfg)
    ocr = backend.run(prep.ocr_image)
    fields = extract_fields(ocr)
    best_source = "preprocessed"
    warnings: list[str] = []

    retry_for_low_conf = cfg.retry_low_conf_on_original and ocr.mean_confidence < cfg.low_conf_threshold
    retry_for_missing_fields = cfg.retry_missing_key_fields and not fields.civil_id

    if retry_for_low_conf or retry_for_missing_fields:
        retry_images: list[tuple[str, object]] = [("original", image)]
        if retry_for_missing_fields and cfg.try_rotations_on_missing_fields:
            retry_images.extend(
                [
                    ("rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                    ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
                    ("rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ]
            )

        for label, retry_image in retry_images:
            try:
                retry_ocr = backend.run(retry_image)
                retry_fields = extract_fields(retry_ocr)
            except Exception as exc:  # noqa: BLE001 - retry should continue on per-candidate errors
                warnings.append(f"OCR retry '{label}' failed: {exc}")
                continue

            if _is_better_candidate(retry_ocr, retry_fields, ocr, fields):
                ocr = retry_ocr
                fields = retry_fields
                best_source = label

        if best_source != "preprocessed":
            warnings.append(f"Selected OCR result from '{best_source}' fallback due to better field extraction.")
        elif retry_for_low_conf and ocr.mean_confidence < cfg.low_conf_threshold:
            warnings.append("Low OCR confidence after fallback attempts.")

    validation = validate_fields(fields, cfg, expected_name=expected_name, expected_dob=expected_dob)
    warnings.extend(validation.warnings)
    if not fields.civil_id:
        warnings.append("Civil ID was not extracted from OCR text.")

    return PipelineResult(
        extracted=fields,
        validation=validation,
        ocr_engine=backend.name,
        ocr_mean_confidence=ocr.mean_confidence,
        preprocess_angle=prep.deskew_angle,
        warnings=warnings,
    )
