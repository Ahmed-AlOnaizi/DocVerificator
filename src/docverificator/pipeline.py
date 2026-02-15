from __future__ import annotations

import cv2
import numpy as np

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
    if fields.birth_date:
        score += 3
    if fields.expiry_date:
        score += 3
    if fields.name:
        score += 2
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


def _clip_crop(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray | None:
    h, w = image.shape[:2]
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    if y1 - y0 < 40 or x1 - x0 < 40:
        return None
    return image[y0:y1, x0:x1]


def _detect_card_like_crops(image: np.ndarray, max_regions: int = 3) -> list[np.ndarray]:
    h, w = image.shape[:2]
    page_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[float, int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area < page_area * 0.08 or area > page_area * 0.95:
            continue
        aspect = cw / max(ch, 1)
        # ID cards are generally landscape-ish; keep broad bounds to tolerate perspective.
        if not (1.1 <= aspect <= 2.8):
            continue
        boxes.append((area, x, y, cw, ch))

    boxes.sort(key=lambda item: item[0], reverse=True)
    crops: list[np.ndarray] = []
    for _, x, y, cw, ch in boxes[: max_regions * 2]:
        pad_x = int(cw * 0.04)
        pad_y = int(ch * 0.06)
        crop = _clip_crop(image, y - pad_y, y + ch + pad_y, x - pad_x, x + cw + pad_x)
        if crop is not None:
            crops.append(crop)
        if len(crops) >= max_regions:
            break
    return crops


def _build_retry_images(image: np.ndarray, include_rotations: bool) -> list[tuple[str, np.ndarray]]:
    h, w = image.shape[:2]
    retry_images: list[tuple[str, np.ndarray]] = [("original", image)]

    # Common for uploaded PDFs with two stacked card captures.
    top60 = _clip_crop(image, 0, int(h * 0.60), 0, w)
    top50 = _clip_crop(image, 0, int(h * 0.50), 0, w)
    mid80 = _clip_crop(image, int(h * 0.10), int(h * 0.90), 0, w)
    bottom60 = _clip_crop(image, int(h * 0.40), h, 0, w)
    for label, crop in (("top60", top60), ("top50", top50), ("mid80", mid80), ("bottom60", bottom60)):
        if crop is not None:
            retry_images.append((label, crop))

    for idx, crop in enumerate(_detect_card_like_crops(image), start=1):
        retry_images.append((f"card{idx}", crop))

    if include_rotations:
        retry_images.extend(
            [
                ("rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                ("rot180", cv2.rotate(image, cv2.ROTATE_180)),
                ("rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
        )
    return retry_images


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
    id_like = bool(fields.civil_id) or fields.doc_type_hint == "civil_id"
    retry_for_missing_fields = cfg.retry_missing_key_fields and (
        not fields.civil_id or (id_like and (not fields.birth_date or not fields.expiry_date or not fields.name))
    )

    if retry_for_low_conf or retry_for_missing_fields:
        retry_images = _build_retry_images(image, include_rotations=cfg.try_rotations_on_missing_fields)

        for label, retry_image in retry_images:
            local_best_ocr: OCRResult | None = None
            local_best_fields: ExtractedFields | None = None
            local_best_label: str | None = None
            try:
                retry_prep = preprocess_for_ocr(retry_image, max_deskew_angle=cfg.max_deskew_angle)
            except Exception as exc:  # noqa: BLE001 - retry should continue on per-candidate errors
                warnings.append(f"OCR retry '{label}' failed: {exc}")
                continue

            candidates = [
                (f"{label}:preprocessed", retry_prep.ocr_image),
                (f"{label}:original", retry_image),
            ]
            for source_label, candidate_img in candidates:
                try:
                    candidate_ocr = backend.run(candidate_img)
                    candidate_fields = extract_fields(candidate_ocr)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"OCR retry '{source_label}' failed: {exc}")
                    continue

                if (
                    local_best_ocr is None
                    or local_best_fields is None
                    or _is_better_candidate(candidate_ocr, candidate_fields, local_best_ocr, local_best_fields)
                ):
                    local_best_ocr = candidate_ocr
                    local_best_fields = candidate_fields
                    local_best_label = source_label

            if (
                local_best_ocr is not None
                and local_best_fields is not None
                and _is_better_candidate(local_best_ocr, local_best_fields, ocr, fields)
            ):
                ocr = local_best_ocr
                fields = local_best_fields
                best_source = local_best_label or label

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
        ocr_source=best_source,
        ocr_mean_confidence=ocr.mean_confidence,
        preprocess_angle=prep.deskew_angle,
        warnings=warnings,
    )
