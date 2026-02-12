from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessArtifacts:
    ocr_image: np.ndarray
    deskew_angle: float
    sharpness_score: float


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.01:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _estimate_skew_angle(gray: np.ndarray) -> float:
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 200:
        return 0.0

    raw = float(cv2.minAreaRect(coords)[-1])
    # Normalize to the closest equivalent angle around 0 degrees.
    candidates = (raw, raw - 90.0, raw + 90.0)
    return min(candidates, key=abs)


def preprocess_for_ocr(image_bgr: np.ndarray, max_deskew_angle: float = 12.0) -> PreprocessArtifacts:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    angle = _estimate_skew_angle(contrast)
    if abs(angle) > max_deskew_angle:
        # Large rotation estimates are often wrong for ID layouts; skip deskew instead of over-rotating.
        angle = 0.0
    deskewed = _rotate(contrast, angle=angle)

    # Mild adaptive threshold keeps text strokes readable for OCR while reducing noise.
    binarized = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    ocr_ready = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    sharpness = float(cv2.Laplacian(deskewed, cv2.CV_64F).var())
    return PreprocessArtifacts(ocr_image=ocr_ready, deskew_angle=angle, sharpness_score=sharpness)
