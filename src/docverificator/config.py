from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    ocr_engine: str = "auto"
    paddle_lang: str = "ar"
    easyocr_langs: tuple[str, ...] = ("ar", "en")
    low_conf_threshold: float = 0.55
    max_file_size_mb: int = 12
    pdf_dpi: int = 220
    min_age: int = 16
    max_age: int = 110
    retry_low_conf_on_original: bool = True
    retry_missing_key_fields: bool = True
    try_rotations_on_missing_fields: bool = True
    max_deskew_angle: float = 12.0
    easyocr_gpu: bool = False


def load_settings() -> Settings:
    langs = os.getenv("EASYOCR_LANGS", "ar,en")
    parsed_langs = tuple(x.strip() for x in langs.split(",") if x.strip())

    return Settings(
        ocr_engine=os.getenv("OCR_ENGINE", "auto").strip().lower(),
        paddle_lang=os.getenv("PADDLE_LANG", "ar").strip().lower(),
        easyocr_langs=parsed_langs or ("ar", "en"),
        low_conf_threshold=float(os.getenv("LOW_CONF_THRESHOLD", "0.55")),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "12")),
        pdf_dpi=int(os.getenv("PDF_DPI", "220")),
        min_age=int(os.getenv("MIN_AGE", "16")),
        max_age=int(os.getenv("MAX_AGE", "110")),
        retry_low_conf_on_original=_env_bool("RETRY_LOW_CONF_ON_ORIGINAL", True),
        retry_missing_key_fields=_env_bool("RETRY_MISSING_KEY_FIELDS", True),
        try_rotations_on_missing_fields=_env_bool("TRY_ROTATIONS_ON_MISSING_FIELDS", True),
        max_deskew_angle=float(os.getenv("MAX_DESKEW_ANGLE", "12")),
        easyocr_gpu=_env_bool("EASYOCR_GPU", False),
    )
