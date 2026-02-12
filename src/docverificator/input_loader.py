from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def _check_size(path: Path, max_file_size_mb: int) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_size_mb:
        raise ValueError(f"File is too large ({size_mb:.2f} MB). Max allowed: {max_file_size_mb} MB.")


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif
        except ImportError as exc:
            raise RuntimeError("HEIC support requires pillow-heif. Install extra: [heic]") from exc
        pillow_heif.register_heif_opener()

    with Image.open(path) as im:
        # Exif transpose neutralizes orientation metadata so OCR sees the real orientation.
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_pdf(path: Path, dpi: int) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PDF support requires PyMuPDF.") from exc

    doc = fitz.open(path)
    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")

    page = doc.load_page(0)
    zoom = max(float(dpi) / 72.0, 1.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_document(file_path: str | Path, max_file_size_mb: int, pdf_dpi: int) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    _check_size(path, max_file_size_mb=max_file_size_mb)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path, dpi=pdf_dpi)
    if ext in SUPPORTED_IMAGE_EXTS:
        return _load_image(path)

    raise ValueError(f"Unsupported file extension: {ext}")

