from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from .config import Settings
from .models import OCRLine, OCRResult


def _mean_conf(lines: list[OCRLine]) -> float:
    if not lines:
        return 0.0
    return float(sum(line.confidence for line in lines) / len(lines))


@dataclass(slots=True)
class OCRBackend:
    name: str

    def run(self, image_bgr: np.ndarray) -> OCRResult:  # pragma: no cover - interface method
        raise NotImplementedError


class PaddleBackend(OCRBackend):
    def __init__(self, lang: str) -> None:
        super().__init__(name="paddle")

        # Windows CPU deployments can fail in oneDNN fused conv ops on some wheel combos.
        # These defaults bias toward stability over peak speed.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["CPU_NUM"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is not installed. Use extra: [ocr-paddle]") from exc

        base = {
            "use_angle_cls": False,
            "lang": lang,
            "use_gpu": False,
            "enable_mkldnn": False,
            "cpu_threads": 1,
        }
        attempts = [
            {**base, "show_log": False, "ir_optim": False},
            {**base, "show_log": False},
            {**base, "ir_optim": False},
            base,
        ]

        last_exc: Exception | None = None
        self._reader = None
        for kwargs in attempts:
            try:
                self._reader = PaddleOCR(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - version-specific PaddleOCR kwargs
                last_exc = exc
                msg = str(exc).lower()
                if "unknown argument" in msg or "unexpected keyword" in msg:
                    continue
                raise

        if self._reader is None:
            assert last_exc is not None
            raise last_exc

    def run(self, image_bgr: np.ndarray) -> OCRResult:
        raw = None
        errors: list[Exception] = []
        for kwargs in ({"cls": False}, {"cls": True}, {}):
            try:
                raw = self._reader.ocr(image_bgr, **kwargs)
                break
            except TypeError:
                # Older signatures may not support `cls`.
                if kwargs:
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - runtime fallback for unstable Paddle CPU kernels
                errors.append(exc)
                continue

        if raw is None:
            text = " | ".join(str(e) for e in errors) if errors else "Unknown OCR runtime error."
            raise RuntimeError(f"PaddleOCR inference failed after fallbacks: {text}")

        lines: list[OCRLine] = []
        texts: list[str] = []

        for block in raw or []:
            if not block:
                continue
            for item in block:
                try:
                    bbox, payload = item
                    text, conf = payload
                except (TypeError, ValueError):
                    continue
                if not text:
                    continue
                bbox_pairs = [(float(x), float(y)) for x, y in bbox]
                lines.append(OCRLine(text=str(text).strip(), confidence=float(conf), bbox=bbox_pairs))
                texts.append(str(text).strip())

        return OCRResult(lines=lines, full_text="\n".join(texts), mean_confidence=_mean_conf(lines))


class EasyBackend(OCRBackend):
    def __init__(self, langs: tuple[str, ...], gpu: bool = False) -> None:
        super().__init__(name="easyocr")
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is not installed. Use extra: [ocr-easy]") from exc
        self._reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def run(self, image_bgr: np.ndarray) -> OCRResult:
        raw = self._reader.readtext(image_bgr, detail=1, paragraph=False)
        lines: list[OCRLine] = []
        texts: list[str] = []

        for item in raw or []:
            try:
                bbox, text, conf = item
            except (TypeError, ValueError):
                continue
            if not text:
                continue
            bbox_pairs = [(float(x), float(y)) for x, y in bbox]
            lines.append(OCRLine(text=str(text).strip(), confidence=float(conf), bbox=bbox_pairs))
            texts.append(str(text).strip())

        return OCRResult(lines=lines, full_text="\n".join(texts), mean_confidence=_mean_conf(lines))


def build_ocr_backend(settings: Settings) -> OCRBackend:
    preferred = settings.ocr_engine.lower()
    if preferred not in {"auto", "paddle", "easy"}:
        raise ValueError(f"Unsupported OCR_ENGINE value: {settings.ocr_engine}")

    order = ["paddle", "easy"] if preferred == "auto" else [preferred]
    errors: list[str] = []

    for engine in order:
        try:
            if engine == "paddle":
                return PaddleBackend(lang=settings.paddle_lang)
            return EasyBackend(langs=settings.easyocr_langs, gpu=settings.easyocr_gpu)
        except Exception as exc:  # noqa: BLE001 - backend fallback logic
            errors.append(f"{engine}: {exc}")

    raise RuntimeError("Could not initialize OCR backend. " + " | ".join(errors))
