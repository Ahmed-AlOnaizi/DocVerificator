# DocVerificator

Python starter project for document checks on JPG/PNG/HEIC/PDF inputs, focused on:

- OCR for Arabic + English documents
- Field extraction (Civil ID, DOB, name)
- Best-effort authenticity validation checks (format/checksum/consistency/fuzzy match)

## Why this design

This implementation uses your guidelines, with production-friendly defaults:

- PaddleOCR first, EasyOCR fallback
- OpenCV preprocessing (denoise, contrast, deskew)
- Regex + heuristics extraction
- Kuwait Civil ID validation:
- 12-digit format
- community-sourced checksum formula (not official PACI spec)
- DOB plausibility and Civil ID vs DOB consistency
- Fuzzy name matching via `rapidfuzz`

Important: this is not a cryptographic authenticity verifier. It is a strong screening layer that reduces fraud risk, but cannot prove originality by itself.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[ocr-paddle,ocr-easy,heic,dev]
```

If PaddleOCR installation is heavy for your environment, install EasyOCR first:

```bash
pip install -e .[ocr-easy,heic,dev]
```

## Run

```bash
docverificator --file path\to\document.jpg --expected-name "John Doe" --expected-dob 1995-04-12
```

Or:

```bash
python -m docverificator.cli --file path\to\document.pdf --ocr-engine paddle
```

Output is JSON with extracted fields, validation decisions, OCR confidence, and warnings.

## Environment configuration

See `.env.example`:

- `OCR_ENGINE`: `auto`, `paddle`, `easy`
- `PADDLE_LANG`: default `ar`
- `EASYOCR_LANGS`: CSV, default `ar,en`
- `LOW_CONF_THRESHOLD`: OCR retry threshold
- `MAX_FILE_SIZE_MB`: upload guardrail
- `PDF_DPI`: render quality for PDFs
- `MIN_AGE` / `MAX_AGE`: DOB plausibility bounds

## Next hardening steps

1. Add region detectors (ID number/name zones) and second-pass OCR on crops.
2. Add anti-tamper signals (copy-move detection, JPEG artifact inconsistency, MRZ checks if available).
3. Add official/partner verification integration when accessible.
4. Add golden test fixtures for your real document templates.

