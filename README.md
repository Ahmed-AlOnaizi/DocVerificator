# DocVerificator

Document verification pipeline in Python for Arabic + English IDs/statements (JPG/PNG/HEIC/PDF).

The system runs OCR, extracts key fields, validates consistency, and now saves each scan result as JSON under `output/`.

## What This Implementation Does

- OCR engine support:
- `paddle` (recommended for Arabic+English documents)
- `easy` (fallback)
- `auto` (tries Paddle first, then Easy)
- Preprocessing:
- denoise
- contrast enhancement (CLAHE)
- deskew with safety cap
- Extraction:
- Civil ID candidate scoring (not first regex hit)
- birth date extraction + Civil ID-derived birth date fallback
- expiry date extraction (label-aware, then positional fallback)
- name extraction anchored to `Name` labels (same line or next line)
- Validation:
- Civil ID format + checksum (community-sourced algorithm)
- DOB plausibility + consistency checks
- optional fuzzy name matching
- Persistence:
- each CLI scan writes a timestamped JSON output file to `output/` (or custom folder via flag)

## A-to-Z Flow

1. Input arrives via CLI (`--file`).
2. File loader checks size/format and loads:
- image directly, or
- first PDF page rendered to image.
3. Preprocessing runs (denoise/contrast/deskew).
4. OCR backend is initialized (`paddle`, `easy`, or `auto`).
5. OCR runs on preprocessed image.
6. Retry logic can run OCR on original/rotated variants when key fields are missing.
7. Field extraction runs:
- Civil ID: labeled-context aware candidate scoring + checksum/DOB-aware ranking.
- Birth date: explicit date patterns first, Civil ID birth-date fallback second.
- Expiry date: explicit/split-label patterns and compact fallback parsing.
- Name: label-guided selection + heuristic fallback.
8. Validation runs:
- format/checksum/DOB checks
- optional expected name/birth-date comparison
9. CLI prints JSON to terminal.
10. Same JSON is saved to disk in `output/<inputname>_<timestamp>.json`.

## Project Structure

- `src/docverificator/input_loader.py`: file ingest (image/PDF/HEIC)
- `src/docverificator/preprocessing.py`: image preprocessing
- `src/docverificator/ocr_backends.py`: Paddle/Easy OCR adapters
- `src/docverificator/extraction.py`: field extraction logic
- `src/docverificator/validation.py`: authenticity-oriented checks
- `src/docverificator/pipeline.py`: orchestrates full scan
- `src/docverificator/cli.py`: CLI + output JSON persistence
- `tests/`: unit and optional integration tests

## Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you use Paddle, verify:

```powershell
python -c "import paddle, paddleocr; print('ok')"
```

## Run

Basic:

```powershell
$env:PYTHONPATH="src"
python -m docverificator.cli --file sample_data\Civil-ID.pdf --ocr-engine paddle
```

With expected values:

```powershell
python -m docverificator.cli `
  --file sample_data\Civil-ID.pdf `
  --ocr-engine paddle `
  --expected-name "John Doe" `
  --expected-dob 2003-09-16
```

Custom output directory:

```powershell
python -m docverificator.cli --file sample_data\Civil-ID.pdf --output-dir scans
```

## Output Behavior

- Terminal output: JSON result.
- Saved output file: JSON file in `output/` by default.
- Filename format: `<input_stem>_<YYYYMMDD_HHMMSS_microseconds>.json`.
- Path is included in terminal JSON as `output_file`.
- Extracted `birth_date` and `expiry_date` are emitted in `dd/mm/yyyy`.

## Config

Environment variables (`.env.example`):

- `OCR_ENGINE`, `PADDLE_LANG`, `EASYOCR_LANGS`
- `LOW_CONF_THRESHOLD`, `MAX_FILE_SIZE_MB`, `PDF_DPI`
- `MIN_AGE`, `MAX_AGE`
- `RETRY_LOW_CONF_ON_ORIGINAL`
- `RETRY_MISSING_KEY_FIELDS`
- `TRY_ROTATIONS_ON_MISSING_FIELDS`
- `MAX_DESKEW_ANGLE`

## Notes

- Checksum logic is best-effort (community reference), not official PACI publication.
- This is a screening pipeline, not cryptographic proof of document authenticity.
