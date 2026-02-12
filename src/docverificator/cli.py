from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document authenticity-oriented checker (OCR + validation).")
    parser.add_argument("--file", required=True, help="Path to input image/pdf document.")
    parser.add_argument("--expected-name", default=None, help="User-provided expected full name.")
    parser.add_argument("--expected-dob", default=None, help="Expected DOB in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where scan JSON outputs are saved (default: output).",
    )
    parser.add_argument(
        "--ocr-engine",
        default=None,
        choices=["auto", "paddle", "easy"],
        help="Override OCR engine for this run.",
    )
    return parser


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "document"


def save_result_json(payload: dict[str, Any], input_file: str, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{_safe_stem(src)}_{timestamp}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    from .pipeline import process_document

    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()
    if args.ocr_engine:
        settings = replace(settings, ocr_engine=args.ocr_engine)

    if not Path(args.file).exists():
        raise SystemExit(f"Input file does not exist: {args.file}")

    result = process_document(
        file_path=args.file,
        expected_name=args.expected_name,
        expected_dob=args.expected_dob,
        settings=settings,
    )
    payload = result.to_dict()
    out_path = save_result_json(payload, input_file=args.file, output_dir=args.output_dir)
    payload["output_file"] = str(out_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
