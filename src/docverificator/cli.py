from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_settings
from .pipeline import process_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document authenticity-oriented checker (OCR + validation).")
    parser.add_argument("--file", required=True, help="Path to input image/pdf document.")
    parser.add_argument("--expected-name", default=None, help="User-provided expected full name.")
    parser.add_argument("--expected-dob", default=None, help="Expected DOB in YYYY-MM-DD format.")
    parser.add_argument(
        "--ocr-engine",
        default=None,
        choices=["auto", "paddle", "easy"],
        help="Override OCR engine for this run.",
    )
    return parser


def main() -> None:
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
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

