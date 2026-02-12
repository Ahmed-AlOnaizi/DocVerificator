from __future__ import annotations

import re
from datetime import date

from .models import ExtractedFields, OCRResult

_ARABIC_DIGITS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")
_CIVIL_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
_DOB_RE = re.compile(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b")
_DOB_COMPACT_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")

_CIVIL_LABELS_EN = ("civil id", "civilid", "id number", "id no", "civil")
_CIVIL_LABELS_AR = (
    "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0645\u062f\u0646\u064a",
    "\u0631\u0642\u0645 \u0645\u062f\u0646\u064a",
    "\u0627\u0644\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0645\u062f\u0646\u064a\u0629",
)
_DOB_LABELS_EN = ("dob", "date of birth", "birth date", "birth")
_DOB_LABELS_AR = ("\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f", "\u0645\u064a\u0644\u0627\u062f")

_NAME_LABELS_EN = ("name", "full name", "customer name", "account holder")
_NAME_LABELS_AR = (
    "\u0627\u0644\u0627\u0633\u0645",
    "\u0627\u0644\u0627\u0633\u0645 \u0627\u0644\u0643\u0627\u0645\u0644",
    "\u0627\u0633\u0645 \u0627\u0644\u0639\u0645\u064a\u0644",
    "\u0627\u0633\u0645 \u0635\u0627\u062d\u0628 \u0627\u0644\u062d\u0633\u0627\u0628",
)

_CHECKSUM_WEIGHTS = (2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


def normalize_digits(text: str) -> str:
    return text.translate(_ARABIC_DIGITS)


def _clean_line(text: str) -> str:
    return normalize_digits(text).strip()


def _normalize_for_digit_recovery(text: str) -> str:
    recovered = text.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "I": "1",
                "l": "1",
                "|": "1",
                "S": "5",
            }
        )
    )
    return recovered


def _parse_dob(day_s: str, month_s: str, year_s: str) -> date | None:
    day = int(day_s)
    month = int(month_s)
    year = int(year_s)
    if year < 100:
        year = 2000 + year if year <= (date.today().year % 100) else 1900 + year
    if year < 1900 or year > date.today().year:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _civil_checksum_valid(civil_id: str) -> bool:
    if not re.fullmatch(r"\d{12}", civil_id):
        return False
    total = sum(int(civil_id[i]) * _CHECKSUM_WEIGHTS[i] for i in range(11))
    expected = 11 - (total % 11)
    return 0 <= expected <= 9 and expected == int(civil_id[-1])


def _civil_id_to_dob(civil_id: str) -> date | None:
    if not re.fullmatch(r"\d{12}", civil_id):
        return None
    century = civil_id[0]
    yy = int(civil_id[1:3])
    mm = int(civil_id[3:5])
    dd = int(civil_id[5:7])
    if century == "2":
        year = 1900 + yy
    elif century == "3":
        year = 2000 + yy
    else:
        return None
    try:
        parsed = date(year, mm, dd)
    except ValueError:
        return None
    if parsed > date.today():
        return None
    return parsed


def _dob_plausible(value: date) -> bool:
    today = date.today()
    age = today.year - value.year - ((today.month, today.day) < (value.month, value.day))
    return 0 <= age <= 120


def _has_civil_label(line: str) -> bool:
    lowered = line.lower()
    compact = re.sub(r"\s+", "", lowered)
    if any(label in lowered for label in _CIVIL_LABELS_EN):
        return True
    if "civilid" in compact or compact.startswith("id") or "idkwt" in compact:
        return True
    return any(label in line for label in _CIVIL_LABELS_AR)


def _has_dob_label(line: str) -> bool:
    lowered = line.lower()
    if any(label in lowered for label in _DOB_LABELS_EN):
        return True
    return any(label in line for label in _DOB_LABELS_AR)


def _candidate_score(civil_id: str, line: str, labeled_context: bool, from_window: bool) -> float:
    score = 0.0
    if labeled_context:
        score += 11.0
    if _has_civil_label(line):
        score += 8.0
    if not from_window:
        score += 0.8
    if civil_id[0] in {"2", "3"}:
        score += 1.0

    cid_dob = _civil_id_to_dob(civil_id)
    if cid_dob:
        score += 2.0
        if _dob_plausible(cid_dob):
            score += 2.0

    if _civil_checksum_valid(civil_id):
        score += 7.0
    else:
        score -= 2.0

    return score


def _push_candidate(
    scoreboard: dict[str, float],
    candidate: str,
    line: str,
    labeled_context: bool,
    from_window: bool,
) -> None:
    if not re.fullmatch(r"\d{12}", candidate):
        return
    score = _candidate_score(candidate, line=line, labeled_context=labeled_context, from_window=from_window)
    current = scoreboard.get(candidate)
    if current is None or score > current:
        scoreboard[candidate] = score


def _collect_line_candidates(line: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    normalized = _normalize_for_digit_recovery(normalize_digits(line))

    for exact in _CIVIL_RE.findall(normalized):
        out.append((exact, False))

    for seq in re.findall(r"\d{12,}", normalized):
        if len(seq) == 12:
            out.append((seq, False))
            continue
        for i in range(0, len(seq) - 11):
            out.append((seq[i : i + 12], True))
    return out


def _pick_civil_id(lines: list[str]) -> str | None:
    if not lines:
        return None

    scoreboard: dict[str, float] = {}

    for idx, line in enumerate(lines):
        for candidate, from_window in _collect_line_candidates(line):
            _push_candidate(
                scoreboard,
                candidate,
                line=line,
                labeled_context=False,
                from_window=from_window,
            )

        if _has_civil_label(line):
            for j in range(idx, min(idx + 3, len(lines))):
                for candidate, from_window in _collect_line_candidates(lines[j]):
                    _push_candidate(
                        scoreboard,
                        candidate,
                        line=lines[j],
                        labeled_context=True,
                        from_window=from_window,
                    )

    # OCR can split ID digits across consecutive lines.
    for i in range(len(lines)):
        chunk = ""
        has_label = _has_civil_label(lines[i])
        for j in range(i, min(i + 4, len(lines))):
            segment = re.sub(r"\D", "", _normalize_for_digit_recovery(lines[j]))
            if not segment:
                break
            chunk += segment
            if len(chunk) < 12:
                continue
            for k in range(0, len(chunk) - 11):
                candidate = chunk[k : k + 12]
                _push_candidate(
                    scoreboard,
                    candidate,
                    line=lines[i],
                    labeled_context=has_label,
                    from_window=True,
                )

    if not scoreboard:
        return None
    best = max(scoreboard.items(), key=lambda item: item[1])[0]
    return best


def _pick_dob(lines: list[str], full_text: str, civil_id: str | None) -> str | None:
    normalized_text = normalize_digits(full_text)

    for day_s, month_s, year_s in _DOB_RE.findall(normalized_text):
        dt = _parse_dob(day_s, month_s, year_s)
        if dt:
            return dt.isoformat()

    for line in lines:
        if not _has_dob_label(line):
            continue
        normalized_line = normalize_digits(line)
        for day_s, month_s, year_s in _DOB_RE.findall(normalized_line):
            dt = _parse_dob(day_s, month_s, year_s)
            if dt:
                return dt.isoformat()
        compact = re.sub(r"\D", "", normalized_line)
        for day_s, month_s, year_s in _DOB_COMPACT_RE.findall(compact):
            dt = _parse_dob(day_s, month_s, year_s)
            if dt:
                return dt.isoformat()

    if civil_id:
        cid_dob = _civil_id_to_dob(civil_id)
        if cid_dob:
            return cid_dob.isoformat()

    return None


def _label_extract_name(lines: list[str]) -> list[str]:
    candidates: list[str] = []
    for line in lines:
        lowered = line.lower()
        has_en_label = any(label in lowered for label in _NAME_LABELS_EN)
        has_ar_label = any(label in line for label in _NAME_LABELS_AR)
        if not (has_en_label or has_ar_label):
            continue

        value = line
        for label in _NAME_LABELS_EN:
            value = re.sub(label, "", value, flags=re.IGNORECASE)
        for label in _NAME_LABELS_AR:
            value = value.replace(label, "")

        value = value.strip(": -\t")
        if value:
            candidates.append(value)
    return candidates


def _name_score(line: str) -> float:
    if len(line) < 3:
        return -1.0
    alpha = sum(1 for ch in line if ch.isalpha())
    digits = sum(1 for ch in line if ch.isdigit())
    if alpha == 0:
        return -1.0
    ratio = alpha / max(len(line), 1)
    return ratio + min(alpha, 36) / 80.0 - (digits * 0.1)


def _pick_name(lines: list[str]) -> tuple[str | None, list[str]]:
    candidates = _label_extract_name(lines)

    if not candidates:
        scored = sorted(
            ((line, _name_score(line)) for line in lines),
            key=lambda x: x[1],
            reverse=True,
        )
        candidates = [line for line, score in scored if score > 0.55][:3]

    if not candidates:
        return None, []
    return candidates[0], candidates


def _doc_hint(full_text: str) -> str | None:
    text = full_text.lower()
    if (
        "civil id" in text
        or "\u0627\u0644\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0645\u062f\u0646\u064a\u0629" in text
        or "\u0628\u0637\u0627\u0642\u0629 \u0645\u062f\u0646\u064a\u0629" in text
    ):
        return "civil_id"
    if "statement" in text or "bank" in text or "\u0643\u0634\u0641 \u062d\u0633\u0627\u0628" in text:
        return "bank_statement"
    return None


def extract_fields(ocr_result: OCRResult) -> ExtractedFields:
    lines = [_clean_line(line.text) for line in ocr_result.lines if line.text.strip()]
    full_text = "\n".join(lines) if lines else _clean_line(ocr_result.full_text)

    civil_id = _pick_civil_id(lines)
    dob = _pick_dob(lines, full_text=full_text, civil_id=civil_id)
    name, candidates = _pick_name(lines)

    return ExtractedFields(
        civil_id=civil_id,
        date_of_birth=dob,
        name=name,
        doc_type_hint=_doc_hint(full_text),
        candidate_names=candidates,
        raw_lines=lines,
    )
