from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .models import ExtractedFields, OCRResult

_ARABIC_DIGITS = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669", "0123456789")
_CIVIL_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
_DATE_RE = re.compile(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b")
_DATE_COMPACT_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")
_MRZ_DATE_PAIR_RE = re.compile(r"[A-Z]([0-9]{6})[A-Z]([0-9]{6})")

_CIVIL_LABELS_EN = ("civil id", "civilid", "id number", "id no", "civil")
_CIVIL_LABELS_AR = (
    "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0645\u062f\u0646\u064a",
    "\u0631\u0642\u0645 \u0645\u062f\u0646\u064a",
    "\u0627\u0644\u0628\u0637\u0627\u0642\u0629 \u0627\u0644\u0645\u062f\u0646\u064a\u0629",
)
_DOB_LABELS_EN = ("dob", "date of birth", "birth date", "birth")
_DOB_LABELS_AR = ("\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f", "\u0645\u064a\u0644\u0627\u062f")
_EXPIRY_LABELS_EN = ("expiry", "expiry date", "expiration", "expiration date", "exp date", "valid until")
_EXPIRY_LABELS_AR = (
    "\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0627\u0646\u062a\u0647\u0627\u0621",
    "\u0627\u0644\u0627\u0646\u062a\u0647\u0627\u0621",
)

_NAME_LABELS_EN = ("name", "full name", "customer name", "account holder")
_NAME_LABELS_AR = (
    "\u0627\u0644\u0627\u0633\u0645",
    "\u0627\u0644\u0627\u0633\u0645 \u0627\u0644\u0643\u0627\u0645\u0644",
    "\u0627\u0633\u0645 \u0627\u0644\u0639\u0645\u064a\u0644",
    "\u0627\u0633\u0645 \u0635\u0627\u062d\u0628 \u0627\u0644\u062d\u0633\u0627\u0628",
)

_CHECKSUM_WEIGHTS = (2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)


@dataclass(slots=True)
class _DateCandidate:
    value: date
    line_index: int
    has_dob_label: bool
    has_expiry_label: bool
    direct_dob_label: bool
    direct_expiry_label: bool
    raw_text: str


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


def _parse_date(day_s: str, month_s: str, year_s: str, *, allow_future: bool) -> date | None:
    day = int(day_s)
    month = int(month_s)
    year = int(year_s)
    if year < 100:
        year = 2000 + year if year <= (date.today().year % 100) else 1900 + year
    max_year = date.today().year + 30 if allow_future else date.today().year
    if year < 1900 or year > max_year:
        return None
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if not allow_future and parsed > date.today():
        return None
    return parsed


def _format_date_output(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def _parse_yymmdd(token: str, *, allow_future: bool) -> date | None:
    if not re.fullmatch(r"\d{6}", token):
        return None
    yy = int(token[0:2])
    month = int(token[2:4])
    day = int(token[4:6])

    # Pivot near current year to map two-digit years.
    current_yy = date.today().year % 100
    year = 2000 + yy if yy <= current_yy + 15 else 1900 + yy
    max_year = date.today().year + 30 if allow_future else date.today().year
    if year < 1900 or year > max_year:
        return None
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if not allow_future and parsed > date.today():
        return None
    return parsed


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
    if "civilid" in compact:
        return True
    if re.search(r"\bid(?:\s*no|\s*number|[:#])", lowered):
        return True
    return any(label in line for label in _CIVIL_LABELS_AR)


def _has_dob_label(line: str) -> bool:
    lowered = line.lower()
    if any(label in lowered for label in _DOB_LABELS_EN):
        return True
    compact = re.sub(r"[^a-z]", "", lowered)
    if "dateofbirth" in compact or "birthdate" in compact or "dob" in compact:
        return True
    return any(label in line for label in _DOB_LABELS_AR)


def _has_expiry_label(line: str) -> bool:
    lowered = line.lower()
    if any(label in lowered for label in _EXPIRY_LABELS_EN):
        return True
    compact = re.sub(r"[^a-z]", "", lowered)
    if "expiry" in compact or "expiration" in compact or "validuntil" in compact or "validtill" in compact:
        return True
    if "date" in compact and any(tok in compact for tok in ("exp", "piy", "iry", "piry")):
        return True
    return any(label in line for label in _EXPIRY_LABELS_AR)


def _has_label_in_context(lines: list[str], idx: int, checker: callable) -> bool:
    if checker(lines[idx]):
        return True
    if idx > 0:
        prev = lines[idx - 1]
        if checker(prev + " " + lines[idx]):
            return True
    if idx + 1 < len(lines):
        nxt = lines[idx + 1]
        if checker(lines[idx] + " " + nxt):
            return True
    return False


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


def _push_candidate_with_bonus(
    scoreboard: dict[str, float],
    candidate: str,
    line: str,
    labeled_context: bool,
    from_window: bool,
    bonus: float,
) -> None:
    if not re.fullmatch(r"\d{12}", candidate):
        return
    score = _candidate_score(candidate, line=line, labeled_context=labeled_context, from_window=from_window) + bonus
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

    # First pass: candidates around explicit Civil ID labels.
    labeled_scoreboard: dict[str, float] = {}
    for idx, line in enumerate(lines):
        if not _has_civil_label(line):
            continue
        for j in range(idx, min(idx + 3, len(lines))):
            normalized = _normalize_for_digit_recovery(normalize_digits(lines[j]))
            for candidate, from_window in _collect_line_candidates(normalized):
                _push_candidate_with_bonus(
                    labeled_scoreboard,
                    candidate,
                    line=lines[j],
                    labeled_context=True,
                    from_window=from_window,
                    bonus=2.0,
                )
            # For long merged strings near label, the right-most 12 digits usually represent Civil ID.
            for seq in re.findall(r"\d{13,}", normalized):
                tail = seq[-12:]
                _push_candidate_with_bonus(
                    labeled_scoreboard,
                    tail,
                    line=lines[j],
                    labeled_context=True,
                    from_window=True,
                    bonus=7.5,
                )

    if labeled_scoreboard:
        best_labeled = max(labeled_scoreboard.items(), key=lambda item: item[1])[0]
        return best_labeled

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


def _collect_date_candidates(lines: list[str]) -> list[_DateCandidate]:
    out: list[_DateCandidate] = []
    for idx, raw in enumerate(lines):
        line = normalize_digits(raw)
        direct_dob_label = _has_dob_label(line)
        direct_expiry_label = _has_expiry_label(line)
        has_dob_label = _has_label_in_context(lines, idx, _has_dob_label)
        has_expiry_label = _has_label_in_context(lines, idx, _has_expiry_label)

        for match in _DATE_RE.finditer(line):
            day_s, month_s, year_s = match.groups()
            dt = _parse_date(day_s, month_s, year_s, allow_future=True)
            if dt:
                out.append(
                    _DateCandidate(
                        dt,
                        idx,
                        has_dob_label,
                        has_expiry_label,
                        direct_dob_label,
                        direct_expiry_label,
                        match.group(0),
                    )
                )

        # Compact pattern is noisy; use it only on labeled lines.
        if has_dob_label or has_expiry_label:
            compact = re.sub(r"\D", "", line)
            for match in _DATE_COMPACT_RE.finditer(compact):
                day_s, month_s, year_s = match.groups()
                dt = _parse_date(day_s, month_s, year_s, allow_future=True)
                if dt:
                    out.append(
                        _DateCandidate(
                            dt,
                            idx,
                            has_dob_label,
                            has_expiry_label,
                            direct_dob_label,
                            direct_expiry_label,
                            match.group(0),
                        )
                    )
    return out


def _pick_dob_and_expiry(lines: list[str], civil_id: str | None) -> tuple[str | None, str | None]:
    candidates = _collect_date_candidates(lines)

    dob_date: date | None = None
    dob_line_index: int | None = None

    if civil_id:
        cid_dob = _civil_id_to_dob(civil_id)
        if cid_dob:
            dob_date = cid_dob
            for c in candidates:
                if c.value == cid_dob:
                    dob_line_index = c.line_index
                    break

    if dob_date is None:
        direct_labeled_dob = [c for c in candidates if c.direct_dob_label and _dob_plausible(c.value)]
        if direct_labeled_dob:
            chosen = direct_labeled_dob[0]
            dob_date, dob_line_index = chosen.value, chosen.line_index
        else:
            labeled_dob = [c for c in candidates if c.has_dob_label and _dob_plausible(c.value)]
            if labeled_dob:
                chosen = labeled_dob[0]
                dob_date, dob_line_index = chosen.value, chosen.line_index
            else:
                generic = [c for c in candidates if _dob_plausible(c.value)]
                if generic:
                    chosen = generic[0]
                    dob_date, dob_line_index = chosen.value, chosen.line_index

    def pick_expiry_value(pool: list[_DateCandidate]) -> date | None:
        if not pool:
            return None
        # For IDs, expiry should be after birth when birth is known.
        if dob_date is not None:
            after_birth = [c for c in pool if c.value > dob_date]
            if after_birth:
                return max(after_birth, key=lambda c: (c.value, c.line_index)).value
            not_same = [c for c in pool if c.value != dob_date]
            if not_same:
                return max(not_same, key=lambda c: (c.value, c.line_index)).value
        return max(pool, key=lambda c: (c.value, c.line_index)).value

    expiry_date: date | None = None
    direct_labeled_expiry = [c for c in candidates if c.direct_expiry_label]
    expiry_date = pick_expiry_value(direct_labeled_expiry)

    if expiry_date is None:
        labeled_expiry = [c for c in candidates if c.has_expiry_label]
        expiry_date = pick_expiry_value(labeled_expiry)

    if expiry_date is None and dob_date is not None:
        post_dob = [
            c for c in candidates if c.value > dob_date and (dob_line_index is None or c.line_index >= dob_line_index)
        ]
        expiry_date = pick_expiry_value(post_dob)

    if expiry_date is None and dob_date is not None:
        post_dob_anywhere = [c for c in candidates if c.value > dob_date]
        expiry_date = pick_expiry_value(post_dob_anywhere)

    if expiry_date is None and candidates:
        # Resilient fallback: choose the latest visible date.
        expiry_date = max((c.value for c in candidates), default=None)

    if expiry_date is None:
        # Fallback for compact MRZ-like OCR lines, e.g. "...D309161M220916..."
        for raw in lines:
            normalized = _normalize_for_digit_recovery(normalize_digits(raw)).upper()
            for dob6, exp6 in _MRZ_DATE_PAIR_RE.findall(normalized):
                mrz_dob = _parse_yymmdd(dob6, allow_future=False)
                mrz_exp = _parse_yymmdd(exp6, allow_future=True)

                if dob_date is None and mrz_dob and _dob_plausible(mrz_dob):
                    dob_date = mrz_dob
                    dob_line_index = 0

                if mrz_exp and (dob_date is None or mrz_exp > dob_date):
                    expiry_date = mrz_exp
                    break
            if expiry_date is not None:
                break

    birth_out = _format_date_output(dob_date) if dob_date else None
    expiry_out = _format_date_output(expiry_date) if expiry_date else None
    return birth_out, expiry_out


def _label_extract_name(lines: list[str]) -> list[str]:
    candidates: list[str] = []

    def looks_like_non_name_label(value: str) -> bool:
        lowered = value.lower()
        compact = re.sub(r"[^a-z]", "", lowered)
        return any(
            token in compact
            for token in (
                "birthdate",
                "dateofbirth",
                "dob",
                "expiry",
                "expiration",
                "civilid",
                "civilnumber",
                "idnumber",
                "serial",
            )
        )

    def clean_name_value(value: str) -> str:
        value = value.strip(": -\t")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def clean_name_tokens(value: str) -> str:
        parts = [part for part in re.split(r"\s+", value) if part]
        if len(parts) < 2:
            return ""
        cleaned: list[str] = []
        for part in parts:
            token = re.sub(r"[^A-Za-z\u0600-\u06FF]", "", part)
            if not token:
                continue
            cleaned.append(token)
        return " ".join(cleaned).strip()

    def has_name_label(line: str) -> bool:
        lowered = line.lower()
        return any(label in lowered for label in _NAME_LABELS_EN) or any(label in line for label in _NAME_LABELS_AR)

    # First pass: strictly same-line value after "Name".
    for idx, line in enumerate(lines):
        if not has_name_label(line):
            continue

        value = line
        lowered = line.lower()
        used_en_label = False
        for label in sorted(_NAME_LABELS_EN, key=len, reverse=True):
            pos = lowered.find(label)
            if pos != -1:
                value = line[pos + len(label) :]
                used_en_label = True
                break
        if not used_en_label:
            for label in _NAME_LABELS_EN:
                value = re.sub(label, "", value, flags=re.IGNORECASE)
        for label in _NAME_LABELS_AR:
            value = value.replace(label, "")
        value = clean_name_tokens(clean_name_value(value))
        if value and not looks_like_non_name_label(value) and _name_score(value) > 0.6:
            candidates.append(value)
    if candidates:
        return candidates

    # Second pass fallback: next line after name label.
    for idx, line in enumerate(lines):
        if not has_name_label(line):
            continue
        # If the label line has no usable value, the next line often holds the full name.
        for j in range(idx + 1, min(idx + 3, len(lines))):
            nxt = clean_name_tokens(clean_name_value(lines[j]))
            if has_name_label(nxt):
                continue
            if looks_like_non_name_label(nxt):
                continue
            if _name_score(nxt) > 0.75:
                candidates.append(nxt)
                break
    return candidates


def _name_score(line: str) -> float:
    if len(line) < 3:
        return -1.0
    lowered = line.lower()
    if any(token in lowered for token in ("birth", "dob", "expiry", "expiration", "civil id", "serial", "date")):
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
    birth_date, expiry = _pick_dob_and_expiry(lines, civil_id=civil_id)
    name, candidates = _pick_name(lines)

    return ExtractedFields(
        civil_id=civil_id,
        birth_date=birth_date,
        expiry_date=expiry,
        name=name,
        doc_type_hint=_doc_hint(full_text),
        candidate_names=candidates,
        raw_lines=lines,
    )
