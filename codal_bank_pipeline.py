#!/usr/bin/env python3
"""Reproducible annual bank-statement pipeline for six Iranian banks.

The run used for the accompanying dashboard reads public annual reports that
originate from CODAL and are exposed by Rahavard365's public report index/file
endpoints.  The raw report index and PDFs are cached, and every extracted value
retains the report id, PDF URL, publication date and extraction line.

Important: the calculated liquidity and credit-exposure measures are transparent
research proxies.  They are not official LCR, NSFR, NPL or capital-adequacy ratios.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scenario_engine import Scenario, evaluate


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
RAW_DIR = ROOT / "data" / "raw"
PDF_DIR = RAW_DIR / "pdfs"
TEXT_DIR = RAW_DIR / "text"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

REPORT_INDEX_URL = "https://rahavard365.com/api/v2/report/reports"
REPORT_PDF_URL = "https://rahavard365.com/api/v2/report/images/{image_id}/file"
CODAL_PORTAL_URL = "https://my.codal.ir/"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CODALBankResearch/1.0)",
    "Accept": "application/json, application/pdf;q=0.9, */*;q=0.8",
    "platform": "web",
    "Application-Name": "rahavard",
}

DIGIT_TRANS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
CHAR_TRANS = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ھ": "ه", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "أ": "ا", "إ": "ا", "ٱ": "ا", "‌": " ",
})
BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
NUM_RE = re.compile(r"\(?\s*-?\s*[0-9][0-9,٬]*\s*\)?")


FIELD_PATTERNS: dict[str, list[str]] = {
    "cash": [r"موجود[یى]?\s*نقد"],
    "due_from_banks": [r"مطالبات از بانک\s*ها و (?:سایر )?موسسات اعتباری(?:\s*غیر\s*بانکی)?"],
    "due_from_central_bank": [r"مطالبات از بانک\s*ها[یی]+ مرکزی"],
    "government_receivables": [r"مطالبات از دولت"],
    "loans_government": [r"تسهیلات اعطا[یی]+ و مطالبات از اشخاص دولتی"],
    "loans_nongovernment": [r"تسهیلات اعطا[یی]+ و مطالبات از اشخاص غیر\s*دولتی"],
    "loans_total": [r"^تسهیلات اعطا[یی]+ و مطالبات$", r"^تسهیلات اعطا[یی]+$"],
    "investments_securities": [r"سرمایه\s*گذاری در سهام و (?:سایر|ساير) اوراق بهادار"],
    "statutory_deposit": [r"سپرده قانونی"],
    "total_assets": [r"جمع دارا[یی]+\s*ها?$"],
    "customer_deposits": [r"سپرده\s*ها[یی]+ مشتریان"],
    "investment_depositor_rights": [r"حقوق صاحبان سپرده\s*ها[یی]+ سرمایه\s*گذاری"],
    "deposits_sight": [r"سپرده\s*ها[یی]+ دیداری و مشابه"],
    "deposits_savings": [r"سپرده\s*ها[یی]+ پس\s*انداز و مشابه"],
    "deposits_term": [r"سپرده\s*ها[یی]+ سرمایه\s*گذاری مدت\s*دار"],
    "deposits_other": [r"سایر سپرده\s*ها"],
    "total_liabilities": [r"جمع بده[یی]+\s*ها?$"],
    "total_equity": [r"جمع حقوق مالکانه$"],
    "liabilities_and_equity": [r"جمع بده[یی]+\s*ها و حقوق مالکانه$"],
}

# pdftotext inserts spaces inside Persian words in some 1397-1398 filings.
# These compact aliases are matched after removing whitespace, while their
# location is mapped back to the original line so the correct adjacent value is
# retained even when assets and liabilities share one printed row.
FIELD_ALIASES_COMPACT: dict[str, list[str]] = {
    "cash": ["موجودینقد"],
    "due_from_banks": ["مطالباتازبانکهاوسایرموسساتاعتباری", "مطالباتازبانکهاوموسساتاعتباریغیربانکی"],
    "due_from_central_bank": ["مطالباتازبانکهایمركزی", "مطالباتازبانکهایمرکزی"],
    "government_receivables": ["مطالباتازدولت"],
    "loans_government": ["تسهیلاتاعطاییومطالباتازاشخاصدولتی", "تسهیالتاعطاییومطالباتازاشخاصدولتی"],
    "loans_nongovernment": ["تسهیلاتاعطاییومطالباتازاشخاصغیردولتی", "تسهیالتاعطاییومطالباتازاشخاصغیردولتی"],
    "loans_total": ["تسهیلاتاعطاییومطالبات", "تسهیالتاعطاییومطالبات"],
    "investments_securities": ["سرمایهگذاریدرسهاموسایراوراقبهادار"],
    "statutory_deposit": ["سپردهقانونی"],
    "total_assets": ["جمعداراییها"],
    "customer_deposits": ["سپردههایمشتریان"],
    "investment_depositor_rights": ["حقوقصاحبانسپردههایسرمایهگذاری"],
    "deposits_sight": ["سپردههایدیداریومشابه"],
    "deposits_savings": ["سپردههایپساندازومشابه"],
    "deposits_term": ["سپردههایسرمایهگذاریمدتدار"],
    "deposits_other": ["سایرسپردهها"],
    "total_liabilities": ["جمعبدهیها"],
    "total_equity": ["جمعحقوقمالکانه", "جمعحقوقصاحبانسهام"],
    "liabilities_and_equity": ["جمعبدهیهاوحقوقمالکانه", "جمعبدهیهاوحقوقصاحبانسهام"],
}

REQUIRED_FIELDS = ["cash", "due_from_banks", "total_assets", "customer_deposits", "total_liabilities", "total_equity"]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PDF_DIR, TEXT_DIR, PROCESSED_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def normalize_fa(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(DIGIT_TRANS).translate(CHAR_TRANS)
    value = BIDI_RE.sub("", value)
    value = value.replace("٬", ",").replace("−", "-")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_report_index(symbol: str, asset_id: int, refresh: bool = False) -> list[dict[str, Any]]:
    path = RAW_DIR / f"report_index_{symbol}.json"
    if not path.exists() or refresh:
        query = urllib.parse.urlencode({"_skip": 0, "_count": 1200, "asset_id": asset_id})
        payload = http_get_bytes(f"{REPORT_INDEX_URL}?{query}")
        path.write_bytes(payload)
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("data", [])


def gregorian_fiscal_to_jalali_year(value: str) -> int:
    """Map Iranian March fiscal year-end metadata to its Jalali year."""
    dt = datetime.fromisoformat(value)
    if dt.month not in (3, 12):
        # Annual bank filings occasionally carry a non-March fiscal date.  This
        # mapping is retained for audit, but March reports are preferred.
        return dt.year - 622 if dt.month <= 3 else dt.year - 621
    return dt.year - 622 if dt.month == 3 else dt.year - 621


def report_score(report: dict[str, Any]) -> tuple[int, int, int, str]:
    subs = set(report.get("subtitle") or [])
    has_main = int("اصلی (غیر تلفیقی)" in subs)
    audited = int("حسابرسی شده" in subs)
    amended = int("اصلاحیه" in subs)
    return (has_main, audited, amended, report.get("date", ""))


def select_annual_reports(
    rows: list[dict[str, Any]], symbol: str, years: Iterable[int],
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    years = set(years)
    candidates: dict[int, list[dict[str, Any]]] = {year: [] for year in years}
    for report in rows:
        subs = report.get("subtitle") or []
        if report.get("trade_symbol") != symbol:
            continue
        if report.get("title") != "صورت های مالی" or "۱۲ ماهه" not in subs:
            continue
        if "شرکت های تابعه" in report.get("title", ""):
            continue
        year = gregorian_fiscal_to_jalali_year(report["fiscal_year"])
        if year in years:
            candidates[year].append(report)

    selected: list[dict[str, Any]] = []
    for year in sorted(years):
        options = candidates.get(year) or []
        # Prefer reports explicitly containing the parent/separate statement;
        # within that set prefer audited, amendment, then latest publication.
        main_options = [r for r in options if "اصلی (غیر تلفیقی)" in (r.get("subtitle") or [])]
        pool = main_options or options
        if not pool:
            selected.append({"symbol": symbol, "jalali_year": year, "selection_status": "missing"})
            continue
        override = (overrides or {}).get(f"{symbol}:{year}")
        override_pick = None
        if override:
            override_pick = next((r for r in options if str(r.get("id")) == str(override["report_id"])), None)
        pick = (override_pick or max(pool, key=report_score)).copy()
        pick.update({
            "symbol": symbol,
            "jalali_year": year,
            "selection_status": "selected",
            "is_audited": "حسابرسی شده" in (pick.get("subtitle") or []),
            "is_amended": "اصلاحیه" in (pick.get("subtitle") or []),
            "has_separate_scope": "اصلی (غیر تلفیقی)" in (pick.get("subtitle") or []),
            "pdf_url": REPORT_PDF_URL.format(image_id=pick["image_id"]),
            "codal_portal_url": CODAL_PORTAL_URL,
            "selection_override": bool(override_pick),
            "selection_reason": (
                override.get("reason") if override_pick else
                ("گزارش حسابرسی‌شده متناظر در نمایه موجود نبود؛ نسخه اصلی ۱۲ماهه انتخاب شد."
                 if "حسابرسی شده" not in (pick.get("subtitle") or []) else
                 "اولویت: اصلی، حسابرسی‌شده، اصلاحیه، آخرین انتشار")
            ),
        })
        selected.append(pick)
    return selected


def download_report(report: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    if report.get("selection_status") != "selected":
        return report
    pdf_path = PDF_DIR / f"{report['symbol']}_{report['jalali_year']}_{report['id']}.pdf"
    report["pdf_path"] = str(pdf_path.relative_to(ROOT))
    try:
        if not pdf_path.exists() or refresh:
            pdf_path.write_bytes(http_get_bytes(report["pdf_url"], timeout=240))
        if not pdf_path.read_bytes()[:5] == b"%PDF-":
            raise ValueError("downloaded file is not a PDF")
        report["pdf_bytes"] = pdf_path.stat().st_size
        report["pdf_sha256"] = sha256_file(pdf_path)
        report["download_status"] = "ok"
    except Exception as exc:  # retained in audit output; not silently swallowed
        report["download_status"] = "error"
        report["download_error"] = f"{type(exc).__name__}: {exc}"
    return report


def extract_pdf_text(report: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    if report.get("download_status") != "ok":
        return report
    pdf_path = ROOT / report["pdf_path"]
    txt_path = TEXT_DIR / f"{pdf_path.stem}.txt"
    report["text_path"] = str(txt_path.relative_to(ROOT))
    try:
        if not txt_path.exists() or refresh:
            subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True, capture_output=True)
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
        report["text_chars"] = len(raw)
        report["text_status"] = "ok" if len(raw.strip()) >= 1000 else "insufficient_text"
    except Exception as exc:
        report["text_status"] = "error"
        report["text_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _parse_number(token: str) -> float:
    cleaned = token.replace("(", "-").replace(")", "").replace(",", "").replace("٬", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return float(cleaned)


def _candidate_statement_blocks(lines: list[str]) -> list[tuple[int, int]]:
    starts = [
        i for i, line in enumerate(lines)
        if "صورت وضعیت مالی" in line or line.strip() in {"ترازنامه", "ترازنامه تلفیقی"}
    ]
    blocks = []
    for start in starts:
        end = min(len(lines), start + 180)
        for j in range(start + 15, end):
            if "صورت تغییرات در حقوق مالکانه" in lines[j] or "صورت سود و زیان" in lines[j]:
                end = j
                break
        blocks.append((start, end))
    return blocks


def _compact_label_position(line: str, aliases: list[str]) -> int | None:
    compact_chars: list[str] = []
    positions: list[int] = []
    for i, char in enumerate(line):
        if not char.isspace():
            compact_chars.append(char)
            positions.append(i)
    compact = "".join(compact_chars)
    for alias in aliases:
        idx = compact.find(alias)
        if idx >= 0:
            return positions[idx]
    return None


def _label_match(line: str, patterns: list[str]) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match
    return None


def _extract_field_from_line(field: str, line: str, next_line: str = "") -> tuple[float, str] | None:
    compact_line = "".join(char for char in line if not char.isspace())
    if field == "total_liabilities" and (
        "جمعبدهیهاوحقوقمالکانه" in compact_line or "جمعبدهیهاوحقوقصاحبانسهام" in compact_line
        or "جمعبدهیهایقبلاز" in compact_line or "جمعبدهیهاقبلاز" in compact_line
    ):
        return None
    if field == "investment_depositor_rights" and "قبلازحقوقصاحبانسپرده" in compact_line:
        return None
    match = _label_match(line, FIELD_PATTERNS[field])
    label_position = match.start() if match else _compact_label_position(
        line, FIELD_ALIASES_COMPACT.get(field, [])
    )
    if label_position is None:
        return None
    nums = list(NUM_RE.finditer(line[:label_position]))
    if nums:
        return _parse_number(nums[-1].group(0)), line
    # Some 1398 statements wrap a long label and place the value at the start
    # of the next printed line.
    next_nums = list(NUM_RE.finditer(next_line))
    if next_nums:
        return _parse_number(next_nums[0].group(0)), f"{line} | {next_line}"
    return None


def _complete_liability_side(
    lines: list[str], found: dict[str, tuple[float, int, str]]
) -> dict[str, tuple[float, int, str]]:
    """Recover a separately printed liability/equity page in 1398 filings."""
    total_assets = found.get("total_assets", (math.nan, 0, ""))[0]
    if not math.isfinite(total_assets) or total_assets <= 0:
        return found
    candidates: list[tuple[float, int, float]] = []
    for i, line in enumerate(lines):
        parsed = _extract_field_from_line(
            "liabilities_and_equity", line, lines[i + 1] if i + 1 < len(lines) else ""
        )
        if parsed:
            value, _ = parsed
            candidates.append((abs(value - total_assets) / total_assets, i, value))
    if not candidates:
        return found
    gap, anchor, _ = min(candidates, key=lambda x: x[0])
    if gap > 0.01:
        return found
    start, end = max(0, anchor - 45), min(len(lines), anchor + 5)
    # The consolidated and parent liability pages can be adjacent.  Start at
    # the nearest liability heading before the matched balancing total.
    headings = [i for i in range(start, anchor) if lines[i].strip() == "بدهی ها"]
    if headings:
        start = headings[-1]
    supplemental_fields = [
        "customer_deposits", "deposits_sight", "deposits_savings", "deposits_term", "deposits_other",
        "total_liabilities", "total_equity", "liabilities_and_equity",
    ]
    for i in range(start, end):
        for field in supplemental_fields:
            if field in found:
                continue
            parsed = _extract_field_from_line(
                field, lines[i], lines[i + 1] if i + 1 < len(lines) else ""
            )
            if parsed:
                item_value, source = parsed
                found[field] = (item_value, i + 1, source)
    return found


def parse_statement(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {
        "symbol": report.get("symbol"), "jalali_year": report.get("jalali_year"),
        "report_id": report.get("id"), "image_id": report.get("image_id"),
        "publication_date": str(report.get("date", ""))[:10],
        "fiscal_year_gregorian": str(report.get("fiscal_year", ""))[:10],
        "is_audited": report.get("is_audited", False),
        "is_amended": report.get("is_amended", False),
        "has_separate_scope": report.get("has_separate_scope", False),
        "scope_used": "separate_parent",
        "unit": "million_rial", "pdf_url": report.get("pdf_url"),
        "pdf_sha256": report.get("pdf_sha256"),
    }
    if report.get("text_status") != "ok":
        return {**base, "parse_status": report.get("text_status", "not_available")}, []

    txt_path = ROOT / report["text_path"]
    normalized = normalize_fa(txt_path.read_text(encoding="utf-8", errors="replace"))
    lines = [normalize_fa(line) for line in normalized.splitlines()]
    blocks = _candidate_statement_blocks(lines)
    if int(report.get("jalali_year") or 0) >= 1399:
        modern_blocks = [(start, end) for start, end in blocks if "صورت وضعیت مالی" in lines[start]]
        if modern_blocks:
            blocks = modern_blocks
    best: tuple[int, int, dict[str, tuple[float, int, str]]] | None = None

    for start, end in blocks:
        found: dict[str, tuple[float, int, str]] = {}
        for line_no in range(start, end):
            line = lines[line_no]
            for field, patterns in FIELD_PATTERNS.items():
                if field in found:
                    continue
                parsed = _extract_field_from_line(
                    field, line, lines[line_no + 1] if line_no + 1 < end else ""
                )
                if not parsed:
                    continue
                value, source = parsed
                found[field] = (value, line_no + 1, source)
        found = _complete_liability_side(lines, found)
        scope_bonus = 5 if "تلفیقی" not in lines[start] else 0
        score = sum(field in found for field in REQUIRED_FIELDS) * 10 + len(found) + scope_bonus
        if best is None or score > best[0]:
            best = (score, start, found)

    if best is None:
        return {**base, "parse_status": "statement_not_found"}, []

    score, statement_start, found = best
    if not found:
        return {**base, "parse_status": "fields_not_found"}, []
    row = {
        **base, "parse_status": "ok", "statement_text_line": statement_start + 1,
        "statement_heading": lines[statement_start], "parser_score": score,
    }
    raw_items: list[dict[str, Any]] = []
    for field, (value, line_no, line) in found.items():
        row[field] = value
        raw_items.append({
            "symbol": row["symbol"], "jalali_year": row["jalali_year"], "report_id": row["report_id"],
            "field": field, "value_million_rial": value, "source_text_line": line_no,
            "source_line": line[:500], "pdf_url": row["pdf_url"], "pdf_sha256": row["pdf_sha256"],
        })
    return row, raw_items


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.div(den.where(den.ne(0)))


def calculate_metrics(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    numeric = list(FIELD_PATTERNS)
    for col in numeric:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    old_deposit_components = ["deposits_sight", "deposits_savings", "deposits_term", "deposits_other"]
    old_deposit_sum = df[old_deposit_components].sum(axis=1, min_count=1)
    df["customer_deposits"] = df["customer_deposits"].fillna(old_deposit_sum)
    # In the 1399-1400 banking template, investment-depositor balances are
    # presented as a distinct claim rather than inside "customer deposits".
    # Total customer funding restores a comparable denominator across templates.
    df["customer_funding"] = df["customer_deposits"] + df["investment_depositor_rights"].fillna(0)
    df["gross_loans_proxy"] = df[["loans_government", "loans_nongovernment"]].sum(axis=1, min_count=1)
    df["gross_loans_proxy"] = df["gross_loans_proxy"].fillna(df["loans_total"])
    df["bank_receivables"] = df[["due_from_banks", "due_from_central_bank"]].sum(axis=1, min_count=1)
    df["cash_and_bank_assets"] = df[["cash", "bank_receivables"]].sum(axis=1, min_count=2)
    df["cash_bank_assets_to_deposits"] = _safe_div(df["cash_and_bank_assets"], df["customer_funding"])
    df["statutory_reserve_to_deposits"] = _safe_div(df["statutory_deposit"], df["customer_funding"])
    df["loan_to_deposit_proxy"] = _safe_div(df["gross_loans_proxy"], df["customer_funding"])
    df["equity_to_assets"] = _safe_div(df["total_equity"], df["total_assets"])
    df["loans_to_assets"] = _safe_div(df["gross_loans_proxy"], df["total_assets"])
    df["balance_gap"] = df["total_assets"] - df["liabilities_and_equity"].fillna(df["total_liabilities"] + df["total_equity"])
    df["balance_gap_pct"] = _safe_div(df["balance_gap"].abs(), df["total_assets"].abs())
    df["required_field_count"] = df[REQUIRED_FIELDS].notna().sum(axis=1)
    df["qc_pass"] = (
        df["parse_status"].eq("ok") & df["required_field_count"].ge(6) &
        df["total_assets"].gt(0) & df["customer_funding"].gt(0) &
        df["balance_gap_pct"].le(0.01)
    )

    scenario_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for name, assumptions in config["scenario_assumptions"].items():
            result = evaluate(r.to_dict(), Scenario(**assumptions))
            scenario_rows.append({
                "symbol": r["symbol"], "jalali_year": r["jalali_year"], "scenario": name,
                **result,
            })
    scenarios = pd.DataFrame(scenario_rows)

    checks: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for check_name, ok, detail in [
            ("required_fields", r["required_field_count"] >= 6, f"{int(r['required_field_count'])}/6"),
            ("positive_assets", bool(r["total_assets"] > 0), str(r["total_assets"])),
            ("positive_customer_funding", bool(r["customer_funding"] > 0), str(r["customer_funding"])),
            ("balance_equation_1pct", bool(r["balance_gap_pct"] <= 0.01), f"{r['balance_gap_pct']:.6f}"),
            ("separate_scope_metadata", bool(r["has_separate_scope"]), str(r["has_separate_scope"])),
            ("nonconsolidated_heading", "تلفیقی" not in str(r.get("statement_heading", "")), str(r.get("statement_heading", ""))),
        ]:
            checks.append({"symbol": r["symbol"], "jalali_year": r["jalali_year"], "check": check_name, "pass": bool(ok), "detail": detail})
    return df, scenarios, pd.DataFrame(checks)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def run(refresh_index: bool = False, refresh_pdfs: bool = False, workers: int = 4) -> None:
    ensure_dirs()
    config = load_config()
    selected: list[dict[str, Any]] = []
    for symbol, bank in config["banks"].items():
        rows = fetch_report_index(symbol, bank["rahavard_asset_id"], refresh=refresh_index)
        selected.extend(select_annual_reports(
            rows, symbol, config["jalali_years"], config.get("report_overrides")
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        selected = list(pool.map(lambda r: download_report(r, refresh_pdfs), selected))
    for report in selected:
        extract_pdf_text(report, refresh=refresh_pdfs)

    report_df = pd.DataFrame(selected)
    report_df["subtitle"] = report_df.get("subtitle", pd.Series(dtype=object)).apply(
        lambda x: " | ".join(x) if isinstance(x, list) else x
    )
    save_csv(report_df, RAW_DIR / "report_index_selected.csv")

    rows, items = [], []
    for report in selected:
        row, raw_items = parse_statement(report)
        rows.append(row)
        items.extend(raw_items)
    raw = pd.DataFrame(rows).sort_values(["symbol", "jalali_year"])
    raw_items = pd.DataFrame(items).sort_values(["symbol", "jalali_year", "field"])
    save_csv(raw, RAW_DIR / "statement_values.csv")
    save_csv(raw_items, RAW_DIR / "source_line_items.csv")

    processed, scenarios, checks = calculate_metrics(raw, config)
    save_csv(processed, PROCESSED_DIR / "bank_year_metrics.csv")
    save_csv(scenarios, PROCESSED_DIR / "stress_scenarios.csv")
    save_csv(checks, PROCESSED_DIR / "qc_checks.csv")
    summary = {
        "bank_years_expected": len(config["banks"]) * len(config["jalali_years"]),
        "reports_selected": int(report_df["selection_status"].eq("selected").sum()),
        "pdf_download_ok": int(report_df.get("download_status", pd.Series(dtype=str)).eq("ok").sum()),
        "parse_ok": int(processed["parse_status"].eq("ok").sum()),
        "qc_pass": int(processed["qc_pass"].sum()),
        "audited_reports": int(processed["is_audited"].sum()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (PROCESSED_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--refresh-pdfs", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.refresh_index, args.refresh_pdfs, args.workers)


if __name__ == "__main__":
    main()
