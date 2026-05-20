"""
Shared helpers for reading trade journals (JSONL) and common indexes used by report / reconcile / mark.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATA_DIR


def parse_iso_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def date_from_journal_filename(path: Path) -> str | None:
    name = path.name
    if not (name.startswith("trades_") and name.endswith(".jsonl")):
        return None
    return name[len("trades_") : -len(".jsonl")]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def journal_paths_sorted() -> list[Path]:
    return sorted(DATA_DIR.glob("trades_*.jsonl"))


def journal_paths_in_date_range(start_d: datetime, end_d: datetime) -> list[tuple[Path, str]]:
    """Paths whose embedded YYYY-MM-DD falls in [start_d, end_d] inclusive."""
    out: list[tuple[Path, str]] = []
    for p in journal_paths_sorted():
        d = date_from_journal_filename(p)
        if not d:
            continue
        try:
            dd = parse_iso_date(d)
        except Exception:
            continue
        if dd < start_d or dd > end_d:
            continue
        out.append((p, d))
    return out


def placed_post_submit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("note") == "post-submit" and r.get("success") is True]


def placed_with_order_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("note") == "post-submit" and r.get("success") is True and r.get("order_id")]


def index_fills_by_order_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("note") == "fill" and r.get("order_id"):
            out[str(r["order_id"])] = r
    return out


def index_marks_by_order_and_label(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("note") == "mark" and r.get("order_id"):
            out[(str(r["order_id"]), str(r.get("mark_label", "")))] = r
    return out


def existing_fill_order_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(r.get("order_id")) for r in rows if r.get("note") == "fill" and r.get("order_id")}


def load_window_rows(paths_with_dates: list[tuple[Path, str]]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    From (path, embedded_date) pairs, load post-submit (with game_date), fill, and mark rows.
    """
    placed: list[dict] = []
    fill_rows: list[dict] = []
    mark_rows: list[dict] = []
    for p, d in paths_with_dates:
        for r in load_jsonl_rows(p):
            if r.get("note") == "post-submit" and r.get("success") is True:
                rr = dict(r)
                rr.setdefault("game_date", d)
                placed.append(rr)
            if r.get("note") == "fill" and r.get("order_id"):
                rr = dict(r)
                rr.setdefault("game_date", d)
                fill_rows.append(rr)
            if r.get("note") == "mark" and r.get("order_id"):
                rr = dict(r)
                rr.setdefault("game_date", d)
                mark_rows.append(rr)
    return placed, fill_rows, mark_rows
