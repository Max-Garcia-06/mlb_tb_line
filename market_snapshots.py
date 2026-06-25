"""
Persist Kalshi total-bases market books for historical backtest replay.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from config import DATA_DIR
from data_engine import slate_schedule_index
from journal_risk import game_slug_from_ticker
from kalshi_bridge import MarketLine

log = logging.getLogger(__name__)

SNAPSHOT_DIR = DATA_DIR / "snapshots"


@dataclass
class MarketSnapshot:
    captured_at: str
    game_date: str
    ticker: str
    event_ticker: str
    player_name: str
    line: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    volume: int = 0
    open_interest: int = 0
    xref_player_id: str = ""

    @classmethod
    def from_market_line(cls, ml: MarketLine, *, captured_at: str | None = None) -> "MarketSnapshot":
        ts = captured_at or datetime.now(timezone.utc).isoformat()
        et = str(getattr(ml, "event_ticker", "") or "")
        if not et and ml.ticker:
            parts = str(ml.ticker).split("-")
            if len(parts) >= 2:
                et = f"{parts[0]}-{parts[1]}"
        return cls(
            captured_at=ts,
            game_date=str(ml.game_date),
            ticker=str(ml.ticker),
            event_ticker=et,
            player_name=str(ml.player_name),
            line=float(ml.line),
            yes_ask=float(ml.yes_ask),
            yes_bid=float(ml.yes_bid),
            no_ask=float(ml.no_ask),
            no_bid=float(ml.no_bid),
            volume=int(getattr(ml, "volume", 0) or 0),
            open_interest=int(getattr(ml, "open_interest", 0) or 0),
            xref_player_id=str(getattr(ml, "xref_player_id", "") or ""),
        )

    def to_market_line(self) -> MarketLine:
        return MarketLine(
            ticker=self.ticker,
            player_name=self.player_name,
            player_id=0,
            game_date=self.game_date,
            line=float(self.line),
            yes_ask=float(self.yes_ask),
            yes_bid=float(self.yes_bid),
            no_ask=float(self.no_ask),
            no_bid=float(self.no_bid),
            volume=int(self.volume),
            open_interest=int(self.open_interest),
            xref_player_id=self.xref_player_id,
            event_ticker=self.event_ticker,
        )


def snapshot_path(game_date: str) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"tb_markets_{game_date}.jsonl"


def append_snapshots(game_date: str, lines: Iterable[MarketLine], *, captured_at: str | None = None) -> int:
    path = snapshot_path(game_date)
    ts = captured_at or datetime.now(timezone.utc).isoformat()
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for ml in lines:
            snap = MarketSnapshot.from_market_line(ml, captured_at=ts)
            f.write(json.dumps(asdict(snap), separators=(",", ":")) + "\n")
            n += 1
    log.info("Appended %s snapshots to %s", n, path)
    return n


def load_snapshots(
    game_date: str,
    *,
    latest_only: bool = True,
    earliest_only: bool = False,
    as_of: str | None = None,
) -> list[MarketSnapshot]:
    """
    Load snapshots for a date.

    - ``latest_only``: last capture per ticker (close proxy)
    - ``earliest_only``: first capture per ticker (open / pre-game proxy)
    - ``as_of``: ISO timestamp — last capture per ticker with ``captured_at <= as_of``
    """
    path = snapshot_path(game_date)
    if not path.exists():
        return []
    snaps: list[MarketSnapshot] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                snaps.append(MarketSnapshot(**d))
            except Exception:
                continue
    if as_of:
        cutoff = as_of.strip()
        snaps = [s for s in snaps if str(s.captured_at) <= cutoff]
    if not snaps:
        return []
    if earliest_only:
        by_ticker: dict[str, MarketSnapshot] = {}
        for s in sorted(snaps, key=lambda x: x.captured_at):
            if s.ticker not in by_ticker:
                by_ticker[s.ticker] = s
        return list(by_ticker.values())
    if latest_only:
        by_ticker = {}
        for s in sorted(snaps, key=lambda x: x.captured_at):
            by_ticker[s.ticker] = s
        return list(by_ticker.values())
    return snaps


def _parse_captured_at_utc(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_snapshots_within_hours_of_start(
    game_date: str,
    *,
    within_hours: float,
    schedule_index: dict | None = None,
) -> list[MarketSnapshot]:
    """
    Per ticker: latest snapshot with ``start - within_hours <= captured_at <= start`` (UTC).
    Mirrors live ``scan`` timing vs first pitch.
    """
    if within_hours <= 0:
        return load_snapshots(game_date, earliest_only=True)

    path = snapshot_path(game_date)
    if not path.exists():
        return []

    all_snaps: list[MarketSnapshot] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_snaps.append(MarketSnapshot(**json.loads(line)))
            except Exception:
                continue
    if not all_snaps:
        return []

    idx = schedule_index if schedule_index is not None else slate_schedule_index(game_date)
    window_sec = float(within_hours) * 3600.0
    by_ticker: dict[str, MarketSnapshot] = {}
    excluded_no_schedule = 0
    excluded_outside_window = 0

    for s in sorted(all_snaps, key=lambda x: x.captured_at):
        slug = game_slug_from_ticker(s.ticker)
        if not slug:
            excluded_no_schedule += 1
            continue
        row = idx.get(slug)
        if not row:
            excluded_no_schedule += 1
            continue
        start = row.get("start_utc")
        if not isinstance(start, datetime):
            excluded_no_schedule += 1
            continue
        start_utc = start.astimezone(timezone.utc)
        cap = _parse_captured_at_utc(s.captured_at)
        if cap is None:
            excluded_outside_window += 1
            continue
        lo = start_utc - timedelta(seconds=window_sec)
        if cap < lo or cap > start_utc:
            excluded_outside_window += 1
            continue
        prev = by_ticker.get(s.ticker)
        if prev is None or str(s.captured_at) >= str(prev.captured_at):
            by_ticker[s.ticker] = s

    if excluded_no_schedule or excluded_outside_window:
        log.info(
            "Snapshot window %s: kept %s tickers (no_schedule=%s outside_window=%s)",
            game_date,
            len(by_ticker),
            excluded_no_schedule,
            excluded_outside_window,
        )
    return list(by_ticker.values())


def load_snapshots_open_and_close(game_date: str) -> tuple[list[MarketSnapshot], list[MarketSnapshot]]:
    """Return (earliest per ticker, latest per ticker) for CLV studies."""
    return (
        load_snapshots(game_date, earliest_only=True),
        load_snapshots(game_date, latest_only=True),
    )


def list_snapshot_dates() -> list[str]:
    if not SNAPSHOT_DIR.exists():
        return []
    out = []
    for p in sorted(SNAPSHOT_DIR.glob("tb_markets_*.jsonl")):
        stem = p.stem.replace("tb_markets_", "")
        if len(stem) == 10:
            out.append(stem)
    return out
