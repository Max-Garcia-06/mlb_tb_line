"""
CLV- and fill-aware segment health for go/no-go trading decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import (
    SEGMENT_MIN_AVG_CLV,
    SEGMENT_MIN_FILL_RATE,
    SEGMENT_MIN_FILLS,
    SEGMENT_MIN_ROI_PCT,
)
from journal_reader import (
    index_fills_by_order_id,
    index_marks_by_order_and_label,
    journal_paths_in_date_range,
    load_window_rows,
    parse_iso_date,
)
from reporting_common import edge_bucket, line_bucket, pnl_per_contract, spread_bucket

log = logging.getLogger(__name__)


@dataclass
class SegmentMetrics:
    side: str
    line: str
    spread: str
    edge: str
    orders: int = 0
    fills: int = 0
    contracts: int = 0
    cost: float = 0.0
    realized_pnl: float = 0.0
    clv_sum: float = 0.0
    clv_contracts: int = 0

    @property
    def fill_rate(self) -> float:
        return float(self.fills) / float(self.orders) if self.orders > 0 else 0.0

    @property
    def roi_pct(self) -> float:
        return (self.realized_pnl / self.cost * 100.0) if self.cost > 0 else 0.0

    @property
    def avg_clv(self) -> float | None:
        if self.clv_contracts <= 0:
            return None
        return self.clv_sum / float(self.clv_contracts)

    @property
    def segment_key(self) -> str:
        return f"{self.side}|{self.line}|{self.spread}|{self.edge}"


@dataclass
class SegmentVerdict:
    metrics: SegmentMetrics
    status: str  # PASS | FAIL | INSUFFICIENT
    reasons: list[str] = field(default_factory=list)


@dataclass
class SegmentHealthReport:
    segments: list[SegmentVerdict]
    recommendation: str  # TRADE | PAUSE
    summary_reasons: list[str] = field(default_factory=list)


def _latest_mark_for_order(marks: dict[tuple[str, str], dict[str, Any]], order_id: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_ts = ""
    for (oid, _lbl), row in marks.items():
        if oid != order_id:
            continue
        ts = str(row.get("ts", "") or "")
        if best is None or ts >= best_ts:
            best = row
            best_ts = ts
    return best


def _clv_from_mark(side: str, entry: float, mark_row: dict[str, Any], contracts: int) -> tuple[float, int]:
    side_l = str(side or "").lower()
    if side_l == "yes":
        mid = float(mark_row.get("mark_yes_mid") or 0.0)
    else:
        mid = float(mark_row.get("mark_no_mid") or 0.0)
    if mid <= 0 or entry <= 0 or contracts <= 0:
        return 0.0, 0
    return (mid - entry) * contracts, contracts


def aggregate_segments(
    placed: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
    mark_rows: list[dict[str, Any]],
    *,
    outcome_for: callable,
) -> dict[str, SegmentMetrics]:
    fills_by_oid = index_fills_by_order_id(fill_rows)
    marks_index = index_marks_by_order_and_label(mark_rows)

    # Map post-submit by order_id for spread/edge context
    submit_by_oid: dict[str, dict[str, Any]] = {}
    for r in placed:
        oid = str(r.get("order_id", "") or "")
        if oid:
            submit_by_oid[oid] = r

    segments: dict[str, SegmentMetrics] = {}

    for r in placed:
        side = str(r.get("side", "") or "").lower()
        line = line_bucket(float(r.get("kalshi_line", r.get("line", 1.5)) or 1.5))
        spread = spread_bucket(float(r.get("book_spread", 0.0) or 0.0))
        edge = edge_bucket(float(r.get("edge", 0.0) or 0.0))
        key = f"{side}|{line}|{spread}|{edge}"
        seg = segments.get(key)
        if seg is None:
            seg = SegmentMetrics(side=side, line=line, spread=spread, edge=edge)
            segments[key] = seg
        seg.orders += 1

    for oid, fill in fills_by_oid.items():
        filled = int(fill.get("filled_contracts", 0) or 0)
        if filled <= 0:
            continue
        sub = submit_by_oid.get(oid) or fill
        side = str(sub.get("side", fill.get("side", "")) or "").lower()
        line = line_bucket(float(sub.get("kalshi_line", sub.get("line", 1.5)) or 1.5))
        spread = spread_bucket(float(sub.get("book_spread", 0.0) or 0.0))
        edge = edge_bucket(float(sub.get("edge", 0.0) or 0.0))
        key = f"{side}|{line}|{spread}|{edge}"
        seg = segments.get(key)
        if seg is None:
            seg = SegmentMetrics(side=side, line=line, spread=spread, edge=edge)
            segments[key] = seg

        price = float(fill.get("avg_fill_price", 0.0) or 0.0)
        if price <= 0:
            price = float(sub.get("limit_price", 0.0) or 0.0)
        seg.fills += 1
        seg.contracts += filled
        seg.cost += price * filled

        ticker = str(sub.get("ticker", fill.get("ticker", "")) or "")
        res = outcome_for(ticker)
        pnlpc = pnl_per_contract(side, price, res)
        if pnlpc is not None:
            seg.realized_pnl += pnlpc * filled

        mk = _latest_mark_for_order(marks_index, oid)
        if mk:
            clv_add, clv_ctr = _clv_from_mark(side, price, mk, filled)
            seg.clv_sum += clv_add
            seg.clv_contracts += clv_ctr

    return segments


def judge_segment(m: SegmentMetrics) -> SegmentVerdict:
    reasons: list[str] = []
    if m.fills < int(SEGMENT_MIN_FILLS):
        return SegmentVerdict(
            metrics=m,
            status="INSUFFICIENT",
            reasons=[f"fills {m.fills} < min {SEGMENT_MIN_FILLS}"],
        )
    if m.fill_rate < float(SEGMENT_MIN_FILL_RATE):
        reasons.append(f"fill_rate {m.fill_rate:.2f} < {SEGMENT_MIN_FILL_RATE}")
    avg_clv = m.avg_clv
    if avg_clv is not None and avg_clv < float(SEGMENT_MIN_AVG_CLV):
        reasons.append(f"avg_clv {avg_clv:+.4f} < {SEGMENT_MIN_AVG_CLV}")
    elif avg_clv is None:
        reasons.append("no mark CLV data")
    if m.cost > 0 and m.roi_pct < float(SEGMENT_MIN_ROI_PCT):
        reasons.append(f"roi {m.roi_pct:.1f}% < {SEGMENT_MIN_ROI_PCT}%")
    status = "PASS" if not reasons else "FAIL"
    return SegmentVerdict(metrics=m, status=status, reasons=reasons)


def build_segment_health_report(
    segments: dict[str, SegmentMetrics],
) -> SegmentHealthReport:
    verdicts = [judge_segment(m) for m in segments.values()]
    verdicts.sort(key=lambda v: (v.status != "PASS", -v.metrics.fills, v.metrics.segment_key))

    passing = [v for v in verdicts if v.status == "PASS"]
    if passing:
        return SegmentHealthReport(
            segments=verdicts,
            recommendation="TRADE",
            summary_reasons=[f"{len(passing)} segment(s) PASS with sufficient fills"],
        )

    fail_reasons: list[str] = []
    for v in verdicts:
        if v.status == "FAIL":
            fail_reasons.append(f"{v.metrics.segment_key}: {'; '.join(v.reasons)}")
        elif v.status == "INSUFFICIENT" and v.metrics.orders > 0:
            fail_reasons.append(f"{v.metrics.segment_key}: {'; '.join(v.reasons)}")
    if not fail_reasons:
        fail_reasons.append("no filled segments in window")
    return SegmentHealthReport(
        segments=verdicts,
        recommendation="PAUSE",
        summary_reasons=fail_reasons[:8],
    )


def segment_health_for_range(
    start: str,
    end: str,
    *,
    client: object | None = None,
) -> SegmentHealthReport:
    start_d = parse_iso_date(start)
    end_d = parse_iso_date(end)
    paths = journal_paths_in_date_range(start_d, end_d)
    placed, fill_rows, mark_rows = load_window_rows(paths)

    tickers: set[str] = set()
    for r in placed + fill_rows:
        t = str(r.get("ticker", "") or "")
        if t:
            tickers.add(t)

    markets: dict[str, dict] = {}
    if client is not None and hasattr(client, "get_market"):
        for t in tickers:
            try:
                markets[t] = client.get_market(t) or {}
            except Exception:
                markets[t] = {}

    from reporting_common import market_yes_no_result

    def outcome_for(ticker: str) -> str:
        return market_yes_no_result(markets.get(ticker, {}))

    segments = aggregate_segments(placed, fill_rows, mark_rows, outcome_for=outcome_for)
    return build_segment_health_report(segments)
