"""
Sync Kalshi order status into trade journals as ``note=fill`` rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from journal_reader import (
    existing_fill_order_ids,
    journal_paths_in_date_range,
    load_jsonl_rows,
    placed_with_order_id,
)
from trade_journal import TradeRow, append_row, journal_path

log = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    updated: int
    skipped: int
    errors: int
    days_processed: int = 0

    def merge(self, other: ReconcileResult) -> ReconcileResult:
        return ReconcileResult(
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            errors=self.errors + other.errors,
            days_processed=self.days_processed + other.days_processed,
        )


def _avg_fill_price_from_order(order: dict[str, Any], placed_row: dict[str, Any]) -> float:
    for k in ("avg_fill_price", "average_fill_price", "avg_price", "fill_price"):
        if order.get(k) is not None:
            return float(order[k])
    return float(placed_row.get("limit_price", 0.0))


def reconcile_fills_for_date(
    game_date: str,
    *,
    client: Any,
    include_resting: bool = False,
) -> ReconcileResult:
    """
    Write ``note=fill`` rows for journaled orders not yet reconciled on ``game_date``.

    Idempotent per ``order_id``. Returns zeros if journal missing or no placed orders.
    """
    path = journal_path(game_date)
    if not path.exists():
        return ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)

    rows = load_jsonl_rows(path)
    placed = placed_with_order_id(rows)
    if not placed:
        return ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)

    existing_fill_ids = existing_fill_order_ids(rows)
    updated = 0
    skipped = 0
    errors = 0

    for r in placed:
        order_id = str(r.get("order_id"))
        if order_id in existing_fill_ids:
            skipped += 1
            continue
        try:
            o = client.get_order(order_id)
        except Exception as e:
            log.warning("Could not fetch order %s (%s)", order_id, e)
            errors += 1
            continue

        count = int(o.get("count", r.get("contracts", 0)) or 0)
        remaining = int(o.get("remaining_count", o.get("remaining", 0)) or 0)
        status = str(o.get("status", "") or "").lower()
        filled = max(0, count - remaining)

        if filled <= 0 and not include_resting:
            continue

        append_row(
            game_date,
            TradeRow(
                game_date=game_date,
                ticker=str(r.get("ticker", "")),
                side=str(r.get("side", "")),
                action=str(r.get("action", "buy")),
                contracts=int(r.get("contracts", 0)),
                limit_price=float(r.get("limit_price", 0.0)),
                order_id=order_id,
                player_name=str(r.get("player_name", "")),
                kalshi_line=float(r.get("kalshi_line", 0.0)),
                predicted_lambda=float(r.get("predicted_lambda", 0.0)),
                p_model=float(r.get("p_model", 0.0)),
                p_model_raw=float(r.get("p_model_raw", 0.0) or 0.0),
                p_model_cal=float(r.get("p_model_cal", 0.0) or 0.0),
                p_market=float(r.get("p_market", 0.0)),
                fee_per_contract=float(r.get("fee_per_contract", 0.0) or 0.0),
                edge=float(r.get("edge", 0.0)),
                ev=float(r.get("ev", 0.0)),
                expected_pnl=float(r.get("expected_pnl", 0.0)),
                book_bid=float(r.get("book_bid", 0.0)),
                book_ask=float(r.get("book_ask", 0.0)),
                book_spread=float(r.get("book_spread", 0.0)),
                filled_contracts=int(filled),
                avg_fill_price=_avg_fill_price_from_order(o, r),
                note="fill",
                success=True if status in {"executed", "filled"} else None,
            ).to_dict(),
        )
        updated += 1

    return ReconcileResult(
        updated=updated,
        skipped=skipped,
        errors=errors,
        days_processed=1,
    )


def reconcile_fills_for_range(
    start_d: datetime,
    end_d: datetime,
    *,
    client: Any,
    include_resting: bool = False,
) -> ReconcileResult:
    """Reconcile each journal day in ``[start_d, end_d]`` (inclusive)."""
    total = ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)
    for _path, date_str in journal_paths_in_date_range(start_d, end_d):
        day_result = reconcile_fills_for_date(
            date_str,
            client=client,
            include_resting=include_resting,
        )
        if day_result.days_processed > 0:
            total = total.merge(day_result)
    return total
