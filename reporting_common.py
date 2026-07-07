"""Shared helpers for trade report / report-range (buckets, PnL, outcomes)."""

from __future__ import annotations

from fees import order_fee_usd


def pnl_per_contract(side: str, price: float, result: str) -> float | None:
    if result not in {"yes", "no"}:
        return None
    return (1.0 - price) if side == result else (-price)


def estimated_order_fee_usd(row: dict, price: float, contracts: int) -> float:
    """
    Kalshi fee for a filled trade. Rows journaled after 2026-07-06 carry
    ``fee_per_contract`` (0.0 for resting/maker fills); older rows always
    crossed at the ask, so estimate the taker fee at the fill price.
    """
    if "fee_per_contract" in row:
        return float(row.get("fee_per_contract", 0.0) or 0.0) * int(contracts)
    return order_fee_usd(price, contracts, is_taker=True)


def market_yes_no_result(market: dict) -> str:
    return str((market or {}).get("result") or "").lower()


def line_bucket(line: float) -> str:
    """Kalshi TB strike bucket for segment reporting."""
    ln = float(line)
    if ln <= 0.5:
        return "0.5"
    if ln <= 1.5:
        return "1.5"
    if ln <= 2.5:
        return "2.5"
    return "3.5+"


def price_bucket(price: float) -> str:
    if price < 0.20:
        return "<0.20"
    if price < 0.50:
        return "0.20-0.49"
    if price < 0.80:
        return "0.50-0.79"
    return ">=0.80"


def edge_bucket(edge: float) -> str:
    if edge < 0.05:
        return "<0.05"
    if edge < 0.10:
        return "0.05-0.09"
    if edge < 0.15:
        return "0.10-0.14"
    if edge < 0.20:
        return "0.15-0.19"
    return ">=0.20"


def spread_bucket(spread: float) -> str:
    if spread <= 0:
        return "n/a"
    if spread < 0.05:
        return "<0.05"
    if spread < 0.10:
        return "0.05-0.09"
    if spread < 0.20:
        return "0.10-0.19"
    return ">=0.20"
