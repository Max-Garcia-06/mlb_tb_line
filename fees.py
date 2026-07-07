"""
fees.py
-------
Kalshi trading-fee model.

Taker fee per Kalshi fee schedule: ``ceil(rate * C * P * (1-P))`` cents per
order, with rate = 0.07 for most series. Resting (maker) orders pay
``KALSHI_MAKER_FEE_RATE`` (0.0 by default).
"""

from __future__ import annotations

import math

from config import KALSHI_MAKER_FEE_RATE, KALSHI_TAKER_FEE_RATE


def fee_per_contract(price: float, *, is_taker: bool = True) -> float:
    """Expected fee in USD per contract at ``price`` (un-ceiled, for EV math)."""
    p = float(min(0.99, max(0.01, price)))
    rate = KALSHI_TAKER_FEE_RATE if is_taker else KALSHI_MAKER_FEE_RATE
    return float(rate) * p * (1.0 - p)


def order_fee_usd(price: float, contracts: int, *, is_taker: bool = True) -> float:
    """Actual fee in USD for one order: per-order ceil to the next cent."""
    n = int(contracts)
    if n <= 0:
        return 0.0
    raw_cents = 100.0 * fee_per_contract(price, is_taker=is_taker) * n
    return math.ceil(raw_cents - 1e-9) / 100.0
