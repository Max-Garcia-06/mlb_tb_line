"""
Shared post-edge pipeline for live scan and backtest (sizing parity).
"""

from __future__ import annotations

from typing import Any

from capital_optimizer import resize_signals_portfolio
from edge_detector import EdgeSignal, dollars_to_contracts


def finalize_signals(
    signals: list[EdgeSignal],
    *,
    bankroll: float,
    one_per_player: bool = True,
    max_signals: int | None = None,
    max_contracts: int | None = None,
) -> list[EdgeSignal]:
    """Mirror live scan sizing: best per player, cap count, portfolio resize, contract caps."""
    if not signals:
        return []

    if one_per_player:
        seen: dict[str, EdgeSignal] = {}
        for s in signals:
            if s.player_name not in seen or s.ev > seen[s.player_name].ev:
                seen[s.player_name] = s
        signals = sorted(seen.values(), key=lambda x: x.ev, reverse=True)

    if max_signals and len(signals) > max_signals:
        signals = signals[: int(max_signals)]

    resize_signals_portfolio(signals, float(bankroll))

    for s in signals:
        if max_contracts:
            s.recommended_contracts = min(int(s.recommended_contracts), int(max_contracts))
        s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    total_raw = sum(s.bet_dollars for s in signals)
    if total_raw > bankroll and total_raw > 0:
        scale = bankroll / total_raw
        for s in signals:
            s.bet_dollars = round(s.bet_dollars * scale, 2)
            s.recommended_contracts = dollars_to_contracts(s.bet_dollars, s.limit_price)
            if max_contracts:
                s.recommended_contracts = min(int(s.recommended_contracts), int(max_contracts))
            s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    return [s for s in signals if s.recommended_contracts > 0]


def fill_probability(spread: float, *, side: str = "yes") -> float:
    """Heuristic limit-fill probability from book spread (tighter = more likely)."""
    sp = max(0.0, float(spread))
    base = max(0.25, 1.0 - sp / 0.25)
    if str(side).lower() == "no" and sp > 0.12:
        base *= 0.92
    return min(1.0, max(0.0, base))
