from __future__ import annotations

from edge_detector import EdgeSignal
from trading_stack import fill_probability, finalize_signals


def _sig(ev: float, player: str) -> EdgeSignal:
    return EdgeSignal(
        player_name=player,
        player_id=1,
        game_date="2024-06-01",
        ticker="T",
        kalshi_line=1.5,
        predicted_lambda=2.0,
        p_model=0.6,
        p_model_raw=0.6,
        p_market=0.5,
        edge=0.1,
        ev=ev,
        kelly_f=0.05,
        recommended_contracts=10,
        recommended_side="yes",
        bet_dollars=5.0,
        limit_price=0.5,
        book_bid=0.48,
        book_ask=0.52,
        book_spread=0.04,
    )


def test_finalize_one_per_player():
    s = finalize_signals([_sig(0.1, "A"), _sig(0.2, "A"), _sig(0.3, "B")], bankroll=1000, one_per_player=True)
    names = {x.player_name for x in s}
    assert names == {"A", "B"}
    assert next(x for x in s if x.player_name == "A").ev == 0.2


def test_fill_probability_tighter_spread():
    assert fill_probability(0.02) > fill_probability(0.20)
