import pytest

import edge_detector
from edge_detector import detect_edge, is_blocked_segment, quote_side_edge
from kalshi_bridge import MarketLine
from probability_engine import ProbabilityResult


@pytest.fixture(autouse=True)
def _model_only_blend(monkeypatch):
    """Neutralize the blend (w=1) so gate tests exercise one thing at a time."""
    monkeypatch.setattr(edge_detector, "load_blend_weight", lambda: 1.0)


def _pr(p_over: float, line: float = 1.5) -> ProbabilityResult:
    pr = ProbabilityResult(
        player_id=1,
        player_name="A",
        game_date="2026-07-06",
        kalshi_line=line,
        predicted_lambda=2.0,
        p_over=p_over,
        p_under=1.0 - p_over,
        distribution="poisson",
        variance=1.5,
    )
    pr.p_over_calibrated = p_over
    pr.p_under_calibrated = 1.0 - p_over
    return pr


def _ml(line: float = 1.5, yes_bid: float = 0.48, yes_ask: float = 0.50) -> MarketLine:
    return MarketLine(
        ticker="T",
        player_name="A",
        player_id=1,
        game_date="2026-07-06",
        line=line,
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=round(1.0 - yes_bid, 2),
        no_bid=round(1.0 - yes_ask, 2),
    )


def test_blocked_segment_parsing():
    # default BLOCKED_SEGMENTS from config is {("1.5", "no")}
    assert is_blocked_segment(1.5, "no") == (("1.5", "no") in edge_detector.BLOCKED_SEGMENTS)


def test_blocked_segment_suppresses_no_side(monkeypatch):
    monkeypatch.setattr(edge_detector, "BLOCKED_SEGMENTS", frozenset({("1.5", "no")}))
    # Strong under edge on the 1.5 line: p_under=0.75 vs no_ask=0.52
    sig = detect_edge(_pr(0.25), _ml(), bankroll=1000.0)
    assert sig is None
    # Same trade allowed on the 2.5 line
    sig2 = detect_edge(_pr(0.25, line=2.5), _ml(line=2.5), bankroll=1000.0)
    assert sig2 is not None and sig2.recommended_side == "no"


def test_fee_reduces_edge_for_taker(monkeypatch):
    monkeypatch.setattr(edge_detector, "MAKER_MODE", False)
    p_blend, limit, fee, edge = quote_side_edge(0.60, bid=0.48, ask=0.50, side="yes")
    assert limit == pytest.approx(0.50)  # crosses at the ask
    assert fee == pytest.approx(0.07 * 0.5 * 0.5)
    assert edge == pytest.approx(0.60 - 0.50 - 0.0175)


def test_maker_mode_rests_inside_ask_with_no_fee(monkeypatch):
    monkeypatch.setattr(edge_detector, "MAKER_MODE", True)
    p_blend, limit, fee, edge = quote_side_edge(0.60, bid=0.48, ask=0.50, side="yes")
    assert limit == pytest.approx(0.49)  # one tick inside the ask
    assert fee == 0.0
    assert edge == pytest.approx(0.60 - 0.49)


def test_blend_shrinks_edge_toward_market(monkeypatch):
    monkeypatch.setattr(edge_detector, "load_blend_weight", lambda: 0.5)
    monkeypatch.setattr(edge_detector, "MAKER_MODE", False)
    p_blend, _, _, _ = quote_side_edge(0.70, bid=0.48, ask=0.50, side="yes")
    assert 0.49 < p_blend < 0.70


def test_signal_carries_fee_and_preblend_prob(monkeypatch):
    monkeypatch.setattr(edge_detector, "BLOCKED_SEGMENTS", frozenset())
    monkeypatch.setattr(edge_detector, "MAKER_MODE", False)
    sig = detect_edge(_pr(0.65), _ml(), bankroll=1000.0)
    assert sig is not None
    assert sig.p_model_cal == pytest.approx(0.65)
    assert sig.fee_per_contract > 0.0
    assert sig.edge == pytest.approx(sig.p_model - sig.limit_price - sig.fee_per_contract)
