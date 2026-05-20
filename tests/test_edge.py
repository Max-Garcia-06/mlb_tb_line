from edge_detector import (
    dollars_to_contracts,
    expected_pnl_per_contract_mean_var,
    expected_pnl_usd,
    expected_pnl_usd_std,
    portfolio_expected_pnl_std,
    fill_calibrated_probabilities,
    EdgeSignal,
)
from probability_engine import ProbabilityResult


def test_dollars_to_contracts_floors_without_minimum():
    assert dollars_to_contracts(0.0, 0.5) == 0
    assert dollars_to_contracts(0.1, 0.5) == 0
    assert dollars_to_contracts(1.0, 0.5) == 2


def test_expected_pnl_usd():
    s = EdgeSignal(
        player_name="X",
        player_id=1,
        game_date="2026-01-01",
        ticker="T",
        kalshi_line=1.5,
        predicted_lambda=1.0,
        p_model=0.6,
        p_model_raw=0.55,
        p_market=0.5,
        edge=0.1,
        ev=0.2,
        kelly_f=0.05,
        recommended_contracts=10,
        recommended_side="yes",
        bet_dollars=5.0,
        limit_price=0.5,
        book_bid=0.48,
        book_ask=0.52,
        book_spread=0.04,
    )
    assert abs(expected_pnl_usd(s) - 10 * (0.6 - 0.5)) < 1e-9
    mu, v = expected_pnl_per_contract_mean_var(0.6, 0.5)
    assert abs(mu - 0.1) < 1e-9
    assert abs(v - 0.24) < 1e-9
    assert abs(expected_pnl_usd_std(s) - (10 * v) ** 0.5) < 1e-6


def test_portfolio_expected_pnl_std_independent_sum():
    s1 = EdgeSignal(
        player_name="A",
        player_id=1,
        game_date="2026-01-01",
        ticker="T1",
        kalshi_line=1.5,
        predicted_lambda=1.0,
        p_model=0.6,
        p_model_raw=0.6,
        p_market=0.5,
        edge=0.1,
        ev=0.2,
        kelly_f=0.05,
        recommended_contracts=10,
        recommended_side="yes",
        bet_dollars=5.0,
        limit_price=0.5,
        book_bid=0.48,
        book_ask=0.52,
        book_spread=0.04,
    )
    s2 = EdgeSignal(
        player_name="B",
        player_id=2,
        game_date="2026-01-01",
        ticker="T2",
        kalshi_line=1.5,
        predicted_lambda=1.0,
        p_model=0.5,
        p_model_raw=0.5,
        p_market=0.48,
        edge=0.02,
        ev=0.05,
        kelly_f=0.01,
        recommended_contracts=1,
        recommended_side="yes",
        bet_dollars=0.24,
        limit_price=0.48,
        book_bid=0.46,
        book_ask=0.48,
        book_spread=0.02,
    )
    _, v1 = expected_pnl_per_contract_mean_var(0.6, 0.5)
    _, v2 = expected_pnl_per_contract_mean_var(0.5, 0.48)
    want = (10 * v1 + 1 * v2) ** 0.5
    assert abs(portfolio_expected_pnl_std([s1, s2]) - want) < 1e-6


def test_fill_calibrated_probabilities_sets_fields():
    pr = ProbabilityResult(
        player_id=1,
        player_name="A",
        game_date="2026-01-01",
        kalshi_line=1.5,
        predicted_lambda=1.2,
        p_over=0.4,
        p_under=0.6,
        distribution="poisson",
        variance=1.5,
    )
    fill_calibrated_probabilities([pr])
    assert pr.p_over_calibrated is not None
    assert pr.p_under_calibrated is not None
    assert abs(pr.p_over_calibrated + pr.p_under_calibrated - 1.0) < 1e-6
