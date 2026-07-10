import json

import pytest

from fees import fee_per_contract, order_fee_usd
from market_blend import blend_probability, fit_blend_weight


def test_taker_fee_matches_kalshi_schedule():
    # 0.07 * 0.5 * 0.5 = 0.0175 per contract
    assert fee_per_contract(0.5, is_taker=True) == pytest.approx(0.0175)
    # 10 contracts at 0.5: 17.5c -> ceil to 18c
    assert order_fee_usd(0.5, 10, is_taker=True) == pytest.approx(0.18)
    assert order_fee_usd(0.5, 0, is_taker=True) == 0.0


def test_maker_fee_is_zero_by_default():
    assert fee_per_contract(0.5, is_taker=False) == 0.0
    assert order_fee_usd(0.5, 100, is_taker=False) == 0.0


def test_blend_endpoints_and_monotonicity():
    assert blend_probability(0.8, 0.5, 1.0) == pytest.approx(0.8, abs=1e-6)
    assert blend_probability(0.8, 0.5, 0.0) == pytest.approx(0.5, abs=1e-6)
    mid = blend_probability(0.8, 0.5, 0.5)
    assert 0.5 < mid < 0.8


def test_blend_damps_saturated_calibrator_output():
    # Isotonic calibrators can emit exactly 1.0; a low w must not let that
    # extreme logit manufacture an edge over the market mid.
    p = blend_probability(1.0, 0.875, 0.02)
    assert p == pytest.approx(blend_probability(0.99, 0.875, 0.02))
    assert p - 0.875 < 0.02


def test_fit_blend_weight_recovers_market_when_model_is_noise():
    # Outcomes drawn to match the market prob exactly; model says 0.9 always.
    rows = []
    for i in range(200):
        y = 1.0 if i % 2 == 0 else 0.0  # 50% base rate
        rows.append({"p": 0.9, "m": 0.5, "y": y, "weight": 1.0})
    w, diag = fit_blend_weight(rows)
    assert w == pytest.approx(0.0, abs=0.02)
    assert diag["logloss_market_only"] <= diag["logloss_model_only"]


def test_fit_blend_weight_recovers_model_when_model_is_right():
    rows = []
    for i in range(200):
        y = 1.0 if i % 10 < 9 else 0.0  # 90% base rate
        rows.append({"p": 0.9, "m": 0.5, "y": y, "weight": 1.0})
    w, _ = fit_blend_weight(rows)
    assert w == pytest.approx(1.0, abs=0.02)


def test_load_blend_weight_reads_meta(tmp_path, monkeypatch):
    import config
    import market_blend

    meta = tmp_path / "blend_meta.json"
    meta.write_text(json.dumps({"w": 0.5}))
    monkeypatch.setattr(market_blend, "BLEND_META_PATH", meta)
    monkeypatch.setattr(market_blend, "USE_MARKET_BLEND", True)
    monkeypatch.setattr(market_blend, "BLEND_WEIGHT_OVERRIDE", None)
    market_blend.reset_blend_cache()
    assert market_blend.load_blend_weight() == pytest.approx(0.5)
    market_blend.reset_blend_cache()


def test_load_blend_weight_applies_floor(tmp_path, monkeypatch):
    import market_blend

    meta = tmp_path / "blend_meta.json"
    meta.write_text(json.dumps({"w": 0.02}))
    monkeypatch.setattr(market_blend, "BLEND_META_PATH", meta)
    monkeypatch.setattr(market_blend, "USE_MARKET_BLEND", True)
    monkeypatch.setattr(market_blend, "BLEND_WEIGHT_OVERRIDE", None)
    monkeypatch.setattr(market_blend, "MIN_BLEND_WEIGHT", 0.3)
    market_blend.reset_blend_cache()
    assert market_blend.load_blend_weight() == pytest.approx(0.3)
    market_blend.reset_blend_cache()
