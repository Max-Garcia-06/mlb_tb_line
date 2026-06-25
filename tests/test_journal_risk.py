from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from journal_risk import (
    daily_risk_snapshot,
    deployed_usd_from_journal,
    game_slug_from_ticker,
    merge_proposed_into_snapshot,
    mtm_pnl_per_contract,
    player_leg_key,
)


@dataclass
class _Sig:
    player_name: str
    kalshi_line: float
    ticker: str
    recommended_contracts: int
    limit_price: float
    bet_dollars: float


def test_game_slug_from_ticker():
    assert game_slug_from_ticker("KXMLBTB-26MAY201940BOSKC-PLAYER-2") == "BOSKC"


def test_mtm_pnl_per_contract_yes():
    mark = {"mark_yes_mid": 0.55, "mark_no_mid": 0.45}
    assert mtm_pnl_per_contract("yes", 0.50, mark) == pytest.approx(0.05)


def test_deployed_usd_from_journal():
    rows = [
        {"note": "post-submit", "success": True, "contracts": 10, "limit_price": 0.4},
        {"note": "post-submit", "success": False, "contracts": 5, "limit_price": 0.3},
    ]
    assert deployed_usd_from_journal(rows) == pytest.approx(4.0)


def test_daily_risk_snapshot_realized(tmp_path, monkeypatch):
    journal = tmp_path / "trades_2026-05-20.jsonl"
    rows = [
        {
            "note": "post-submit",
            "success": True,
            "order_id": "o1",
            "ticker": "KXMLBTB-26MAY201940BOSKC-P-2",
            "side": "yes",
            "contracts": 5,
            "limit_price": 0.40,
            "player_name": "Player A",
            "kalshi_line": 1.5,
        },
        {
            "note": "fill",
            "order_id": "o1",
            "filled_contracts": 5,
            "avg_fill_price": 0.40,
        },
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr("journal_risk.journal_path", lambda _d: journal)

    class _Client:
        def get_market(self, ticker: str):
            return {"result": "yes"}

        def get_orders(self, status: str = "resting", limit: int = 200):
            return []

    snap = daily_risk_snapshot("2026-05-20", client=_Client())
    assert snap.realized_pnl == pytest.approx(5 * (1.0 - 0.40))
    assert snap.mtm_pnl == 0.0
    assert snap.orders_placed == 1
    assert snap.contracts_by_game.get("BOSKC") == 5


def test_daily_risk_snapshot_mtm(tmp_path, monkeypatch):
    journal = tmp_path / "trades_2026-05-20.jsonl"
    rows = [
        {
            "note": "post-submit",
            "success": True,
            "order_id": "o1",
            "ticker": "KXMLBTB-26MAY201940BOSKC-P-2",
            "side": "yes",
            "contracts": 4,
            "limit_price": 0.40,
            "player_name": "Player A",
            "kalshi_line": 1.5,
            "ts": "2026-05-20T10:00:00+00:00",
        },
        {
            "note": "fill",
            "order_id": "o1",
            "filled_contracts": 4,
            "avg_fill_price": 0.40,
        },
        {
            "note": "mark",
            "order_id": "o1",
            "mark_label": "30m",
            "mark_yes_mid": 0.50,
            "mark_no_mid": 0.50,
            "ts": "2026-05-20T10:30:00+00:00",
        },
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr("journal_risk.journal_path", lambda _d: journal)

    class _Client:
        def get_market(self, ticker: str):
            return {"result": ""}

        def get_orders(self, status: str = "resting", limit: int = 200):
            return []

    snap = daily_risk_snapshot("2026-05-20", client=_Client())
    assert snap.realized_pnl == 0.0
    assert snap.mtm_pnl == pytest.approx(4 * 0.10)
    assert snap.total_pnl_for_limit == pytest.approx(0.40)


def test_merge_proposed_concentration():
    from journal_risk import DailyRiskSnapshot

    base = DailyRiskSnapshot(
        player_legs={player_leg_key("Judge", 1.5)},
        contracts_by_game={"BOSKC": 10},
    )
    sigs = [
        _Sig("Judge", 2.5, "KXMLBTB-26MAY201940BOSKC-J-3", 5, 0.45, 2.25),
        _Sig("Soto", 1.5, "KXMLBTB-26MAY201940BOSKC-S-2", 3, 0.50, 1.50),
    ]
    merged = merge_proposed_into_snapshot(base, sigs)  # type: ignore[arg-type]
    assert len(merged.player_legs) == 3
    assert merged.contracts_by_game["BOSKC"] == 18
    assert merged.deployed_usd == pytest.approx(3.75)
