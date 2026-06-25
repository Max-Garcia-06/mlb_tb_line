from __future__ import annotations

import json

import pytest

from journal_risk import DailyRiskSnapshot
from risk_manager import (
    activate_kill_switch,
    check_pre_trade_risk,
    effective_bankroll_from,
    kill_switch_active,
    resolve_bankroll,
)


class _MockClient:
    def get_balance(self):
        return 500.0

    def get_orders(self, status: str = "resting", limit: int = 200):
        return []

    def get_market(self, ticker: str):
        return {"result": ""}


def test_resolve_bankroll_cli_when_live_off():
    assert resolve_bankroll(500.0, None, use_live_balance=False) == 500.0


def test_effective_bankroll_reserves_resting(monkeypatch):
    monkeypatch.setattr("risk_manager.RESERVE_RESTING_FROM_BANKROLL", True)
    assert effective_bankroll_from(500.0, 50.0) == 450.0


def test_kill_switch(tmp_path, monkeypatch):
    p = tmp_path / "KILL"
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", p)
    assert not kill_switch_active()
    p.write_text("halt")
    assert kill_switch_active()


def test_check_daily_loss_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr("risk_manager.DAILY_LOSS_LIMIT_USD", 50.0)
    monkeypatch.setattr("risk_manager.AUTO_KILL_ON_RISK_BREACH", False)
    monkeypatch.setattr("risk_manager.MAX_ORDERS_PER_DAY", 0)
    monkeypatch.setattr("risk_manager.MAX_DAILY_DEPLOYED_USD", 0)
    monkeypatch.setattr("risk_manager.MAX_OPEN_RESTING_USD", 0)
    monkeypatch.setattr("risk_manager.MAX_CONTRACTS_PER_GAME", 0)
    monkeypatch.setattr("risk_manager.MAX_LEGS_PER_PLAYER_DAY", 0)

    snap = DailyRiskSnapshot(total_pnl_for_limit=-60.0, realized_pnl=-60.0)
    result = check_pre_trade_risk(
        game_date="2026-05-20",
        cli_bankroll=500.0,
        client=_MockClient(),
        snapshot=snap,
    )
    assert not result.ok
    assert "Daily loss limit" in result.reason


def test_check_deployment_cap_with_proposed(tmp_path, monkeypatch):
    monkeypatch.setattr("risk_manager.DAILY_LOSS_LIMIT_USD", 0)
    monkeypatch.setattr("risk_manager.AUTO_KILL_ON_RISK_BREACH", False)
    monkeypatch.setattr("risk_manager.MAX_ORDERS_PER_DAY", 0)
    monkeypatch.setattr("risk_manager.MAX_DAILY_DEPLOYED_USD", 10.0)
    monkeypatch.setattr("risk_manager.MAX_OPEN_RESTING_USD", 0)
    monkeypatch.setattr("risk_manager.MAX_CONTRACTS_PER_GAME", 0)
    monkeypatch.setattr("risk_manager.MAX_LEGS_PER_PLAYER_DAY", 0)

    from dataclasses import dataclass

    @dataclass
    class _Sig:
        player_name: str = "A"
        kalshi_line: float = 1.5
        ticker: str = "KXMLBTB-26MAY201940BOSKC-P-2"
        recommended_contracts: int = 20
        limit_price: float = 0.5
        bet_dollars: float = 10.0

    snap = DailyRiskSnapshot(deployed_usd=5.0)
    result = check_pre_trade_risk(
        game_date="2026-05-20",
        cli_bankroll=500.0,
        client=_MockClient(),
        proposed_signals=[_Sig()],
        snapshot=snap,
    )
    assert not result.ok
    assert "Max daily deployed" in result.reason


def test_auto_kill_on_breach(tmp_path, monkeypatch):
    kill_path = tmp_path / "KILL_SWITCH"
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", kill_path)
    monkeypatch.setattr("risk_manager.DAILY_LOSS_LIMIT_USD", 10.0)
    monkeypatch.setattr("risk_manager.AUTO_KILL_ON_RISK_BREACH", True)
    monkeypatch.setattr("risk_manager.MAX_ORDERS_PER_DAY", 0)
    monkeypatch.setattr("risk_manager.MAX_DAILY_DEPLOYED_USD", 0)
    monkeypatch.setattr("risk_manager.MAX_OPEN_RESTING_USD", 0)
    monkeypatch.setattr("risk_manager.MAX_CONTRACTS_PER_GAME", 0)
    monkeypatch.setattr("risk_manager.MAX_LEGS_PER_PLAYER_DAY", 0)

    snap = DailyRiskSnapshot(total_pnl_for_limit=-20.0)
    result = check_pre_trade_risk(
        game_date="2026-05-20",
        cli_bankroll=500.0,
        client=_MockClient(),
        snapshot=snap,
    )
    assert not result.ok
    assert result.auto_killed
    assert kill_path.exists()
