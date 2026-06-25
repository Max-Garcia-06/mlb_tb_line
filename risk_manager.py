"""
Live trading risk checks: balance, daily loss cap, order limits, kill switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from config import (
    AUTO_KILL_ON_RISK_BREACH,
    DAILY_LOSS_LIMIT_USD,
    KILL_SWITCH_PATH,
    MAX_CONTRACTS_PER_GAME,
    MAX_DAILY_DEPLOYED_USD,
    MAX_LEGS_PER_PLAYER_DAY,
    MAX_OPEN_RESTING_USD,
    MAX_ORDERS_PER_DAY,
    RESERVE_RESTING_FROM_BANKROLL,
    USE_LIVE_BALANCE,
)
from journal_risk import (
    DailyRiskSnapshot,
    count_new_player_legs,
    daily_risk_snapshot,
    merge_proposed_into_snapshot,
    proposed_deployed_usd,
    proposed_resting_usd,
    total_player_legs_after_proposed,
)

if TYPE_CHECKING:
    from edge_detector import EdgeSignal

log = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    ok: bool
    bankroll: float
    effective_bankroll: float
    reason: str = ""
    snapshot: DailyRiskSnapshot | None = None
    auto_killed: bool = False


def kill_switch_active(path: Path | None = None) -> bool:
    p = path or KILL_SWITCH_PATH
    return p.exists()


def activate_kill_switch(path: Path | None = None, *, reason: str = "") -> Path:
    p = path or KILL_SWITCH_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"halted_at={datetime.now(timezone.utc).isoformat()}\nreason={reason}\n")
    log.warning("Kill switch activated: %s", p)
    return p


def resolve_bankroll(
    cli_bankroll: float,
    client: object | None,
    *,
    use_live_balance: bool | None = None,
) -> float:
    """Prefer exchange balance when configured and client supports it."""
    use_live = USE_LIVE_BALANCE if use_live_balance is None else use_live_balance
    if use_live and client is not None and hasattr(client, "get_balance"):
        try:
            bal = float(client.get_balance())
            if bal > 0:
                log.info("Using Kalshi available balance: $%.2f", bal)
                return bal
            log.warning("Kalshi balance returned %.2f; using CLI bankroll $%.2f", bal, cli_bankroll)
        except Exception as e:
            log.warning("get_balance failed, using CLI bankroll $%.2f: %s", cli_bankroll, e)
    return float(cli_bankroll)


def effective_bankroll_from(balance: float, open_resting_usd: float) -> float:
    if RESERVE_RESTING_FROM_BANKROLL and open_resting_usd > 0:
        return max(0.0, float(balance) - float(open_resting_usd))
    return float(balance)


def _maybe_auto_kill(reason: str) -> bool:
    if not AUTO_KILL_ON_RISK_BREACH:
        return False
    if kill_switch_active():
        return True
    activate_kill_switch(reason=reason)
    return True


def _check_limits(
    snapshot: DailyRiskSnapshot,
    *,
    proposed_signals: list["EdgeSignal"] | None,
    n_new_orders: int,
) -> str:
    """Return empty string if ok, else human-readable block reason."""
    merged = (
        merge_proposed_into_snapshot(snapshot, proposed_signals)
        if proposed_signals
        else snapshot
    )

    if DAILY_LOSS_LIMIT_USD > 0 and snapshot.total_pnl_for_limit <= -float(DAILY_LOSS_LIMIT_USD):
        return (
            f"Daily loss limit hit (P&L ${snapshot.total_pnl_for_limit:.2f} "
            f"<= -${DAILY_LOSS_LIMIT_USD:.2f}; realized ${snapshot.realized_pnl:.2f}, "
            f"mtm ${snapshot.mtm_pnl:.2f})"
        )

    if MAX_ORDERS_PER_DAY > 0:
        placed = snapshot.orders_placed
        if placed + int(n_new_orders) > int(MAX_ORDERS_PER_DAY):
            return f"Max orders/day ({MAX_ORDERS_PER_DAY}) — already placed {placed}"

    if proposed_signals:
        prop_deploy = proposed_deployed_usd(proposed_signals)
        if MAX_DAILY_DEPLOYED_USD > 0 and merged.deployed_usd > float(MAX_DAILY_DEPLOYED_USD):
            return (
                f"Max daily deployed ${MAX_DAILY_DEPLOYED_USD:.2f} exceeded "
                f"(would be ${merged.deployed_usd:.2f} incl. ${prop_deploy:.2f} proposed)"
            )

        prop_rest = proposed_resting_usd(proposed_signals)
        if MAX_OPEN_RESTING_USD > 0 and merged.open_resting_usd > float(MAX_OPEN_RESTING_USD):
            return (
                f"Max open resting ${MAX_OPEN_RESTING_USD:.2f} exceeded "
                f"(would be ${merged.open_resting_usd:.2f} incl. ${prop_rest:.2f} proposed)"
            )

        if MAX_CONTRACTS_PER_GAME > 0:
            for slug, cnt in merged.contracts_by_game.items():
                if cnt > int(MAX_CONTRACTS_PER_GAME):
                    return (
                        f"Max contracts per game ({MAX_CONTRACTS_PER_GAME}) exceeded "
                        f"for {slug} (would be {cnt})"
                    )

        if MAX_LEGS_PER_PLAYER_DAY > 0:
            total_legs = total_player_legs_after_proposed(snapshot, proposed_signals)
            if total_legs > int(MAX_LEGS_PER_PLAYER_DAY):
                new_legs = count_new_player_legs(snapshot, proposed_signals)
                return (
                    f"Max player legs/day ({MAX_LEGS_PER_PLAYER_DAY}) exceeded "
                    f"(would be {total_legs} legs, +{new_legs} new from this scan)"
                )

    return ""


def check_pre_trade_risk(
    *,
    game_date: str,
    cli_bankroll: float,
    client: object | None = None,
    n_new_orders: int = 0,
    proposed_signals: list["EdgeSignal"] | None = None,
    snapshot: DailyRiskSnapshot | None = None,
) -> RiskCheckResult:
    if kill_switch_active():
        bal = resolve_bankroll(cli_bankroll, client)
        eff = effective_bankroll_from(bal, 0.0)
        return RiskCheckResult(
            False,
            bal,
            eff,
            "Kill switch file present — trading halted",
        )

    snap = snapshot if snapshot is not None else daily_risk_snapshot(game_date, client=client)
    bankroll = resolve_bankroll(cli_bankroll, client)
    eff = effective_bankroll_from(bankroll, snap.open_resting_usd)

    reason = _check_limits(
        snap,
        proposed_signals=proposed_signals,
        n_new_orders=n_new_orders,
    )
    if reason:
        auto_killed = _maybe_auto_kill(reason)
        return RiskCheckResult(
            False,
            bankroll,
            eff,
            reason,
            snapshot=snap,
            auto_killed=auto_killed,
        )

    return RiskCheckResult(True, bankroll, eff, "", snapshot=snap)
