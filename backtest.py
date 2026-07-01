"""
Replay stored Kalshi snapshots through the scan stack and settle vs actual TB.

Supports point-in-time model training, earliest vs latest snapshots, live-stack parity
(portfolio sizing, VPIN), and fill-probability adjustment.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import sqlalchemy as sa

from config import (
    BACKTEST_FILL_MODEL,
    BACKTEST_PIT_RETRAIN_DAYS,
    BACKTEST_PIT_TRAIN,
    BACKTEST_USE_EARLIEST_SNAPSHOT,
    DB_PATH,
    EDGE_THRESHOLD,
    GAMES_FOR_LAMBDA_SANITY,
    LAMBDA_SANITY_MAX,
    MIN_P,
    SCAN_WITHIN_HOURS,
    TAIL_EDGE_MULT,
    TAIL_P_CUTOFF,
)
from edge_detector import (
    apply_flow_guard,
    fill_calibrated_probabilities,
    scan_for_edges,
)
from identity_bridge import norm_player_name, resolve_mlb_player_id
from kalshi_bridge import MarketLine
from market_snapshots import (
    load_snapshots,
    load_snapshots_open_and_close,
    load_snapshots_within_hours_of_start,
)
from model import load_model, predict_lambda, predict_tb_pmf_row, train_as_of
from probability_engine import calculate_probabilities
from feature_store import MODEL_FEATURES, build_feature_table, load_distinct_training_player_ids, load_features_as_of
from trading_stack import fill_probability, finalize_signals

log = logging.getLogger(__name__)

_pit_model_cache: dict[str, tuple[Any, dict]] = {}


@dataclass
class BacktestTrade:
    game_date: str
    player_name: str
    ticker: str
    side: str
    line: float
    limit_price: float
    contracts: int
    p_model: float
    edge: float
    ev: float
    actual_tb: int
    won: bool
    pnl_usd: float
    fill_prob: float = 1.0
    clv: float | None = None


@dataclass
class BacktestReport:
    game_date: str
    n_markets: int
    n_signals: int
    n_trades: int
    total_pnl: float
    total_cost: float
    roi_pct: float
    win_rate: float
    mean_clv: float | None
    trades: list[BacktestTrade]
    by_line: dict[float, dict[str, float]] = field(default_factory=dict)
    by_spread: dict[str, dict[str, float]] = field(default_factory=dict)
    pit_as_of: str = ""


def _pit_cache_key(game_date: str, retrain_days: int) -> str:
    d = pd.Timestamp(game_date)
    if retrain_days <= 1:
        return str(d.date())
    epoch = (d - pd.Timestamp("1970-01-01")).days
    bucket = epoch // int(retrain_days)
    return f"bucket_{retrain_days}_{bucket}"


def _get_model_for_backtest(
    game_date: str,
    *,
    pit_train: bool,
    retrain_days: int,
) -> tuple[Any, dict]:
    if not pit_train:
        return load_model()
    key = _pit_cache_key(game_date, retrain_days)
    if key in _pit_model_cache:
        return _pit_model_cache[key]
    try:
        mdl, meta = train_as_of(game_date)
        meta = dict(meta)
        meta["trained_on"] = game_date
        meta["pit_as_of"] = game_date
        _pit_model_cache[key] = (mdl, meta)
        return mdl, meta
    except Exception as e:
        log.warning("PIT train failed for %s (%s); falling back to saved model", game_date, e)
        return load_model()


def _actual_tb_by_player(game_date: str) -> dict[str, int]:
    eng = sa.create_engine(f"sqlite:///{DB_PATH}")
    try:
        df = pd.read_sql(
            "SELECT player_name, tb FROM batter_games WHERE date(game_date) = date(:gd)",
            eng,
            params={"gd": game_date},
        )
    except Exception:
        return {}
    out: dict[str, int] = {}
    for _, r in df.iterrows():
        out[norm_player_name(str(r["player_name"]))] = int(r["tb"])
    return out


def _settle(side: str, line: float, actual_tb: int) -> bool:
    if side == "yes":
        return actual_tb > float(line)
    return actual_tb <= float(line)


def _pnl(side: str, won: bool, price: float, contracts: int) -> float:
    c = int(contracts)
    p = float(price)
    if side == "yes":
        return c * ((1.0 - p) if won else -p)
    return c * (p if won else -(1.0 - p))


def _clv_yes(entry_yes_price: float, close_yes_mid: float) -> float:
    """Positive CLV = we beat the close on YES contracts."""
    return float(close_yes_mid) - float(entry_yes_price)


def _games_played_map(feat_df: pd.DataFrame | None) -> dict[str, int]:
    if feat_df is None or feat_df.empty:
        return {}
    out: dict[str, int] = {}
    if "player_name" not in feat_df.columns or "games_played" not in feat_df.columns:
        return out
    for _, r in feat_df.groupby("player_name").tail(1).iterrows():
        out[str(r["player_name"])] = int(r.get("games_played", 0) or 0)
    return out


def _build_predictions(
    game_date: str,
    market_lines: list[MarketLine],
    trained_model: object,
    meta: dict,
    feat_df: pd.DataFrame | None,
    training_ids: set[int],
) -> list[dict]:
    from matchup_features import (
        apply_live_feature_overrides,
        build_opp_tb_allowed_lookup,
        build_slate_matchup_index,
        slate_teams,
    )
    from feature_store import build_live_pitcher_features
    from data_engine import get_confirmed_lineups

    feat_names = list(meta.get("feature_names") or MODEL_FEATURES)
    variance = meta["residual_var"]
    matchup_slate = build_slate_matchup_index(game_date)
    opp_tb_lookup = build_opp_tb_allowed_lookup(game_date, slate_teams(matchup_slate))
    try:
        live_sp_by_team = build_live_pitcher_features(game_date)
    except Exception:
        live_sp_by_team = {}
    try:
        confirmed_lineups = get_confirmed_lineups(game_date)
    except Exception:
        confirmed_lineups = {}
    by_slate: dict[tuple, list] = defaultdict(list)
    for ml in market_lines:
        by_slate[(norm_player_name(ml.player_name), str(ml.game_date))].append(ml)

    predictions = []
    for ml in market_lines:
        key = (norm_player_name(ml.player_name), str(ml.game_date))
        mls = by_slate[key]
        ml0 = mls[0]
        pid = resolve_mlb_player_id(
            player_name=ml0.player_name,
            xref_player_id=ml0.xref_player_id,
            feat_df=feat_df,
            allowed_player_ids=training_ids or None,
        )
        lam = float(ml.line * ml.implied_prob / 0.5)
        pmf = None
        if feat_df is not None and not feat_df.empty:
            if pid:
                pr = feat_df[feat_df["player_id"] == int(pid)]
            else:
                pr = feat_df[feat_df["player_name"].str.lower() == ml0.player_name.lower()]
            if not pr.empty:
                latest = pr.sort_values("game_date").iloc[-1]
                row_features = latest[feat_names].fillna(0).to_dict()
                et = str(ml0.event_ticker or "")
                if not et and ml0.ticker:
                    parts = str(ml0.ticker).split("-")
                    if len(parts) >= 2:
                        et = f"{parts[0]}-{parts[1]}"
                row_features = apply_live_feature_overrides(
                    row_features,
                    game_date=game_date,
                    player_id=int(latest.get("player_id", 0) or 0),
                    player_team=str(latest.get("team", "") or ""),
                    bats_hand=str(latest.get("bats_hand", "R") or "R"),
                    tb_roll=float(latest.get("tb_roll", 0) or 0),
                    event_ticker=et,
                    matchup_slate=matchup_slate,
                    opp_tb_lookup=opp_tb_lookup,
                    confirmed_lineups=confirmed_lineups,
                    live_sp_by_team=live_sp_by_team,
                )
                lam_raw = float(predict_lambda(row_features, trained_model, feature_names=feat_names, meta=meta))
                gplayed = int(latest.get("games_played", 999) or 999)
                if lam_raw > LAMBDA_SANITY_MAX and gplayed < GAMES_FOR_LAMBDA_SANITY:
                    lam = float(ml.line * ml.implied_prob / 0.5)
                else:
                    lam = lam_raw
                    pmf = predict_tb_pmf_row(row_features, trained_model, feature_names=feat_names, meta=meta)
        pred = {
            "player_id": int(pid or 0),
            "player_name": ml.player_name,
            "game_date": ml.game_date,
            "kalshi_line": ml.line,
            "predicted_lambda": lam,
        }
        if pmf is not None:
            pred["tb_pmf"] = pmf.tolist()
        predictions.append(pred)
    return predictions


def run_backtest_day(
    game_date: str,
    *,
    bankroll: float = 1000.0,
    edge_threshold: float = EDGE_THRESHOLD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
    one_per_player: bool = True,
    max_signals: int | None = None,
    max_contracts: int | None = 250,
    require_snapshots: bool = True,
    pit_train: bool | None = None,
    pit_retrain_days: int | None = None,
    use_earliest_snapshot: bool | None = None,
    use_fill_model: bool | None = None,
    within_hours: float | None = None,
    use_time_window: bool | None = None,
) -> BacktestReport | None:
    pit = BACKTEST_PIT_TRAIN if pit_train is None else pit_train
    retrain_d = BACKTEST_PIT_RETRAIN_DAYS if pit_retrain_days is None else int(pit_retrain_days)
    earliest = BACKTEST_USE_EARLIEST_SNAPSHOT if use_earliest_snapshot is None else use_earliest_snapshot
    fill_m = BACKTEST_FILL_MODEL if use_fill_model is None else use_fill_model
    window_on = use_time_window if use_time_window is not None else (SCAN_WITHIN_HOURS > 0)
    hours = float(SCAN_WITHIN_HOURS if within_hours is None else within_hours)

    close_snaps = load_snapshots(game_date, latest_only=True)
    if window_on and hours > 0:
        snaps = load_snapshots_within_hours_of_start(game_date, within_hours=hours)
        if not snaps:
            log.warning(
                "No snapshots within %.1fh of first pitch for %s — "
                "run schedule-snapshots near each slate wave",
                hours,
                game_date,
            )
            return None
    elif earliest:
        open_snaps, _ = load_snapshots_open_and_close(game_date)
        snaps = open_snaps if open_snaps else load_snapshots(game_date, earliest_only=True)
    else:
        snaps = close_snaps if close_snaps else load_snapshots(game_date, latest_only=True)

    if not snaps and require_snapshots:
        log.warning("No snapshots for %s", game_date)
        return None

    close_by_ticker = {s.ticker: s for s in close_snaps}
    market_lines = [s.to_market_line() for s in snaps]
    actual = _actual_tb_by_player(game_date)
    if not actual:
        log.warning("No batter_games outcomes for %s — run etl first", game_date)
        return None

    trained_model, meta = _get_model_for_backtest(game_date, pit_train=pit, retrain_days=retrain_d)
    variance = meta["residual_var"]
    training_ids = load_distinct_training_player_ids()
    need_ids: set[int] = set()
    for ml in market_lines:
        pid = resolve_mlb_player_id(
            player_name=ml.player_name, xref_player_id=ml.xref_player_id, allowed_player_ids=training_ids or None
        )
        if pid:
            need_ids.add(int(pid))

    feat_df = None
    if need_ids:
        try:
            feat_df = load_features_as_of(game_date, player_ids=need_ids)
            if feat_df is None or feat_df.empty:
                feat_df = build_feature_table(player_ids=need_ids)
                feat_df = feat_df[pd.to_datetime(feat_df["game_date"]) < pd.Timestamp(game_date)]
        except Exception:
            feat_df = build_feature_table(player_ids=need_ids) if need_ids else None

    predictions = _build_predictions(game_date, market_lines, trained_model, meta, feat_df, training_ids)
    prob_results = calculate_probabilities(predictions, variance)
    fill_calibrated_probabilities(prob_results, games_played_by_player=_games_played_map(feat_df))

    signals = scan_for_edges(
        prob_results,
        market_lines,
        bankroll,
        edge_threshold=edge_threshold,
        min_p=min_p,
        tail_p_cutoff=tail_p_cutoff,
        tail_edge_mult=tail_edge_mult,
    )
    signals = apply_flow_guard(signals, market_lines)
    signals = finalize_signals(
        signals,
        bankroll=bankroll,
        one_per_player=one_per_player,
        max_signals=max_signals,
        max_contracts=max_contracts,
    )

    trades: list[BacktestTrade] = []
    for s in signals:
        key = norm_player_name(s.player_name)
        if key not in actual:
            continue
        tb = actual[key]
        won = _settle(s.recommended_side, s.kalshi_line, tb)
        fp = fill_probability(s.book_spread, side=s.recommended_side) if fill_m else 1.0
        pnl_raw = _pnl(s.recommended_side, won, s.limit_price, s.recommended_contracts)
        pnl = float(pnl_raw) * float(fp)

        clv_val = None
        cs = close_by_ticker.get(s.ticker)
        if cs is not None:
            close_mid = (float(cs.yes_bid) + float(cs.yes_ask)) / 2.0
            if str(s.recommended_side).lower() == "yes":
                clv_val = _clv_yes(s.limit_price, close_mid)
            else:
                clv_val = _clv_yes(1.0 - s.limit_price, 1.0 - close_mid)

        trades.append(
            BacktestTrade(
                game_date=game_date,
                player_name=s.player_name,
                ticker=s.ticker,
                side=s.recommended_side,
                line=float(s.kalshi_line),
                limit_price=float(s.limit_price),
                contracts=int(s.recommended_contracts),
                p_model=float(s.p_model),
                edge=float(s.edge),
                ev=float(s.ev),
                actual_tb=int(tb),
                won=bool(won),
                pnl_usd=float(pnl),
                fill_prob=float(fp),
                clv=clv_val,
            )
        )

    total_pnl = sum(t.pnl_usd for t in trades)
    total_cost = sum(t.limit_price * t.contracts for t in trades)
    roi = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0
    win_rate = sum(1 for t in trades if t.won) / len(trades) if trades else 0.0
    clvs = [t.clv for t in trades if t.clv is not None]
    mean_clv = float(sum(clvs) / len(clvs)) if clvs else None

    by_line: dict[float, dict[str, float]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        bl = by_line[float(t.line)]
        bl["n"] += 1
        bl["pnl"] += t.pnl_usd
        bl["wins"] += 1 if t.won else 0

    return BacktestReport(
        game_date=game_date,
        n_markets=len(market_lines),
        n_signals=len(signals),
        n_trades=len(trades),
        total_pnl=float(total_pnl),
        total_cost=float(total_cost),
        roi_pct=float(roi),
        win_rate=float(win_rate),
        mean_clv=mean_clv,
        trades=trades,
        by_line=dict(by_line),
        pit_as_of=str(meta.get("pit_as_of") or meta.get("trained_on") or ""),
    )


def run_backtest_range(
    start: str,
    end: str,
    **kwargs: Any,
) -> list[BacktestReport]:
    from journal_reader import parse_iso_date

    s = parse_iso_date(start)
    e = parse_iso_date(end)
    reports: list[BacktestReport] = []
    cur = s
    while cur <= e:
        gd = cur.strftime("%Y-%m-%d")
        rep = run_backtest_day(gd, **kwargs)
        if rep is not None:
            reports.append(rep)
        cur = cur + pd.Timedelta(days=1)
    return reports
