"""
model_vs_market.py
------------------
Score model probabilities against the Kalshi book on FULL snapshot slates —
every market, no trade filter — settled vs actual TB.

This answers the question the fill-based blend fit cannot: fills are a
selection-biased subset (only where the model disagreed with the market), so a
low fitted ``w`` there says the model loses *where it used to trade*. Here we
score all markets, then slice by line and by model-vs-market disagreement to
find segments (if any) where the model genuinely adds information.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from backtest import (
    _actual_tb_by_player,
    _build_predictions,
    _games_played_map,
    _get_model_for_backtest,
)
from edge_detector import fill_calibrated_probabilities
from feature_store import build_feature_table, load_distinct_training_player_ids, load_features_as_of
from identity_bridge import norm_player_name, resolve_mlb_player_id
from market_blend import blend_probability, fit_blend_weight
from market_snapshots import load_snapshots
from probability_engine import calculate_probabilities

log = logging.getLogger(__name__)

MAX_BOOK_SPREAD = 0.25


@dataclass(frozen=True)
class ScoreRow:
    game_date: str
    player_name: str
    ticker: str
    line: float
    p_model_raw: float
    p_model_cal: float
    p_market_mid: float
    actual_tb: int
    y: float  # 1.0 if TB > line


def evaluate_day(game_date: str, *, pit_train: bool = False, earliest: bool = False) -> list[ScoreRow]:
    """Score one day's snapshot slate. Returns [] when snapshots/outcomes are missing."""
    snaps = load_snapshots(game_date, latest_only=not earliest, earliest_only=earliest)
    if not snaps:
        return []
    actual = _actual_tb_by_player(game_date)
    if not actual:
        log.warning("No batter_games outcomes for %s — run etl first", game_date)
        return []

    market_lines = []
    for s in snaps:
        bid, ask = float(s.yes_bid), float(s.yes_ask)
        if not (0.0 < bid < ask < 1.0) or (ask - bid) > MAX_BOOK_SPREAD:
            continue
        if norm_player_name(s.player_name) not in actual:
            continue  # player didn't bat (scratched / not in boxscore)
        market_lines.append(s.to_market_line())
    if not market_lines:
        return []

    trained_model, meta = _get_model_for_backtest(game_date, pit_train=pit_train, retrain_days=7)
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
    prob_results = calculate_probabilities(predictions, meta["residual_var"])
    fill_calibrated_probabilities(prob_results, games_played_by_player=_games_played_map(feat_df))

    rows: list[ScoreRow] = []
    for pr, ml in zip(prob_results, market_lines):
        tb = actual[norm_player_name(ml.player_name)]
        rows.append(
            ScoreRow(
                game_date=game_date,
                player_name=ml.player_name,
                ticker=ml.ticker,
                line=float(ml.line),
                p_model_raw=float(pr.p_over),
                p_model_cal=float(pr.p_over_calibrated if pr.p_over_calibrated is not None else pr.p_over),
                p_market_mid=float(ml.yes_mid),
                actual_tb=int(tb),
                y=1.0 if tb > float(ml.line) else 0.0,
            )
        )
    return rows


def _logloss(rows: list[ScoreRow], key: str) -> float:
    import math

    eps = 1e-4
    tot = 0.0
    for r in rows:
        p = min(1.0 - eps, max(eps, getattr(r, key)))
        tot += -(r.y * math.log(p) + (1.0 - r.y) * math.log(1.0 - p))
    return tot / len(rows)


def _brier(rows: list[ScoreRow], key: str) -> float:
    return sum((getattr(r, key) - r.y) ** 2 for r in rows) / len(rows)


def disagreement_bucket(r: ScoreRow) -> str:
    d = abs(r.p_model_cal - r.p_market_mid)
    if d < 0.05:
        return "<0.05"
    if d < 0.10:
        return "0.05-0.10"
    if d < 0.15:
        return "0.10-0.15"
    return ">=0.15"


def summarize_slice(rows: list[ScoreRow]) -> dict:
    """n, log-losses/Briers for model vs market, and the fitted blend weight for this slice."""
    out = {
        "n": len(rows),
        "ll_model": _logloss(rows, "p_model_cal"),
        "ll_market": _logloss(rows, "p_market_mid"),
        "brier_model": _brier(rows, "p_model_cal"),
        "brier_market": _brier(rows, "p_market_mid"),
        "base_rate": sum(r.y for r in rows) / len(rows),
    }
    fit = [{"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "weight": 1.0} for r in rows]
    try:
        w, diag = fit_blend_weight(fit)
        out["w_fit"] = w
        out["ll_blend"] = diag["logloss_best"]
    except ValueError:
        out["w_fit"] = float("nan")
        out["ll_blend"] = float("nan")
    return out


def summarize(rows: list[ScoreRow]) -> dict[str, dict[str, dict]]:
    """Slices: overall, by line, by model-market disagreement, by date."""
    groups: dict[str, dict[str, list[ScoreRow]]] = {
        "overall": {"all": rows},
        "line": defaultdict(list),
        "disagreement": defaultdict(list),
        "date": defaultdict(list),
    }
    for r in rows:
        groups["line"][f"{r.line:g}"].append(r)
        groups["disagreement"][disagreement_bucket(r)].append(r)
        groups["date"][r.game_date].append(r)
    return {
        gname: {k: summarize_slice(v) for k, v in sorted(g.items()) if v}
        for gname, g in groups.items()
    }
