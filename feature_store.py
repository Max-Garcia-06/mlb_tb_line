"""
feature_store.py (MLB TB)
------------------------
Builds a feature table from batter game logs for training and live prediction.
"""

from __future__ import annotations

from typing import Collection

import numpy as np
import pandas as pd
import sqlalchemy as sa

from config import DB_PATH, FEATURES_TABLE, ROLLING_WINDOW, MIN_GAMES
from venue_physics import lookup_park_physics
from matchup_features import (
    MATCHUP_FEATURE_NAMES,
    attach_platoon_features,
    attach_team_defense_features,
    finalize_matchup_columns,
)

PITCHER_ROLLING_WINDOW = 5

MODEL_FEATURES = [
    "tb_roll",
    "tb_season_avg",
    "h_roll",
    "ab_roll",
    "hr_roll",
    "bb_roll",
    "so_roll",
    "tb_per_ab_roll",
    "h_per_ab_roll",
    "hr_per_ab_roll",
    "bb_per_ab_roll",
    "so_per_ab_roll",
    "tb_roll_std",
    "days_since_last_game",
    "game_month",
    "game_dow",
    # Statcast-style environment (venue priors + game-day weather; see data_engine ETL)
    "venue_distance_added_index",
    "venue_elevation_ft",
    "game_temp_norm",
    "game_wind_carry",
    "statcast_env_lift",
    # Opposing starter rolling stats (last 5 starts, shifted to avoid leakage)
    "opp_sp_era_roll",
    "opp_sp_k9_roll",
    "opp_sp_hr9_roll",
    "opp_sp_bb9_roll",
] + MATCHUP_FEATURE_NAMES


def _engine():
    return sa.create_engine(f"sqlite:///{DB_PATH}")


def _build_pitcher_rolling(engine: sa.Engine) -> pd.DataFrame:
    """Build shifted rolling ERA/K9/BB9/HR9 for starters from pitcher_games table."""
    try:
        df = pd.read_sql("SELECT * FROM pitcher_games WHERE is_starter=1", engine)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    g = df.groupby("pitcher_id", group_keys=False)
    for col in ("er", "so", "bb", "hr", "ip"):
        s = g[col].shift(1)
        df[f"{col}_roll"] = (
            s.groupby(df["pitcher_id"], group_keys=False)
            .rolling(PITCHER_ROLLING_WINDOW, min_periods=2)
            .sum()
            .reset_index(level=0, drop=True)
        )
    ip_safe = df["ip_roll"].clip(lower=1.0)
    df["opp_sp_era_roll"] = (df["er_roll"] / ip_safe * 9).clip(0, 15)
    df["opp_sp_k9_roll"]  = (df["so_roll"] / ip_safe * 9).clip(0, 20)
    df["opp_sp_bb9_roll"] = (df["bb_roll"] / ip_safe * 9).clip(0, 15)
    df["opp_sp_hr9_roll"] = (df["hr_roll"] / ip_safe * 9).clip(0, 10)
    keep = ["team", "game_date", "opp_sp_era_roll", "opp_sp_k9_roll", "opp_sp_bb9_roll", "opp_sp_hr9_roll"]
    return df.dropna(subset=["opp_sp_era_roll"])[keep].copy()


def build_live_pitcher_features(game_date: str) -> dict[str, dict]:
    """
    Return {opp_team_abbr: {opp_sp_era_roll, opp_sp_k9_roll, ...}} for today's starters.
    Uses the pitcher_games rolling history up to game_date.
    """
    from data_engine import get_probable_starters

    engine = _engine()
    pitcher_roll = _build_pitcher_rolling(engine)
    if pitcher_roll.empty:
        return {}

    probable = get_probable_starters(game_date)
    if not probable:
        return {}

    # Latest rolling stats per pitcher (as of game_date)
    gd_ts = pd.Timestamp(game_date)
    pitcher_roll = pitcher_roll[pitcher_roll["game_date"] < gd_ts]
    if pitcher_roll.empty:
        return {}

    # pitcher_games.team = the pitcher's own team; we need the opponent's pitcher
    # probable: {team_abbr -> pitcher_id}; we need pitcher career stats by pitcher_id
    try:
        all_starters = pd.read_sql(
            "SELECT * FROM pitcher_games WHERE is_starter=1", engine
        )
    except Exception:
        return {}
    if all_starters.empty:
        return {}
    all_starters["game_date"] = pd.to_datetime(all_starters["game_date"])

    result: dict[str, dict] = {}
    for team_abbr, pitcher_id in probable.items():
        rows = all_starters[
            (all_starters["pitcher_id"] == pitcher_id)
            & (all_starters["game_date"] < gd_ts)
        ].sort_values("game_date")
        if len(rows) < 2:
            continue
        recent = rows.tail(PITCHER_ROLLING_WINDOW)
        ip_total = max(recent["ip"].sum(), 1.0)
        era = float((recent["er"].sum() / ip_total * 9).clip(0, 15))
        k9  = float((recent["so"].sum() / ip_total * 9).clip(0, 20))
        bb9 = float((recent["bb"].sum() / ip_total * 9).clip(0, 15))
        hr9 = float((recent["hr"].sum() / ip_total * 9).clip(0, 10))
        result[team_abbr] = {
            "opp_sp_era_roll": era,
            "opp_sp_k9_roll":  k9,
            "opp_sp_bb9_roll": bb9,
            "opp_sp_hr9_roll": hr9,
        }
    return result


def load_distinct_training_player_ids() -> set[int]:
    """All batter ``player_id`` values in the DB (cheap query for identity disambiguation)."""
    engine = _engine()
    df = pd.read_sql("SELECT DISTINCT player_id FROM batter_games", engine)
    if df.empty or "player_id" not in df.columns:
        return set()
    return set(pd.to_numeric(df["player_id"], errors="coerce").dropna().astype(int).tolist())


def _ensure_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c, default in [
        ("park_team_abbr", ""),
        ("venue_id", np.nan),
        ("temp_f", np.nan),
        ("wind_mph", np.nan),
        ("wind_l_to_r", 0),
        ("venue_distance_added_index", np.nan),
        ("venue_elevation_ft", np.nan),
    ]:
        if c not in df.columns:
            df[c] = default
    # Backfill physics from park_team if older DB rows lack merged indices
    mask = df["venue_distance_added_index"].isna() & df["park_team_abbr"].astype(str).str.len().gt(0)
    if mask.any():
        idxs = []
        elevs = []
        for ab in df.loc[mask, "park_team_abbr"].astype(str):
            dai, el = lookup_park_physics(ab)
            idxs.append(dai)
            elevs.append(el)
        df.loc[mask, "venue_distance_added_index"] = idxs
        df.loc[mask, "venue_elevation_ft"] = elevs
    df["venue_distance_added_index"] = pd.to_numeric(df["venue_distance_added_index"], errors="coerce").fillna(0.0)
    df["venue_elevation_ft"] = pd.to_numeric(df["venue_elevation_ft"], errors="coerce").fillna(0.0)
    tf = pd.to_numeric(df["temp_f"], errors="coerce")
    df["game_temp_norm"] = ((tf.fillna(72.0) - 72.0) / 15.0).clip(-2.5, 2.5)
    wm = pd.to_numeric(df["wind_mph"], errors="coerce").fillna(0.0)
    wlr = pd.to_numeric(df["wind_l_to_r"], errors="coerce").fillna(0).clip(0, 1)
    df["game_wind_carry"] = (wlr * (wm / 20.0)).clip(0, 2.5)
    elev_n = df["venue_elevation_ft"] / 5280.0
    df["statcast_env_lift"] = (
        df["venue_distance_added_index"]
        + 0.015 * df["game_temp_norm"]
        + 0.10 * elev_n.clip(0, 1.5)
        + 0.012 * df["game_wind_carry"]
    )
    return df


def build_feature_table(player_ids: Collection[int] | None = None) -> pd.DataFrame:
    engine = _engine()
    if player_ids is not None and len(list(player_ids)) > 0:
        ids_sql = ",".join(str(int(x)) for x in sorted(set(int(x) for x in player_ids)))
        df = pd.read_sql(f"SELECT * FROM batter_games WHERE player_id IN ({ids_sql})", engine)
    else:
        df = pd.read_sql("SELECT * FROM batter_games", engine)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    df = _ensure_env_columns(df)

    # Basic rollups
    g = df.groupby("player_id", group_keys=False)
    df["games_played"] = g.cumcount() + 1

    for col, out in [
        ("tb", "tb_roll"),
        ("h", "h_roll"),
        ("ab", "ab_roll"),
        ("hr", "hr_roll"),
        ("bb", "bb_roll"),
        ("so", "so_roll"),
    ]:
        # Use only prior games for features (avoid target leakage).
        s = g[col].shift(1)
        df[out] = s.groupby(df["player_id"], group_keys=False).rolling(ROLLING_WINDOW, min_periods=3).mean().reset_index(
            level=0, drop=True
        )

    df["tb_season_avg"] = (
        g["tb"].shift(1).groupby(df["player_id"], group_keys=False).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    )
    df["tb_per_ab_roll"] = (df["tb_roll"] / df["ab_roll"]).replace([float("inf"), -float("inf")], pd.NA)
    df["h_per_ab_roll"] = (df["h_roll"] / df["ab_roll"]).replace([float("inf"), -float("inf")], pd.NA)
    df["hr_per_ab_roll"] = (df["hr_roll"] / df["ab_roll"]).replace([float("inf"), -float("inf")], pd.NA)
    df["bb_per_ab_roll"] = (df["bb_roll"] / df["ab_roll"]).replace([float("inf"), -float("inf")], pd.NA)
    df["so_per_ab_roll"] = (df["so_roll"] / df["ab_roll"]).replace([float("inf"), -float("inf")], pd.NA)

    # Trailing volatility (using only prior games)
    tb_lag = g["tb"].shift(1)
    df["tb_roll_std"] = (
        tb_lag.groupby(df["player_id"], group_keys=False)
        .rolling(ROLLING_WINDOW, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Recency / calendar features
    prev_date = g["game_date"].shift(1)
    df["days_since_last_game"] = (df["game_date"] - prev_date).dt.days.astype("float")
    df["days_since_last_game"] = df["days_since_last_game"].clip(lower=0, upper=14)
    df["game_month"] = df["game_date"].dt.month.astype(int)
    df["game_dow"] = df["game_date"].dt.dayofweek.astype(int)

    for c, default in [
        ("is_home", 0),
        ("opponent_team", ""),
        ("lineup_slot", 5),
        ("bats_hand", "R"),
        ("opp_sp_hand_L", 0.5),
    ]:
        if c not in df.columns:
            df[c] = default

    df = attach_team_defense_features(df)
    df = attach_platoon_features(df)
    df = finalize_matchup_columns(df)

    # Join pitcher rolling stats keyed on (opponent_team, game_date)
    pitcher_roll = _build_pitcher_rolling(engine)
    if not pitcher_roll.empty:
        pitcher_roll = pitcher_roll.rename(columns={"team": "opponent_team"})
        df = df.merge(pitcher_roll, on=["opponent_team", "game_date"], how="left")
    else:
        for c in ("opp_sp_era_roll", "opp_sp_k9_roll", "opp_sp_bb9_roll", "opp_sp_hr9_roll"):
            df[c] = np.nan

    # Filter for training cohort
    df = df[df["games_played"] >= MIN_GAMES].copy()
    return df


def materialize_feature_table(*, as_of_date: str | None = None) -> int:
    """
    Persist gold feature rows to SQLite for point-in-time replay and faster scan/train.
    """
    df = build_feature_table()
    if as_of_date:
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(as_of_date)].copy()
    if df.empty:
        return 0
    keep = ["player_id", "player_name", "game_date", "game_id", "tb"] + list(MODEL_FEATURES)
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    out = df[keep].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    engine = _engine()
    with engine.begin() as conn:
        out.to_sql(FEATURES_TABLE, conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_{FEATURES_TABLE}_player_date "
            f"ON {FEATURES_TABLE}(player_id, game_date)"
        )
    return len(out)


def load_features_as_of(
    as_of_date: str,
    player_ids: Collection[int] | None = None,
) -> pd.DataFrame:
    """
    Latest feature row per player with ``game_date < as_of_date`` from materialized table.
    Falls back to ``build_feature_table`` if table missing.
    """
    engine = _engine()
    try:
        if player_ids is not None and len(list(player_ids)) > 0:
            ids_sql = ",".join(str(int(x)) for x in sorted(set(int(x) for x in player_ids)))
            q = (
                f"SELECT * FROM {FEATURES_TABLE} WHERE player_id IN ({ids_sql}) "
                "AND date(game_date) < date(:gd)"
            )
            df = pd.read_sql(q, engine, params={"gd": as_of_date})
        else:
            df = pd.read_sql(
                f"SELECT * FROM {FEATURES_TABLE} WHERE date(game_date) < date(:gd)",
                engine,
                params={"gd": as_of_date},
            )
    except Exception:
        df = build_feature_table(player_ids=player_ids)
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(as_of_date)].copy()
        return df

    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).groupby("player_id", as_index=False).tail(1)
    return df

