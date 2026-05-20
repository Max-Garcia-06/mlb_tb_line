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

from config import DB_PATH, ROLLING_WINDOW, MIN_GAMES
from venue_physics import lookup_park_physics


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
]


def _engine():
    return sa.create_engine(f"sqlite:///{DB_PATH}")


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

    # Filter for training cohort
    df = df[df["games_played"] >= MIN_GAMES].copy()
    return df

