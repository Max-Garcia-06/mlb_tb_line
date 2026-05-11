"""
feature_store.py (MLB TB)
------------------------
Builds a feature table from batter game logs for training and live prediction.
"""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

from config import DB_PATH, ROLLING_WINDOW, MIN_GAMES


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
]


def _engine():
    return sa.create_engine(f"sqlite:///{DB_PATH}")


def build_feature_table() -> pd.DataFrame:
    engine = _engine()
    df = pd.read_sql("SELECT * FROM batter_games", engine)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

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

