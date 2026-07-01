"""
statcast_engine.py (MLB TB)
---------------------------
Pull Baseball Savant (Statcast) pitch-level data via pybaseball and aggregate it
into per-batter-game and per-pitcher-game quality-of-contact tables.

Savant uses MLBAM player ids, which match ``player_id`` / ``pitcher_id`` already
stored in ``batter_games`` / ``pitcher_games`` -- no identity bridge is needed.

Memory safety: pitch-level pulls are large, so we fetch in date-range chunks,
aggregate each chunk immediately to the (small) per-game grain, discard the raw
frame, and cache the aggregated chunk to parquet so re-runs skip the network.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import sqlalchemy as sa

from config import (
    DB_PATH,
    SEASONS,
    STATCAST_BATTER_TABLE,
    STATCAST_CACHE_DIR,
    STATCAST_CHUNK_DAYS,
    STATCAST_MIN_BBE,
    STATCAST_PITCHER_TABLE,
)

log = logging.getLogger(__name__)

# Columns required from the Savant pull; absence means the upstream schema changed.
_REQUIRED_COLUMNS = frozenset(
    {
        "game_date",
        "game_pk",
        "batter",
        "pitcher",
        "type",
        "description",
        "launch_speed",
        "launch_speed_angle",
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
    }
)

BATTER_METRIC_COLUMNS = ["xwoba", "xslg", "avg_ev", "barrel_rate", "hardhit_rate", "bbe"]
PITCHER_METRIC_COLUMNS = [
    "xwoba_against",
    "xslg_against",
    "barrel_rate_allowed",
    "csw_pct",
    "whiff_pct",
    "pitches",
    "bbe",
]

# Statcast ``description`` categories used for plate-discipline rates.
_SWINGING_STRIKE = frozenset({"swinging_strike", "swinging_strike_blocked"})
_WHIFF = _SWINGING_STRIKE | {"foul_tip", "missed_bunt"}
_FOUL = frozenset({"foul", "foul_bunt"})
_HARD_HIT_MPH = 95.0
_BARREL_CODE = 6  # launch_speed_angle == 6 marks a barrel


def _engine() -> sa.Engine:
    return sa.create_engine(f"sqlite:///{DB_PATH}")


def _validate_raw(raw: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Statcast pull missing required columns: {sorted(missing)}")


def aggregate_batter_games(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Savant rows to one row per (player_id, game_id).

    Returns columns: player_id, game_id, game_date + ``BATTER_METRIC_COLUMNS``.
    Metrics are computed over batted-ball events (``type == 'X'``) only.
    """
    _validate_raw(raw)
    bip = raw[raw["type"].astype(str) == "X"]
    if bip.empty:
        return pd.DataFrame(columns=["player_id", "game_id", "game_date"] + BATTER_METRIC_COLUMNS)

    ls = pd.to_numeric(bip["launch_speed"], errors="coerce")  # Shape: (n_bbe,)
    lsa = pd.to_numeric(bip["launch_speed_angle"], errors="coerce")
    xw = pd.to_numeric(bip["estimated_woba_using_speedangle"], errors="coerce")
    xs = pd.to_numeric(bip["estimated_slg_using_speedangle"], errors="coerce")
    work = bip.assign(
        _ls=ls,
        _xw=xw,
        _xs=xs,
        _barrel=(lsa == _BARREL_CODE).astype(float),
        _hard=(ls >= _HARD_HIT_MPH).astype(float),
    )
    grp = work.groupby(["batter", "game_pk", "game_date"], as_index=False)
    out = grp.agg(
        xwoba=("_xw", "mean"),
        xslg=("_xs", "mean"),
        avg_ev=("_ls", "mean"),
        barrel_rate=("_barrel", "mean"),
        hardhit_rate=("_hard", "mean"),
        bbe=("_ls", "size"),
    )
    out = out.rename(columns={"batter": "player_id", "game_pk": "game_id"})
    return _finalize_grain(out)


def aggregate_pitcher_games(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level Savant rows to one row per (pitcher_id, game_id).

    Returns columns: player_id, game_id, game_date + ``PITCHER_METRIC_COLUMNS``.
    Contact-quality metrics use batted balls; discipline rates use all pitches.
    """
    _validate_raw(raw)
    if raw.empty:
        return pd.DataFrame(columns=["player_id", "game_id", "game_date"] + PITCHER_METRIC_COLUMNS)

    desc = raw["description"].astype(str)
    bip_mask = raw["type"].astype(str) == "X"
    ls = pd.to_numeric(raw["launch_speed"], errors="coerce")
    lsa = pd.to_numeric(raw["launch_speed_angle"], errors="coerce")
    xw = pd.to_numeric(raw["estimated_woba_using_speedangle"], errors="coerce")
    xs = pd.to_numeric(raw["estimated_slg_using_speedangle"], errors="coerce")

    is_called = desc.eq("called_strike")
    is_sw = desc.isin(_SWINGING_STRIKE)
    is_whiff = desc.isin(_WHIFF)
    is_swing = is_whiff | desc.isin(_FOUL) | desc.eq("hit_into_play")

    work = raw.assign(
        _pitch=1.0,
        _csw=(is_called | is_sw).astype(float),
        _whiff=is_whiff.astype(float),
        _swing=is_swing.astype(float),
        _bip=bip_mask.astype(float),
        _barrel=np.where(bip_mask, (lsa == _BARREL_CODE).astype(float), np.nan),
        _xw=np.where(bip_mask, xw, np.nan),
        _xs=np.where(bip_mask, xs, np.nan),
    )
    grp = work.groupby(["pitcher", "game_pk", "game_date"], as_index=False)
    agg = grp.agg(
        pitches=("_pitch", "sum"),
        _csw_n=("_csw", "sum"),
        _whiff_n=("_whiff", "sum"),
        _swing_n=("_swing", "sum"),
        bbe=("_bip", "sum"),
        _barrel_n=("_barrel", "sum"),
        xwoba_against=("_xw", "mean"),
        xslg_against=("_xs", "mean"),
    )
    pitches_safe = agg["pitches"].clip(lower=1.0)
    agg["csw_pct"] = (agg["_csw_n"] / pitches_safe).clip(0, 1)
    agg["whiff_pct"] = (agg["_whiff_n"] / agg["_swing_n"].clip(lower=1.0)).clip(0, 1)
    agg["barrel_rate_allowed"] = (agg["_barrel_n"] / agg["bbe"].clip(lower=1.0)).clip(0, 1)
    out = agg.rename(columns={"pitcher": "player_id", "game_pk": "game_id"})
    keep = ["player_id", "game_id", "game_date"] + PITCHER_METRIC_COLUMNS
    return _finalize_grain(out[keep])


def _finalize_grain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "game_id"])
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"] = df["game_id"].astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.reset_index(drop=True)


def _date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    step = max(1, int(days))
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=step - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _cache_paths(start: date, end: date) -> tuple[Path, Path]:
    STATCAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.isoformat()}_{end.isoformat()}"
    return (
        STATCAST_CACHE_DIR / f"batter_{tag}.parquet",
        STATCAST_CACHE_DIR / f"pitcher_{tag}.parquet",
    )


def _fetch_savant_chunk(start: date, end: date) -> pd.DataFrame:
    """Pull raw pitch-level Savant data for an inclusive date range (network)."""
    from pybaseball import statcast

    raw = statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
    if raw is None:
        return pd.DataFrame()
    return raw


def _aggregate_chunk(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (batter_agg, pitcher_agg) for a chunk, using parquet cache if present."""
    b_path, p_path = _cache_paths(start, end)
    if b_path.exists() and p_path.exists():
        return pd.read_parquet(b_path), pd.read_parquet(p_path)

    raw = _fetch_savant_chunk(start, end)
    if raw is None or raw.empty:
        b_df = pd.DataFrame(columns=["player_id", "game_id", "game_date"] + BATTER_METRIC_COLUMNS)
        p_df = pd.DataFrame(columns=["player_id", "game_id", "game_date"] + PITCHER_METRIC_COLUMNS)
    else:
        b_df = aggregate_batter_games(raw)
        p_df = aggregate_pitcher_games(raw)
    del raw  # release pitch-level frame before next chunk

    b_df.to_parquet(b_path, index=False)
    p_df.to_parquet(p_path, index=False)
    return b_df, p_df


def _max_game_date(engine: sa.Engine, table: str) -> date | None:
    try:
        df = pd.read_sql(f"SELECT MAX(game_date) AS m FROM {table}", engine)
    except Exception:
        return None
    if df.empty or pd.isna(df.iloc[0]["m"]):
        return None
    return pd.to_datetime(df.iloc[0]["m"]).date()


def _existing_game_ids(engine: sa.Engine, table: str) -> set[int]:
    try:
        df = pd.read_sql(f"SELECT DISTINCT game_id FROM {table}", engine)
    except Exception:
        return set()
    return {int(x) for x in df["game_id"].dropna().tolist()}


def _write_table(
    engine: sa.Engine,
    table: str,
    frame: pd.DataFrame,
    *,
    incremental: bool,
) -> int:
    if frame.empty:
        return 0
    frame = frame.drop_duplicates(subset=["player_id", "game_id"], keep="last")
    if incremental:
        existing = _existing_game_ids(engine, table)
        if existing:
            frame = frame[~frame["game_id"].isin(existing)]
    if frame.empty:
        return 0
    if_exists = "append" if incremental else "replace"
    with engine.begin() as conn:
        frame.to_sql(table, conn, if_exists=if_exists, index=False)
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_player_game ON {table}(player_id, game_id)"
        )
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_player_date ON {table}(player_id, game_date)"
        )
    return len(frame)


def build_statcast_store(
    seasons: list[int] | None = None,
    *,
    incremental: bool = False,
    chunk_days: int = STATCAST_CHUNK_DAYS,
) -> dict[str, int]:
    """
    Populate ``statcast_batter_games`` / ``statcast_pitcher_games`` for ``seasons``.

    Aggregated chunks are accumulated (small) and written once per run; raw
    pitch-level frames are never held across chunks.
    """
    seasons = seasons or SEASONS
    engine = _engine()
    start_floor = _max_game_date(engine, STATCAST_BATTER_TABLE) if incremental else None

    batter_frames: list[pd.DataFrame] = []
    pitcher_frames: list[pd.DataFrame] = []
    today = date.today()

    for season in sorted(int(s) for s in seasons):
        s_start = date(season, 3, 1)
        s_end = min(date(season, 11, 30), today)
        if incremental and start_floor is not None:
            s_start = max(s_start, start_floor + timedelta(days=1))
        if s_start > s_end:
            continue
        log.info("Statcast season %s: %s -> %s", season, s_start, s_end)
        for c_start, c_end in _date_chunks(s_start, s_end, chunk_days):
            try:
                b_df, p_df = _aggregate_chunk(c_start, c_end)
            except Exception as e:
                log.warning("Statcast chunk %s..%s failed: %s", c_start, c_end, e)
                continue
            if not b_df.empty:
                batter_frames.append(b_df)
            if not p_df.empty:
                pitcher_frames.append(p_df)
            log.info("  chunk %s..%s: %s batter-games, %s pitcher-games", c_start, c_end, len(b_df), len(p_df))

    b_all = (
        pd.concat(batter_frames, ignore_index=True)
        if batter_frames
        else pd.DataFrame(columns=["player_id", "game_id", "game_date"] + BATTER_METRIC_COLUMNS)
    )
    p_all = (
        pd.concat(pitcher_frames, ignore_index=True)
        if pitcher_frames
        else pd.DataFrame(columns=["player_id", "game_id", "game_date"] + PITCHER_METRIC_COLUMNS)
    )
    if STATCAST_MIN_BBE > 0 and not b_all.empty:
        b_all = b_all[pd.to_numeric(b_all["bbe"], errors="coerce").fillna(0) >= STATCAST_MIN_BBE]

    n_b = _write_table(engine, STATCAST_BATTER_TABLE, b_all, incremental=incremental)
    n_p = _write_table(engine, STATCAST_PITCHER_TABLE, p_all, incremental=incremental)
    log.info("Statcast store: wrote %s batter-game rows, %s pitcher-game rows", n_b, n_p)
    return {"batter_rows": n_b, "pitcher_rows": n_p}
