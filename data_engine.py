"""
data_engine.py (MLB)
--------------------
Build a historical batter game-log store for Total Bases (TB) using MLB Stats API.

Stores one row per player-game into SQLite at config.DB_PATH.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import sqlalchemy as sa

import statsapi

from config import DB_PATH, SEASONS

log = logging.getLogger(__name__)


def _engine():
    return sa.create_engine(f"sqlite:///{DB_PATH}")


def _compute_tb(row: dict) -> int:
    # TB = 1B + 2*2B + 3*3B + 4*HR where 1B = H - 2B - 3B - HR
    h = int(row.get("hits", 0) or 0)
    d2 = int(row.get("doubles", 0) or 0)
    d3 = int(row.get("triples", 0) or 0)
    hr = int(row.get("homeRuns", 0) or 0)
    singles = max(0, h - d2 - d3 - hr)
    return int(singles + 2 * d2 + 3 * d3 + 4 * hr)


def fetch_games_for_season(season: int) -> list[dict]:
    """
    Return a list of scheduled games with game_id and date.
    Uses regular season by default; you can expand later.
    """
    # statsapi.schedule returns list[dict] with keys including 'game_id' and 'game_date'
    games = statsapi.schedule(start_date=f"{season}-03-01", end_date=f"{season}-11-30")
    # Filter out non-MLB or missing ids
    out = []
    for g in games:
        gid = g.get("game_id")
        gdate = g.get("game_date")
        if not gid or not gdate:
            continue
        out.append({"game_id": int(gid), "game_date": str(gdate)[:10]})
    return out


def fetch_batter_rows_for_game(game_id: int, game_date: str) -> list[dict]:
    """
    Pull boxscore data and return batter stat rows for the game.
    """
    box = statsapi.boxscore_data(game_id)
    # NOTE: statsapi.boxscore_data returns top-level keys like "away"/"home",
    # each containing a "players" dict keyed by "ID{player_id}".
    rows: list[dict] = []
    for side in ("away", "home"):
        team = box.get(side, {}) or {}
        players = team.get("players", {}) or {}
        team_abbr = (team.get("team", {}) or {}).get("abbreviation", "")
        for _, p in players.items():
            person = p.get("person", {}) or {}
            stats = (p.get("stats", {}) or {}).get("batting", {}) or {}
            # Only include players who actually batted (AB or PA-like presence)
            ab = int(stats.get("atBats", 0) or 0)
            if ab <= 0 and int(stats.get("plateAppearances", 0) or 0) <= 0:
                continue

            row = {
                "game_id": game_id,
                "game_date": game_date,
                "season": int(game_date[:4]),
                "team": team_abbr,
                "player_id": int(person.get("id", 0) or 0),
                "player_name": str(person.get("fullName", "") or ""),
                "ab": ab,
                "h": int(stats.get("hits", 0) or 0),
                "2b": int(stats.get("doubles", 0) or 0),
                "3b": int(stats.get("triples", 0) or 0),
                "hr": int(stats.get("homeRuns", 0) or 0),
                "bb": int(stats.get("baseOnBalls", 0) or 0),
                "so": int(stats.get("strikeOuts", 0) or 0),
                "rbi": int(stats.get("rbi", 0) or 0),
                "sb": int(stats.get("stolenBases", 0) or 0),
            }
            row["tb"] = _compute_tb(
                {"hits": row["h"], "doubles": row["2b"], "triples": row["3b"], "homeRuns": row["hr"]}
            )
            rows.append(row)
    return rows


def build_historical_store(seasons: list[int] | None = None) -> None:
    seasons = seasons or SEASONS
    engine = _engine()

    all_rows: list[dict] = []
    for season in seasons:
        log.info(f"Fetching MLB games for season {season}...")
        games = fetch_games_for_season(int(season))
        log.info(f"Season {season}: {len(games)} games in schedule window")
        for i, g in enumerate(games, start=1):
            gid = g["game_id"]
            gdate = g["game_date"]
            try:
                rows = fetch_batter_rows_for_game(gid, gdate)
                all_rows.extend(rows)
            except Exception as e:
                # API hiccups happen; skip and continue
                log.debug(f"Skip game_id={gid} date={gdate}: {e}")

            if i % 250 == 0:
                log.info(f"  processed {i}/{len(games)} games...")

    if not all_rows:
        raise RuntimeError("No batter rows collected. Check Stats API connectivity.")

    df = pd.DataFrame(all_rows)
    df["game_date"] = pd.to_datetime(df["game_date"])

    with engine.begin() as conn:
        df.to_sql("batter_games", conn, if_exists="replace", index=False)

        # Helpful indexes
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_batter_games_player_date ON batter_games(player_id, game_date)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_batter_games_date ON batter_games(game_date)")

    log.info(f"ETL complete: wrote {len(df):,} batter-game rows to {DB_PATH}")

