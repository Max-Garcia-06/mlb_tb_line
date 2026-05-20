"""
data_engine.py (MLB)
--------------------
Build a historical batter game-log store for Total Bases (TB) using MLB Stats API.

Stores one row per player-game into SQLite at config.DB_PATH.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import lru_cache

import pandas as pd
import sqlalchemy as sa

import statsapi

from config import DB_PATH, SEASONS
from venue_physics import lookup_park_physics

log = logging.getLogger(__name__)


def _engine():
    return sa.create_engine(f"sqlite:///{DB_PATH}")


def fetch_game_environment(game_id: int) -> dict:
    """
    Venue + weather snapshot for a gamePk (best-effort; defaults if API fails).

    park_team_abbr is the home team's abbreviation (park context for all PAs).
    """
    default = {
        "park_team_abbr": "",
        "venue_id": None,
        "temp_f": None,
        "wind_mph": None,
        "wind_l_to_r": 0,
    }
    try:
        d = statsapi.get("game", {"gamePk": int(game_id)})
    except Exception:
        return default
    gd = (d or {}).get("gameData") or {}
    teams = gd.get("teams") or {}
    home = teams.get("home") or {}
    weather = gd.get("weather") or {}
    venue = gd.get("venue") or {}
    abbr = str(home.get("abbreviation") or "").strip().upper()
    temp_raw = weather.get("temp")
    temp_f = None
    if temp_raw is not None:
        try:
            temp_f = float(str(temp_raw).replace("°", "").strip())
        except ValueError:
            temp_f = None
    wind_s = str(weather.get("wind") or "")
    wind_mph = None
    m = re.search(r"(\d+)\s*mph", wind_s, re.I)
    if m:
        try:
            wind_mph = float(m.group(1))
        except ValueError:
            wind_mph = None
    l_to_r = 1 if re.search(r"\b[LR]\s+To\s+R\b", wind_s, re.I) else 0
    vid = venue.get("id")
    try:
        venue_id = int(vid) if vid is not None else None
    except (TypeError, ValueError):
        venue_id = None
    return {
        "park_team_abbr": abbr,
        "venue_id": venue_id,
        "temp_f": temp_f,
        "wind_mph": wind_mph,
        "wind_l_to_r": int(l_to_r),
    }


def _compute_tb(row: dict) -> int:
    # TB = 1B + 2*2B + 3*3B + 4*HR where 1B = H - 2B - 3B - HR
    h = int(row.get("hits", 0) or 0)
    d2 = int(row.get("doubles", 0) or 0)
    d3 = int(row.get("triples", 0) or 0)
    hr = int(row.get("homeRuns", 0) or 0)
    singles = max(0, h - d2 - d3 - hr)
    return int(singles + 2 * d2 + 3 * d3 + 4 * hr)


# MLB schedule statuses where the game has started or finished (exclude from live scan).
_STARTED_GAME_STATUSES = frozenset(
    {
        "In Progress",
        "Final",
        "Game Over",
        "Completed Early",
    }
)


@lru_cache(maxsize=1)
def _mlb_team_abbreviations() -> frozenset[str]:
    teams = statsapi.get("teams", {"sportId": 1}).get("teams", []) or []
    return frozenset(str(t.get("abbreviation", "")).strip().upper() for t in teams if t.get("abbreviation"))


def parse_kalshi_event_matchup(event_ticker: str) -> tuple[str, str] | None:
    """
    Parse away/home team abbreviations from a Kalshi event ticker, e.g.
    ``KXMLBTB-26MAY201940BOSKC`` -> (``BOS``, ``KC``).
    """
    et = (event_ticker or "").strip()
    m = re.search(r"KXMLBTB-\d{2}[A-Z]{3}\d{2}\d{4}([A-Z]+)$", et)
    if not m:
        return None
    blob = m.group(1)
    abbrs = _mlb_team_abbreviations()
    for i in range(2, len(blob)):
        away = blob[:i]
        home = blob[i:]
        if away in abbrs and home in abbrs:
            return away, home
    return None


def matchup_slug(away_abbr: str, home_abbr: str) -> str:
    return f"{away_abbr}{home_abbr}"


def game_status_allows_scan(status: str) -> bool:
    return (status or "").strip() not in _STARTED_GAME_STATUSES


def matchup_status_map(game_date: str) -> dict[str, str]:
    """Map ``TEXCOL``-style keys to MLB schedule status for ``game_date`` (YYYY-MM-DD)."""
    teams = statsapi.get("teams", {"sportId": 1}).get("teams", []) or []
    abbr_by_id = {int(t["id"]): str(t["abbreviation"]).strip().upper() for t in teams if t.get("id")}
    out: dict[str, str] = {}
    for g in statsapi.schedule(date=game_date) or []:
        away_id = g.get("away_id")
        home_id = g.get("home_id")
        if away_id is None or home_id is None:
            continue
        away = abbr_by_id.get(int(away_id), "")
        home = abbr_by_id.get(int(home_id), "")
        if not away or not home:
            continue
        out[matchup_slug(away, home)] = str(g.get("status", "") or "").strip()
    return out


def filter_market_lines_pregame(
    market_lines: list,
    game_date: str,
) -> tuple[list, list[tuple[str, str, str]]]:
    """
    Drop markets tied to MLB games that have already started or finished.

    Returns ``(kept_lines, excluded)`` where each excluded entry is
    ``(event_ticker, matchup_slug, status)``.
    """
    status_by_matchup = matchup_status_map(game_date)
    kept: list = []
    excluded: list[tuple[str, str, str]] = []
    seen_events: set[str] = set()
    for ml in market_lines:
        et = str(getattr(ml, "event_ticker", "") or "").strip()
        if not et and getattr(ml, "ticker", ""):
            parts = str(ml.ticker).split("-")
            if len(parts) >= 2:
                et = f"{parts[0]}-{parts[1]}"
        matchup = parse_kalshi_event_matchup(et) if et else None
        if not matchup:
            kept.append(ml)
            continue
        key = matchup_slug(*matchup)
        status = status_by_matchup.get(key, "")
        if not status or game_status_allows_scan(status):
            kept.append(ml)
            continue
        if et not in seen_events:
            seen_events.add(et)
            excluded.append((et, key, status))
    return kept, excluded


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
    env = fetch_game_environment(game_id)
    dai, elev = lookup_park_physics(env.get("park_team_abbr"))
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
                "park_team_abbr": env.get("park_team_abbr") or "",
                "venue_id": env.get("venue_id"),
                "temp_f": env.get("temp_f"),
                "wind_mph": env.get("wind_mph"),
                "wind_l_to_r": int(env.get("wind_l_to_r") or 0),
                "venue_distance_added_index": float(dai),
                "venue_elevation_ft": float(elev),
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

