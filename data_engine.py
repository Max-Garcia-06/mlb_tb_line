"""
data_engine.py (MLB)
--------------------
Build a historical batter game-log store for Total Bases (TB) using MLB Stats API.

Stores one row per player-game into SQLite at config.DB_PATH.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable

import pandas as pd
import sqlalchemy as sa

import statsapi

from config import DB_PATH, MLB_API_MAX_RETRIES, MLB_API_RETRY_SLEEP_SEC, SEASONS
from venue_physics import lookup_park_physics
from matchup_features import (
    _pitcher_hand_from_name,
    build_bats_hand_cache,
    enrich_batter_row_from_boxscore,
    schedule_games_by_date,
)

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


def _parse_game_datetime_utc(raw: str) -> datetime | None:
    """Parse MLB ``game_datetime`` (e.g. ``2026-05-24T16:15:00Z``) to aware UTC."""
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_ticker_from_market_line(ml) -> str:
    et = str(getattr(ml, "event_ticker", "") or "").strip()
    if not et and getattr(ml, "ticker", ""):
        parts = str(ml.ticker).split("-")
        if len(parts) >= 2:
            et = f"{parts[0]}-{parts[1]}"
    return et


def slate_schedule_index(game_date: str) -> dict[str, dict]:
    """
    Map matchup slug (e.g. ``BOSKC``) -> ``{status, start_utc}`` for ``game_date``.
    """
    teams = statsapi.get("teams", {"sportId": 1}).get("teams", []) or []
    abbr_by_id = {int(t["id"]): str(t["abbreviation"]).strip().upper() for t in teams if t.get("id")}
    out: dict[str, dict] = {}
    for g in schedule_games_by_date(game_date):
        away_id = g.get("away_id")
        home_id = g.get("home_id")
        if away_id is None or home_id is None:
            continue
        away = abbr_by_id.get(int(away_id), "")
        home = abbr_by_id.get(int(home_id), "")
        if not away or not home:
            continue
        slug = matchup_slug(away, home)
        out[slug] = {
            "status": str(g.get("status", "") or "").strip(),
            "start_utc": _parse_game_datetime_utc(str(g.get("game_datetime", "") or "")),
        }
    return out


def matchup_status_map(game_date: str) -> dict[str, str]:
    """Map ``TEXCOL``-style keys to MLB schedule status for ``game_date`` (YYYY-MM-DD)."""
    return {slug: str(row.get("status", "") or "") for slug, row in slate_schedule_index(game_date).items()}


def matchup_start_time_map(game_date: str) -> dict[str, datetime]:
    """Map matchup slug -> first-pitch time (UTC) for ``game_date``."""
    out: dict[str, datetime] = {}
    for slug, row in slate_schedule_index(game_date).items():
        start = row.get("start_utc")
        if isinstance(start, datetime):
            out[slug] = start
    return out


def filter_market_lines_by_start_window(
    market_lines: list,
    game_date: str,
    *,
    within_hours: float,
    now: datetime | None = None,
    schedule_index: dict[str, dict] | None = None,
) -> tuple[list, list[tuple[str, str, str, str]]]:
    """
    Keep only markets whose MLB game starts within ``within_hours`` of ``now`` (UTC).

    Returns ``(kept_lines, excluded)`` where each excluded entry is
    ``(event_ticker, matchup_slug, game_datetime_iso, reason)``.
    Unparseable event tickers or missing schedule rows are excluded (fail closed).
    """
    if within_hours <= 0:
        return list(market_lines), []

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    idx = schedule_index if schedule_index is not None else slate_schedule_index(game_date)
    max_sec = float(within_hours) * 3600.0
    kept: list = []
    excluded: list[tuple[str, str, str, str]] = []
    seen_events: set[str] = set()

    for ml in market_lines:
        et = _event_ticker_from_market_line(ml)
        matchup = parse_kalshi_event_matchup(et) if et else None
        if not matchup:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, "", "", "unparseable_event"))
            continue
        key = matchup_slug(*matchup)
        row = idx.get(key)
        if not row:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, "", "no_schedule"))
            continue
        start = row.get("start_utc")
        if not isinstance(start, datetime):
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, "", "no_start_time"))
            continue
        start_utc = start.astimezone(timezone.utc)
        delta_sec = (start_utc - now_utc).total_seconds()
        start_iso = start_utc.isoformat().replace("+00:00", "Z")
        if delta_sec < 0:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, start_iso, "already_started"))
            continue
        if delta_sec > max_sec:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, start_iso, "too_far"))
            continue
        kept.append(ml)

    return kept, excluded


def filter_market_lines_pregame(
    market_lines: list,
    game_date: str,
    schedule_index: dict[str, dict] | None = None,
) -> tuple[list, list[tuple[str, str, str]]]:
    """
    Drop markets tied to MLB games that have already started or finished.

    Returns ``(kept_lines, excluded)`` where each excluded entry is
    ``(event_ticker, matchup_slug, status)``.
    """
    if schedule_index is not None:
        status_by_matchup = {slug: str(row.get("status", "") or "") for slug, row in schedule_index.items()}
    else:
        status_by_matchup = matchup_status_map(game_date)
    kept: list = []
    excluded: list[tuple[str, str, str]] = []
    seen_events: set[str] = set()
    for ml in market_lines:
        et = _event_ticker_from_market_line(ml)
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


@lru_cache(maxsize=512)
def _opp_sp_hand_L_by_game_id(game_date: str) -> dict[int, tuple[float, float]]:
    """game_id -> (away_sp_hand_L, home_sp_hand_L) from schedule probable pitchers."""
    out: dict[int, tuple[float, float]] = {}
    for g in schedule_games_by_date(game_date):
        gid = int(g.get("game_id", 0) or 0)
        if not gid:
            continue
        away_sp = str(g.get("away_probable_pitcher", "") or "").strip()
        home_sp = str(g.get("home_probable_pitcher", "") or "").strip()
        ah = _pitcher_hand_from_name(away_sp)
        hh = _pitcher_hand_from_name(home_sp)
        out[gid] = (
            1.0 if str(ah).upper().startswith("L") else 0.0,
            1.0 if str(hh).upper().startswith("L") else 0.0,
        )
    return out


_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})


def _retry_mlb_api(fn: Callable[[], object], *, label: str) -> object:
    """Retry transient MLB Stats API failures (502/503/etc.)."""
    last_exc: Exception | None = None
    attempts = max(1, int(MLB_API_MAX_RETRIES))
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status in _RETRYABLE_HTTP if status is not None else True
            if not retryable or attempt >= attempts - 1:
                raise
            sleep_sec = float(MLB_API_RETRY_SLEEP_SEC) * (2**attempt)
            log.warning(
                "%s failed (%s); retry %s/%s in %.1fs",
                label,
                e,
                attempt + 2,
                attempts,
                sleep_sec,
            )
            time.sleep(sleep_sec)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with no exception")


def fetch_games_for_season(season: int) -> list[dict]:
    """
    Return a list of scheduled games with game_id and date.
    Uses regular season by default; you can expand later.
    """
    # statsapi.schedule returns list[dict] with keys including 'game_id' and 'game_date'
    raw = _retry_mlb_api(
        lambda: statsapi.schedule(start_date=f"{season}-03-01", end_date=f"{season}-11-30"),
        label=f"MLB schedule season {season}",
    )
    games = list(raw) if raw is not None else []
    # Filter out non-MLB or missing ids
    out = []
    for g in games:
        gid = g.get("game_id")
        gdate = g.get("game_date")
        if not gid or not gdate:
            continue
        out.append({"game_id": int(gid), "game_date": str(gdate)[:10]})
    return out


def _parse_ip(ip_str: str) -> float:
    """Convert MLB IP string ('6.2' = 6⅔ innings) to decimal float."""
    try:
        parts = str(ip_str or "0").split(".")
        whole = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return whole + thirds / 3.0
    except (ValueError, IndexError):
        return 0.0


def fetch_batter_rows_for_game(
    game_id: int,
    game_date: str,
    *,
    fetch_weather: bool = False,
    bats_hand_cache: dict[int, str] | None = None,
) -> list[dict]:
    """
    Pull boxscore data and return batter stat rows for the game.

    By default skips the per-game ``game`` API call (weather); park context uses the
    home team from the boxscore. Set ``fetch_weather=True`` for temp/wind fields.
    """
    box = _retry_mlb_api(
        lambda: statsapi.boxscore_data(game_id),
        label=f"boxscore game_id={game_id}",
    )
    if not isinstance(box, dict):
        raise TypeError(f"boxscore_data returned {type(box)!r}, expected dict")
    team_info = box.get("teamInfo", {}) or {}
    away_abbr = str((team_info.get("away", {}) or {}).get("abbreviation", "") or "").upper()
    home_abbr = str((team_info.get("home", {}) or {}).get("abbreviation", "") or "").upper()
    if fetch_weather:
        env = fetch_game_environment(game_id)
        park_abbr = str(env.get("park_team_abbr") or home_abbr)
    else:
        env = {
            "park_team_abbr": home_abbr,
            "venue_id": None,
            "temp_f": None,
            "wind_mph": None,
            "wind_l_to_r": 0,
        }
        park_abbr = home_abbr
    dai, elev = lookup_park_physics(park_abbr)
    sp_map = _opp_sp_hand_L_by_game_id(str(game_date)[:10])
    away_sp_L, home_sp_L = sp_map.get(int(game_id), (0.5, 0.5))
    # NOTE: statsapi.boxscore_data returns top-level keys like "away"/"home",
    # each containing a "players" dict keyed by "ID{player_id}".
    rows: list[dict] = []
    for side in ("away", "home"):
        team = box.get(side, {}) or {}
        players = team.get("players", {}) or {}
        team_abbr = away_abbr if side == "away" else home_abbr
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
            enrich_batter_row_from_boxscore(
                row=row,
                side=side,
                team_abbr=team_abbr,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                player_meta=p,
                bats_hand_cache=bats_hand_cache,
            )
            is_home = int(row.get("is_home", 0) or 0)
            row["opp_sp_hand_L"] = float(away_sp_L if is_home else home_sp_L)
            rows.append(row)
    return rows


def _existing_game_ids(engine: sa.Engine) -> set[int]:
    try:
        ids = pd.read_sql("SELECT DISTINCT game_id FROM batter_games", engine)["game_id"]
        return {int(x) for x in ids.dropna().tolist()}
    except Exception:
        return set()


def fetch_pitcher_rows_for_game(game_id: int, game_date: str) -> list[dict]:
    """Return one row per pitcher-appearance for a game."""
    box = _retry_mlb_api(
        lambda: statsapi.boxscore_data(game_id),
        label=f"boxscore(pitcher) game_id={game_id}",
    )
    if not isinstance(box, dict):
        return []
    team_info = box.get("teamInfo", {}) or {}
    rows: list[dict] = []
    for side in ("away", "home"):
        team_data = box.get(side, {}) or {}
        players = team_data.get("players", {}) or {}
        team_abbr = str((team_info.get(side, {}) or {}).get("abbreviation", "") or "").upper()
        pitcher_ids: list[int] = team_data.get("pitchers", []) or []
        for rank, pid in enumerate(pitcher_ids):
            p = players.get(f"ID{pid}", {}) or {}
            person = p.get("person", {}) or {}
            stats = (p.get("stats", {}) or {}).get("pitching", {}) or {}
            ip = _parse_ip(str(stats.get("inningsPitched", "0") or "0"))
            pitcher_id = int(person.get("id", 0) or 0)
            if pitcher_id <= 0:
                continue
            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "season": int(game_date[:4]),
                "team": team_abbr,
                "pitcher_id": pitcher_id,
                "pitcher_name": str(person.get("fullName", "") or ""),
                "is_starter": int(rank == 0),
                "ip": ip,
                "h": int(stats.get("hits", 0) or 0),
                "er": int(stats.get("earnedRuns", 0) or 0),
                "bb": int(stats.get("baseOnBalls", 0) or 0),
                "so": int(stats.get("strikeOuts", 0) or 0),
                "hr": int(stats.get("homeRuns", 0) or 0),
            })
    return rows


def get_probable_starters(game_date: str) -> dict[str, int]:
    """Return {team_abbr: pitcher_id} for probable starters on game_date."""
    try:
        data = statsapi.get("schedule", {
            "date": game_date,
            "sportId": 1,
            "hydrate": "probablePitcher,team",
        })
        result: dict[str, int] = {}
        for date_entry in (data.get("dates") or []):
            for game in (date_entry.get("games") or []):
                for side in ("away", "home"):
                    team = (game.get(side) or {}).get("team") or {}
                    abbr = str(team.get("abbreviation", "") or "").strip().upper()
                    prob = (game.get(side) or {}).get("probablePitcher") or {}
                    pid = int(prob.get("id", 0) or 0)
                    if abbr and pid:
                        result[abbr] = pid
        return result
    except Exception as e:
        log.debug("get_probable_starters failed: %s", e)
        return {}


def get_confirmed_lineups(game_date: str) -> dict[int, int]:
    """
    Return {player_id: batting_slot (1-9)} for confirmed lineups on ``game_date``.

    Lineups typically post ~2-3h before first pitch. Returns empty when not yet set.
    """
    try:
        data = statsapi.get(
            "schedule",
            {"date": game_date, "sportId": 1, "hydrate": "lineups"},
        )
    except Exception as e:
        log.debug("get_confirmed_lineups failed: %s", e)
        return {}
    out: dict[int, int] = {}
    for date_entry in (data.get("dates") or []):
        for game in (date_entry.get("games") or []):
            lineups = game.get("lineups") or {}
            for side in ("homePlayers", "awayPlayers"):
                players = lineups.get(side) or []
                for slot, person in enumerate(players[:9], start=1):
                    pid = int((person or {}).get("id", 0) or 0)
                    if pid:
                        out[pid] = slot
    return out


def _existing_pitcher_game_ids(engine: sa.Engine) -> set[int]:
    try:
        ids = pd.read_sql("SELECT DISTINCT game_id FROM pitcher_games", engine)["game_id"]
        return {int(x) for x in ids.dropna().tolist()}
    except Exception:
        return set()


def _fetch_pitcher_games_parallel(
    games: list[dict],
    *,
    workers: int,
) -> list[dict]:
    if not games:
        return []
    if workers <= 1:
        rows: list[dict] = []
        for g in games:
            try:
                rows.extend(fetch_pitcher_rows_for_game(int(g["game_id"]), str(g["game_date"])))
            except Exception as e:
                log.debug("Skip pitcher game_id=%s: %s", g.get("game_id"), e)
        return rows

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_pitcher_rows_for_game, int(g["game_id"]), str(g["game_date"])): g
            for g in games
        }
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as e:
                g = futures[fut]
                log.debug("Skip pitcher game_id=%s: %s", g.get("game_id"), e)
    return rows


def _fetch_games_parallel(
    games: list[dict],
    *,
    workers: int,
    fetch_weather: bool,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    if workers <= 1:
        rows: list[dict] = []
        total = len(games)
        for i, g in enumerate(games, start=1):
            try:
                rows.extend(
                    fetch_batter_rows_for_game(
                        int(g["game_id"]),
                        str(g["game_date"]),
                        fetch_weather=fetch_weather,
                    )
                )
            except Exception as e:
                log.debug("Skip game_id=%s date=%s: %s", g.get("game_id"), g.get("game_date"), e)
            if on_progress and (i % 250 == 0 or i == total):
                on_progress(i, total)
        return rows

    rows: list[dict] = []
    done = 0
    total = len(games)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                fetch_batter_rows_for_game,
                int(g["game_id"]),
                str(g["game_date"]),
                fetch_weather=fetch_weather,
            ): g
            for g in games
        }
        for fut in as_completed(futures):
            done += 1
            try:
                rows.extend(fut.result())
            except Exception as e:
                g = futures[fut]
                log.debug("Skip game_id=%s date=%s: %s", g.get("game_id"), g.get("game_date"), e)
            if on_progress and (done % 250 == 0 or done == total):
                on_progress(done, total)
    return rows


def build_historical_store(
    seasons: list[int] | None = None,
    *,
    workers: int = 8,
    incremental: bool = False,
    fetch_weather: bool = False,
) -> None:
    seasons = seasons or SEASONS
    engine = _engine()
    existing_batter_ids = _existing_game_ids(engine) if incremental else set()
    existing_pitcher_ids = _existing_pitcher_game_ids(engine) if incremental else set()
    if incremental and existing_batter_ids:
        log.info("Incremental ETL: %s batter game(s) already in DB", len(existing_batter_ids))

    total_batter_rows = 0
    total_pitcher_rows = 0
    new_player_ids: set[int] = set()

    for season in seasons:
        log.info("Fetching MLB games for season %s...", season)
        try:
            games = fetch_games_for_season(int(season))
        except Exception as e:
            log.error("Season %s schedule fetch failed: %s — skipping", season, e)
            continue

        new_games = [g for g in games if int(g["game_id"]) not in existing_batter_ids] if incremental else games
        log.info("Season %s: %s game(s) to fetch", season, len(new_games))
        if not new_games:
            continue

        def _progress(i: int, total: int) -> None:
            log.info("  processed %s/%s games...", i, total)

        season_rows = _fetch_games_parallel(
            new_games,
            workers=max(1, int(workers)),
            fetch_weather=fetch_weather,
            on_progress=_progress,
        )
        if season_rows:
            df_s = pd.DataFrame(season_rows)
            df_s["game_date"] = pd.to_datetime(df_s["game_date"])
            df_s = df_s.drop_duplicates(subset=["player_id", "game_id"], keep="last")
            new_player_ids.update(int(x) for x in df_s["player_id"].dropna().tolist())
            with engine.begin() as conn:
                df_s.to_sql("batter_games", conn, if_exists="append", index=False)
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_batter_games_player_date "
                    "ON batter_games(player_id, game_date)"
                )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_batter_games_date ON batter_games(game_date)"
                )
            total_batter_rows += len(df_s)
            log.info("Season %s: wrote %s batter rows", season, len(df_s))

        new_pitcher_games = [g for g in new_games if int(g["game_id"]) not in existing_pitcher_ids]
        pitcher_rows = _fetch_pitcher_games_parallel(new_pitcher_games, workers=max(1, int(workers)))
        if pitcher_rows:
            dfp = pd.DataFrame(pitcher_rows)
            dfp["game_date"] = pd.to_datetime(dfp["game_date"])
            with engine.begin() as conn:
                dfp.to_sql("pitcher_games", conn, if_exists="append", index=False)
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_pitcher_games_team_date "
                    "ON pitcher_games(team, game_date)"
                )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS idx_pitcher_games_pitcher "
                    "ON pitcher_games(pitcher_id, game_date)"
                )
            total_pitcher_rows += len(dfp)
            log.info("Season %s: wrote %s pitcher rows", season, len(dfp))

    if total_batter_rows == 0 and not (incremental and existing_batter_ids):
        raise RuntimeError("No batter rows collected. Check Stats API connectivity.")

    if total_batter_rows == 0:
        log.info("No new games to ingest.")
        return

    log.info("Resolving bat side for %s new player(s)...", len(new_player_ids))
    hand_cache = build_bats_hand_cache(list(new_player_ids), max_workers=max(1, int(workers)))
    if hand_cache:
        with engine.begin() as conn:
            for pid, hand in hand_cache.items():
                conn.exec_driver_sql(
                    "UPDATE batter_games SET bats_hand=? WHERE player_id=? "
                    "AND (bats_hand IS NULL OR bats_hand='')",
                    (hand, int(pid)),
                )

    log.info(
        "ETL complete: %s new batter rows, %s new pitcher rows → %s",
        total_batter_rows,
        total_pitcher_rows,
        DB_PATH,
    )

    try:
        from config import MATERIALIZE_FEATURES_ON_ETL
        if MATERIALIZE_FEATURES_ON_ETL:
            from feature_store import materialize_feature_table

            n_feat = materialize_feature_table()
            log.info("Materialized %s feature rows", n_feat)
    except Exception as e:
        log.warning("Feature materialization skipped: %s", e)

