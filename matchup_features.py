"""
Same-day and historical matchup context for TB prediction.

Adds opponent team defense, home/away, lineup slot, platoon, and probable SP hand.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import re

import numpy as np
import pandas as pd
import statsapi

log = logging.getLogger(__name__)


def parse_kalshi_event_matchup(event_ticker: str) -> tuple[str, str] | None:
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

MATCHUP_FEATURE_NAMES = [
    "is_home",
    "lineup_slot_norm",
    "expected_pa_proxy",
    "opp_tb_allowed_roll",
    "opp_sp_hand_L",
    "platoon_tb_adj",
]


@lru_cache(maxsize=1)
def _team_id_to_abbr() -> dict[int, str]:
    teams = statsapi.get("teams", {"sportId": 1}).get("teams", []) or []
    return {int(t["id"]): str(t["abbreviation"]).strip().upper() for t in teams if t.get("id")}


@lru_cache(maxsize=1)
def _mlb_team_abbreviations() -> frozenset[str]:
    return frozenset(_team_id_to_abbr().values())


@lru_cache(maxsize=4096)
def _player_bats_hand(player_id: int) -> str:
    """R, L, or S (switch)."""
    try:
        p = statsapi.get("person", {"personId": int(player_id)})
        people = (p or {}).get("people") or []
        person = people[0] if people else {}
        bat = ((person or {}).get("batSide") or {}).get("code", "") or ""
        return str(bat).strip().upper()[:1] or "R"
    except Exception:
        return "R"


@lru_cache(maxsize=512)
def _pitcher_hand_from_name(name: str) -> str:
    key = (name or "").strip()
    if not key:
        return "R"
    try:
        hits = statsapi.lookup_player(key)
    except Exception:
        hits = []
    if not hits:
        return "R"
    pid = int(hits[0].get("id", 0) or 0)
    if not pid:
        return "R"
    try:
        p = statsapi.get("person", {"personId": pid})
        people = (p or {}).get("people") or []
        person = people[0] if people else {}
        throw_code = ((person or {}).get("pitchHand") or {}).get("code", "") or ""
        return str(throw_code).strip().upper()[:1] or "R"
    except Exception:
        return "R"


def schedule_games_by_date(game_date: str) -> list[dict]:
    return list(statsapi.schedule(date=game_date) or [])


def _game_matchup_row(game: dict) -> dict[str, Any]:
    abbr = _team_id_to_abbr()
    away_id = int(game.get("away_id", 0) or 0)
    home_id = int(game.get("home_id", 0) or 0)
    away = abbr.get(away_id, "")
    home = abbr.get(home_id, "")
    away_sp = str(game.get("away_probable_pitcher", "") or "").strip()
    home_sp = str(game.get("home_probable_pitcher", "") or "").strip()
    return {
        "game_id": int(game.get("game_id", 0) or 0),
        "away": away,
        "home": home,
        "away_sp_hand": _pitcher_hand_from_name(away_sp),
        "home_sp_hand": _pitcher_hand_from_name(home_sp),
    }


@lru_cache(maxsize=16)
def build_slate_matchup_index(game_date: str) -> dict[str, dict]:
    """
    Map Kalshi matchup slug (e.g. ``BOSKC``) -> home/away, SP hands.
    """
    out: dict[str, dict] = {}
    for g in schedule_games_by_date(game_date):
        row = _game_matchup_row(g)
        if not row["away"] or not row["home"]:
            continue
        slug = f"{row['away']}{row['home']}"
        out[slug] = row
    return out


def slate_teams(slate: dict[str, dict]) -> frozenset[str]:
    """All team abbreviations appearing in a slate matchup index."""
    teams: set[str] = set()
    for row in slate.values():
        for k in ("away", "home"):
            ab = str(row.get(k, "") or "").strip().upper()
            if ab:
                teams.add(ab)
    return frozenset(teams)


def build_opp_tb_allowed_lookup(
    game_date: str,
    teams: frozenset[str] | None = None,
) -> dict[str, float]:
    """
    Mean batter TB vs each opponent team before ``game_date`` (one SQL round-trip).
    """
    try:
        import sqlalchemy as sa

        from config import DB_PATH

        eng = sa.create_engine(f"sqlite:///{DB_PATH}")
        if teams:
            teams_u = sorted(t for t in teams if t)
            if not teams_u:
                return {}
            placeholders = ",".join("?" * len(teams_u))
            q = (
                f"SELECT opponent_team, AVG(tb) AS m FROM batter_games "
                f"WHERE date(game_date) < date(?) AND opponent_team IN ({placeholders}) "
                "GROUP BY opponent_team"
            )
            df = pd.read_sql(q, eng, params=[game_date, *teams_u])
        else:
            q = (
                "SELECT opponent_team, AVG(tb) AS m FROM batter_games "
                "WHERE date(game_date) < date(?) GROUP BY opponent_team"
            )
            df = pd.read_sql(q, eng, params=[game_date])
        if df.empty:
            return {}
        out: dict[str, float] = {}
        for _, row in df.iterrows():
            opp = str(row.get("opponent_team", "") or "").strip().upper()
            if opp and pd.notna(row.get("m")):
                out[opp] = float(row["m"])
        return out
    except Exception:
        return {}


def bats_hand_from_boxscore_person(person: dict) -> str | None:
    """Read bat side from boxscore person when present (avoids a person API call)."""
    bat = ((person or {}).get("batSide") or {}).get("code", "") or ""
    s = str(bat).strip().upper()[:1]
    return s if s in {"R", "L", "S"} else None


def build_bats_hand_cache(player_ids: list[int], *, max_workers: int = 16) -> dict[int, str]:
    """Resolve bat side once per player_id (parallel Stats API person lookups)."""
    pids = sorted({int(p) for p in player_ids if int(p) > 0})
    if not pids:
        return {}
    cache: dict[int, str] = {}
    if max_workers <= 1:
        for pid in pids:
            cache[pid] = _player_bats_hand(pid)
        return cache
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_player_bats_hand, pid): pid for pid in pids}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                cache[pid] = str(fut.result() or "R").upper()[:1] or "R"
            except Exception:
                cache[pid] = "R"
    return cache


def enrich_batter_row_from_boxscore(
    *,
    row: dict,
    side: str,
    team_abbr: str,
    home_abbr: str,
    away_abbr: str,
    player_meta: dict,
    bats_hand_cache: dict[int, str] | None = None,
) -> dict:
    """Attach matchup fields to a batter-game row during ETL."""
    is_home = 1 if str(team_abbr).upper() == str(home_abbr).upper() else 0
    opponent = away_abbr if is_home else home_abbr
    bo = player_meta.get("battingOrder")
    lineup_slot = 0
    if bo is not None:
        try:
            lineup_slot = int(int(bo) // 100)
        except (TypeError, ValueError):
            lineup_slot = 0
    lineup_slot = max(0, min(9, lineup_slot))
    pid = int(row.get("player_id", 0) or 0)
    person = (player_meta or {}).get("person", {}) or {}
    hand = bats_hand_from_boxscore_person(person)
    if not hand and bats_hand_cache is not None and pid:
        hand = bats_hand_cache.get(pid)
    row["team"] = str(team_abbr).upper()
    row["is_home"] = int(is_home)
    row["opponent_team"] = str(opponent).upper()
    row["lineup_slot"] = int(lineup_slot)
    row["bats_hand"] = hand or "R"
    return row


def attach_team_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean TB allowed by opponent (pitching) team, using only prior games.
    """
    if df.empty or "opponent_team" not in df.columns:
        return df
    df = df.sort_values(["opponent_team", "game_date", "player_id"]).reset_index(drop=True)
    # Team-game total TB allowed = sum of batter TB in rows facing that opponent_team
    tg = (
        df.groupby(["game_date", "opponent_team"], as_index=False)["tb"]
        .sum()
        .rename(columns={"tb": "team_tb_allowed"})
        .sort_values(["opponent_team", "game_date"])
    )
    tg["opp_tb_allowed_roll"] = (
        tg.groupby("opponent_team", group_keys=False)["team_tb_allowed"]
        .shift(1)
        .rolling(30, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df = df.merge(tg[["game_date", "opponent_team", "opp_tb_allowed_roll"]], on=["game_date", "opponent_team"], how="left")
    return df


def attach_platoon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Platoon adjustment vs opposing starter hand (historical rows use neutral 0.5 if unknown)."""
    df = df.copy()
    if "opp_sp_hand_L" not in df.columns:
        df["opp_sp_hand_L"] = 0.5
    df["opp_sp_hand_L"] = pd.to_numeric(df["opp_sp_hand_L"], errors="coerce").fillna(0.5)
    tb_roll = pd.to_numeric(df.get("tb_roll", 0), errors="coerce").fillna(0)
    hand = df["opp_sp_hand_L"]
    bats = df.get("bats_hand", pd.Series(["R"] * len(df))).astype(str).str.upper().str[:1]
    # LHB vs RHP boost; RHB vs LHP boost
    boost = np.where(
        (bats == "L") & (hand < 0.5),
        1.08,
        np.where((bats == "R") & (hand > 0.5), 1.06, 1.0),
    )
    df["platoon_tb_adj"] = tb_roll * boost
    return df


def finalize_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive normalized matchup columns used by the model."""
    df = df.copy()
    slot = pd.to_numeric(df.get("lineup_slot", 0), errors="coerce").fillna(5)
    df["lineup_slot_norm"] = (slot / 9.0).clip(0, 1)
    df["expected_pa_proxy"] = (4.5 - 0.28 * (slot - 5.0)).clip(3.2, 5.2)
    df["is_home"] = pd.to_numeric(df.get("is_home", 0), errors="coerce").fillna(0)
    df["opp_tb_allowed_roll"] = pd.to_numeric(df.get("opp_tb_allowed_roll", 0), errors="coerce").fillna(0)
    if "opp_sp_hand_L" not in df.columns:
        df["opp_sp_hand_L"] = 0.5
    if "platoon_tb_adj" not in df.columns:
        df["platoon_tb_adj"] = pd.to_numeric(df.get("tb_roll", 0), errors="coerce").fillna(0)
    return df


def live_matchup_overrides(
    *,
    game_date: str,
    player_team: str,
    event_ticker: str,
    tb_roll: float,
    bats_hand: str = "R",
    slate: dict[str, dict] | None = None,
    opp_tb_allowed_by_team: dict[str, float] | None = None,
    lineup_slot: int | None = None,
) -> dict[str, float]:
    """
    Build matchup feature overrides for tonight's game (not from stale history row).

    Pass ``slate`` and ``opp_tb_allowed_by_team`` from scan/backtest to avoid repeated
    MLB schedule and per-player SQL lookups. When ``lineup_slot`` (1-9) is supplied
    from a confirmed lineup, it drives ``lineup_slot_norm`` / ``expected_pa_proxy``
    instead of the neutral 0.55 default.
    """
    matchup = parse_kalshi_event_matchup(event_ticker) if event_ticker else None
    slate_idx = slate if slate is not None else build_slate_matchup_index(game_date)
    opp_hand_L = 0.5
    is_home = 0
    opp_allowed = 0.0
    if matchup:
        away, home = matchup
        slug = f"{away}{home}"
        info = slate_idx.get(slug, {})
        team = str(player_team or "").upper()
        if team == home:
            is_home = 1
            opp_hand_L = 1.0 if str(info.get("away_sp_hand", "R")).upper().startswith("L") else 0.0
            opp_team = away
        elif team == away:
            is_home = 0
            opp_hand_L = 1.0 if str(info.get("home_sp_hand", "R")).upper().startswith("L") else 0.0
            opp_team = home
        else:
            opp_team = ""
        if opp_team:
            if opp_tb_allowed_by_team is not None:
                opp_allowed = float(opp_tb_allowed_by_team.get(str(opp_team).upper(), 0.0))
            else:
                lookup = build_opp_tb_allowed_lookup(game_date, frozenset({str(opp_team).upper()}))
                opp_allowed = float(lookup.get(str(opp_team).upper(), 0.0))
    bats = str(bats_hand or "R").upper()[:1]
    boost = 1.08 if bats == "L" and opp_hand_L < 0.5 else (1.06 if bats == "R" and opp_hand_L > 0.5 else 1.0)
    if lineup_slot is not None and 1 <= int(lineup_slot) <= 9:
        slot = float(int(lineup_slot))
        slot_norm = slot / 9.0
        expected_pa = float(min(5.2, max(3.2, 4.5 - 0.28 * (slot - 5.0))))
    else:
        slot_norm = 0.55
        expected_pa = float(4.5 - 0.28 * (slot_norm * 9 - 5))
    return {
        "is_home": float(is_home),
        "lineup_slot_norm": float(slot_norm),
        "expected_pa_proxy": float(expected_pa),
        "opp_tb_allowed_roll": float(opp_allowed),
        "opp_sp_hand_L": float(opp_hand_L),
        "platoon_tb_adj": float(tb_roll * boost),
    }


def merge_live_into_feature_dict(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k in MATCHUP_FEATURE_NAMES:
        if k in overrides:
            out[k] = overrides[k]
    return out


def opponent_team_for(event_ticker: str, player_team: str) -> str:
    """Opposing team abbreviation for ``player_team`` given a Kalshi event ticker."""
    matchup = parse_kalshi_event_matchup(event_ticker) if event_ticker else None
    if not matchup:
        return ""
    away, home = matchup
    team = str(player_team or "").upper()
    if team == home:
        return away
    if team == away:
        return home
    return ""


def apply_live_feature_overrides(
    row_features: dict,
    *,
    game_date: str,
    player_id: int,
    player_team: str,
    bats_hand: str,
    tb_roll: float,
    event_ticker: str,
    matchup_slate: dict[str, dict] | None,
    opp_tb_lookup: dict[str, float] | None = None,
    confirmed_lineups: dict[int, int] | None = None,
    live_sp_by_team: dict[str, dict] | None = None,
) -> dict:
    """
    Replace a stale last-game feature row with tonight's matchup, confirmed lineup
    slot, and the opposing probable starter's rolling stats. Shared by the live
    scan and the backtest so both predict on identical inputs.
    """
    if not matchup_slate:
        return row_features

    lineup_slot = confirmed_lineups.get(int(player_id)) if confirmed_lineups else None
    overrides = live_matchup_overrides(
        game_date=str(game_date),
        player_team=str(player_team or ""),
        event_ticker=str(event_ticker or ""),
        tb_roll=float(tb_roll or 0.0),
        bats_hand=str(bats_hand or "R"),
        slate=matchup_slate,
        opp_tb_allowed_by_team=opp_tb_lookup,
        lineup_slot=lineup_slot,
    )
    row_features = merge_live_into_feature_dict(row_features, overrides)

    if live_sp_by_team:
        opp_team = opponent_team_for(str(event_ticker or ""), str(player_team or ""))
        sp = live_sp_by_team.get(opp_team) if opp_team else None
        if sp:
            for k, v in sp.items():
                if k in row_features:
                    row_features[k] = float(v)
    return row_features
