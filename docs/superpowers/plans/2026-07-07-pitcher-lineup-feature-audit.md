# Pitcher/Lineup Feature Audit & Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three concrete, evidence-backed defects in the pitcher/lineup feature block (`opp_sp_hand_L`/platoon collinearity with `tb_roll`, `expected_pa_proxy` collinearity with `lineup_slot_norm`, and a ~47%-failure-rate name-based pitcher-hand lookup), backfill the corrected hand label across history, retrain, and judge the result with `model-vs-market`.

**Architecture:** All changes live in the existing feature-engineering layer (`matchup_features.py`, `data_engine.py`, `model.py`) — no new modules, no schema changes beyond overwriting one existing column (`batter_games.opp_sp_hand_L`). A new one-shot root-level script (`backfill_opp_sp_hand.py`, following the existing `backfill_pitchers.py` convention) performs the historical correction.

**Tech Stack:** Python, pandas, SQLAlchemy/SQLite, statsapi (MLB Stats API client), pytest, statsmodels (via existing `ordinal_core.py`).

## Global Constraints

- No Statcast data pull or `USE_STATCAST_FEATURES` changes (out of scope per design spec).
- No changes to `opp_sp_era_roll`/`opp_sp_hr9_roll`/`opp_sp_bb9_roll` (flagged, not fixed).
- No changes to `BLEND_WEIGHT`, gates, sizing, or fee logic.
- Follow existing repo conventions: underscore-prefixed helpers imported across modules (already done for `_pitcher_hand_from_name`, `build_bats_hand_cache`, etc.), deferred (function-local) imports where a top-level import would create a cycle (see `feature_store.py:216`'s `from data_engine import get_probable_starters` inside `build_live_pitcher_features` — same pattern needed here since `data_engine.py` imports from `matchup_features.py` at module level).
- Reference design spec: `docs/superpowers/specs/2026-07-07-pitcher-lineup-feature-audit-design.md`.

---

### Task 1: ID-based pitcher-hand resolution + decorrelated platoon feature in `matchup_features.py`

**Files:**
- Modify: `matchup_features.py`
- Test: `tests/test_matchup_features_hand.py` (new)
- Test: `tests/test_matchup_scan_cache.py` (existing — must still pass unmodified)

**Interfaces:**
- Produces: `_pitcher_hand_from_id(pitcher_id: int) -> str` (module-level, `lru_cache`, returns `"R"` or `"L"`) — consumed by Task 2.
- Produces: `MATCHUP_FEATURE_NAMES = ["is_home", "lineup_slot_norm", "opp_tb_allowed_roll", "opp_sp_hand_L", "platoon_edge"]` — consumed by Task 3 and by `feature_store.MODEL_FEATURES` (no `feature_store.py` edit needed; it imports this list directly).
- Removes: `_pitcher_hand_from_name` (dead after this task — no remaining callers once Task 2 lands, but Task 1 alone already removes its only call site in this file).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matchup_features_hand.py`:

```python
"""ID-based pitcher-hand resolution and the decorrelated platoon feature."""

from unittest.mock import patch

import pandas as pd

from matchup_features import (
    MATCHUP_FEATURE_NAMES,
    _pitcher_hand_from_id,
    attach_platoon_features,
    finalize_matchup_columns,
    live_matchup_overrides,
)


def test_pitcher_hand_from_id_resolves_left():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get") as mock_get:
        mock_get.return_value = {"people": [{"pitchHand": {"code": "L"}}]}
        assert _pitcher_hand_from_id(477132) == "L"
    mock_get.assert_called_once_with("person", {"personId": 477132})


def test_pitcher_hand_from_id_defaults_to_right_on_missing_data():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get") as mock_get:
        mock_get.return_value = {"people": []}
        assert _pitcher_hand_from_id(1) == "R"


def test_pitcher_hand_from_id_defaults_to_right_on_exception():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get", side_effect=Exception("boom")):
        assert _pitcher_hand_from_id(2) == "R"


def test_matchup_feature_names_drops_expected_pa_proxy_and_renames_platoon():
    assert "expected_pa_proxy" not in MATCHUP_FEATURE_NAMES
    assert "platoon_tb_adj" not in MATCHUP_FEATURE_NAMES
    assert "platoon_edge" in MATCHUP_FEATURE_NAMES


def test_attach_platoon_features_produces_decorrelated_edge():
    df = pd.DataFrame(
        {
            "opp_sp_hand_L": [1.0, 0.0, 0.0],
            "bats_hand": ["L", "R", "L"],
            "tb_roll": [2.0, 1.0, 0.5],
        }
    )
    out = attach_platoon_features(df)
    # LHB vs LHP (opp_sp_hand_L=1.0) -> no platoon edge
    assert out["platoon_edge"].iloc[0] == 0.0
    # RHB vs RHP (opp_sp_hand_L=0.0) -> no platoon edge
    assert out["platoon_edge"].iloc[1] == 0.0
    # LHB vs RHP (opp_sp_hand_L=0.0) -> platoon edge, independent of tb_roll
    assert out["platoon_edge"].iloc[2] == 0.08


def test_finalize_matchup_columns_defaults_platoon_edge_to_zero():
    df = pd.DataFrame({"lineup_slot": [5]})
    out = finalize_matchup_columns(df)
    assert "expected_pa_proxy" not in out.columns
    assert out["platoon_edge"].iloc[0] == 0.0


def test_live_matchup_overrides_returns_platoon_edge_not_pa_proxy():
    # BOS (away) batter is L, facing NYY's (home) starter who is also L per the
    # slate ("home_sp_hand": "L") -> opp_sp_hand_L=1.0 -> same-handed matchup
    # (LHB vs LHP), no platoon edge.
    slate = {"BOSNYY": {"away": "BOS", "home": "NYY", "away_sp_hand": "R", "home_sp_hand": "L"}}
    opp = {"NYY": 1.42}
    with patch("matchup_features.build_slate_matchup_index") as mock_build:
        out = live_matchup_overrides(
            game_date="2026-05-23",
            player_team="BOS",
            event_ticker="KXMLBTB-26MAY201940BOSNYY",
            tb_roll=1.1,
            bats_hand="L",
            slate=slate,
            opp_tb_allowed_by_team=opp,
        )
        mock_build.assert_not_called()
    assert "expected_pa_proxy" not in out
    assert out["opp_sp_hand_L"] == 1.0
    assert out["platoon_edge"] == 0.0

    # Same slate, but a right-handed batter facing that same left-handed NYY
    # starter -> RHB vs LHP is the opposite-handed (platoon-advantage) matchup,
    # boost=1.06 -> platoon_edge=0.06.
    out_r = live_matchup_overrides(
        game_date="2026-05-23",
        player_team="BOS",
        event_ticker="KXMLBTB-26MAY201940BOSNYY",
        tb_roll=1.1,
        bats_hand="R",
        slate=slate,
        opp_tb_allowed_by_team=opp,
    )
    assert abs(out_r["platoon_edge"] - 0.06) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_matchup_features_hand.py -v`
Expected: FAIL — `_pitcher_hand_from_id` doesn't exist (ImportError), `platoon_edge` key/column doesn't exist.

- [ ] **Step 3: Add ID-based hand resolution, remove name-based lookup**

In `matchup_features.py`, delete the `_pitcher_hand_from_name` function (currently the `@lru_cache(maxsize=512)`-decorated function around line 70):

```python
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
```

Replace it with:

```python
@lru_cache(maxsize=2048)
def _pitcher_hand_from_id(pitcher_id: int) -> str:
    """R or L throwing hand, resolved by MLBAM person id.

    Deliberately not a name search: ``statsapi.lookup_player(name)`` only
    covers the current-season active-roster snapshot and silently returns no
    hits for anyone not rostered at call time (IL, traded, optioned) --
    measured at ~47% failure rate on a sample of real starting pitchers.
    """
    try:
        p = statsapi.get("person", {"personId": int(pitcher_id)})
        people = (p or {}).get("people") or []
        person = people[0] if people else {}
        throw_code = ((person or {}).get("pitchHand") or {}).get("code", "") or ""
        return str(throw_code).strip().upper()[:1] or "R"
    except Exception:
        return "R"
```

- [ ] **Step 4: Rewire `_game_matchup_row` and `build_slate_matchup_index` to use pitcher IDs**

Replace:

```python
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
```

with:

```python
def _game_matchup_row(game: dict, probable: dict[str, int]) -> dict[str, Any]:
    abbr = _team_id_to_abbr()
    away_id = int(game.get("away_id", 0) or 0)
    home_id = int(game.get("home_id", 0) or 0)
    away = abbr.get(away_id, "")
    home = abbr.get(home_id, "")
    away_pid = int(probable.get(away, 0) or 0)
    home_pid = int(probable.get(home, 0) or 0)
    return {
        "game_id": int(game.get("game_id", 0) or 0),
        "away": away,
        "home": home,
        "away_sp_hand": _pitcher_hand_from_id(away_pid) if away_pid else "R",
        "home_sp_hand": _pitcher_hand_from_id(home_pid) if home_pid else "R",
    }


@lru_cache(maxsize=16)
def build_slate_matchup_index(game_date: str) -> dict[str, dict]:
    """
    Map Kalshi matchup slug (e.g. ``BOSKC``) -> home/away, SP hands.

    SP hand is resolved by MLBAM pitcher id via ``data_engine.get_probable_starters``,
    not by searching the probable-pitcher name string.
    """
    from data_engine import get_probable_starters

    probable = get_probable_starters(game_date)
    out: dict[str, dict] = {}
    for g in schedule_games_by_date(game_date):
        row = _game_matchup_row(g, probable)
        if not row["away"] or not row["home"]:
            continue
        slug = f"{row['away']}{row['home']}"
        out[slug] = row
    return out
```

- [ ] **Step 5: Drop `expected_pa_proxy`, redesign `platoon_tb_adj` → `platoon_edge`**

Update `MATCHUP_FEATURE_NAMES`:

```python
MATCHUP_FEATURE_NAMES = [
    "is_home",
    "lineup_slot_norm",
    "opp_tb_allowed_roll",
    "opp_sp_hand_L",
    "platoon_edge",
]
```

Update `attach_platoon_features`:

```python
def attach_platoon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Platoon edge vs opposing starter hand: a boost delta, decorrelated from tb_roll.

    Previously this was ``tb_roll * boost`` (r=0.999 with tb_roll, already in
    the model), which made the platoon-advantage signal statistically
    invisible. The delta isolates the handedness-matchup effect on its own.
    """
    df = df.copy()
    if "opp_sp_hand_L" not in df.columns:
        df["opp_sp_hand_L"] = 0.5
    df["opp_sp_hand_L"] = pd.to_numeric(df["opp_sp_hand_L"], errors="coerce").fillna(0.5)
    hand = df["opp_sp_hand_L"]
    bats = df.get("bats_hand", pd.Series(["R"] * len(df))).astype(str).str.upper().str[:1]
    # LHB vs RHP boost; RHB vs LHP boost
    boost = np.where(
        (bats == "L") & (hand < 0.5),
        1.08,
        np.where((bats == "R") & (hand > 0.5), 1.06, 1.0),
    )
    df["platoon_edge"] = boost - 1.0
    return df
```

Update `finalize_matchup_columns` (remove the `expected_pa_proxy` line, rename the fallback):

```python
def finalize_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive normalized matchup columns used by the model."""
    df = df.copy()
    slot = pd.to_numeric(df.get("lineup_slot", 0), errors="coerce").fillna(5)
    df["lineup_slot_norm"] = (slot / 9.0).clip(0, 1)
    df["is_home"] = pd.to_numeric(df.get("is_home", 0), errors="coerce").fillna(0)
    df["opp_tb_allowed_roll"] = pd.to_numeric(df.get("opp_tb_allowed_roll", 0), errors="coerce").fillna(0)
    if "opp_sp_hand_L" not in df.columns:
        df["opp_sp_hand_L"] = 0.5
    if "platoon_edge" not in df.columns:
        df["platoon_edge"] = 0.0
    return df
```

Update `live_matchup_overrides` (remove `expected_pa`/`slot_norm` PA computation, return `platoon_edge`):

```python
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
    from a confirmed lineup, it drives ``lineup_slot_norm`` instead of the neutral
    0.55 default.
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
        slot_norm = float(int(lineup_slot)) / 9.0
    else:
        slot_norm = 0.55
    return {
        "is_home": float(is_home),
        "lineup_slot_norm": float(slot_norm),
        "opp_tb_allowed_roll": float(opp_allowed),
        "opp_sp_hand_L": float(opp_hand_L),
        "platoon_edge": float(boost - 1.0),
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_matchup_features_hand.py tests/test_matchup_scan_cache.py -v`
Expected: PASS (7 new tests + 3 existing tests, 10 passed)

- [ ] **Step 7: Commit**

```bash
git add matchup_features.py tests/test_matchup_features_hand.py
git commit -m "fix: ID-based pitcher-hand resolution, decorrelate platoon feature, drop redundant expected_pa_proxy"
```

---

### Task 2: ID-based hand resolution in the `data_engine.py` ETL path

**Files:**
- Modify: `data_engine.py`
- Test: `tests/test_data_engine_sp_hand.py` (new)

**Interfaces:**
- Consumes: `matchup_features._pitcher_hand_from_id(pitcher_id: int) -> str` and `matchup_features._team_id_to_abbr() -> dict[int, str]` (both from Task 1; `_team_id_to_abbr` already exists, unchanged).
- Consumes: `data_engine.get_probable_starters(game_date: str) -> dict[str, int]` (already exists in this file, unchanged).
- Produces: `_opp_sp_hand_L_by_game_id(game_date: str) -> dict[int, tuple[float, float]]` — same signature as before, only the resolution method inside changes. No downstream callers need updating.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_engine_sp_hand.py`:

```python
"""ETL-path pitcher-hand resolution must use pitcher ids, not name search."""

from unittest.mock import patch

from data_engine import _opp_sp_hand_L_by_game_id


def test_opp_sp_hand_l_uses_id_based_lookup_not_name_search():
    _opp_sp_hand_L_by_game_id.cache_clear()
    game = {
        "game_id": 824089,
        "away_id": 143,
        "home_id": 118,
        "away_probable_pitcher": "Cristopher Sánchez",
        "home_probable_pitcher": "Noah Cameron",
    }
    with patch("data_engine.schedule_games_by_date", return_value=[game]), patch(
        "data_engine._team_id_to_abbr", return_value={143: "PHI", 118: "KC"}
    ), patch(
        "data_engine.get_probable_starters", return_value={"PHI": 111, "KC": 222}
    ), patch(
        "data_engine._pitcher_hand_from_id", side_effect=lambda pid: "L" if pid == 111 else "R"
    ) as mock_hand, patch(
        "data_engine.statsapi.lookup_player"
    ) as mock_lookup:
        out = _opp_sp_hand_L_by_game_id("2026-07-06")

    mock_lookup.assert_not_called()
    mock_hand.assert_any_call(111)
    mock_hand.assert_any_call(222)
    assert out[824089] == (1.0, 0.0)  # away (PHI, pid 111) = L, home (KC, pid 222) = R
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_data_engine_sp_hand.py -v`
Expected: FAIL — `data_engine._team_id_to_abbr` / `data_engine.get_probable_starters` / `data_engine._pitcher_hand_from_id` not patchable (not imported into `data_engine` namespace yet) or `_opp_sp_hand_L_by_game_id` still calls the old name-based path.

- [ ] **Step 3: Update the import block**

In `data_engine.py`, replace:

```python
from matchup_features import (
    _pitcher_hand_from_name,
    build_bats_hand_cache,
    enrich_batter_row_from_boxscore,
    schedule_games_by_date,
)
```

with:

```python
from matchup_features import (
    _pitcher_hand_from_id,
    _team_id_to_abbr,
    build_bats_hand_cache,
    enrich_batter_row_from_boxscore,
    schedule_games_by_date,
)
```

- [ ] **Step 4: Rewrite `_opp_sp_hand_L_by_game_id`**

Replace:

```python
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
```

with:

```python
@lru_cache(maxsize=512)
def _opp_sp_hand_L_by_game_id(game_date: str) -> dict[int, tuple[float, float]]:
    """game_id -> (away_sp_hand_L, home_sp_hand_L), resolved by pitcher id.

    Uses ``get_probable_starters`` (id-based, from the hydrated schedule
    endpoint) instead of searching the probable-pitcher name string through
    the current-season active-roster snapshot, which silently misses anyone
    not rostered at call time.
    """
    probable = get_probable_starters(game_date)
    abbr = _team_id_to_abbr()
    out: dict[int, tuple[float, float]] = {}
    for g in schedule_games_by_date(game_date):
        gid = int(g.get("game_id", 0) or 0)
        if not gid:
            continue
        away_abbr = abbr.get(int(g.get("away_id", 0) or 0), "")
        home_abbr = abbr.get(int(g.get("home_id", 0) or 0), "")
        away_pid = int(probable.get(away_abbr, 0) or 0)
        home_pid = int(probable.get(home_abbr, 0) or 0)
        ah = _pitcher_hand_from_id(away_pid) if away_pid else "R"
        hh = _pitcher_hand_from_id(home_pid) if home_pid else "R"
        out[gid] = (
            1.0 if str(ah).upper().startswith("L") else 0.0,
            1.0 if str(hh).upper().startswith("L") else 0.0,
        )
    return out
```

Note: `get_probable_starters` is defined later in this same file (module-level function, called at runtime — definition order doesn't matter for a function body reference).

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_data_engine_sp_hand.py -v`
Expected: PASS

- [ ] **Step 6: Run the full existing test suite to check for regressions**

Run: `python3 -m pytest tests/ -v -x --ignore=tests/test_matchup_features_hand.py --ignore=tests/test_data_engine_sp_hand.py -k "not kalshi_balance"`
Expected: PASS (pre-existing suite unaffected; `-k "not kalshi_balance"` skips any test requiring live credentials if present — check output for any unexpected failures tied to `opp_sp_hand_L` or `_pitcher_hand_from_name` and resolve before proceeding)

- [ ] **Step 7: Commit**

```bash
git add data_engine.py tests/test_data_engine_sp_hand.py
git commit -m "fix: resolve ETL opp_sp_hand_L by pitcher id instead of name search"
```

---

### Task 3: Update `model.py` ablation feature list

**Files:**
- Modify: `model.py:571-573`

**Interfaces:**
- Consumes: `feature_store.MODEL_FEATURES` (unchanged signature, contents now reflect Task 1's `MATCHUP_FEATURE_NAMES`).
- No new interfaces produced — this task only keeps `tune_hyperparameters`'s hardcoded ablation list in sync with the renamed/removed feature names so the "with vs without matchup features" trial doesn't silently reference dropped columns.

- [ ] **Step 1: Update the ablation list**

In `model.py`, replace:

```python
        base_matchup = [f for f in MODEL_FEATURES if f in (
            "is_home", "lineup_slot_norm", "expected_pa_proxy", "opp_tb_allowed_roll", "opp_sp_hand_L", "platoon_tb_adj"
        )]
```

with:

```python
        base_matchup = [f for f in MODEL_FEATURES if f in (
            "is_home", "lineup_slot_norm", "opp_tb_allowed_roll", "opp_sp_hand_L", "platoon_edge"
        )]
```

- [ ] **Step 2: Verify no other stale references remain**

Run: `grep -rn "expected_pa_proxy\|platoon_tb_adj\|_pitcher_hand_from_name" --include="*.py" . --exclude-dir=.venv --exclude-dir=__pycache__`
Expected: no output (all three names fully removed from the codebase)

- [ ] **Step 3: Commit**

```bash
git add model.py
git commit -m "fix: sync tune_hyperparameters ablation list with renamed matchup features"
```

---

### Task 4: Capture the pre-fix `model-vs-market` baseline

**Files:** none (read-only command run; output captured to a scratch file for later comparison)

**Interfaces:**
- Consumes: `python3 run_pipeline.py model-vs-market --start <date> --end <date>` (existing CLI command, unchanged).

This must run *before* Task 5's backfill and retrain overwrite `models/tb_model.pkl`, so the "before" and "after" scoreboard numbers are comparable under the same evaluation code with only the model/data differing.

- [ ] **Step 1: Run the baseline scoreboard and save output**

Run:
```bash
python3 run_pipeline.py model-vs-market --start 2026-05-21 --end 2026-07-07 2>&1 | tee /tmp/model_vs_market_before.txt
```
Expected: prints four tables (Overall, By Kalshi line, By disagreement, By date). Confirm the "Overall" row's `LL model`/`LL market`/`w fit` values are captured in `/tmp/model_vs_market_before.txt` — these are the pre-fix baseline to diff against in Task 6.

- [ ] **Step 2: No commit** (read-only verification step; nothing to commit)

---

### Task 5: Historical backfill of `batter_games.opp_sp_hand_L`

**Files:**
- Create: `backfill_opp_sp_hand.py`

**Interfaces:**
- Consumes: `matchup_features._pitcher_hand_from_id(pitcher_id: int) -> str` (Task 1).
- Consumes: `config.DB_PATH` (existing).
- Mutates: `batter_games.opp_sp_hand_L` column in place, for every row whose `(game_id, opponent_team)` has a matching starter in `pitcher_games`.

- [ ] **Step 1: Write the script**

Create `backfill_opp_sp_hand.py`:

```python
"""
One-shot backfill: recompute batter_games.opp_sp_hand_L using ID-based
pitcher-hand resolution instead of the fragile name-search lookup it was
originally stamped with. Joins batter_games to pitcher_games
(game_id, opponent_team=team, is_starter=1) to recover each row's opposing
starter's pitcher_id, resolves hand via matchup_features._pitcher_hand_from_id
(cached per distinct pitcher_id), and overwrites opp_sp_hand_L in place.

Run once after the pitcher-hand resolution fix (matchup_features.py /
data_engine.py). Rows whose game_id has no matching starter in pitcher_games
are left unchanged.
"""
from __future__ import annotations

import logging

import pandas as pd
import sqlalchemy as sa

from config import DB_PATH
from matchup_features import _pitcher_hand_from_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    engine = sa.create_engine(f"sqlite:///{DB_PATH}")

    starters = pd.read_sql(
        "SELECT game_id, team, pitcher_id FROM pitcher_games WHERE is_starter=1", engine
    )
    starters = starters.dropna(subset=["game_id", "pitcher_id"])
    starter_by_game_team: dict[tuple[int, str], int] = {
        (int(r.game_id), str(r.team)): int(r.pitcher_id) for r in starters.itertuples()
    }
    log.info("Loaded %s (game_id, team) -> starting pitcher_id pairs", len(starter_by_game_team))

    distinct_pids = sorted({pid for pid in starter_by_game_team.values()})
    log.info("Resolving hand for %s distinct starting pitcher ids (cached, one API call each)...", len(distinct_pids))
    hand_by_pid: dict[int, str] = {}
    for i, pid in enumerate(distinct_pids, start=1):
        hand_by_pid[pid] = _pitcher_hand_from_id(pid)
        if i % 200 == 0:
            log.info("  resolved %s/%s pitcher hands", i, len(distinct_pids))
    n_left = sum(1 for h in hand_by_pid.values() if h == "L")
    log.info("Resolved %s pitcher hands (%s L, %s R)", len(hand_by_pid), n_left, len(hand_by_pid) - n_left)

    batters = pd.read_sql("SELECT game_id, opponent_team, opp_sp_hand_L FROM batter_games", engine)
    log.info("Backfilling opp_sp_hand_L for %s batter_games rows", len(batters))

    def resolve(row) -> float | None:
        pid = starter_by_game_team.get((int(row.game_id), str(row.opponent_team)))
        if pid is None:
            return None  # no matching starter -- leave existing value alone
        hand = hand_by_pid.get(pid, "R")
        return 1.0 if str(hand).upper().startswith("L") else 0.0

    new_vals = batters.apply(resolve, axis=1)
    resolvable = new_vals.notna()
    old_vals = pd.to_numeric(batters["opp_sp_hand_L"], errors="coerce")
    changed = int(((new_vals != old_vals) & resolvable).sum())
    log.info(
        "%s/%s rows have a resolvable starter; %s of those change opp_sp_hand_L",
        int(resolvable.sum()),
        len(batters),
        changed,
    )

    update_df = pd.DataFrame(
        {
            "game_id": batters.loc[resolvable, "game_id"].astype(int),
            "opponent_team": batters.loc[resolvable, "opponent_team"].astype(str),
            "new_hand": new_vals[resolvable].astype(float),
        }
    ).drop_duplicates(subset=["game_id", "opponent_team"])

    with engine.begin() as conn:
        update_df.to_sql("tmp_sp_hand_backfill", conn, if_exists="replace", index=False)
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_tmp_sp_hand_backfill "
            "ON tmp_sp_hand_backfill(game_id, opponent_team)"
        )
        conn.exec_driver_sql(
            """
            UPDATE batter_games
            SET opp_sp_hand_L = (
                SELECT new_hand FROM tmp_sp_hand_backfill t
                WHERE t.game_id = batter_games.game_id
                  AND t.opponent_team = batter_games.opponent_team
            )
            WHERE EXISTS (
                SELECT 1 FROM tmp_sp_hand_backfill t
                WHERE t.game_id = batter_games.game_id
                  AND t.opponent_team = batter_games.opponent_team
            )
            """
        )
        conn.exec_driver_sql("DROP TABLE tmp_sp_hand_backfill")

    log.info("Backfill complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run sanity check on a tiny slice before running against the full table**

Run:
```bash
python3 -c "
import pandas as pd, sqlalchemy as sa
from config import DB_PATH
engine = sa.create_engine(f'sqlite:///{DB_PATH}')
starters = pd.read_sql('SELECT game_id, team, pitcher_id FROM pitcher_games WHERE is_starter=1 LIMIT 5', engine)
print(starters)
"
```
Expected: prints 5 rows with non-null `game_id`, `team`, `pitcher_id` — confirms the join key exists and is populated before running the full backfill.

- [ ] **Step 3: Back up the database before mutating it**

Run:
```bash
cp data/mlb_tb.db data/mlb_tb.db.pre-sp-hand-backfill.bak
ls -la data/mlb_tb.db.pre-sp-hand-backfill.bak
```
Expected: backup file exists with a size close to `data/mlb_tb.db`.

- [ ] **Step 4: Run the backfill**

Run:
```bash
python3 backfill_opp_sp_hand.py 2>&1 | tee /tmp/backfill_opp_sp_hand.log
```
Expected: logs "Loaded N (game_id, team) -> starting pitcher_id pairs", "Resolved M distinct starting pitcher ids (~X L, ~Y R)" with an L-rate in the neighborhood of MLB's real ~25-28% left-handed-starter base rate, "Backfilling opp_sp_hand_L for ~1.1M rows", and "Backfill complete." with no exceptions.

- [ ] **Step 5: Verify the change landed**

Run:
```bash
python3 -c "
import sqlite3
con = sqlite3.connect('data/mlb_tb.db')
cur = con.cursor()
cur.execute('SELECT opp_sp_hand_L, COUNT(*) FROM batter_games GROUP BY opp_sp_hand_L')
print(cur.fetchall())
"
```
Expected: two rows (0.0 and 1.0 counts), with the 1.0 (left-handed) share now noticeably higher than before the fix (previously suppressed toward 0 by the ~47%-failure default-to-'R' behavior).

- [ ] **Step 6: Commit**

```bash
git add backfill_opp_sp_hand.py
git commit -m "feat: backfill opp_sp_hand_L via ID-based pitcher-hand resolution"
```

(The `data/mlb_tb.db` mutation itself is not committed — it's a local SQLite file; confirm it's covered by `.gitignore` before staging anything in `data/`.)

---

### Task 6: Re-materialize, retrain, and judge with the scoreboard

**Files:** none (operational commands only)

**Interfaces:**
- Consumes: `feature_store.materialize_feature_table()` via `python3 run_pipeline.py train` (existing CLI, unchanged) — confirm whether `train` calls this automatically or whether a separate materialize step is needed (check `pipeline/commands.py`'s `train()` before running).

- [ ] **Step 1: Confirm whether `train` re-materializes the feature table**

Run: `grep -n "materialize_feature_table" pipeline/commands.py`
If `train()` already calls `materialize_feature_table()`, skip to Step 2. If not, run:
```bash
python3 -c "from feature_store import materialize_feature_table; n = materialize_feature_table(); print(f'materialized {n} rows')"
```
Expected: prints a row count close to the `batter_games` row count (~1.1M, filtered by `MIN_GAMES`).

- [ ] **Step 2: Retrain**

Run:
```bash
python3 run_pipeline.py train 2>&1 | tee /tmp/train_after_fix.log
```
Expected: completes without error, overwrites `models/tb_model.pkl`. Note the training log's row count and any printed fit diagnostics.

- [ ] **Step 3: Re-run the coefficient audit**

Run:
```bash
python3 -c "
import pickle
import pandas as pd
with open('models/tb_model.pkl','rb') as f:
    m = pickle.load(f)
res = m['result']
params, bse, pvals = res.params, res.bse, res.pvalues
names = [n for n in params.index if any(k in n for k in ['sp_', 'lineup', 'platoon', 'is_home', 'opp_tb_allowed'])]
df = pd.DataFrame({'coef': params, 'se': bse, 'p': pvals}).loc[names]
df['abs_z'] = (df['coef']/df['se']).abs()
print(df.sort_values('abs_z', ascending=False).to_string())
"
```
Expected: `expected_pa_proxy`/`platoon_tb_adj` no longer appear (replaced by `platoon_edge`); compare `platoon_edge`'s p-value/z-score against the pre-fix `platoon_tb_adj` baseline recorded in the design spec (coef +0.075, p=0.31, z=1.0) — record whether it's now more informative. `opp_sp_hand_L`'s coefficient/z-score should also be recorded and compared against the pre-fix baseline (coef -0.030, z=3.55).

- [ ] **Step 4: Re-run the collinearity check**

Run:
```bash
python3 -c "
import pandas as pd
from feature_store import build_feature_table
df = build_feature_table()
cols = ['lineup_slot_norm','platoon_edge','tb_roll','opp_sp_hand_L']
sub = df[cols].dropna()
print(sub.corr().round(3).to_string())
"
```
Expected: `platoon_edge` vs `tb_roll` correlation should be far below the pre-fix `platoon_tb_adj` vs `tb_roll` value of 0.999 (near-zero is expected, since the edge is now a pure handedness-matchup signal independent of the batter's own rate).

- [ ] **Step 5: Run the post-fix `model-vs-market` scoreboard**

Run:
```bash
python3 run_pipeline.py model-vs-market --start 2026-05-21 --end 2026-07-07 2>&1 | tee /tmp/model_vs_market_after.txt
```
Expected: same four tables as Task 4. Compare the "Overall" `LL model` and `w fit` values against `/tmp/model_vs_market_before.txt` from Task 4.

- [ ] **Step 6: Diff before/after and report**

Run:
```bash
diff /tmp/model_vs_market_before.txt /tmp/model_vs_market_after.txt || true
```
Summarize in the final report: whether overall `LL model` moved closer to `LL market`, whether `w fit` increased (model earning more blend weight), and whether the `disagreement`/`line` slices most related to pitcher/lineup signal (e.g. high-disagreement bucket) improved. This diff is the scoreboard's verdict on the fix — report it plainly, including if the result is a wash or a regression (per `SYSTEM.md`, a market win stays decisive; a model win should be re-checked with `--pit-train` before trusting it, since this run used the saved/look-ahead model).

- [ ] **Step 7: No code commit for this task** (the retrained `models/tb_model.pkl` and `models/model_meta.pkl` are binary artifacts already gitignored/regenerated locally — confirm with `git status` that nothing unexpected is staged, then stop.)

---

## Final report format

After Task 6, produce a short before/after summary covering:
1. The three fixes shipped (redundant feature dropped, platoon feature decorrelated, pitcher-hand resolution corrected).
2. Backfill scope (rows changed / total, L-rate before vs after).
3. Coefficient audit before/after for `opp_sp_hand_L` and `platoon_edge`/`platoon_tb_adj`.
4. `model-vs-market` overall and sliced before/after (`LL model`, `LL market`, `w fit`).
5. A plain verdict: did the scoreboard move, and by how much — not just "tests pass."
