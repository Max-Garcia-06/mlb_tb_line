"""
Static venue physics table (Statcast-style distance/elevation priors).

Used at feature time; values are slow-moving park priors, not live Statcast pulls.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from config import BASE_DIR

_VENUE_CSV = BASE_DIR / "data" / "venue_physics.csv"


@lru_cache(maxsize=1)
def load_venue_table() -> dict[str, tuple[float, float]]:
    """
    team_abbr -> (distance_added_index, elevation_ft)
    """
    out: dict[str, tuple[float, float]] = {}
    if not _VENUE_CSV.exists():
        return out
    with _VENUE_CSV.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ab = (row.get("team_abbr") or "").strip().upper()
            if not ab:
                continue
            try:
                dai = float(row.get("distance_added_index") or 0.0)
                el = float(row.get("elevation_ft") or 0.0)
            except ValueError:
                continue
            out[ab] = (dai, el)
    return out


def lookup_park_physics(park_team_abbr: str | None) -> tuple[float, float]:
    if not park_team_abbr:
        return 0.0, 0.0
    t = load_venue_table()
    return t.get(str(park_team_abbr).strip().upper(), (0.0, 0.0))
