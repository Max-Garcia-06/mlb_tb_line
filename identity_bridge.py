"""
xrefId / name resolution to MLB Stats API player_id.

Prefers IDs that appear in the training feature table to avoid wrong-player collisions
(e.g. MiLB vs MLB with the same name).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import statsapi

if TYPE_CHECKING:
    import pandas as pd


def norm_player_name(s: str) -> str:
    """Normalize display names for matching (whitespace, case)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def resolve_mlb_player_id(
    *,
    player_name: str,
    xref_player_id: str | None = None,
    feat_df: "pd.DataFrame | None" = None,
    allowed_player_ids: set[int] | frozenset[int] | None = None,
) -> int:
    x = (xref_player_id or "").strip()
    if x.isdigit():
        return int(x)

    name = (player_name or "").strip()
    if not name:
        return 0

    ids_in_data: set[int] = set()
    if feat_df is not None and "player_id" in feat_df.columns:
        ids_in_data = set(feat_df["player_id"].dropna().astype(int).unique())
    elif allowed_player_ids is not None:
        ids_in_data = set(allowed_player_ids)

    def _pick_from_hits(hits: list) -> int:
        if not hits:
            return 0
        if ids_in_data:
            for h in hits:
                hid = int(h.get("id", 0) or 0)
                if hid in ids_in_data:
                    return hid
        return int(hits[0]["id"])

    try:
        hits = statsapi.lookup_player(name.strip())
    except Exception:
        hits = []
    pid = _pick_from_hits(hits or [])
    if pid:
        return pid

    base = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv)\s*$", "", name, flags=re.I).strip()
    if base != name:
        try:
            hits2 = statsapi.lookup_player(base)
        except Exception:
            hits2 = []
        return _pick_from_hits(hits2 or [])
    return 0
