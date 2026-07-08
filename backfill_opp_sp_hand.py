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

Note: this uses the ACTUAL starter (pitcher_games.is_starter=1), while the
forward-looking ETL (data_engine._opp_sp_hand_L_by_game_id) uses the
ANNOUNCED PROBABLE starter -- the two can differ when a probable is scratched
for an opposite-handed replacement. A full non-incremental re-run of
data_engine.build_historical_store will re-stamp historical rows from the
probable starter, overwriting this backfill's corrections.
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
