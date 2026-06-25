"""
One-shot backfill: fetch pitcher_games rows for all game_ids already in batter_games
that are NOT yet in pitcher_games. Run once after adding the pitcher ETL feature.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
import sqlalchemy as sa

from config import DB_PATH
from data_engine import _fetch_pitcher_games_parallel, _existing_pitcher_game_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main() -> None:
    engine = sa.create_engine(f"sqlite:///{DB_PATH}")

    batter_games = pd.read_sql("SELECT DISTINCT game_id, game_date FROM batter_games", engine)
    all_game_ids = {int(r.game_id) for _, r in batter_games.iterrows()}

    existing_pitcher_ids = _existing_pitcher_game_ids(engine)
    missing = {gid for gid in all_game_ids if gid not in existing_pitcher_ids}
    log.info("Pitcher backfill: %s game(s) to fetch (of %s total)", len(missing), len(all_game_ids))

    if not missing:
        log.info("Nothing to backfill.")
        return

    date_map = {int(r.game_id): str(r.game_date)[:10] for _, r in batter_games.iterrows()}
    games = [{"game_id": gid, "game_date": date_map[gid]} for gid in sorted(missing)]

    CHUNK = 500
    total_written = 0
    for i in range(0, len(games), CHUNK):
        chunk = games[i : i + CHUNK]
        log.info("Fetching chunk %s–%s of %s...", i + 1, min(i + CHUNK, len(games)), len(games))
        rows = _fetch_pitcher_games_parallel(chunk, workers=WORKERS)
        if rows:
            dfp = pd.DataFrame(rows)
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
            total_written += len(dfp)
            log.info("  wrote %s pitcher rows (total so far: %s)", len(dfp), total_written)

    log.info("Backfill complete: %s pitcher rows written.", total_written)


if __name__ == "__main__":
    main()
