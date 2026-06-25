"""
Supervised scheduling of post-trade mark snapshots (replaces silent subprocess.Popen).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger(__name__)

SCHEDULE_PATH = DATA_DIR / "mark_schedule.jsonl"


def schedule_marks(
    *,
    game_date: str,
    delays_minutes: list[int],
    script_path: str | None = None,
) -> list[dict]:
    """
    Record mark jobs and optionally spawn supervised workers that log outcomes.
    Returns list of scheduled job dicts.
    """
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    script = script_path or str(Path(__file__).resolve().parent / "run_pipeline.py")
    jobs: list[dict] = []
    for d in sorted(set(int(x) for x in delays_minutes if int(x) > 0)):
        label = f"{d}m"
        job = {
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "game_date": game_date,
            "delay_minutes": d,
            "label": label,
            "status": "pending",
        }
        with open(SCHEDULE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(job, separators=(",", ":")) + "\n")
        argv = [sys.executable, script, "mark", "--date", game_date, "--label", label]
        # Supervised: log stderr to data/mark_logs/
        log_dir = DATA_DIR / "mark_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mark_{game_date}_{label}.log"

        def _run(delay_sec: int, command: list[str], out_path: Path) -> None:
            time.sleep(delay_sec)
            try:
                with open(out_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n--- run {datetime.now(timezone.utc).isoformat()} ---\n")
                    subprocess.run(command, check=False, stdout=lf, stderr=lf)
            except Exception as e:
                with open(out_path, "a", encoding="utf-8") as lf:
                    lf.write(f"ERROR: {e}\n")

        import threading

        t = threading.Thread(
            target=_run,
            args=(int(d) * 60, argv, log_file),
            daemon=True,
            name=f"mark-{game_date}-{label}",
        )
        t.start()
        job["log_file"] = str(log_file)
        job["status"] = "spawned"
        jobs.append(job)
        log.info("Scheduled mark %s for %s in %sm → %s", label, game_date, d, log_file)
    return jobs
