#!/usr/bin/env bash
# Run a single pipeline step for cron (logging, venv, ET dates).
# Usage: scripts/cron_job.sh <snapshot|etl|reconcile|report|nightly|refit-blend>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
  echo "Usage: $0 <snapshot|etl|reconcile|report|nightly|refit-blend>" >&2
  exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

LOG_FILE="${LOG_DIR}/cron.log"
TS="$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"

log() {
  echo "[$TS] [$JOB] $*" | tee -a "$LOG_FILE"
}

run_py() {
  log "START: $*"
  set +e
  "$PYTHON" run_pipeline.py "$@" >>"$LOG_FILE" 2>&1
  local code=$?
  set -e
  if [[ $code -eq 0 ]]; then
    log "OK (exit 0)"
  else
    log "FAILED (exit $code)"
  fi
  return $code
}

# Calendar dates in US/Eastern (MLB slate context)
TODAY_ET="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
tz = ZoneInfo('America/New_York')
print(datetime.now(tz).date().isoformat())
")"

YESTERDAY_ET="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
tz = ZoneInfo('America/New_York')
print((datetime.now(tz).date() - timedelta(days=1)).isoformat())
")"

SCAN_LOG="${LOG_DIR}/scan.log"

case "$JOB" in
  snapshot)
    # Pre-game tape for today's slate
    run_py snapshot --date "$TODAY_ET"
    ;;
  etl)
    run_py etl --incremental --workers 8
    ;;
  reconcile)
    # After games: journal for yesterday's slate (2 AM job is usually post-evening slate)
    run_py reconcile --date "$YESTERDAY_ET"
    ;;
  report)
    run_py report --date "$YESTERDAY_ET"
    ;;
  nightly)
    # ETL, reconcile yesterday's fills, then report against them
    run_py etl --incremental --workers 8
    run_py reconcile --date "$YESTERDAY_ET"
    run_py report --date "$YESTERDAY_ET"
    ;;
  refit-blend)
    # Re-fit market-blend weight (global + per-segment) from a trailing window of
    # full-slate scoring, so w tracks the model's actual recent performance vs.
    # the market instead of staying pinned wherever it was last set by hand.
    run_py refit-blend
    ;;
  scan)
    # Dry-run edge scan — logs to scan.log, no orders placed
    log "START scan (dry-run) for $TODAY_ET" | tee -a "$SCAN_LOG"
    set +e
    "$PYTHON" run_pipeline.py scan --date "$TODAY_ET" --dry-run >>"$SCAN_LOG" 2>&1
    code=$?
    set -e
    echo "[$TS] [scan] EXIT $code" | tee -a "$SCAN_LOG"
    ;;
  scan-live)
    # Live scan — places orders on Kalshi
    log "START scan (LIVE) for $TODAY_ET" | tee -a "$SCAN_LOG"
    set +e
    "$PYTHON" run_pipeline.py scan --date "$TODAY_ET" --live >>"$SCAN_LOG" 2>&1
    code=$?
    set -e
    echo "[$TS] [scan-live] EXIT $code" | tee -a "$SCAN_LOG"
    ;;
  *)
    log "Unknown job: $JOB"
    exit 1
    ;;
esac
