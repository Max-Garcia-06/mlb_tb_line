#!/usr/bin/env bash
# Watchdog: kill any run_pipeline.py scan process that has run longer than
# MAX_AGE_SECONDS. A normal scan (including its 30m/120m auto-mark sleep)
# finishes well under this; anything older is stuck (e.g. survived a
# system sleep/wake with a hung network call) and blocks reconcile/report
# from seeing that day's fills.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${REPO_ROOT}/logs/cron.log"
MAX_AGE_SECONDS=$((3 * 3600))
TS="$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"

# macOS ps has no "etimes" (seconds) keyword, only "etime" as
# [[dd-]hh:]mm:ss — parse it by hand.
etime_to_seconds() {
  local etime="$1" days=0 rest="$1" h=0 m=0 s=0
  if [[ "$etime" == *-* ]]; then
    days="${etime%%-*}"
    rest="${etime#*-}"
  fi
  IFS=':' read -r a b c <<<"$rest"
  if [[ -n "${c:-}" ]]; then
    h="$a"; m="$b"; s="$c"
  else
    m="$a"; s="$b"
  fi
  echo $((10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

matches="$(ps -eo pid,etime,command | grep "run_pipeline.py scan" | grep -v grep || true)"
if [[ -n "$matches" ]]; then
  while read -r pid etime _cmd; do
    age="$(etime_to_seconds "$etime")"
    if [[ "$age" -gt "$MAX_AGE_SECONDS" ]]; then
      echo "[$TS] [watchdog] killing stuck PID $pid (running ${age}s > ${MAX_AGE_SECONDS}s)" | tee -a "$LOG_FILE"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done <<<"$matches"
fi
