#!/usr/bin/env bash
# Install Phase 1 cron entries (snapshot 10:00 ET, nightly 02:00 ET).
# Backs up existing crontab and replaces any prior mlb_tb_line block.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARKER_BEGIN="# BEGIN mlb_tb_line phase1"
MARKER_END="# END mlb_tb_line phase1"
EXAMPLE="${REPO_ROOT}/scripts/crontab.phase1.example"
CRON_BLOCK="$(sed "s|@@REPO@@|${REPO_ROOT}|g" "$EXAMPLE" | grep -v '^#' | grep -v '^$' | grep -v '^SHELL=' | grep -v '^PATH=' || true)"

chmod +x "${REPO_ROOT}/scripts/cron_job.sh"

TMP="$(mktemp)"
(crontab -l 2>/dev/null || true) | awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
  $0 == b { skip=1; next }
  $0 == e { skip=0; next }
  !skip { print }
' >"$TMP"

{
  echo "$MARKER_BEGIN"
  echo "SHELL=/bin/bash"
  echo "PATH=/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"
  echo "$CRON_BLOCK"
  echo "$MARKER_END"
} >>"$TMP"

echo "Installing crontab block:"
echo "---"
grep -A20 "$MARKER_BEGIN" "$TMP" || true
echo "---"
crontab "$TMP"
rm -f "$TMP"
echo "Done. Logs: ${REPO_ROOT}/logs/cron.log"
echo "Test now:"
echo "  ${REPO_ROOT}/scripts/cron_job.sh snapshot"
echo "  ${REPO_ROOT}/scripts/cron_job.sh nightly"
