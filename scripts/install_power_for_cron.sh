#!/usr/bin/env bash
# Keep Mac awake for cron windows, on AC or battery.
# - AC power: system does not sleep (display may dim)
# - Battery power: relies on the wake alarm + caffeinate bridge below,
#   since pmset -c (AC-only) settings don't apply on battery.
# - launchd: single caffeinate bridge spanning every job that needs the
#   machine awake (see com.mlb_tb_line.keep-awake-overnight.plist for why
#   it's one consolidated window instead of per-job windows).
# - pmset: one repeating wake alarm at 22:50 (system-local time). macOS
#   only supports a single repeat wake system-wide, so this same alarm
#   also covers ca-jepa's 6:20am pre-market cron job, which used to own
#   this slot with its own 6:15am wake.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_DIR="${HOME}/Library/LaunchAgents"
mkdir -p "$LAUNCH_DIR" "${REPO_ROOT}/logs"

install_plist() {
  local src="$1"
  local name="$(basename "$src")"
  local dest="${LAUNCH_DIR}/${name}"
  sed "s|@@REPO@@|${REPO_ROOT}|g" "$src" >"$dest"
  launchctl bootout "gui/$(id -u)" "$dest" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$dest"
  echo "Loaded $dest"
}

echo "=== LaunchAgent (caffeinate bridge, works on AC or battery) ==="
install_plist "${REPO_ROOT}/scripts/com.mlb_tb_line.keep-awake-overnight.plist"

echo ""
echo "=== pmset (requires sudo) ==="
echo "AC power: system sleep disabled entirely (belt-and-suspenders on top of the bridge)."
echo "Battery power: normal sleep still applies outside the 22:50-07:50 bridge window."
echo ""

if [[ "$(uname)" != "Darwin" ]]; then
  echo "pmset steps skipped (not macOS)."
  exit 0
fi

sudo pmset -c sleep 0
sudo pmset -c disksleep 0
sudo pmset -c displaysleep 15
sudo pmset -c standby 0
sudo pmset -c autopoweroff 0

# One repeating wake (macOS allows exactly one); fires before the bridge's
# StartCalendarInterval so launchd's job has an awake machine to fire on.
sudo pmset repeat cancel 2>/dev/null || true
sudo pmset repeat wakeorpoweron MTWRFSU 22:50:00

echo ""
echo "Current custom power settings:"
pmset -g custom || pmset -g
echo ""
pmset -g sched 2>/dev/null || true

echo ""
echo "Done."
echo "  22:50 (system-local) — wake + caffeinate for 9h, covering:"
echo "    23:00 mlb_tb_line nightly (etl/reconcile/report)"
echo "    06:20 ca-jepa pre-market order submission (weekdays)"
echo "    07:00 mlb_tb_line morning snapshot + first scan-live"
echo "  On AC power — system sleep additionally disabled outright"
echo ""
echo "Revert AC sleep later: sudo pmset -c sleep 1"
echo "Remove wake agent: launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.mlb_tb_line.keep-awake-overnight.plist"
