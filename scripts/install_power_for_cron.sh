#!/usr/bin/env bash
# Keep Mac awake for Phase 1 cron windows (plugged in).
# - AC power: system does not sleep (display may dim)
# - launchd: caffeinate before 10:00 and 2:00 ET jobs
# - pmset: daily wake at 1:50 ET (backup if machine did sleep)

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

echo "=== LaunchAgents (caffeinate before cron) ==="
install_plist "${REPO_ROOT}/scripts/com.mlb_tb_line.keep-awake-snapshot.plist"
install_plist "${REPO_ROOT}/scripts/com.mlb_tb_line.keep-awake-nightly.plist"

echo ""
echo "=== pmset (requires sudo) — AC power: system stays awake ==="
echo "When plugged in: no system sleep; display may turn off after 15 minutes."
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

# One repeating wake (macOS allows one repeat); 1:50 ET before 2:00 nightly cron
sudo pmset repeat cancel 2>/dev/null || true
sudo pmset repeat wakeorpoweron MTWRFSU 01:50:00

echo ""
echo "Current AC (charger) settings:"
pmset -g custom | sed -n '/Battery Power/,/AC Power/p' | tail -n +1 || pmset -g
echo ""
pmset -g sched 2>/dev/null || true

echo ""
echo "Done."
echo "  9:50 ET — caffeinate ~45m (covers 10:00 snapshot cron)"
echo "  1:50 ET — caffeinate ~2h + wake repeat (covers 2:00 nightly cron)"
echo "  On AC power — system sleep disabled while plugged in"
echo ""
echo "Revert AC sleep later: sudo pmset -c sleep 1"
echo "Remove wake agents: launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.mlb_tb_line.keep-awake-*.plist"
