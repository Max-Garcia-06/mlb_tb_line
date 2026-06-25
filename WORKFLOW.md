# MLB Total Bases Pipeline — Full Workflow

End-to-end guide for research, live trading, and performance review on Kalshi MLB total-bases markets.

Entry point for all commands:

```bash
python3 run_pipeline.py <command> [options]
```

---

## 1. What the system does

```text
MLB Stats API  ──etl──►  SQLite (batter_games)
                              │
                              ▼
                         train / evaluate
                              │
                              ▼
                         models/*.pkl + calibrators
                              │
Kalshi API ◄──scan────────────┘
     │
     ├──► orders (live) + trade journal (JSONL)
     └──► market snapshots (JSONL) ──► backtest vs actual TB
```

| Phase | Command(s) | Output |
|-------|------------|--------|
| Data | `etl` | `data/mlb_tb.db` — one row per batter-game |
| Model | `train`, `evaluate`, `tune` | `models/tb_model.pkl`, OOF calibrator |
| Markets | `snapshot`, `schedule-snapshots` | `data/snapshots/tb_markets_YYYY-MM-DD.jsonl` |
| Trade | `scan`, `mark`, `reconcile` | `data/trades_YYYY-MM-DD.jsonl`, `data/execution_ledger.json` |
| Review | `report`, `report-range`, `calibrate` | Terminal tables; updated calibrators |
| Research | `backtest`, `segment-report` | Simulated P&L + CLV; segment go/no-go |

Default model: **ordinal logistic** full TB distribution (set `USE_LEGACY_XGB=1` for XGB + Poisson/NB).

---

## 2. One-time setup

### 2.1 Install

```bash
cd /path/to/mlb_tb_line
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2.2 Configure `.env`

Minimum for **live Kalshi**:

- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH` (RSA PEM)
- `KALSHI_BASE_URL` / `KALSHI_ORDER_URL` (production or demo)

Recommended for **small live bankroll** (~$160):

```bash
USE_LIVE_BALANCE=true          # scan sizes off Kalshi balance, not --bankroll default
EDGE_THRESHOLD=0.07
MIN_P=0.55
MAX_BET_PCT=0.04
DAILY_LOSS_LIMIT_USD=50        # halt live orders if journal shows worse
MAX_ORDERS_PER_DAY=20
```

Set `REQUIRE_KALSHI_CREDENTIALS=1` in production so missing keys fail instead of using the mock client.

### 2.3 Halt switch (optional)

Create or touch to stop all live orders:

```bash
touch data/KILL_SWITCH
```

Remove the file to resume trading.

With `AUTO_KILL_ON_RISK_BREACH=true` (default), breaching deployment, resting, concentration, or daily loss limits during a live scan **creates** this file automatically. Delete it manually to resume.

---

## 3. Cold start (first time or full rebuild)

Run once when setting up a new machine or rebuilding history.

```bash
# 1) Pull batter-game history (parallel; 4 seasons by default)
python3 run_pipeline.py etl --workers 12
#    Faster dev:  python3 run_pipeline.py etl --seasons 2024,2025 --workers 12

# 2) Train model + OOF calibrator + materialize feature table
python3 run_pipeline.py train

# 3) Walk-forward CV metrics (sanity check)
python3 run_pipeline.py evaluate

# 4) Optional: hyperparameter search (XGB legacy only benefits most)
python3 run_pipeline.py tune --trials 25
```

**Artifacts created:**

- `data/mlb_tb.db` — table `batter_games` (+ `batter_features` after train/etl)
- `models/tb_model.pkl`, `models/model_meta.pkl`
- `models/p_calibrator_oof.pkl` (if enough CV rows)
- `models/p_calibrator_isotonic.pkl` / segmented (after `calibrate` with fills)

---

## 4. Daily trading workflow (game day)

Typical slate for **today** (`YYYY-MM-DD`). Order matters for honest backtest later.

### Morning / pre-game

```text
┌─────────────────────────────────────────────────────────────┐
│  A. Capture market tape (for backtest + CLV later)          │
│  B. Dry-run scan (review edges, no orders)                  │
│  C. Live scan (place limits)                                │
│  D. Auto-mark scheduled (30m, 120m) if --live               │
└─────────────────────────────────────────────────────────────┘
```

**A. Snapshot** (before first pitch — records books you could trade):

```bash
python3 run_pipeline.py snapshot
# or explicit date:
python3 run_pipeline.py snapshot --date 2026-05-21
```

Optional intraday tape:

```bash
python3 run_pipeline.py schedule-snapshots --date 2026-05-21 --interval 15 --count 4
```

**B. Dry-run scan** (default; uses Kalshi balance if `USE_LIVE_BALANCE=true`):

```bash
python3 run_pipeline.py scan --dry-run --one-per-player --max-contracts 25
```

Review the table: green `eY`/`eN` = passes edge gates (not the same as “will place order” in dry-run).

**C. Live scan**:

```bash
python3 run_pipeline.py scan --live \
  --one-per-player \
  --max-signals 10 \
  --max-contracts 25 \
  --within-hours 2
```

What live scan does:

1. Loads model + features (PIT feature table if present)
2. Fetches Kalshi TB markets; **keeps only games starting within `SCAN_WITHIN_HOURS` (default 2h)** using MLB `game_datetime`, then **drops started/final games**
3. VPIN flow guard on books
4. Edge detection + fractional Kelly + portfolio cap
5. **Risk checks**: kill switch, daily loss, max orders/day, **live balance**
6. Limit orders + `data/trades_YYYY-MM-DD.jsonl` journal
7. Schedules `mark` at 30m and 120m (logs in `data/mark_logs/`)

#### Slate waves (multi-scan days)

By default, `scan` only trades games whose **first pitch is 0–3 hours away** (UTC schedule from MLB Stats API). A single morning run will **not** place orders on night games — run again closer to each wave.

| Approx. scan (ET) | Window includes (first pitch) |
|-------------------|----------------------------------|
| 9:30 AM | ~9:30 AM–12:30 PM |
| 1:00 PM | ~1:00–4:00 PM |
| 4:30 PM | ~4:30–7:30 PM |

```bash
# Default: 2h window (or SCAN_WITHIN_HOURS in .env)
python3 run_pipeline.py scan --live --one-per-player --max-contracts 25 --max-signals 10

# Wider/narrower window
python3 run_pipeline.py scan --dry-run --within-hours 2

# Legacy: full pre-game slate in one run
python3 run_pipeline.py scan --dry-run --no-time-window
```

Set `SCAN_WITHIN_HOURS=0` in `.env` to disable the window by default. Backtest replay is unchanged; live windowed scans are not comparable to a single full-slate snapshot unless you captured books near each wave.

**D. Manual mark** (if auto-mark off):

```bash
python3 run_pipeline.py mark --date 2026-05-21 --label 30m
python3 run_pipeline.py mark --date 2026-05-21 --label 120m
```

### After games finish

```text
┌─────────────────────────────────────────────────────────────┐
│  E. Incremental ETL (new game outcomes)                   │
│  F. Reconcile fills from Kalshi                             │
│  G. Report (P&L, calibration, CLV)                          │
│  H. Optional: recalibrate from your fills                   │
└─────────────────────────────────────────────────────────────┘
```

**E. ETL** (only new `game_id`s):

```bash
python3 run_pipeline.py etl --incremental --workers 12
```

**F. Report** (reconciles Kalshi fills first by default):

```bash
python3 run_pipeline.py report --date 2026-05-21
```

Multi-day (reconciles every journal day in range before aggregating):

```bash
python3 run_pipeline.py report-range --start 2026-05-01 --end 2026-05-24
```

Use `--no-reconcile` for a fast re-print when fills are already synced. Fill-only sync without tables: `reconcile --date ...` or `reconcile --start ... --end ...`.

Primary metric in report: **avg CLV vs close** (from mark rows). Also ROI, buckets by line/edge/spread.

**H. Calibrate** (after you have enough reconciled fills):

```bash
python3 run_pipeline.py calibrate
```

---

## 5. Backtest workflow (research)

Backtest does **not** use your trade journal. It replays **saved snapshots** + **ETL outcomes**.

**Prerequisites per date:**

1. `snapshot` (or `schedule-snapshots`) **before** games
2. `etl` **after** games (actual TB in `batter_games`)

```bash
# Example week
python3 run_pipeline.py snapshot --date 2024-06-01   # repeat for each day OR schedule-snapshots
# ... games play ...
python3 run_pipeline.py etl --incremental

# Run backtest
python3 run_pipeline.py backtest \
  --start 2024-06-01 \
  --end 2024-06-07 \
  --bankroll 160 \
  --pit-train \
  --max-signals 5 \
  --max-contracts 25
```

**What backtest uses:**

| Piece | Source |
|-------|--------|
| Entry prices | **Latest** snapshot per ticker within `SCAN_WITHIN_HOURS` of first pitch (default; matches live `scan`) |
| Close / CLV | **Latest** snapshot per ticker that day |
| Model | PIT retrain on data **before** each date (weekly buckets by default) |
| Outcomes | `batter_games.tb` for that `game_date` |
| Sizing | Same Kelly / portfolio / VPIN / fill-prob as config (not live balance) |

Capture snapshots near each slate wave (`schedule-snapshots` or manual `snapshot` before each `scan`). A single morning tape will exclude afternoon games from windowed backtest.

Legacy earliest-of-day entry: `backtest --no-time-window`.

Disable PIT: `--no-pit-train` (uses single `train` model — can leak future data).

### Segment health (go/no-go)

After `reconcile` and `mark`:

```bash
python3 run_pipeline.py segment-report --start 2026-05-01 --end 2026-05-24
```

**TRADE** if at least one segment (side × line × spread × edge) passes fill-rate, CLV, and ROI gates. **PAUSE** otherwise. Live `scan` prints a dim warning when lookback is PAUSE (does not block).

### Calibration (fills vs OOF)

| Artifact | Source | Used at scan |
|----------|--------|--------------|
| `p_calibrator_oof.pkl` | Walk-forward CV (`train --fit-oof`) | **First** when `USE_OOF_CALIBRATION=true` |
| `p_calibrator_segmented.pkl` | Your fills only (`reconcile` → `calibrate`) | Fallback |

`calibrate` fits **only** reconciled `note=fill` rows. Re-run after ~50+ resolved fills; metadata in `models/calibrator_meta.json`. Set `REQUIRE_FILL_CALIB_FOR_LIVE=1` to block live scan if fill calibrator is missing or older than `CALIBRATE_MAX_AGE_DAYS`.

---

## 6. Cron automation (Phase 1)

Low-risk jobs only: **snapshot** + **nightly close** (ETL, report with implicit reconcile). No live `scan`.

### Install (once)

```bash
chmod +x scripts/cron_job.sh scripts/install_phase1_cron.sh
./scripts/install_phase1_cron.sh
```

This installs:

| Schedule (US/Eastern) | Job | What it runs |
|------------------------|-----|----------------|
| **10:00** daily | `snapshot` | `snapshot --date` today (pre-game tape) |
| **02:00** daily | `nightly` | `etl --incremental` → `report` for **yesterday** (reconcile on) |

Logs append to `logs/cron.log`.

### Manual test

```bash
./scripts/cron_job.sh snapshot
./scripts/cron_job.sh nightly
```

### Uninstall

Edit crontab and remove the block between `# BEGIN mlb_tb_line phase1` and `# END mlb_tb_line phase1`, or run:

```bash
crontab -l | awk '/BEGIN mlb_tb_line phase1/,/END mlb_tb_line phase1/{next} {print}' | crontab -
```

Phase 2 (optional later): add `scan --live` to cron — see earlier ops notes; not installed by default.

### Keep Mac awake at cron times (plugged in)

```bash
./scripts/install_power_for_cron.sh
```

This configures:

| Mechanism | Schedule (ET) | Effect |
|-----------|---------------|--------|
| **LaunchAgent + caffeinate** | 9:50 | Blocks idle sleep ~45m → 10:00 `snapshot` cron |
| **LaunchAgent + caffeinate** | 1:50 | Blocks idle sleep ~2h → 2:00 `nightly` cron |
| **pmset (on AC power)** | While charging | System does not sleep (display may dim after 15m) |
| **pmset repeat wake** | 1:50 daily | Backup wake before nightly job |

Requires your Mac to be **plugged in** (and ideally lid open or clamshell with external display). Revert system sleep: `sudo pmset -c sleep 1`.

---

## 7. Weekly / maintenance

| Task | Command | When |
|------|---------|------|
| Refresh history | `etl --incremental` | Daily after slates (also in cron `nightly`) |
| Retrain model | `train` | Weekly or after large ETL |
| Re-evaluate | `evaluate` | After retrain |
| Multi-day P&L | `report-range --start ... --end ...` | Anytime |
| Rebuild features only | `materialize-features` | After ETL if table stale |
| Retune XGB | `tune` | Occasionally if using legacy head |

Full ETL rebuild (slow):

```bash
python3 run_pipeline.py etl --workers 12
python3 run_pipeline.py train
```

---

## 8. Data files reference

| Path | Purpose |
|------|---------|
| `data/mlb_tb.db` | SQLite: `batter_games`, `batter_features` |
| `data/snapshots/tb_markets_*.jsonl` | Market snapshot tape |
| `data/trades_*.jsonl` | Trade journal (`pre-submit`, `post-submit`, `fill`, `mark`) |
| `data/execution_ledger.json` | Dedup keys for live orders |
| `data/mark_schedule.jsonl` | Scheduled mark jobs |
| `data/mark_logs/*.log` | Mark subprocess output |
| `data/KILL_SWITCH` | Present = no live orders |
| `data/pipeline.jsonl` | Structured logs if `STRUCTURED_LOG=1` |
| `models/*.pkl` | Model + calibrators |

---

## 9. Command reference

| Command | Description |
|---------|-------------|
| `etl` | MLB boxscores → `batter_games` |
| `train` | Fit model; OOF calibrator; materialize features |
| `evaluate` | Walk-forward CV |
| `tune` | Hyperparameter search |
| `snapshot` | One-time Kalshi book capture |
| `schedule-snapshots` | Repeated captures |
| `scan` | Edge detection + optional live orders |
| `mark` | Post-trade price snapshot for CLV |
| `reconcile` | Sync Kalshi fills → journal (fill-only; optional `--start`/`--end`) |
| `report` | Reconcile + single-day performance (default) |
| `report-range` | Reconcile each day in range + multi-day aggregate (default) |
| `calibrate` | Fit isotonic calibrator from fills |
| `backtest` | Replay snapshots vs actual TB |
| `materialize-features` | Rebuild `batter_features` table |

### Common flags

**ETL:** `--seasons 2024,2025`, `--incremental`, `--workers 12`, `--fetch-weather`

**Scan:** `--live`, `--date`, `--max-signals 5`, `--one-per-player`, `--max-contracts 25`, `--threshold`, `--min-p`, `--no-auto-mark`

**Backtest:** `--start`, `--end`, `--bankroll`, `--pit-train` / `--no-pit-train`, `--max-signals`, `--max-contracts`

---

## 10. Risk desk (live)

### Sizing

With `USE_LIVE_BALANCE=true`:

- Bankroll = Kalshi **available balance**
- When `RESERVE_RESTING_FROM_BANKROLL=true`, Kelly sizing uses `balance − open_resting_usd`
- Per-leg cap: `MAX_BET_PCT × effective_bankroll`
- Portfolio cap: `MAX_PORTFOLIO_PCT` (defaults to `MAX_BET_PCT`)
- CLI `--bankroll` is fallback only if balance fetch fails

### Hard stops (checked at scan start and again before live orders)

| Control | Env | Notes |
|---------|-----|--------|
| Kill switch | `KILL_SWITCH_PATH` | Manual file or auto-created on breach |
| Daily P&L | `DAILY_LOSS_LIMIT_USD` | Settled fills via Kalshi `result` + mark MTM on open fills |
| Order count | `MAX_ORDERS_PER_DAY` | Successful `post-submit` rows today |
| Daily deployment | `MAX_DAILY_DEPLOYED_USD` | Sum of `limit×contracts` (journal + proposed scan) |
| Resting capital | `MAX_OPEN_RESTING_USD` | Exchange resting orders + proposed |
| Per game | `MAX_CONTRACTS_PER_GAME` | Filled + proposed contracts per matchup slug |
| Per player/day | `MAX_LEGS_PER_PLAYER_DAY` | Distinct `(player, line)` legs |

Pre-mark sessions: daily loss uses **realized** settlement only; unsettled fills count toward deployment/resting caps but not MTM loss until `mark` runs.

### Suggested small bankroll (~$160)

```bash
DAILY_LOSS_LIMIT_USD=50
MAX_ORDERS_PER_DAY=20
MAX_DAILY_DEPLOYED_USD=25
MAX_OPEN_RESTING_USD=15
MAX_CONTRACTS_PER_GAME=30
MAX_LEGS_PER_PLAYER_DAY=2
AUTO_KILL_ON_RISK_BREACH=true
RESERVE_RESTING_FROM_BANKROLL=true
```

Multi-scan days: each `scan` wave adds to **daily deployed**; run separate waves only if caps allow.

---

## 11. Decision checklist before going live

- [ ] `etl` + `train` + `evaluate` completed
- [ ] `.env` has real Kalshi keys and conservative `MIN_P` / `EDGE_THRESHOLD`
- [ ] `scan --dry-run` looks reasonable on today's slate
- [ ] `USE_LIVE_BALANCE=true` shows correct balance (not $1000 fallback)
- [ ] `snapshot` run for today (if you care about backtest/CLV later)
- [ ] `MAX_SIGNALS` / `MAX_CONTRACTS` appropriate for ~$160 bankroll
- [ ] No `data/KILL_SWITCH` file (remove after auto-kill if limits tripped)
- [ ] `MAX_DAILY_DEPLOYED_USD` / `MAX_OPEN_RESTING_USD` set for intraday multi-scan slates

---

## 12. Quick command cheat sheet

```bash
# Today — research only
python3 run_pipeline.py snapshot
python3 run_pipeline.py scan --dry-run

# Today — live
python3 run_pipeline.py snapshot
python3 run_pipeline.py scan --live --max-signals 5 --one-per-player --max-contracts 25

# Tonight — close the loop
python3 run_pipeline.py etl --incremental
python3 run_pipeline.py report

# Research — past week
python3 run_pipeline.py backtest --start 2024-06-01 --end 2024-06-07 --bankroll 160 --pit-train
```

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Mock markets only (3 players) | Missing Kalshi creds | Set `.env` keys; `REQUIRE_KALSHI_CREDENTIALS=1` |
| Bankroll shows $1000 | Balance parse failed | Fixed in `get_balance`; re-run scan; check logs |
| No edges | Threshold / `MIN_P` too strict | Lower in `.env` or `--threshold` / `--min-p` |
| Backtest empty | No snapshot for date | Run `snapshot` before games for that date |
| Backtest empty | No outcomes | Run `etl` after games |
| Train/serve mismatch | Old ETL without `opp_sp_hand_L` | Full or incremental `etl` to refresh SP hand |
| Live blocked | Risk check | Read warning; remove kill switch or loss limit |

---

*Generated for the `mlb_tb_line` repo. See `README.md` for install notes and `config.py` / `.env.example` for all tunables.*
