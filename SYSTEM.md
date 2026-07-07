# SYSTEM.md — How the MLB TB pipeline works, and why it changed on 2026-07-06

This document explains the system end to end and gives the context behind the
profitability overhaul shipped on 2026-07-06. Read this before touching the
edge/sizing/reporting code. Operational how-to lives in `WORKFLOW.md`; this
file is the *why*.

---

## 1. What this system is

An automated pipeline that trades **Kalshi MLB player total-bases (TB)
markets** (e.g. "Francisco Lindor: 3+ total bases"):

```
MLB Stats API ──etl──► SQLite (batter_games, batter_features, statcast_*)
                            │
                            ▼
                    train (ordinal logistic full-TB-distribution model)
                            │
                            ▼
              models/tb_model.pkl + calibrators + blend_meta.json
                            │
Kalshi API ◄──scan──────────┘
    │  P(TB > k) per player/line → calibrate → blend with market → fee-adjusted
    │  edge vs limit price → Kelly sizing → risk gates → limit orders
    │
    ├──► trade journal (data/trades_YYYY-MM-DD.jsonl)
    ├──► CLV marks (scheduled re-quotes after entry)
    └──► reconcile fills → nightly report / report-range / calibrate / fit-blend
```

Live cadence (crontab): hourly `scan-live` 10:00–22:00 ET, a 10:00 ET book
snapshot, and a 23:00 PT nightly job (ETL + reconcile + report). Bankroll is
read from the live Kalshi balance (~$135–160).

### Key modules

| Module | Role |
|---|---|
| `data_engine.py` | MLB schedule/boxscores, pregame + start-window filters, lineups |
| `feature_store.py`, `statcast_engine.py`, `matchup_features.py` | feature building |
| `model.py`, `ordinal_core.py`, `probability_engine.py` | λ / full TB PMF → P(over line) |
| `calibration.py` | isotonic calibrators (OOF from CV; segmented from fills) |
| `market_blend.py` | **new** — logit-space shrinkage of model prob toward market mid |
| `fees.py` | **new** — Kalshi taker/maker fee model |
| `edge_detector.py` | gates, blend, fee-adjusted edge, Kelly sizing, order placement |
| `execution_engine.py` | limit-price suggestion (incl. maker mode), execution ledger |
| `risk_manager.py`, `journal_risk.py` | daily loss/deployment caps, kill switch |
| `trade_journal.py`, `reconcile_fills.py` | JSONL journal, fill reconciliation |
| `pipeline/commands.py` | all CLI commands (`run_pipeline.py <cmd>`) |

---

## 2. Why the 2026-07-06 overhaul happened

Performance audit of **612 resolved orders, 2026-04-27 → 2026-07-05**
(`report-range`):

- Realized P&L **-$66.21 on $1,217.89 cost (-5.4% ROI)** — *before* fees,
  which the code did not model anywhere (~$43 more, ≈3.4% of cost).
- **Systematic overconfidence exactly where we bet**: every p_model bucket won
  less than predicted — 0.7–0.8 bucket: predicted 75%, actual 56.6% (n=143,
  >4 SE); 0.8–0.9: predicted 85%, actual 64.5%.
- **ROI decayed monotonically with stated edge** (edge <0.05: +17%;
  0.05–0.10: -4%; 0.10–0.15: -9%; 0.15–0.20: -16%). Classic winner's curse:
  the bigger the disagreement with the market, the more it was model error.
- **Negative CLV on early entries**: fills marked 120 min later showed
  **-10.7¢/contract** vs +2.1¢ at 30 min.
- **Line-1.5 NO was the bleed segment**: line 1.5 = 417 orders, -$103
  (-11.7% ROI), concentrated on the NO side.
- Every order **crossed the spread at the ask** (taker) and paid the unmodeled
  taker fee.

Conclusion: the market was better than the model; unshrunk "edges" were mostly
model error, and execution gave away spread + fees on top.

---

## 3. What changed (and where)

### 3.1 Market blend — `market_blend.py`, applied in `edge_detector.quote_side_edge`

Model probabilities are shrunk toward the side's market mid in logit space:

```
p_final = sigmoid( w·logit(p_model_cal) + (1−w)·logit(market_mid) )
```

- `w` is fit on resolved fills by contract-weighted log-loss:
  `python3 run_pipeline.py fit-blend` → writes `models/blend_meta.json`,
  which `scan` reads automatically (env `BLEND_WEIGHT` overrides; kill with
  `USE_MARKET_BLEND=false`).
- Fitting uses `p_model_cal` (calibrated, **pre-blend**, journaled per trade
  since this change) so periodic re-fits never double-shrink.
- This is the fix for the winner's curse: isotonic calibration on all
  historical games (OOF) cannot correct probabilities *conditional on the
  trade filter*; shrinkage toward the price can.

**Current fitted value: `w = 0.02`** (2026-07-06, 612 fills; log-loss market
0.6218 vs model 0.6887). Read that plainly: **conditional on the old trade
filter, the model carried ~no information beyond the Kalshi price.** With
w=0.02 the scan emits almost no signals — the system is intentionally in
"prove it first" mode. Trading resumes in size only if a better model (or a
segment where it demonstrably wins) pushes the fitted `w` up. Do **not**
"fix" low volume by overriding `BLEND_WEIGHT` upward without new evidence.

### 3.2 Fees — `fees.py`, in both the EV gate and reports

- Kalshi taker fee `0.07·C·P·(1−P)` (ceil to cent per order) is subtracted
  from every candidate's edge before thresholding, and EV/Kelly use net odds
  `b = (1−L−fee)/L`. Maker (resting) fills are fee-free
  (`KALSHI_MAKER_FEE_RATE=0`).
- `report` / `report-range` now print `Kalshi fees (est)` and
  `Net P&L after fees`. Legacy journal rows (pre-2026-07-06) are estimated as
  taker at the fill price; new rows journal `fee_per_contract` exactly.

### 3.3 Edge is now measured vs the limit price, not the ask

`edge = p_blend − limit − fee`. Under the old code every signal's limit
equalled the ask anyway (the gate `p − ask > thr` forced it), so this is a
generalization, not a behavior break — it's what makes maker mode coherent.

### 3.4 Maker mode — `execution_engine.suggest_limit_price(maker=True)`

`MAKER_MODE=true` (default): orders rest at least one tick inside the ask
instead of lifting it. Saves the taker fee and a tick of spread; costs fill
probability. Turn off with `MAKER_MODE=false` per order-flow results.

### 3.5 Entry-time window enforced in live scan

`SCAN_WITHIN_HOURS` (default tightened **2 → 1.5**) is now actually applied in
`scan` via `data_engine.filter_market_lines_by_start_window` (previously only
the backtest used it — the live scan bought all day). Rationale: the CLV
numbers above.

### 3.6 Segment gate — `BLOCKED_SEGMENTS` (default `"1.5:no"`)

Line-1.5 NO is blocked from signal generation (`edge_detector.is_blocked_segment`).
Format: comma-separated `line:side` pairs; empty string disables.

### 3.7 More CLV marks

`scan --mark-delays` default changed from `30,120` to `15,30,60,90` minutes
after a live scan. CLV (mark mid vs fill price) is the fastest honest feedback
signal this system has — far faster than settlement variance. Only ~40 trades
had marks in the first 10 weeks; that starves `segment-report`.

### 3.8 Journal schema additions (backward compatible)

New per-trade fields: `p_model_cal` (calibrated pre-blend side prob) and
`fee_per_contract`. `p_model` remains the probability actually used for
sizing (now post-blend). Old rows simply lack the fields; readers fall back.

---

## 4. Probability plumbing, exactly

For each player/line/side at scan time:

1. `model.py` → λ and full TB PMF → `probability_engine` → raw `P(over)`.
2. `edge_detector._calibrate` → isotonic calibration (fill-based segmented
   first, OOF fallback) → `p_model_cal`. `P(under) = 1 − P(over)` for coherence.
3. `market_blend.blend_probability(p_model_cal, side_mid, w)` → `p_blend`
   (empty bid ⇒ shrink toward the ask instead of a fake mid).
4. `suggest_limit_price` (maker-aware) → limit; `fees.fee_per_contract` at that
   limit (taker iff it crosses) → `edge = p_blend − limit − fee`.
5. Gates: `min_p`, edge threshold (×`TAIL_EDGE_MULT` below `TAIL_P_CUTOFF`),
   spread ≤ 0.25, realistic ask, `MAX_YES_LINE`, `BLOCKED_SEGMENTS`, VPIN flow
   guard, positive net EV.
6. Sizing: fractional Kelly (`KELLY_FRACTION`, net-of-fee odds), risky-band
   haircut (0.6–0.9 × 0.4), `MAX_BET_PCT` / portfolio caps, then
   `risk_manager` daily limits and the kill switch.

---

## 5. Feedback loops (run in this order after fills settle)

| Command | What it learns | Cadence |
|---|---|---|
| `reconcile` | actual fills per order | nightly (cron) |
| `report` / `report-range` | P&L, fee drag, calibration gaps, segment slices | nightly / weekly |
| `calibrate` | segmented isotonic from fills (needs ≥50 rows) | weekly-ish |
| `fit-blend` | blend weight `w` from resolved fills | weekly-ish; after model changes |
| `model-vs-market` | model vs book log-loss/Brier on FULL snapshot slates (no trade-selection bias), sliced by line/disagreement/date with per-slice fitted `w` | after model changes; as snapshot data accumulates |
| `segment-report` | go/no-go per line/side segment (uses CLV) | weekly |

`model-vs-market` is the primary model-research scoreboard: `fit-blend` can only
say the model lost *where it used to trade*; this scores every snapshotted
market and shows where (if anywhere) the model earns blend weight. Snapshots
are captured **every 2h, 10:00–22:00 ET** (cron, since 2026-07-06 — previously
once/day at 10:00 ET), so slice quality improves daily. Default run uses the
saved model (look-ahead in the model's favor — a market win is decisive);
confirm any model win with `--pit-train`.

The honest scoreboard is **net-of-fee P&L and CLV**, not model-expected P&L.

---

## 6. Current status & the path back to positive expectancy

As of 2026-07-06 the system is live but effectively **paused by its own
statistics** (w=0.02 ⇒ almost no qualifying edges). That is by design.

First full-slate `model-vs-market` run (5,489 markets, 16 snapshot days,
saved model = look-ahead in the model's favor) confirmed it independently of
the fill-based fit: market log-loss 0.4644 vs model 0.5853, w_fit=0.01
overall, and the model loses *more* the more it disagrees with the book
(ΔLL +0.15 at ≥0.15 disagreement). Worst segment: line 4.5 (ΔLL +0.33 —
tail-line saturation; the 2026-07-06 logit clamp in `market_blend.py` guards
the blend against exactly that). Line 1.5 is the least bad (w_fit=0.10).
The apparent model "wins" at 6.5/7.5 are N≤35 markets with 5¢ floor asks —
longshot bias, untradable under `MIN_LIMIT_PRICE`.

To earn trading volume back:

1. Improve the model (features, pitcher/lineup information, market-timing) and
   re-run `fit-blend` — rising `w` = the model demonstrably adds information.
2. Watch `segment-report` + CLV for any segment where blended edges show
   positive CLV; consider unblocking/widening only those.
3. Keep `report-range` net-of-fee ROI as the single source of truth.
4. If 2–3 weeks of marks show maker orders never fill, revisit `MAKER_MODE`
   (the fee/spread saving is worthless at zero fill rate).

## 7. Knob reference (new/changed)

| Env var | Default | Meaning |
|---|---|---|
| `USE_MARKET_BLEND` | `true` | enable logit blend toward market mid |
| `BLEND_WEIGHT` | *(unset)* | manual override of fitted `w` (use with evidence) |
| `DEFAULT_BLEND_WEIGHT` | `0.35` | fallback when no fit/override exists |
| `MIN_BLEND_ROWS` | `200` | min resolved fills for `fit-blend` |
| `KALSHI_TAKER_FEE_RATE` | `0.07` | fee factor, `fee = rate·P·(1−P)` per contract |
| `KALSHI_MAKER_FEE_RATE` | `0.0` | fee factor for resting fills |
| `MAKER_MODE` | `true` | rest ≥1 tick inside the ask instead of crossing |
| `SCAN_WITHIN_HOURS` | `1.5` | only trade games starting within this window |
| `BLOCKED_SEGMENTS` | `1.5:no` | `line:side` pairs excluded from signals |
