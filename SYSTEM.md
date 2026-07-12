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

**Current state (2026-07-12): segmented, auto-refitting — see §3.9 and §8.**
The single global `w` described above is now only the fallback for buckets
without enough full-slate rows; per-bucket weights (§3.9) are what `scan`
actually uses. §8 documents a 2026-07-10 regression where this "prove it
first" principle was overridden by hand — read it before ever touching
`MIN_BLEND_WEIGHT` or `BLEND_WEIGHT` again. Do **not** "fix" low volume by
overriding either upward without new evidence.

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

### 3.9 Segmented blend weights + automatic refit — `market_blend.py`, `model_vs_market.py`, `refit-blend` (2026-07-12)

One global `w` hid that the model's performance vs. market is not uniform —
it degrades monotonically with |p_model − p_market| disagreement, and the
worst bucket is exactly the one the strategy trades (see §8). Fix:

- `market_blend.disagreement_bucket(p_model, p_market)` buckets into
  `<0.05`, `0.05-0.10`, `0.10-0.15`, `>=0.15`.
- `edge_detector.quote_side_edge` now calls
  `load_blend_weight(p_side_cal, mid)`, which looks up a per-bucket weight
  from `models/blend_meta_segments.json` (floored at `MIN_BLEND_WEIGHT`,
  falling back to the global `blend_meta.json` weight when a bucket has
  fewer than `MIN_BLEND_ROWS_SEGMENT` rows).
- Both files are fit from **full-slate scoring** (`model_vs_market.evaluate_day`
  over every snapshotted market, not just fills) — fills are edge-selected,
  so low-disagreement buckets would otherwise have near-zero coverage. This
  is the same reasoning as `model-vs-market` in §5, just applied per-bucket
  and actually persisted for `scan` to use, instead of only reported.
- `refit-blend` (new CLI command) runs this full-slate fit over a trailing
  window (`BLEND_REFIT_LOOKBACK_DAYS`, default 30d, lagged
  `BLEND_REFIT_LAG_DAYS`=2d so boxscores have settled) and overwrites both
  meta files — the **automatic-adjustment mechanism**: if the model starts
  beating the market in some bucket, that bucket's `w` rises on the next
  refit with no manual step; if not, it stays low. Scheduled weekly, Sundays
  05:00 (system-local, not ET — this job isn't slate-sensitive) via
  `scripts/cron_job.sh refit-blend`.
- `fit-blend-segments` is the same fit as a one-off manual command (explicit
  `--start`/`--end`, no lookback default) — useful for re-checking a specific
  window without waiting for the weekly job.

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
| `fit-blend` | global blend weight `w` from resolved fills (selection-biased — see §3.9) | ad hoc diagnostic only; not in cron |
| `refit-blend` | global + per-disagreement-bucket `w` from **full-slate** scoring, trailing window; writes both `blend_meta.json` and `blend_meta_segments.json` | **weekly, automatic (cron, Sun 05:00)** |
| `fit-blend-segments` | same per-bucket fit as `refit-blend`, explicit date range, one-off | ad hoc, e.g. re-checking a specific window |
| `model-vs-market` | model vs book log-loss/Brier on FULL snapshot slates (no trade-selection bias), sliced by line/disagreement/date with per-slice fitted `w` — read-only report, doesn't write meta files | after model changes; as snapshot data accumulates |
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

As of 2026-07-06 the system went live effectively **paused by its own
statistics** (w=0.02 ⇒ almost no qualifying edges). That was by design — see
§8 for what happened when that design was overridden four days later, and
the current (2026-07-12) state after the fix.

First full-slate `model-vs-market` run (5,489 markets, 16 snapshot days,
saved model = look-ahead in the model's favor) confirmed it independently of
the fill-based fit: market log-loss 0.4644 vs model 0.5853, w_fit=0.01
overall, and the model loses *more* the more it disagrees with the book
(ΔLL +0.15 at ≥0.15 disagreement). Worst segment: line 4.5 (ΔLL +0.33 —
tail-line saturation; the 2026-07-06 logit clamp in `market_blend.py` guards
the blend against exactly that). Line 1.5 is the least bad (w_fit=0.10).
The apparent model "wins" at 6.5/7.5 are N≤35 markets with 5¢ floor asks —
longshot bias, untradable under `MIN_LIMIT_PRICE`. (A larger re-run,
2026-04-27→07-06, N=5,794, reproduced this almost exactly — see §8.)

To earn trading volume back:

1. Improve the model (features, pitcher/lineup information, market-timing).
   `refit-blend` now runs weekly and automatically — a real improvement
   shows up as a rising per-bucket `w` with no manual step required. Don't
   force `w` up by hand to "test" the model; §8 is what that costs.
2. Watch `segment-report` + CLV for any segment where blended edges show
   positive CLV; consider unblocking/widening only those.
3. Keep `report-range` net-of-fee ROI as the single source of truth.
4. If 2–3 weeks of marks show maker orders never fill, revisit `MAKER_MODE`
   (the fee/spread saving is worthless at zero fill rate).

## 7. Knob reference (new/changed)

| Env var | Default | Meaning |
|---|---|---|
| `USE_MARKET_BLEND` | `true` | enable logit blend toward market mid |
| `BLEND_WEIGHT` | *(unset)* | manual override of fitted `w`, global AND all segments (use with evidence — see §8) |
| `DEFAULT_BLEND_WEIGHT` | `0.35` | fallback when no fit/override exists |
| `MIN_BLEND_WEIGHT` | `0.05` | floor under the fitted global `w` (was `0.3` 2026-07-10→12; see §8 for why that was wrong) |
| `MIN_BLEND_ROWS` | `200` | min resolved fills for `fit-blend` (fills-based, diagnostic only) |
| `MIN_BLEND_ROWS_SEGMENT` | `100` | min full-slate rows per disagreement bucket for `refit-blend`/`fit-blend-segments` to trust that segment |
| `BLEND_REFIT_LOOKBACK_DAYS` | `30` | trailing window `refit-blend` scores each run |
| `BLEND_REFIT_LAG_DAYS` | `2` | days excluded from the end of that window (boxscore/ETL settling) |
| `KALSHI_TAKER_FEE_RATE` | `0.07` | fee factor, `fee = rate·P·(1−P)` per contract |
| `KALSHI_MAKER_FEE_RATE` | `0.0` | fee factor for resting fills |
| `MAKER_MODE` | `true` | rest ≥1 tick inside the ask instead of crossing |
| `SCAN_WITHIN_HOURS` | `1.5` | only trade games starting within this window |
| `BLOCKED_SEGMENTS` | `1.5:no` | `line:side` pairs excluded from signals |

---

## 8. Incident: 2026-07-10 floor regression → 2026-07-12 fix

**What happened.** §2's fit landed at w=0.02 on 2026-07-09 (615 fills,
04-27→07-06). Commit `0a42792` (2026-07-10 09:56) added `MIN_BLEND_WEIGHT`
and floored the loaded weight at **0.3** — a 15x increase — reasoning that a
615-row grid search could land on "noisy near-zero w." That floor directly
overrode the §3.1 design intent ("do not fix low volume by overriding
upward without new evidence") using the exact mechanism it warned against.

Trading volume makes the causal chain visible: scans placed **zero orders**
2026-07-07 through 07-09 (w≈0.02 ⇒ blended probabilities collapse onto the
market price ⇒ nothing clears the edge threshold — a de facto circuit
breaker). The moment the floor commit landed, volume snapped back to 33
orders (07-10) and 56 (07-11).

**Why the floor was wrong, not just aggressive.** The commit's own
diagnostics contradicted its reasoning: at w=0 (pure market) log-loss was
0.6219; at w=1 (pure model) it was 0.6792 — the model was *worse* than the
market alone, not merely a noisy near-tie. 615 rows is also 3x the
codebase's own `MIN_BLEND_ROWS=200` significance bar — the commit message's
"small sample" framing wasn't consistent with the threshold already defined
for that judgment.

**Verification, without staking capital.** Fills are selection-biased (only
scored where the model already disagreed with the market enough to trade),
so before touching `MIN_BLEND_WEIGHT`, `model-vs-market` was re-run
full-slate (no trade filter) over the identical window, N=5,794 — 10x the
fill sample. It reproduced the fill-based conclusion and sharpened it: the
model's disadvantage vs. market **grows monotonically with disagreement**,
and is worst by a wide margin in the `>=0.15` bucket (ΔLL +0.10) — the only
bucket the strategy actually trades. Every date in the window individually
showed market beating model; this was never a "last two days" problem, the
floor just re-armed a structural loser that had been losing the whole time.
Realized/MTM P&L over 07-10/07-11 was small either way (~flat to -$2 on
~$76 deployed) — the danger was structural (bad EV going forward), not an
observed blowup, which is why the fix mattered even though the 2-day dollar
damage was minor.

**Fix (§3.9): segment instead of floor, and make it self-correcting.**
Rather than one global floor, weight is now fit **per disagreement bucket**
from full-slate scoring, floored much lower (`MIN_BLEND_WEIGHT=0.05`, chosen
because the data supports ~0.05 global, not because it's an arbitrarily
smaller number), and re-fit automatically every week (`refit-blend`) so a
future model improvement raises its own weight without anyone touching the
floor by hand. First live fit (2026-04-27→07-06, N=5,794): global w=0.05,
`<0.05`/`0.05-0.10` buckets floored to 0.05 (fit was 0), `0.10-0.15`
w=0.08, `>=0.15` w=0.11 — the traded bucket ended up at roughly a third the
influence the 0.3 floor gave it, and for a reason traceable to data instead
of a round number.

**Takeaway for future changes to this knob:** a fitted weight sitting near
the floor is not automatically a bug to "fix" by raising the floor — check
whether raw fit diagnostics (`logloss_model_only` vs `logloss_market_only`)
say the model is actually competitive first. If they don't, the floor is
working as designed and the volume drop is the signal, not the problem.
