# Pitcher/lineup feature audit and strengthening — design

Date: 2026-07-07
Status: approved

## Context

`SYSTEM.md` documents that the model currently carries almost no information
beyond the Kalshi market price (`fit-blend` w=0.02; full-slate
`model-vs-market` w_fit=0.01, model log-loss 0.5853 vs market 0.4644). Per
`SYSTEM.md` §6, the path back to trading volume starts with "improve the
model (features, pitcher/lineup information...) and re-run fit-blend."

This spec covers an audit of the pitcher/lineup feature block already live in
`MODEL_FEATURES` (`opp_sp_era_roll`, `opp_sp_k9_roll`, `opp_sp_hr9_roll`,
`opp_sp_bb9_roll`, `opp_sp_hand_L`, `platoon_tb_adj`, `lineup_slot_norm`,
`expected_pa_proxy`, `is_home`, `opp_tb_allowed_roll`) and fixes for concrete,
evidence-backed defects found in it. Statcast pitcher/batter xStats
(`opp_sp_xwoba_roll` etc.) are explicitly out of scope — those tables don't
exist yet (`etl-statcast` has never been run) and populating them is a
separate, much larger effort.

## Findings

Extracted from the currently trained ordinal model's coefficients
(`models/tb_model.pkl`, statsmodels `OrderedModel`) and a correlation pass
over `build_feature_table()` (4.2M rows):

1. **`lineup_slot_norm` vs `expected_pa_proxy`: r = -0.988.**
   `expected_pa_proxy` is a hardcoded affine function of the same batting
   slot (`4.5 - 0.28*(slot-5)`, clipped) that `lineup_slot_norm` already
   encodes (`slot/9`). It is a near-exact duplicate, not new information.
   Its coefficient is small, wrong-signed, and not significant (p=0.61) —
   consistent with a redundant, unstable column rather than a real null
   effect.

2. **`platoon_tb_adj` vs `tb_roll`: r = 0.999.** It is computed as
   `tb_roll * boost` where `boost ∈ {1.0, 1.06, 1.08}` — i.e. a scaled copy
   of a feature already in the model. The platoon-advantage hypothesis is
   real in baseball, but it's statistically invisible here (coef p=0.31)
   because almost all of its variance is explained by `tb_roll` itself.

3. **Pitcher-hand resolution fails ~47% of the time.** Both the live matchup
   index (`matchup_features._pitcher_hand_from_name`, feeding
   `opp_sp_hand_L` and `platoon_tb_adj`) and the ETL path that stamps
   `opp_sp_hand_L` onto `batter_games`
   (`data_engine._opp_sp_hand_L_by_game_id`) resolve a pitcher's throwing
   hand by searching `statsapi.lookup_player(name)` — which queries the
   *current-season active-roster snapshot* and silently returns no hits for
   anyone not rostered at call time (injured, traded, optioned, etc.).
   Measured directly: sampling 60 real starting pitchers (known id+name from
   `pitcher_games`) and running the actual lookup code, **28/60 (47%)**
   returned zero hits — including Clayton Kershaw, Charlie Morton, Noah
   Syndergaard. Every failed lookup falls back to a hardcoded `'R'`. Because
   ~75% of MLB starters are right-handed, this defaulting masks most of the
   damage, but it still produces an estimated ~13% mislabel rate on
   `opp_sp_hand_L` (0.47 unresolved × ~0.28 true-lefty rate). The existing
   ID-based `person`-lookup pattern (already used for `opp_sp_era_roll` etc.
   via `get_probable_starters`) resolved all 60/60 correctly in the same
   test.

4. **Not fixing, flagged only:** `opp_sp_era_roll` / `opp_sp_hr9_roll` /
   `opp_sp_bb9_roll` are correlated with each other (r=0.37–0.61) — expected
   collinearity among a "pitcher quality" block built from the same 5-start
   rolling window. This is normal, not a bug, and redesigning it (e.g. wider
   window, composite index) is speculative without evidence it hurts
   predictive log-loss. Out of scope for this pass.

## Fix

1. **Drop `expected_pa_proxy`** from `MODEL_FEATURES` (`feature_store.py`).
   Pure duplicate of `lineup_slot_norm`; remove rather than recompute, since
   any slot-derived formula will remain collinear with `lineup_slot_norm`.

2. **Redesign `platoon_tb_adj` → `platoon_edge`.** New value is the boost
   delta itself (`boost - 1.0` ∈ {0, 0.06, 0.08}) instead of
   `tb_roll * boost`, decorrelating it from `tb_roll` so the model can
   estimate an independent platoon-advantage coefficient. Touches
   `matchup_features.attach_platoon_features`,
   `matchup_features.live_matchup_overrides`,
   `matchup_features.finalize_matchup_columns`, and the name in
   `MATCHUP_FEATURE_NAMES`.

3. **ID-based pitcher-hand resolution**, replacing
   `_pitcher_hand_from_name`(name-search) with a cached `person`-by-ID
   lookup wherever a `pitcher_id` is available:
   - Live path: `matchup_features.build_slate_matchup_index` /
     `_game_matchup_row` currently derive hand from
     `away_probable_pitcher`/`home_probable_pitcher` name strings pulled
     from `schedule_games_by_date`. Switch to the hydrated schedule call
     already used by `data_engine.get_probable_starters` (which carries
     `probablePitcher.id`), and resolve hand by ID.
   - ETL path: `data_engine._opp_sp_hand_L_by_game_id` gets the same
     treatment for live/near-term ETL runs.
   - Historical backfill: for existing `batter_games` rows, don't
     re-query the schedule API per historical date. Instead join
     `batter_games` to `pitcher_games` on `(game_id, opponent_team=team)`
     filtered to `is_starter=1` to recover each historical row's starting
     `pitcher_id` (already stored), then resolve hand via the same
     ID-cached lookup (a few hundred to ~2,000 distinct pitcher_ids across
     history, cached, not one call per row). Overwrite `opp_sp_hand_L` on
     `batter_games` in place.

4. **Re-materialize + retrain.** Re-run `materialize_feature_table()` (gold
   feature table reads from `batter_games`, needs the corrected column),
   then `python3 run_pipeline.py train`.

5. **Judge with the scoreboard.** Re-run the coefficient audit
   (verify `expected_pa_proxy`/old `platoon_tb_adj` collinearity is gone,
   check `platoon_edge` and `opp_sp_hand_L` significance) and re-run
   `model-vs-market` to compare log-loss/Brier/`w_fit` before vs. after,
   specifically on the `disagreement` and `line` slices where pitcher/lineup
   signal should matter most. Net-of-fee `report-range` is not affected by
   this change (no live trading volume yet at w=0.02) — the scoreboard here
   is `model-vs-market`, per `SYSTEM.md` §5.

## Non-goals

- No Statcast data pull (`etl-statcast`) or enabling `USE_STATCAST_FEATURES`.
- No changes to `opp_sp_era_roll`/`hr9_roll`/`bb9_roll` (flagged, not fixed).
- No changes to `BLEND_WEIGHT`, gates, sizing, or fee logic.
- No changes to the widened rolling window for pitcher stats.

## Testing

- Unit-level: a small script validating the ID-based hand lookup against the
  same 60-pitcher sample used in the audit (expect 60/60 correct, vs. the
  measured 32/60 correct for the name-based path before the fix).
- Correlation check: re-run the `build_feature_table()` correlation pass
  post-fix to confirm `lineup_slot_norm`/`expected_pa_proxy` collinearity is
  resolved (feature removed) and `platoon_edge`/`tb_roll` correlation drops
  well below the prior 0.999.
- Coefficient check: re-fit and inspect `opp_sp_hand_L` and `platoon_edge`
  coefficients/p-values against the pre-fix baseline captured in this doc.
- Scoreboard check: `model-vs-market` before/after comparison (overall +
  sliced), per `SYSTEM.md`'s stated primary research scoreboard.
