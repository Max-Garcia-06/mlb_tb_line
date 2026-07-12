"""
Typer command implementations for the MLB total-bases pipeline.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime

import typer
from rich import box
from rich.table import Table

from cli.output import console, _header, _success, _warn

log = logging.getLogger(__name__)


def _email_scan_summary(game_date, signals, total_xp, results=None) -> None:
    from notify import send_email

    mode = "LIVE" if results is not None else "DRY RUN"
    subj = f"[MLB TB] {game_date} — {len(signals)} edge(s) ({mode})"

    lines = [f"Date: {game_date}  Mode: {mode}", f"Sum E[PnL]: ${total_xp:+.2f}\n", "=== Signals ==="]
    for s in signals:
        lines.append(
            f"{s.player_name:<24} line={s.kalshi_line} {s.recommended_side.upper():<3} "
            f"edge={s.edge:+.3f} contracts={s.recommended_contracts} ${s.bet_dollars:.2f}"
        )

    if results is not None:
        lines.append("\n=== Order Results ===")
        for r in results:
            status = "OK" if r.success else "FAILED"
            lines.append(f"{status:<6} {r.ticker} {r.side} x{r.contracts} @ {r.price:.2f} {r.message}")

    send_email(subj, "\n".join(lines))


def _event_ticker_from_ml(ml0) -> str:
    et = str(getattr(ml0, "event_ticker", "") or "")
    if not et and getattr(ml0, "ticker", ""):
        parts = str(ml0.ticker).split("-")
        if len(parts) >= 2:
            et = f"{parts[0]}-{parts[1]}"
    return et


def etl(seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons (years), e.g. '2023,2024'")):
    _header("Phase 1 — ETL: Pull MLB Batter Games")
    import data_engine as de
    from config import SEASONS, DB_PATH

    season_list = [int(s.strip()) for s in seasons.split(",")] if seasons else SEASONS
    console.print(f"Seasons  : {season_list}")
    console.print(f"Database : {DB_PATH}\n")
    de.build_historical_store(season_list)
    _success("ETL complete.")


def etl_statcast(
    seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons (years), e.g. '2024,2025'"),
    incremental: bool = typer.Option(False, "--incremental", help="Only pull dates after the latest stored game_date."),
    chunk_days: int = typer.Option(None, "--chunk-days", help="Date-range chunk size for Savant pulls."),
):
    _header("Phase 1b — ETL: Statcast Quality of Contact")
    import statcast_engine as se
    from config import SEASONS, STATCAST_CHUNK_DAYS

    season_list = [int(s.strip()) for s in seasons.split(",")] if seasons else SEASONS
    chunk = int(chunk_days) if chunk_days else STATCAST_CHUNK_DAYS
    console.print(f"Seasons    : {season_list}")
    console.print(f"Incremental: {incremental}")
    console.print(f"Chunk days : {chunk}\n")
    counts = se.build_statcast_store(season_list, incremental=incremental, chunk_days=chunk)
    _success(f"Statcast ETL complete: {counts['batter_rows']:,} batter-games, {counts['pitcher_rows']:,} pitcher-games.")


def train():
    _header("Phase 2 — Model Training")
    import model as m

    mdl, meta = m.train(save=True)
    _success(f"Model saved. Train rows: {meta['train_rows']:,}")
    console.print(f"  Residual σ  : {meta['residual_std']:.3f}")
    console.print(f"  Residual var: {meta['residual_var']:.3f}\n")

    fi = m.get_feature_importance(mdl, meta=meta)
    t = Table(title="Feature Importance (gain)", box=box.SIMPLE)
    t.add_column("Feature", style="cyan")
    t.add_column("Importance", justify="right")
    for _, row in fi.head(12).iterrows():
        t.add_row(row["feature"], f"{row['importance']:.1f}")
    console.print(t)


def evaluate(splits: int = typer.Option(5, "--splits", help="Walk-forward CV folds")):
    _header("Phase 3 — Walk-Forward Evaluation")
    import model as m

    X, y, _ = m.prepare_data()
    res = m.walk_forward_cv(X, y, n_splits=splits)
    console.print(f"Mean MAE: {res['mean_mae']:.3f} TB")
    console.print(f"Std  MAE: {res['std_mae']:.3f} TB")
    if "oof_residual_var" in res:
        console.print(f"OOF residual var: {res['oof_residual_var']:.3f}")
    for k in sorted([k for k in res.keys() if k.startswith("brier@")]):
        console.print(f"{k:>12}: {res[k]:.4f}")
    for k in sorted([k for k in res.keys() if k.startswith("logloss@")]):
        console.print(f"{k:>12}: {res[k]:.4f}")


def tune(
    trials: int = typer.Option(25, "--trials", help="Number of hyperparameter trials to run."),
    splits: int = typer.Option(4, "--splits", help="Walk-forward CV folds per trial (tradeoff: speed vs reliability)."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
):
    _header("Phase 2b — Hyperparameter Tuning (walk-forward CV)")
    import model as m

    best_params, best_score = m.tune_hyperparameters(trials=trials, n_splits=splits, random_seed=seed)
    _success("Saved best params to models/best_params.json (auto-used by training & evaluation).")
    console.print("\nBest params:")
    for k in sorted(best_params.keys()):
        console.print(f"  {k:>22} = {best_params[k]}")
    console.print("\nBest CV score:")
    for k in ["mean_logloss", "mean_mae", "oof_residual_var"]:
        if k in best_score:
            console.print(f"  {k:>22}: {best_score[k]:.6f}" if isinstance(best_score[k], float) else f"  {k:>22}: {best_score[k]}")
    for k in sorted([k for k in best_score.keys() if k.startswith("brier@")]):
        console.print(f"  {k:>22}: {best_score[k]:.6f}")
    for k in sorted([k for k in best_score.keys() if k.startswith("logloss@")]):
        console.print(f"  {k:>22}: {best_score[k]:.6f}")


def scan(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    bankroll: float = typer.Option(1000.0, "--bankroll"),
    threshold: float = typer.Option(None, "--threshold"),
    min_p: float = typer.Option(None, "--min-p"),
    tail_p_cutoff: float = typer.Option(None, "--tail-p-cutoff"),
    tail_edge_mult: float = typer.Option(None, "--tail-edge-mult"),
    max_signals: int = typer.Option(None, "--max-signals"),
    one_per_player: bool = typer.Option(True, "--one-per-player/--no-one-per-player"),
    max_contracts: int = typer.Option(250, "--max-contracts"),
    series_ticker: str = typer.Option("KXMLBTB", "--series-ticker", help="Kalshi series ticker for MLB total bases markets."),
    dry_run: bool = typer.Option(True, "--dry-run/--live"),
    auto_mark: bool = typer.Option(True, "--auto-mark/--no-auto-mark", help="Automatically record CLV marks after scan (live mode only)."),
    mark_delays: str = typer.Option("15,30,60,90", "--mark-delays", help="Comma-separated minutes after scan to record marks, e.g. '15,30,60,90'."),
):
    from config import (
        BASE_DIR,
        EDGE_THRESHOLD,
        GAMES_FOR_LAMBDA_SANITY,
        LAMBDA_SANITY_MAX,
        MAKER_MODE,
        MAX_YES_LINE,
        MIN_LIMIT_PRICE,
        MIN_P,
        SCAN_WITHIN_HOURS,
        TAIL_P_CUTOFF,
        TAIL_EDGE_MULT,
    )
    from model import load_model, predict_lambda, predict_tb_pmf_row
    from feature_store import MODEL_FEATURES, build_feature_table, load_distinct_training_player_ids
    from probability_engine import calculate_probabilities
    from edge_detector import (
        MAX_BID_ASK_SPREAD,
        apply_flow_guard,
        dollars_to_contracts,
        execute_signals,
        expected_pnl_usd,
        expected_pnl_usd_std,
        is_blocked_segment,
        portfolio_expected_pnl_std,
        fill_calibrated_probabilities,
        quote_side_edge,
        scan_for_edges,
    )
    from market_blend import load_blend_weight
    from kalshi_bridge import MockKalshiClient, attach_vpin_proxy_batch, get_client
    from capital_optimizer import resize_signals_portfolio
    from identity_bridge import norm_player_name, resolve_mlb_player_id

    _header("Phase 4 — Edge Scan")
    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    edge_thr = threshold or EDGE_THRESHOLD
    min_p_eff = MIN_P if min_p is None else float(min_p)
    tail_p_eff = TAIL_P_CUTOFF if tail_p_cutoff is None else float(tail_p_cutoff)
    tail_mult_eff = TAIL_EDGE_MULT if tail_edge_mult is None else float(tail_edge_mult)

    console.print(f"Date      : {game_date}")
    console.print(f"Bankroll  : ${bankroll:,.2f}")
    console.print(f"Threshold : {edge_thr}")
    console.print(f"Min P     : {min_p_eff:.3f}  (tail<{tail_p_eff:.3f} ⇒ edge×{tail_mult_eff:.2f})")
    console.print(f"Orders    : {'[yellow]DRY RUN[/yellow]' if dry_run else '[red]LIVE[/red]'}\n")

    trained_model, meta = load_model()
    variance = meta["residual_var"]
    feat_names = list(meta.get("feature_names") or MODEL_FEATURES)

    client = get_client()
    if isinstance(client, MockKalshiClient):
        _warn(
            "Kalshi MOCK client is active (missing or empty KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH in "
            f"{BASE_DIR}/.env). You only see 3 placeholder markets — not the real exchange."
        )
    else:
        from risk_manager import resolve_bankroll
        bankroll = resolve_bankroll(bankroll, client)
        console.print(f"[dim]Live balance: ${bankroll:,.2f}[/dim]")
    market_lines = client.get_total_bases_lines(game_date, series_ticker=series_ticker)
    if not market_lines:
        _warn(f"No total bases markets found for date {game_date} (series={series_ticker}).")
        raise typer.Exit(0)

    from data_engine import filter_market_lines_pregame

    market_lines, excluded_games = filter_market_lines_pregame(market_lines, game_date)
    if excluded_games:
        for et, slug, status in excluded_games:
            console.print(f"[dim]Skipping started game {slug} ({status}) — {et}[/dim]")
        console.print(
            f"[dim]Excluded {len(excluded_games)} in-progress/final game(s); "
            f"{len(market_lines)} market(s) remain.[/dim]\n"
        )
    if not market_lines:
        _warn("No pre-game total bases markets left after excluding started games.")
        raise typer.Exit(0)

    if SCAN_WITHIN_HOURS > 0:
        from data_engine import filter_market_lines_by_start_window

        market_lines, excluded_window = filter_market_lines_by_start_window(
            market_lines, game_date, within_hours=SCAN_WITHIN_HOURS
        )
        if excluded_window:
            console.print(
                f"[dim]Entry window: excluded {len(excluded_window)} game(s) not starting within "
                f"{SCAN_WITHIN_HOURS:g}h; {len(market_lines)} market(s) remain.[/dim]\n"
            )
        if not market_lines:
            _warn(f"No markets starting within {SCAN_WITHIN_HOURS:g}h (SCAN_WITHIN_HOURS).")
            raise typer.Exit(0)

    console.print(f"Found {len(market_lines)} total bases markets on Kalshi (pre-game only)\n")

    attach_vpin_proxy_batch(client, market_lines, max_workers=8)

    training_player_ids = load_distinct_training_player_ids()

    # One model output per (player, game_date): same λ/PMF for all strikes; best xref across tickers.
    by_slate = defaultdict(list)
    for ml in market_lines:
        by_slate[(norm_player_name(ml.player_name), str(ml.game_date))].append(ml)

    slate_pid: dict[tuple, int] = {}
    for key, mls in by_slate.items():
        xref = next((str(m.xref_player_id).strip() for m in mls if str(getattr(m, "xref_player_id", "") or "").strip()), None)
        ml0 = mls[0]
        pid = resolve_mlb_player_id(
            player_name=ml0.player_name,
            xref_player_id=xref,
            feat_df=None,
            allowed_player_ids=training_player_ids or None,
        )
        slate_pid[key] = int(pid or ml0.player_id or 0)

    need_ids = frozenset(p for p in slate_pid.values() if p > 0)
    feat_df = None
    if need_ids:
        try:
            feat_df = build_feature_table(player_ids=need_ids)
            if feat_df is not None and feat_df.empty:
                feat_df = None
        except Exception as e:
            _warn(f"Could not load feature table for slate players ({e}). Using λ fallback.")
            feat_df = None

    if feat_df is not None and not feat_df.empty and "player_id" in feat_df.columns:
        for key, mls in by_slate.items():
            xref = next(
                (str(m.xref_player_id).strip() for m in mls if str(getattr(m, "xref_player_id", "") or "").strip()),
                None,
            )
            ml0 = mls[0]
            pid2 = resolve_mlb_player_id(
                player_name=ml0.player_name,
                xref_player_id=xref,
                feat_df=feat_df,
                allowed_player_ids=training_player_ids or None,
            )
            if int(pid2 or 0) > 0:
                slate_pid[key] = int(pid2)

    from matchup_features import (
        apply_live_feature_overrides,
        build_opp_tb_allowed_lookup,
        build_slate_matchup_index,
        slate_teams,
    )

    try:
        matchup_slate = build_slate_matchup_index(game_date)
        opp_tb_lookup = build_opp_tb_allowed_lookup(game_date, slate_teams(matchup_slate))
    except Exception as e:
        _warn(f"Could not build tonight's matchup slate ({e}); falling back to stale row matchup features.")
        matchup_slate, opp_tb_lookup = {}, {}

    try:
        from data_engine import get_confirmed_lineups

        confirmed_lineups = get_confirmed_lineups(game_date)
    except Exception as e:
        log.debug("Confirmed lineups unavailable: %s", e)
        confirmed_lineups = {}

    try:
        from feature_store import build_live_pitcher_features

        live_sp_by_team = build_live_pitcher_features(game_date)
    except Exception as e:
        log.debug("Live probable-starter features unavailable: %s", e)
        live_sp_by_team = {}

    slate_payload: dict[tuple, dict] = {}
    for key, mls in by_slate.items():
        xref = next((str(m.xref_player_id).strip() for m in mls if str(getattr(m, "xref_player_id", "") or "").strip()), None)
        ml0 = mls[0]
        pid = slate_pid[key]
        lam = float(statistics.median([m.line * m.implied_prob / 0.5 for m in mls]))
        pmf = None
        player_id_out = int(pid or ml0.player_id or 0)
        if feat_df is not None:
            if pid and "player_id" in feat_df.columns:
                player_rows = feat_df[feat_df["player_id"] == int(pid)]
            else:
                player_rows = feat_df[feat_df["player_name"].str.lower() == ml0.player_name.lower()]
            if not player_rows.empty:
                latest = player_rows.sort_values("game_date").iloc[-1]
                row_features = latest[feat_names].fillna(0).to_dict()
                row_features = apply_live_feature_overrides(
                    row_features,
                    game_date=game_date,
                    player_id=int(latest.get("player_id", 0) or 0),
                    player_team=str(latest.get("team", "") or ""),
                    bats_hand=str(latest.get("bats_hand", "R") or "R"),
                    tb_roll=float(latest.get("tb_roll", 0) or 0),
                    event_ticker=_event_ticker_from_ml(ml0),
                    matchup_slate=matchup_slate,
                    opp_tb_lookup=opp_tb_lookup,
                    confirmed_lineups=confirmed_lineups,
                    live_sp_by_team=live_sp_by_team,
                )
                lam_raw = float(predict_lambda(row_features, trained_model, feature_names=feat_names, meta=meta))
                pmf_arr = predict_tb_pmf_row(row_features, trained_model, feature_names=feat_names, meta=meta)
                gplayed = int(latest.get("games_played", 999) or 999)
                if lam_raw > LAMBDA_SANITY_MAX and gplayed < GAMES_FOR_LAMBDA_SANITY:
                    mid_implied = float(statistics.median([m.implied_prob for m in mls]))
                    med_line = float(statistics.median([float(m.line) for m in mls]))
                    lam = med_line * mid_implied / 0.5
                    pmf = None
                    log.warning(
                        "λ sanity fallback for %s (E[TB]=%.2f, games_played=%s)",
                        ml0.player_name,
                        lam_raw,
                        gplayed,
                    )
                else:
                    lam = lam_raw
                    pmf = pmf_arr
                player_id_out = int(pid or latest.get("player_id", 0) or ml0.player_id or 0)

        slate_payload[key] = {
            "player_id": player_id_out,
            "lam": lam,
            "pmf": pmf.tolist() if pmf is not None else None,
        }

    predictions = []
    for ml in market_lines:
        key = (norm_player_name(ml.player_name), str(ml.game_date))
        base = slate_payload[key]
        pred_row = {
            "player_id": int(base["player_id"]),
            "player_name": ml.player_name,
            "game_date": ml.game_date,
            "kalshi_line": ml.line,
            "predicted_lambda": float(base["lam"]),
        }
        if base.get("pmf") is not None:
            pred_row["tb_pmf"] = base["pmf"]
        predictions.append(pred_row)

    prob_results = calculate_probabilities(predictions, variance)
    fill_calibrated_probabilities(prob_results)

    def _edge_thr_for_p(p: float) -> float:
        return float(edge_thr) * (float(tail_mult_eff) if float(p) < float(tail_p_eff) else 1.0)

    def _fmt_edge_cell(
        e: float,
        p_side: float,
        ask: float,
        spread: float,
        *,
        side: str,
        kalshi_line: float,
    ) -> str:
        """Color fee-adjusted blended edge using the same pre-EV gates as detect_edge (not EV)."""
        need = _edge_thr_for_p(p_side)
        spread_ok = spread <= MAX_BID_ASK_SPREAD
        ask_ok = ask >= MIN_LIMIT_PRICE
        line_ok = float(kalshi_line) <= float(MAX_YES_LINE) if side == "yes" else True
        seg_ok = not is_blocked_segment(kalshi_line, side)
        min_ok = p_side >= min_p_eff
        edge_ok = e > need and spread_ok and ask_ok and line_ok and seg_ok
        color = "green" if edge_ok and min_ok else ("yellow" if e > 0 else "red")
        return f"[{color}]{e:+.3f}[/{color}]"

    console.print(
        f"[dim]Blend w={load_blend_weight():.2f} (p = w·model + (1−w)·market mid, logit space) · "
        f"maker mode {'ON' if MAKER_MODE else 'OFF'} · edges net of Kalshi fee at limit.[/dim]"
    )
    console.print(
        "[dim]Legend: Pr/Pc/Pb = P(over) raw / calibrated / blended-with-market; "
        f"eY/eN = blended edge net of fee vs suggested limit (same as signals; max spread {MAX_BID_ASK_SPREAD:.2f}). "
        "Green eY/eN = passes min_p + edge vs threshold + spread + ask + segment gates, not EV.[/dim]\n"
    )
    t = Table(title=f"TB vs Kalshi — {game_date}", box=box.SIMPLE_HEAD)
    t.add_column("Player", style="cyan", min_width=14, overflow="ellipsis")
    t.add_column("Ln", justify="right", min_width=3)
    t.add_column("lam", justify="right", min_width=4)
    t.add_column("Pr", justify="right", min_width=5)
    t.add_column("Pc", justify="right", min_width=5)
    t.add_column("Pb", justify="right", min_width=5)
    t.add_column("Y\nmid", justify="right", min_width=5)
    t.add_column("Y\nask", justify="right", min_width=5)
    t.add_column("eY", justify="right", min_width=6)
    t.add_column("Ysp", justify="right", min_width=4)
    t.add_column("N\nask", justify="right", min_width=5)
    t.add_column("eN", justify="right", min_width=6)
    t.add_column("Nsp", justify="right", min_width=4)

    for pr, ml in zip(prob_results, market_lines):
        p_or = float(pr.p_over)
        p_oc = float(pr.p_over_calibrated) if pr.p_over_calibrated is not None else p_or
        p_uc = float(pr.p_under_calibrated) if pr.p_under_calibrated is not None else (1.0 - p_or)
        p_ob, _, _, yes_edge = quote_side_edge(p_oc, bid=float(ml.yes_bid), ask=float(ml.yes_ask), side="yes")
        p_ub, _, _, no_edge = quote_side_edge(p_uc, bid=float(ml.no_bid), ask=float(ml.no_ask), side="no")
        eY_cell = _fmt_edge_cell(
            yes_edge, p_ob, float(ml.yes_ask), float(ml.yes_spread), side="yes", kalshi_line=float(ml.line)
        )
        eN_cell = _fmt_edge_cell(
            no_edge, p_ub, float(ml.no_ask), float(ml.no_spread), side="no", kalshi_line=float(ml.line)
        )
        t.add_row(
            pr.player_name,
            f"{ml.line:g}",
            f"{pr.predicted_lambda:.2f}",
            f"{p_or:.3f}",
            f"{p_oc:.3f}",
            f"{p_ob:.3f}",
            f"{ml.implied_prob:.3f}",
            f"{ml.yes_ask:.2f}",
            eY_cell,
            f"{ml.yes_spread:.2f}",
            f"{ml.no_ask:.2f}",
            eN_cell,
            f"{ml.no_spread:.2f}",
        )
    console.print(t)

    signals = scan_for_edges(
        prob_results,
        market_lines,
        bankroll,
        edge_threshold=edge_thr,
        min_p=min_p_eff,
        tail_p_cutoff=tail_p_eff,
        tail_edge_mult=tail_mult_eff,
    )
    signals = apply_flow_guard(signals, market_lines)

    if not signals:
        _warn("No edges found.")
        raise typer.Exit(0)

    if one_per_player:
        seen = {}
        for s in signals:
            if s.player_name not in seen or s.ev > seen[s.player_name].ev:
                seen[s.player_name] = s
        signals = sorted(seen.values(), key=lambda s: s.ev, reverse=True)

    if max_signals and len(signals) > max_signals:
        signals = signals[:max_signals]

    resize_signals_portfolio(signals, bankroll)

    for s in signals:
        if max_contracts:
            s.recommended_contracts = min(s.recommended_contracts, max_contracts)
        s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    total_raw = sum(s.bet_dollars for s in signals)
    if total_raw > bankroll:
        scale = bankroll / total_raw
        for s in signals:
            s.bet_dollars = round(s.bet_dollars * scale, 2)
            s.recommended_contracts = dollars_to_contracts(s.bet_dollars, s.limit_price)
            if max_contracts:
                s.recommended_contracts = min(s.recommended_contracts, max_contracts)
            s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    signals = [s for s in signals if s.recommended_contracts > 0]
    if not signals:
        _warn("No edges remained after sizing / contract filters.")
        raise typer.Exit(0)

    total_deployed = sum(s.bet_dollars for s in signals)
    console.print(f"\n[bold green]{len(signals)} edge(s) detected — ${total_deployed:.2f} total deployed:[/bold green]\n")

    sig_table = Table(box=box.SIMPLE_HEAD)
    sig_table.add_column("Player", style="cyan", min_width=24)
    sig_table.add_column("Line", justify="right")
    sig_table.add_column("Side", justify="center")
    sig_table.add_column("Edge", justify="right")
    sig_table.add_column("EV", justify="right")
    sig_table.add_column("Contracts", justify="right")
    sig_table.add_column("Limit", justify="right")
    sig_table.add_column("$Bet", justify="right")
    sig_table.add_column("E[PnL] ±1σ", justify="right")
    for s in signals:
        xp = expected_pnl_usd(s)
        sig = expected_pnl_usd_std(s)
        xp_cell = f"[green]{xp:+.2f} ±{sig:.2f}[/green]" if xp >= 0 else f"[red]{xp:+.2f} ±{sig:.2f}[/red]"
        sig_table.add_row(
            s.player_name,
            str(s.kalshi_line),
            f"[green]{s.recommended_side.upper()}[/green]",
            f"[green]{s.edge:+.3f}[/green]",
            f"{s.ev:.3f}",
            str(s.recommended_contracts),
            f"{s.limit_price:.2f}",
            f"${s.bet_dollars:.2f}",
            xp_cell,
        )
    console.print(sig_table)
    total_xp = sum(expected_pnl_usd(s) for s in signals)
    port_sig = portfolio_expected_pnl_std(signals)
    console.print(
        "[dim]E[PnL] = N×E[per contract]; σ = √(N·Var per contract) with Var from Bernoulli payoffs at limit. "
        "Portfolio σ = √(Σ leg variances), treating legs as independent (same-slate correlation usually increases true risk). "
        f"Sum E[PnL] = ${total_xp:+.2f}; idiosyncratic σ ≈ ${port_sig:.2f} (~68% band ±1σ if normal).[/dim]"
    )

    if not dry_run:
        console.print("\n[bold red]LIVE MODE — placing orders...[/bold red]")
        todays_tickers = {ml.ticker for ml in market_lines}
        results = execute_signals(signals, dry_run=False, todays_tickers=todays_tickers, cancel_stale=True)
        if auto_mark and results:
            import subprocess
            import sys
            from pathlib import Path

            import run_pipeline as _rp

            delays = []
            for part in str(mark_delays).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    delays.append(int(part))
                except Exception:
                    continue
            delays = sorted(set([d for d in delays if d > 0]))
            if delays:
                _script = str(Path(_rp.__file__).resolve())
                for d in delays:
                    label = f"{d}m"
                    argv = [sys.executable, _script, "mark", "--date", game_date, "--label", label]
                    cmd = [
                        sys.executable,
                        "-c",
                        "import time,subprocess,sys; "
                        f"time.sleep({int(d)}*60); "
                        f"subprocess.run({repr(argv)}, check=False)",
                    ]
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                console.print(f"\n[dim]Auto-mark scheduled at: {', '.join(f'{d}m' for d in delays)}[/dim]")
        _email_scan_summary(game_date, signals, total_xp, results=results)
    else:
        _warn("Dry run — no orders placed. Use --live to execute.")
        _email_scan_summary(game_date, signals, total_xp, results=None)


def mark(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    label: str = typer.Option("30m", "--label", help="Mark label, e.g. '30m', '120m'."),
):
    """
    Snapshot current market prices for placed orders and write `note=mark` rows.
    Used for CLV / mark-to-market diagnostics.
    """
    from journal_reader import load_jsonl_rows, placed_with_order_id
    from kalshi_bridge import get_client, read_price_from_market_dict
    from trade_journal import TradeRow, append_row, journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Mark snapshot — {game_date} ({label})")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = load_jsonl_rows(path)
    placed = placed_with_order_id(rows)
    if not placed:
        _warn("No placed orders found to mark.")
        raise typer.Exit(0)

    existing = {(str(r.get("order_id")), str(r.get("mark_label"))) for r in rows if r.get("note") == "mark" and r.get("order_id")}

    client = get_client()

    wrote = 0
    for r in placed:
        oid = str(r.get("order_id"))
        if (oid, str(label)) in existing:
            continue
        ticker = str(r.get("ticker", ""))
        try:
            m = client.get_market(ticker) or {}
        except Exception:
            m = {}

        yes_bid = read_price_from_market_dict(m, "yes_bid_dollars", "yes_bid")
        yes_ask = read_price_from_market_dict(m, "yes_ask_dollars", "yes_ask")
        no_bid = read_price_from_market_dict(m, "no_bid_dollars", "no_bid")
        no_ask = read_price_from_market_dict(m, "no_ask_dollars", "no_ask")
        # Mid-price fallback: prefer (bid+ask)/2, else fall back to whichever side we have.
        yes_mid = (yes_bid + yes_ask) / 2 if (yes_bid > 0 and yes_ask > 0) else (yes_ask or yes_bid or 0.0)
        no_mid = (no_bid + no_ask) / 2 if (no_bid > 0 and no_ask > 0) else (no_ask or no_bid or 0.0)

        append_row(
            game_date,
            TradeRow(
                game_date=game_date,
                ticker=ticker,
                side=str(r.get("side", "")),
                action=str(r.get("action", "buy")),
                contracts=int(r.get("contracts", 0) or 0),
                limit_price=float(r.get("limit_price", 0.0) or 0.0),
                order_id=oid,
                player_name=str(r.get("player_name", "")),
                kalshi_line=float(r.get("kalshi_line", 0.0) or 0.0),
                predicted_lambda=float(r.get("predicted_lambda", 0.0) or 0.0),
                p_model=float(r.get("p_model", 0.0) or 0.0),
                p_model_raw=float(r.get("p_model_raw", 0.0) or 0.0),
                p_market=float(r.get("p_market", 0.0) or 0.0),
                mark_yes_bid=float(yes_bid),
                mark_yes_ask=float(yes_ask),
                mark_no_bid=float(no_bid),
                mark_no_ask=float(no_ask),
                mark_yes_mid=float(yes_mid),
                mark_no_mid=float(no_mid),
                note="mark",
                success=True,
            ).to_dict(),
        )
        wrote += 1

    if wrote:
        _success(f"Wrote {wrote} mark row(s).")
    else:
        _warn("No new mark rows written (already marked or no prices).")

def report(game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today.")):
    from collections import defaultdict

    from journal_reader import (
        index_fills_by_order_id,
        index_marks_by_order_and_label,
        load_jsonl_rows,
        placed_post_submit,
    )
    from kalshi_bridge import get_client
    from reporting_common import edge_bucket, estimated_order_fee_usd, market_yes_no_result, pnl_per_contract, price_bucket, spread_bucket
    from trade_journal import journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Report — {game_date}")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = load_jsonl_rows(path)
    placed = placed_post_submit(rows)
    if not placed:
        _warn("No successful placed orders in journal for this date.")
        raise typer.Exit(0)

    fills = index_fills_by_order_id(rows)
    marks = index_marks_by_order_and_label(rows)

    client = get_client()
    by_ticker = {}
    for r in placed:
        t = r["ticker"]
        if t in by_ticker:
            continue
        try:
            by_ticker[t] = client.get_market(t)
        except Exception:
            by_ticker[t] = {}

    def outcome_for(ticker: str) -> str:
        return market_yes_no_result(by_ticker.get(ticker, {}))

    total_cost = 0.0
    total_realized = 0.0
    total_fees = 0.0
    total_expected = 0.0
    total_expected_resolved = 0.0
    unresolved = 0
    resolved_n = 0
    wins = 0

    bucket = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "p_sum": 0.0})
    by_side = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_line = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_price = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_edge = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_spread = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    resolved_trades: list[dict] = []

    for r in placed:
        side = r.get("side", "")
        contracts = int(r.get("contracts", 0))
        p_model = float(r.get("p_model", 0.0))
        ticker = r.get("ticker", "")
        line = float(r.get("kalshi_line", 0.0))
        edge = float(r.get("edge", 0.0))
        exp = float(r.get("expected_pnl", edge * contracts))
        spread = float(r.get("book_spread", 0.0))

        order_id = str(r.get("order_id", "") or "")
        fill = fills.get(order_id)
        filled_contracts = int(fill.get("filled_contracts", 0)) if fill else 0
        fill_price = float(fill.get("avg_fill_price", 0.0)) if fill else 0.0
        # If we haven't reconciled fills yet, treat as 0 filled (prevents counting resting orders as losses).
        if filled_contracts <= 0:
            continue
        price = fill_price if fill_price > 0 else float(r.get("limit_price", 0.0))

        cost = price * filled_contracts
        total_cost += cost
        # Scale expected pnl to filled volume.
        total_expected += float(edge) * filled_contracts

        res = outcome_for(ticker)
        pnlpc = pnl_per_contract(side, price, res)
        if pnlpc is None:
            unresolved += 1
            continue
        pnl = pnlpc * filled_contracts
        total_realized += pnl
        total_fees += estimated_order_fee_usd(fill or r, price, filled_contracts)
        total_expected_resolved += float(edge) * filled_contracts
        resolved_n += 1
        is_win = side == res
        wins += 1 if is_win else 0

        p_bucket = f"{int(p_model*10)/10:.1f}-{int(p_model*10)/10 + 0.1:.1f}"
        b = bucket[p_bucket]
        b["n"] += 1
        b["contracts"] += filled_contracts
        b["cost"] += cost
        b["pnl"] += pnl
        b["resolved"] += 1
        b["wins"] += 1 if is_win else 0
        b["p_sum"] += p_model

        s = by_side[str(side).lower() or "unknown"]
        s["n"] += 1
        s["contracts"] += filled_contracts
        s["cost"] += cost
        s["pnl"] += pnl
        s["resolved"] += 1
        s["wins"] += 1 if is_win else 0
        s["edge_sum"] += edge

        ln = by_line[str(line)]
        ln["n"] += 1
        ln["contracts"] += filled_contracts
        ln["cost"] += cost
        ln["pnl"] += pnl
        ln["resolved"] += 1
        ln["wins"] += 1 if is_win else 0
        ln["edge_sum"] += edge

        pb = by_price[price_bucket(price)]
        pb["n"] += 1
        pb["contracts"] += filled_contracts
        pb["cost"] += cost
        pb["pnl"] += pnl
        pb["resolved"] += 1
        pb["wins"] += 1 if is_win else 0
        pb["edge_sum"] += edge

        eb = by_edge[edge_bucket(edge)]
        eb["n"] += 1
        eb["contracts"] += filled_contracts
        eb["cost"] += cost
        eb["pnl"] += pnl
        eb["resolved"] += 1
        eb["wins"] += 1 if is_win else 0
        eb["edge_sum"] += edge

        sb = by_spread[spread_bucket(spread)]
        sb["n"] += 1
        sb["contracts"] += filled_contracts
        sb["cost"] += cost
        sb["pnl"] += pnl
        sb["resolved"] += 1
        sb["wins"] += 1 if is_win else 0
        sb["edge_sum"] += edge

        resolved_trades.append(
            {
                "ticker": ticker,
                "player": r.get("player_name", ""),
                "side": side,
                "line": line,
                "price": price,
                "contracts": filled_contracts,
                "p_model": p_model,
                "edge": edge,
                "pnl": pnl,
                "order_id": order_id,
            }
        )

    console.print(f"Orders placed (successful): {len(placed)}")
    console.print(f"Orders filled (reconciled): {resolved_n + unresolved}")
    console.print(f"Total cost (est)          : ${total_cost:,.2f}")
    console.print(f"Expected P&L (model est)  : ${total_expected:,.2f}")
    if total_cost > 0 and (len(placed) - unresolved) > 0:
        roi = (total_realized / total_cost) * 100
        console.print(f"Realized P&L (resolved)   : ${total_realized:,.2f}  (ROI {roi:.2f}%)")
        net = total_realized - total_fees
        net_roi = (net / total_cost) * 100
        console.print(
            f"Kalshi fees (est)         : ${total_fees:,.2f}  →  Net P&L after fees: ${net:,.2f}  (ROI {net_roi:.2f}%)"
        )
        if abs(total_expected_resolved) > 1e-9:
            ratio = total_realized / total_expected_resolved
            console.print(f"Realized / Expected (res) : {ratio:.2f}×")
        if resolved_n:
            console.print(f"Win rate (resolved)       : {wins}/{resolved_n} ({(wins/resolved_n)*100:.1f}%)")
    if unresolved:
        _warn(f"{unresolved} order(s) not resolved yet. Re-run report later.")
    if not fills:
        _warn("No fill rows found for this date. Run `python3 run_pipeline.py reconcile --date YYYY-MM-DD` before trusting ROI.")

    # CLV summary (uses filled trades only).
    clv_by_label = {}
    for tr in resolved_trades:
        oid = tr.get("order_id", "")
        side = str(tr.get("side", "")).lower()
        entry = float(tr.get("price", 0.0))
        ctr = int(tr.get("contracts", 0) or 0)
        for lbl in ("30m", "120m"):
            mk = marks.get((oid, lbl))
            if not mk:
                continue
            mid = float(mk.get("mark_yes_mid", 0.0) if side == "yes" else mk.get("mark_no_mid", 0.0))
            if mid <= 0 or entry <= 0 or ctr <= 0:
                continue
            clv = mid - entry
            agg = clv_by_label.setdefault(lbl, {"n": 0, "contracts": 0, "clv_sum": 0.0})
            agg["n"] += 1
            agg["contracts"] += ctr
            agg["clv_sum"] += clv * ctr

    # CLV coverage: how many resolved fills have at least one mark with a non-zero mid?
    total_fills = resolved_n + unresolved
    clv_n = sum(
        1
        for tr in resolved_trades
        if any(
            marks.get((tr["order_id"], lbl)) and (
                float(marks[(tr["order_id"], lbl)].get("mark_yes_mid", 0.0) or 0.0) > 0
                or float(marks[(tr["order_id"], lbl)].get("mark_no_mid", 0.0) or 0.0) > 0
            )
            for lbl in ("30m", "120m")
        )
    )
    if total_fills > 0:
        cov_pct = clv_n / total_fills * 100
        cov_line = f"CLV coverage: {clv_n}/{total_fills} ({cov_pct:.0f}%)"
        if cov_pct < 70:
            console.print(f"[yellow]Warning: {cov_line} — CLV average excludes unmarked trades and may be unreliable.[/yellow]")
        else:
            console.print(f"[dim]{cov_line}[/dim]")

    if clv_by_label:
        tc = Table(title="CLV / mark-to-market (filled trades only)", box=box.SIMPLE_HEAD)
        tc.add_column("Label", style="cyan")
        tc.add_column("Trades", justify="right")
        tc.add_column("Contracts", justify="right")
        tc.add_column("Avg CLV/ctr", justify="right")
        for lbl in sorted(clv_by_label.keys()):
            a = clv_by_label[lbl]
            avg = (a["clv_sum"] / a["contracts"]) if a["contracts"] else 0.0
            tc.add_row(lbl, str(a["n"]), str(a["contracts"]), f"{avg:+.4f}")
        console.print(tc)

    t = Table(title="Performance by p_model bucket (resolved only)", box=box.SIMPLE_HEAD)
    t.add_column("p_model bucket", style="cyan")
    t.add_column("Orders", justify="right")
    t.add_column("Contracts", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("P&L", justify="right")
    t.add_column("ROI", justify="right")
    t.add_column("Win%", justify="right")
    t.add_column("Avg p", justify="right")
    t.add_column("Calib gap", justify="right")
    for k in sorted(bucket.keys()):
        b = bucket[k]
        cost = b["cost"]
        pnl = b["pnl"]
        roi = (pnl / cost) * 100 if cost > 0 else 0.0
        win_pct = (b["wins"] / b["resolved"]) * 100 if b["resolved"] else 0.0
        avg_p = (b["p_sum"] / b["resolved"]) if b["resolved"] else 0.0
        calib = (win_pct / 100.0) - avg_p
        t.add_row(
            k,
            str(b["n"]),
            str(b["contracts"]),
            f"${cost:,.2f}",
            f"${pnl:,.2f}",
            f"{roi:.2f}%",
            f"{win_pct:.1f}%",
            f"{avg_p:.3f}",
            f"{calib:+.3f}",
        )
    console.print(t)

    # Call out the buckets where the model appears most overconfident.
    calib_rows = []
    for k, b in bucket.items():
        if b["resolved"] < 5:
            continue
        win_rate = (b["wins"] / b["resolved"]) if b["resolved"] else 0.0
        avg_p = (b["p_sum"] / b["resolved"]) if b["resolved"] else 0.0
        calib_gap = win_rate - avg_p
        calib_rows.append((calib_gap, k, b, win_rate, avg_p))
    calib_rows.sort(key=lambda x: x[0])  # most negative first
    if calib_rows:
        tcg = Table(title="Most negative calibration gaps (min 5 resolved)", box=box.SIMPLE_HEAD)
        tcg.add_column("Bucket", style="cyan")
        tcg.add_column("Resolved", justify="right")
        tcg.add_column("Win%", justify="right")
        tcg.add_column("Avg p", justify="right")
        tcg.add_column("Gap", justify="right")
        tcg.add_column("Cost", justify="right")
        tcg.add_column("P&L", justify="right")
        tcg.add_column("ROI", justify="right")
        for calib_gap, k, b, win_rate, avg_p in calib_rows[:5]:
            cost = b["cost"]
            pnl = b["pnl"]
            roi = (pnl / cost) * 100 if cost > 0 else 0.0
            tcg.add_row(
                k,
                str(b["resolved"]),
                f"{win_rate*100:.1f}%",
                f"{avg_p:.3f}",
                f"{calib_gap:+.3f}",
                f"${cost:,.2f}",
                f"${pnl:,.2f}",
                f"{roi:.2f}%",
            )
        console.print(tcg)

    def _slice_table(title: str, group: dict):
        tt = Table(title=title, box=box.SIMPLE_HEAD)
        tt.add_column("Group", style="cyan")
        tt.add_column("Orders", justify="right")
        tt.add_column("Contracts", justify="right")
        tt.add_column("Cost", justify="right")
        tt.add_column("P&L", justify="right")
        tt.add_column("ROI", justify="right")
        tt.add_column("Win%", justify="right")
        tt.add_column("Avg edge", justify="right")
        for g in sorted(group.keys(), key=str):
            b = group[g]
            cost = b["cost"]
            pnl = b["pnl"]
            roi = (pnl / cost) * 100 if cost > 0 else 0.0
            win_pct = (b["wins"] / b["resolved"]) * 100 if b["resolved"] else 0.0
            avg_edge = (b["edge_sum"] / b["resolved"]) if b["resolved"] else 0.0
            tt.add_row(str(g), str(b["n"]), str(b["contracts"]), f"${cost:,.2f}", f"${pnl:,.2f}", f"{roi:.2f}%", f"{win_pct:.1f}%", f"{avg_edge:+.3f}")
        console.print(tt)

    _slice_table("Slice: side", by_side)
    _slice_table("Slice: kalshi line", by_line)
    _slice_table("Slice: entry price bucket (limit_price)", by_price)
    _slice_table("Slice: edge bucket", by_edge)
    if any(k != "n/a" for k in by_spread.keys()):
        _slice_table("Slice: book spread bucket", by_spread)

    # Segment P&L table: line × edge_bucket (fills ≥ 2 only).
    if resolved_trades:
        seg: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "wins": 0, "edge_sum": 0.0})
        for tr in resolved_trades:
            key = (str(tr["line"]), edge_bucket(tr["edge"]))
            s = seg[key]
            s["n"] += 1
            s["contracts"] += tr["contracts"]
            s["cost"] += tr["price"] * tr["contracts"]
            s["pnl"] += tr["pnl"]
            s["wins"] += 1 if tr["pnl"] > 0 else 0
            s["edge_sum"] += tr["edge"]
        visible = {k: v for k, v in seg.items() if v["n"] >= 2}
        if visible:
            tseg = Table(title="P&L by line × edge bucket (≥2 fills)", box=box.SIMPLE_HEAD)
            tseg.add_column("line", style="cyan", justify="right")
            tseg.add_column("edge_bucket", style="cyan")
            tseg.add_column("fills", justify="right")
            tseg.add_column("contracts", justify="right")
            tseg.add_column("roi%", justify="right")
            tseg.add_column("win_rate", justify="right")
            tseg.add_column("avg_edge", justify="right")
            for (line_key, eb_key) in sorted(visible.keys(), key=lambda x: (float(x[0]), x[1])):
                sv = visible[(line_key, eb_key)]
                roi = (sv["pnl"] / sv["cost"]) * 100 if sv["cost"] > 0 else 0.0
                wr = sv["wins"] / sv["n"] * 100
                avg_e = sv["edge_sum"] / sv["n"]
                tseg.add_row(line_key, eb_key, str(sv["n"]), str(sv["contracts"]), f"{roi:.2f}%", f"{wr:.1f}%", f"{avg_e:+.3f}")
            console.print(tseg)

    if resolved_trades:
        worst = sorted(resolved_trades, key=lambda x: x["pnl"])[:10]
        tw = Table(title="Worst 10 trades (resolved P&L)", box=box.SIMPLE_HEAD)
        tw.add_column("P&L", justify="right")
        tw.add_column("Edge", justify="right")
        tw.add_column("p_model", justify="right")
        tw.add_column("Price", justify="right")
        tw.add_column("Ctr", justify="right")
        tw.add_column("Side", justify="center")
        tw.add_column("Line", justify="right")
        tw.add_column("Player", style="cyan", min_width=18)
        tw.add_column("Ticker", style="dim")
        for r0 in worst:
            tw.add_row(
                f"${r0['pnl']:.2f}",
                f"{r0['edge']:+.3f}",
                f"{r0['p_model']:.3f}",
                f"{r0['price']:.2f}",
                str(r0["contracts"]),
                str(r0["side"]).upper(),
                str(r0["line"]),
                str(r0["player"])[:28],
                str(r0["ticker"])[:42],
            )
        console.print(tw)

        edge_losers = sorted([x for x in resolved_trades if x["pnl"] < 0], key=lambda x: x["edge"], reverse=True)[:10]
        if edge_losers:
            te = Table(title="Biggest-edge losers (resolved, pnl<0)", box=box.SIMPLE_HEAD)
            te.add_column("Edge", justify="right")
            te.add_column("P&L", justify="right")
            te.add_column("p_model", justify="right")
            te.add_column("Price", justify="right")
            te.add_column("Ctr", justify="right")
            te.add_column("Side", justify="center")
            te.add_column("Line", justify="right")
            te.add_column("Player", style="cyan", min_width=18)
            te.add_column("Ticker", style="dim")
            for r0 in edge_losers:
                te.add_row(
                    f"{r0['edge']:+.3f}",
                    f"${r0['pnl']:.2f}",
                    f"{r0['p_model']:.3f}",
                    f"{r0['price']:.2f}",
                    str(r0["contracts"]),
                    str(r0["side"]).upper(),
                    str(r0["line"]),
                    str(r0["player"])[:28],
                    str(r0["ticker"])[:42],
                )
            console.print(te)

        conf_losers = sorted([x for x in resolved_trades if x["pnl"] < 0], key=lambda x: x["p_model"], reverse=True)[:10]
        if conf_losers:
            tc = Table(title="Highest-confidence losers (resolved, pnl<0)", box=box.SIMPLE_HEAD)
            tc.add_column("p_model", justify="right")
            tc.add_column("Edge", justify="right")
            tc.add_column("P&L", justify="right")
            tc.add_column("Price", justify="right")
            tc.add_column("Ctr", justify="right")
            tc.add_column("Side", justify="center")
            tc.add_column("Line", justify="right")
            tc.add_column("Player", style="cyan", min_width=18)
            tc.add_column("Ticker", style="dim")
            for r0 in conf_losers:
                tc.add_row(
                    f"{r0['p_model']:.3f}",
                    f"{r0['edge']:+.3f}",
                    f"${r0['pnl']:.2f}",
                    f"{r0['price']:.2f}",
                    str(r0["contracts"]),
                    str(r0["side"]).upper(),
                    str(r0["line"]),
                    str(r0["player"])[:28],
                    str(r0["ticker"])[:42],
                )
            console.print(tc)


def report_range(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Default: earliest journal found."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: latest journal found."),
    per_day: bool = typer.Option(True, "--per-day/--no-per-day", help="Print a per-day summary table in addition to combined totals."),
):
    """
    Aggregate trade performance across multiple days by scanning `data/trades_YYYY-MM-DD.jsonl` journals.
    """
    from collections import defaultdict

    from config import DATA_DIR
    from journal_reader import (
        date_from_journal_filename,
        journal_paths_in_date_range,
        journal_paths_sorted,
        load_window_rows,
        parse_iso_date,
    )
    from kalshi_bridge import get_client
    from reporting_common import edge_bucket, estimated_order_fee_usd, market_yes_no_result, pnl_per_contract, price_bucket, spread_bucket

    journal_paths = journal_paths_sorted()
    if not journal_paths:
        _warn(f"No trade journals found under {DATA_DIR}")
        raise typer.Exit(0)

    # Determine date window defaults from available journals.
    dates = []
    for p in journal_paths:
        d = date_from_journal_filename(p)
        if not d:
            continue
        try:
            dates.append(d)
        except Exception:
            continue
    dates = sorted(set(dates))
    if not dates:
        _warn(f"No valid journal filenames under {DATA_DIR} (expected trades_YYYY-MM-DD.jsonl)")
        raise typer.Exit(0)

    start_d = parse_iso_date(start) if start else parse_iso_date(dates[0])
    end_d = parse_iso_date(end) if end else parse_iso_date(dates[-1])
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    window_paths = journal_paths_in_date_range(start_d, end_d)
    placed, fill_rows, mark_rows = load_window_rows(window_paths)

    window_label = f"{start_d.strftime('%Y-%m-%d')} → {end_d.strftime('%Y-%m-%d')}"
    _header(f"Report (range) — {window_label}")

    if not placed:
        _warn("No successful placed orders found in journals for this date range.")
        raise typer.Exit(0)

    fills: dict[str, dict] = {str(r["order_id"]): r for r in fill_rows if r.get("order_id")}
    marks: dict[tuple[str, str], dict] = {(str(r["order_id"]), str(r.get("mark_label", ""))): r for r in mark_rows if r.get("order_id")}

    # Fetch outcomes once per ticker (across the whole range).
    client = get_client()
    by_ticker = {}
    for r in placed:
        t = r.get("ticker", "")
        if not t or t in by_ticker:
            continue
        try:
            by_ticker[t] = client.get_market(t)
        except Exception:
            by_ticker[t] = {}

    def outcome_for(ticker: str) -> str:
        return market_yes_no_result(by_ticker.get(ticker, {}))

    total_cost = 0.0
    total_realized = 0.0
    total_fees = 0.0
    total_expected = 0.0
    total_expected_resolved = 0.0
    unresolved = 0
    resolved_n = 0
    wins = 0

    bucket = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "p_sum": 0.0})
    by_day = defaultdict(lambda: {"orders": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "unresolved": 0})
    by_side = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_line = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_price = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_edge = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    by_spread = defaultdict(lambda: {"n": 0, "contracts": 0, "cost": 0.0, "pnl": 0.0, "resolved": 0, "wins": 0, "edge_sum": 0.0})
    resolved_trades: list[dict] = []

    for r in placed:
        day = str(r.get("game_date", ""))
        side = r.get("side", "")
        contracts = int(r.get("contracts", 0))
        p_model = float(r.get("p_model", 0.0))
        ticker = r.get("ticker", "")
        line = float(r.get("kalshi_line", 0.0))
        edge = float(r.get("edge", 0.0))
        exp = float(r.get("expected_pnl", edge * contracts))
        spread = float(r.get("book_spread", 0.0))

        order_id = str(r.get("order_id", "") or "")
        fill = fills.get(order_id)
        filled_contracts = int(fill.get("filled_contracts", 0)) if fill else 0
        fill_price = float(fill.get("avg_fill_price", 0.0)) if fill else 0.0
        if filled_contracts <= 0:
            continue
        price = fill_price if fill_price > 0 else float(r.get("limit_price", 0.0))

        cost = price * filled_contracts
        total_cost += cost
        total_expected += float(edge) * filled_contracts
        by_day[day]["orders"] += 1
        by_day[day]["contracts"] += filled_contracts
        by_day[day]["cost"] += cost

        res = outcome_for(ticker)
        pnlpc = pnl_per_contract(side, price, res)
        if pnlpc is None:
            unresolved += 1
            by_day[day]["unresolved"] += 1
            continue

        pnl = pnlpc * filled_contracts
        total_realized += pnl
        total_fees += estimated_order_fee_usd(fill or r, price, filled_contracts)
        total_expected_resolved += float(edge) * filled_contracts
        by_day[day]["pnl"] += pnl
        resolved_n += 1
        is_win = side == res
        wins += 1 if is_win else 0

        p_bucket = f"{int(p_model*10)/10:.1f}-{int(p_model*10)/10 + 0.1:.1f}"
        b = bucket[p_bucket]
        b["n"] += 1
        b["contracts"] += filled_contracts
        b["cost"] += cost
        b["pnl"] += pnl
        b["resolved"] += 1
        b["wins"] += 1 if is_win else 0
        b["p_sum"] += p_model

        s = by_side[str(side).lower() or "unknown"]
        s["n"] += 1
        s["contracts"] += filled_contracts
        s["cost"] += cost
        s["pnl"] += pnl
        s["resolved"] += 1
        s["wins"] += 1 if is_win else 0
        s["edge_sum"] += edge

        ln = by_line[str(line)]
        ln["n"] += 1
        ln["contracts"] += filled_contracts
        ln["cost"] += cost
        ln["pnl"] += pnl
        ln["resolved"] += 1
        ln["wins"] += 1 if is_win else 0
        ln["edge_sum"] += edge

        pb = by_price[price_bucket(price)]
        pb["n"] += 1
        pb["contracts"] += filled_contracts
        pb["cost"] += cost
        pb["pnl"] += pnl
        pb["resolved"] += 1
        pb["wins"] += 1 if is_win else 0
        pb["edge_sum"] += edge

        eb = by_edge[edge_bucket(edge)]
        eb["n"] += 1
        eb["contracts"] += filled_contracts
        eb["cost"] += cost
        eb["pnl"] += pnl
        eb["resolved"] += 1
        eb["wins"] += 1 if is_win else 0
        eb["edge_sum"] += edge

        sb = by_spread[spread_bucket(spread)]
        sb["n"] += 1
        sb["contracts"] += filled_contracts
        sb["cost"] += cost
        sb["pnl"] += pnl
        sb["resolved"] += 1
        sb["wins"] += 1 if is_win else 0
        sb["edge_sum"] += edge

        resolved_trades.append(
            {
                "game_date": day,
                "ticker": ticker,
                "player": r.get("player_name", ""),
                "side": side,
                "line": line,
                "price": price,
                "contracts": filled_contracts,
                "p_model": p_model,
                "edge": edge,
                "pnl": pnl,
                "order_id": order_id,
            }
        )

    console.print(f"Orders placed (successful): {len(placed)}")
    console.print(f"Orders filled (reconciled): {resolved_n + unresolved}")
    console.print(f"Total cost (est)          : ${total_cost:,.2f}")
    console.print(f"Expected P&L (model est)  : ${total_expected:,.2f}")
    if total_cost > 0 and (len(placed) - unresolved) > 0:
        roi = (total_realized / total_cost) * 100
        console.print(f"Realized P&L (resolved)   : ${total_realized:,.2f}  (ROI {roi:.2f}%)")
        net = total_realized - total_fees
        net_roi = (net / total_cost) * 100
        console.print(
            f"Kalshi fees (est)         : ${total_fees:,.2f}  →  Net P&L after fees: ${net:,.2f}  (ROI {net_roi:.2f}%)"
        )
        if abs(total_expected_resolved) > 1e-9:
            ratio = total_realized / total_expected_resolved
            console.print(f"Realized / Expected (res) : {ratio:.2f}×")
        if resolved_n:
            console.print(f"Win rate (resolved)       : {wins}/{resolved_n} ({(wins/resolved_n)*100:.1f}%)")
    if unresolved:
        _warn(f"{unresolved} order(s) not resolved yet. Re-run report later.")
    if not fills:
        _warn("No fill rows found in this range. Run `python3 run_pipeline.py reconcile --date ...` for each day before trusting ROI.")

    clv_by_label = {}
    for tr in resolved_trades:
        oid = tr.get("order_id", "")
        side = str(tr.get("side", "")).lower()
        entry = float(tr.get("price", 0.0))
        ctr = int(tr.get("contracts", 0) or 0)
        for lbl in ("30m", "120m"):
            mk = marks.get((oid, lbl))
            if not mk:
                continue
            mid = float(mk.get("mark_yes_mid", 0.0) if side == "yes" else mk.get("mark_no_mid", 0.0))
            if mid <= 0 or entry <= 0 or ctr <= 0:
                continue
            clv = mid - entry
            agg = clv_by_label.setdefault(lbl, {"n": 0, "contracts": 0, "clv_sum": 0.0})
            agg["n"] += 1
            agg["contracts"] += ctr
            agg["clv_sum"] += clv * ctr

    if clv_by_label:
        tc = Table(title="CLV / mark-to-market (filled trades only)", box=box.SIMPLE_HEAD)
        tc.add_column("Label", style="cyan")
        tc.add_column("Trades", justify="right")
        tc.add_column("Contracts", justify="right")
        tc.add_column("Avg CLV/ctr", justify="right")
        for lbl in sorted(clv_by_label.keys()):
            a = clv_by_label[lbl]
            avg = (a["clv_sum"] / a["contracts"]) if a["contracts"] else 0.0
            tc.add_row(lbl, str(a["n"]), str(a["contracts"]), f"{avg:+.4f}")
        console.print(tc)

    if per_day:
        td = Table(title="Per-day summary (resolved P&L shown; unresolved excluded)", box=box.SIMPLE_HEAD)
        td.add_column("Date", style="cyan")
        td.add_column("Orders", justify="right")
        td.add_column("Contracts", justify="right")
        td.add_column("Cost", justify="right")
        td.add_column("P&L", justify="right")
        td.add_column("ROI", justify="right")
        td.add_column("Unresolved", justify="right")
        for day in sorted(by_day.keys()):
            b = by_day[day]
            cost = b["cost"]
            pnl = b["pnl"]
            roi = (pnl / cost) * 100 if cost > 0 else 0.0
            td.add_row(
                day,
                str(b["orders"]),
                str(b["contracts"]),
                f"${cost:,.2f}",
                f"${pnl:,.2f}",
                f"{roi:.2f}%",
                str(b["unresolved"]),
            )
        console.print(td)

    n_days_with_fills = len(by_day)
    if n_days_with_fills > 0:
        avg_realized_per_day = total_realized / n_days_with_fills
        daily_rois: list[float] = []
        for b in by_day.values():
            cost = float(b["cost"])
            pnl = float(b["pnl"])
            daily_rois.append((pnl / cost) * 100.0 if cost > 0 else 0.0)
        avg_roi_per_day = sum(daily_rois) / len(daily_rois)
        console.print(
            f"[dim]Realized P&L avg / day ({n_days_with_fills} day(s) with ≥1 reconciled fill): "
            f"${avg_realized_per_day:,.2f}  ·  "
            f"Avg ROI / day (unweighted mean of each day's ROI): {avg_roi_per_day:.2f}%[/dim]"
        )

    t = Table(title="Performance by p_model bucket (resolved only)", box=box.SIMPLE_HEAD)
    t.add_column("p_model bucket", style="cyan")
    t.add_column("Orders", justify="right")
    t.add_column("Contracts", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("P&L", justify="right")
    t.add_column("ROI", justify="right")
    t.add_column("Win%", justify="right")
    t.add_column("Avg p", justify="right")
    t.add_column("Calib gap", justify="right")
    for k in sorted(bucket.keys()):
        b = bucket[k]
        cost = b["cost"]
        pnl = b["pnl"]
        roi = (pnl / cost) * 100 if cost > 0 else 0.0
        win_pct = (b["wins"] / b["resolved"]) * 100 if b["resolved"] else 0.0
        avg_p = (b["p_sum"] / b["resolved"]) if b["resolved"] else 0.0
        calib = (win_pct / 100.0) - avg_p
        t.add_row(
            k,
            str(b["n"]),
            str(b["contracts"]),
            f"${cost:,.2f}",
            f"${pnl:,.2f}",
            f"{roi:.2f}%",
            f"{win_pct:.1f}%",
            f"{avg_p:.3f}",
            f"{calib:+.3f}",
        )
    console.print(t)

    calib_rows = []
    for k, b in bucket.items():
        if b["resolved"] < 10:
            continue
        win_rate = (b["wins"] / b["resolved"]) if b["resolved"] else 0.0
        avg_p = (b["p_sum"] / b["resolved"]) if b["resolved"] else 0.0
        calib_gap = win_rate - avg_p
        calib_rows.append((calib_gap, k, b, win_rate, avg_p))
    calib_rows.sort(key=lambda x: x[0])
    if calib_rows:
        tcg = Table(title="Most negative calibration gaps (min 10 resolved)", box=box.SIMPLE_HEAD)
        tcg.add_column("Bucket", style="cyan")
        tcg.add_column("Resolved", justify="right")
        tcg.add_column("Win%", justify="right")
        tcg.add_column("Avg p", justify="right")
        tcg.add_column("Gap", justify="right")
        tcg.add_column("Cost", justify="right")
        tcg.add_column("P&L", justify="right")
        tcg.add_column("ROI", justify="right")
        for calib_gap, k, b, win_rate, avg_p in calib_rows[:8]:
            cost = b["cost"]
            pnl = b["pnl"]
            roi = (pnl / cost) * 100 if cost > 0 else 0.0
            tcg.add_row(
                k,
                str(b["resolved"]),
                f"{win_rate*100:.1f}%",
                f"{avg_p:.3f}",
                f"{calib_gap:+.3f}",
                f"${cost:,.2f}",
                f"${pnl:,.2f}",
                f"{roi:.2f}%",
            )
        console.print(tcg)

    def _slice_table(title: str, group: dict):
        tt = Table(title=title, box=box.SIMPLE_HEAD)
        tt.add_column("Group", style="cyan")
        tt.add_column("Orders", justify="right")
        tt.add_column("Contracts", justify="right")
        tt.add_column("Cost", justify="right")
        tt.add_column("P&L", justify="right")
        tt.add_column("ROI", justify="right")
        tt.add_column("Win%", justify="right")
        tt.add_column("Avg edge", justify="right")
        for g in sorted(group.keys(), key=str):
            b = group[g]
            cost = b["cost"]
            pnl = b["pnl"]
            roi = (pnl / cost) * 100 if cost > 0 else 0.0
            win_pct = (b["wins"] / b["resolved"]) * 100 if b["resolved"] else 0.0
            avg_edge = (b["edge_sum"] / b["resolved"]) if b["resolved"] else 0.0
            tt.add_row(str(g), str(b["n"]), str(b["contracts"]), f"${cost:,.2f}", f"${pnl:,.2f}", f"{roi:.2f}%", f"{win_pct:.1f}%", f"{avg_edge:+.3f}")
        console.print(tt)

    _slice_table("Slice: side", by_side)
    _slice_table("Slice: kalshi line", by_line)
    _slice_table("Slice: entry price bucket (limit_price)", by_price)
    _slice_table("Slice: edge bucket", by_edge)
    if any(k != "n/a" for k in by_spread.keys()):
        _slice_table("Slice: book spread bucket", by_spread)

    if resolved_trades:
        worst = sorted(resolved_trades, key=lambda x: x["pnl"])[:15]
        tw = Table(title="Worst 15 trades (resolved P&L)", box=box.SIMPLE_HEAD)
        tw.add_column("Date", style="cyan")
        tw.add_column("P&L", justify="right")
        tw.add_column("Edge", justify="right")
        tw.add_column("p_model", justify="right")
        tw.add_column("Price", justify="right")
        tw.add_column("Ctr", justify="right")
        tw.add_column("Side", justify="center")
        tw.add_column("Line", justify="right")
        tw.add_column("Player", style="cyan", min_width=18)
        for r0 in worst:
            tw.add_row(
                str(r0["game_date"]),
                f"${r0['pnl']:.2f}",
                f"{r0['edge']:+.3f}",
                f"{r0['p_model']:.3f}",
                f"{r0['price']:.2f}",
                str(r0["contracts"]),
                str(r0["side"]).upper(),
                str(r0["line"]),
                str(r0["player"])[:28],
            )
        console.print(tw)

        edge_losers = sorted([x for x in resolved_trades if x["pnl"] < 0], key=lambda x: x["edge"], reverse=True)[:15]
        if edge_losers:
            te = Table(title="Biggest-edge losers (resolved, pnl<0)", box=box.SIMPLE_HEAD)
            te.add_column("Date", style="cyan")
            te.add_column("Edge", justify="right")
            te.add_column("P&L", justify="right")
            te.add_column("p_model", justify="right")
            te.add_column("Price", justify="right")
            te.add_column("Ctr", justify="right")
            te.add_column("Side", justify="center")
            te.add_column("Line", justify="right")
            te.add_column("Player", style="cyan", min_width=18)
            for r0 in edge_losers:
                te.add_row(
                    str(r0["game_date"]),
                    f"{r0['edge']:+.3f}",
                    f"${r0['pnl']:.2f}",
                    f"{r0['p_model']:.3f}",
                    f"{r0['price']:.2f}",
                    str(r0["contracts"]),
                    str(r0["side"]).upper(),
                    str(r0["line"]),
                    str(r0["player"])[:28],
                )
            console.print(te)

        conf_losers = sorted([x for x in resolved_trades if x["pnl"] < 0], key=lambda x: x["p_model"], reverse=True)[:15]
        if conf_losers:
            tc = Table(title="Highest-confidence losers (resolved, pnl<0)", box=box.SIMPLE_HEAD)
            tc.add_column("Date", style="cyan")
            tc.add_column("p_model", justify="right")
            tc.add_column("Edge", justify="right")
            tc.add_column("P&L", justify="right")
            tc.add_column("Price", justify="right")
            tc.add_column("Ctr", justify="right")
            tc.add_column("Side", justify="center")
            tc.add_column("Line", justify="right")
            tc.add_column("Player", style="cyan", min_width=18)
            for r0 in conf_losers:
                tc.add_row(
                    str(r0["game_date"]),
                    f"{r0['p_model']:.3f}",
                    f"{r0['edge']:+.3f}",
                    f"${r0['pnl']:.2f}",
                    f"{r0['price']:.2f}",
                    str(r0["contracts"]),
                    str(r0["side"]).upper(),
                    str(r0["line"]),
                    str(r0["player"])[:28],
                )
            console.print(tc)


def reconcile(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    include_resting: bool = typer.Option(False, "--include-resting", help="Also write fill rows for currently resting orders (0 filled)."),
):
    """
    Reconcile journaled orders with Kalshi order status and write `note=fill` rows.

    This prevents `report` from counting resting/unfilled orders as if they were positions.
    """
    from journal_reader import existing_fill_order_ids, load_jsonl_rows, placed_with_order_id
    from kalshi_bridge import get_client
    from trade_journal import TradeRow, append_row, journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Reconcile fills — {game_date}")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = load_jsonl_rows(path)
    placed = placed_with_order_id(rows)
    if not placed:
        _warn("No placed orders with order_id found to reconcile.")
        raise typer.Exit(0)

    existing_fill_ids = existing_fill_order_ids(rows)

    client = get_client()
    updated = 0
    skipped = 0
    for r in placed:
        order_id = str(r.get("order_id"))
        if order_id in existing_fill_ids:
            skipped += 1
            continue
        try:
            o = client.get_order(order_id)
        except Exception as e:
            _warn(f"Could not fetch order {order_id} ({e})")
            continue

        count = int(o.get("count", r.get("contracts", 0)) or 0)
        remaining = int(o.get("remaining_count", o.get("remaining", 0)) or 0)
        status = str(o.get("status", "") or "").lower()
        filled = max(0, count - remaining)

        if filled <= 0 and not include_resting:
            continue

        # Best-effort average fill price. If the API doesn't provide fill price, fall back to journal limit_price.
        avg_px = None
        for k in ("avg_fill_price", "average_fill_price", "avg_price", "fill_price"):
            if o.get(k) is not None:
                avg_px = float(o.get(k))
                break
        if avg_px is None:
            avg_px = float(r.get("limit_price", 0.0))

        append_row(
            game_date,
            TradeRow(
                game_date=game_date,
                ticker=str(r.get("ticker", "")),
                side=str(r.get("side", "")),
                action=str(r.get("action", "buy")),
                contracts=int(r.get("contracts", 0)),
                limit_price=float(r.get("limit_price", 0.0)),
                order_id=order_id,
                player_name=str(r.get("player_name", "")),
                kalshi_line=float(r.get("kalshi_line", 0.0)),
                predicted_lambda=float(r.get("predicted_lambda", 0.0)),
                p_model=float(r.get("p_model", 0.0)),
                p_model_raw=float(r.get("p_model_raw", 0.0) or 0.0),
                p_market=float(r.get("p_market", 0.0)),
                edge=float(r.get("edge", 0.0)),
                ev=float(r.get("ev", 0.0)),
                expected_pnl=float(r.get("expected_pnl", 0.0)),
                book_bid=float(r.get("book_bid", 0.0)),
                book_ask=float(r.get("book_ask", 0.0)),
                book_spread=float(r.get("book_spread", 0.0)),
                filled_contracts=int(filled),
                avg_fill_price=float(avg_px),
                note="fill",
                success=True if status in {"executed", "filled"} else None,
            ).to_dict(),
        )
        updated += 1

    if updated:
        _success(f"Wrote {updated} fill row(s).")
    if skipped:
        console.print(f"Skipped (already reconciled): {skipped}")
    if not updated:
        _warn("No new fill rows written. (Maybe none filled yet, or already reconciled.)")


def calibrate(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Default: earliest journal found."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: latest journal found."),
    min_rows: int = typer.Option(None, "--min-rows", help="Minimum resolved filled trades for global calibrator (default: MIN_CALIB_ROWS_GLOBAL)."),
):
    """
    Fit segmented isotonic calibrators from reconciled **fill** rows only.

    Uses journal ``note=fill`` (run ``reconcile`` first) and Kalshi market ``result`` for outcomes.
    Writes ``models/p_calibrator_segmented.pkl`` and ``models/calibrator_meta.json``.

    At live ``scan``, this fill-based bundle is applied first when present;
    the OOF calibrator is the fallback (see ``edge_detector._calibrate``).
    """
    import numpy as np

    from calibration import fit_isotonic, fit_segmented, save, save_segmented
    from config import DATA_DIR, MIN_CALIB_ROWS_GLOBAL, MIN_CALIB_ROWS_SEGMENT
    from journal_reader import date_from_journal_filename, journal_paths_sorted, load_jsonl_rows, parse_iso_date
    from kalshi_bridge import get_client
    from reporting_common import market_yes_no_result

    journal_paths = journal_paths_sorted()
    if not journal_paths:
        _warn(f"No trade journals found under {DATA_DIR}")
        raise typer.Exit(0)

    dates = sorted({d for p in journal_paths if (d := date_from_journal_filename(p))})
    if not dates:
        _warn(f"No valid journal filenames under {DATA_DIR} (expected trades_YYYY-MM-DD.jsonl)")
        raise typer.Exit(0)

    start_d = parse_iso_date(start) if start else parse_iso_date(dates[0])
    end_d = parse_iso_date(end) if end else parse_iso_date(dates[-1])
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    _header(f"Calibrate p_model — {start_d.strftime('%Y-%m-%d')} → {end_d.strftime('%Y-%m-%d')}")

    fill_rows: list[dict] = []
    for p in journal_paths:
        d = date_from_journal_filename(p)
        if not d:
            continue
        try:
            dd = parse_iso_date(d)
        except Exception:
            continue
        if dd < start_d or dd > end_d:
            continue
        for r in load_jsonl_rows(p):
            if r.get("note") == "fill" and r.get("order_id"):
                if int(r.get("filled_contracts", 0) or 0) > 0:
                    fill_rows.append(r)

    if not fill_rows:
        _warn("No fill rows with filled_contracts>0 found. Run reconcile first, then re-run calibrate.")
        raise typer.Exit(0)

    client = get_client()
    by_ticker = {}
    for r in fill_rows:
        t = str(r.get("ticker", ""))
        if not t or t in by_ticker:
            continue
        try:
            by_ticker[t] = client.get_market(t)
        except Exception:
            by_ticker[t] = {}

    def outcome_for(ticker: str) -> str:
        return market_yes_no_result(by_ticker.get(ticker, {}))

    min_global = int(min_rows) if min_rows is not None else int(MIN_CALIB_ROWS_GLOBAL)
    fit_rows: list[dict] = []
    for r in fill_rows:
        ticker = str(r.get("ticker", ""))
        res = outcome_for(ticker)
        if res not in {"yes", "no"}:
            continue
        side = str(r.get("side", "")).lower()
        y = 1.0 if side == res else 0.0
        p_cal = float(r.get("p_model", 0.0))
        raw = r.get("p_model_raw")
        p_for_fit = float(raw) if raw is not None else p_cal
        filled = int(r.get("filled_contracts", 0) or 0)
        if filled <= 0:
            continue
        line = float(r.get("kalshi_line", r.get("line", 1.5)) or 1.5)
        gp = int(r.get("games_played", 0) or 0)
        fit_rows.append({"p": p_for_fit, "y": y, "weight": float(filled), "line": line, "side": side, "games_played": gp})

    if len(fit_rows) < min_global:
        _warn(
            f"Not enough resolved filled trades to calibrate ({len(fit_rows)}<{min_global}). "
            "Try a wider date range or pass --min-rows lower (risk: overfitting)."
        )
        raise typer.Exit(0)

    bundle = fit_segmented(fit_rows, min_global=min_global, min_segment=MIN_CALIB_ROWS_SEGMENT)
    if bundle is None:
        _warn("Segmented calibration fit failed.")
        raise typer.Exit(1)
    save_segmented(bundle)
    n_seg = len(bundle.segments)
    from calibrate_preflight import write_calibrator_meta
    write_calibrator_meta(
        n_rows=len(fit_rows),
        n_segments=n_seg,
        start=start_d.strftime("%Y-%m-%d"),
        end=end_d.strftime("%Y-%m-%d"),
    )
    _success(
        f"Saved segmented fill calibrator ({n_seg} segment(s) with n>={MIN_CALIB_ROWS_SEGMENT}) "
        f"from {len(fit_rows)} resolved fills. "
        "At scan, this fill calibrator is applied first; OOF calibrator is fallback."
    )


def fit_blend(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Default: earliest journal found."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: latest journal found."),
    min_rows: int = typer.Option(None, "--min-rows", help="Minimum resolved fills to fit (default: MIN_BLEND_ROWS)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fit and report, but do not write models/blend_meta.json."),
):
    """
    Fit the market-blend weight ``w`` on resolved reconciled fills.

    Minimizes contract-weighted log-loss of
    ``sigmoid(w*logit(p_model) + (1-w)*logit(market_mid))`` over the traded
    side's outcome. Uses ``p_model_cal`` (pre-blend) when journaled so re-fits
    never double-shrink; older rows fall back to ``p_model``.
    Writes ``models/blend_meta.json``, which ``scan`` picks up automatically.
    """
    from config import DATA_DIR, MIN_BLEND_ROWS
    from journal_reader import date_from_journal_filename, journal_paths_sorted, load_jsonl_rows, parse_iso_date
    from kalshi_bridge import get_client
    from market_blend import fit_blend_weight, save_blend_meta
    from reporting_common import market_yes_no_result

    journal_paths = journal_paths_sorted()
    if not journal_paths:
        _warn(f"No trade journals found under {DATA_DIR}")
        raise typer.Exit(0)
    dates = sorted({d for p in journal_paths if (d := date_from_journal_filename(p))})
    start_d = parse_iso_date(start) if start else parse_iso_date(dates[0])
    end_d = parse_iso_date(end) if end else parse_iso_date(dates[-1])
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    _header(f"Fit market-blend weight — {start_d.strftime('%Y-%m-%d')} → {end_d.strftime('%Y-%m-%d')}")

    fill_rows: list[dict] = []
    for p in journal_paths:
        d = date_from_journal_filename(p)
        if not d:
            continue
        try:
            dd = parse_iso_date(d)
        except Exception:
            continue
        if dd < start_d or dd > end_d:
            continue
        for r in load_jsonl_rows(p):
            if r.get("note") == "fill" and r.get("order_id") and int(r.get("filled_contracts", 0) or 0) > 0:
                fill_rows.append(r)
    if not fill_rows:
        _warn("No fill rows found. Run reconcile first.")
        raise typer.Exit(0)

    client = get_client()
    by_ticker: dict[str, dict] = {}
    for r in fill_rows:
        t = str(r.get("ticker", ""))
        if t and t not in by_ticker:
            try:
                by_ticker[t] = client.get_market(t)
            except Exception:
                by_ticker[t] = {}

    rows: list[dict] = []
    for r in fill_rows:
        res = market_yes_no_result(by_ticker.get(str(r.get("ticker", "")), {}))
        if res not in {"yes", "no"}:
            continue
        side = str(r.get("side", "")).lower()
        p_cal = float(r.get("p_model_cal", 0.0) or 0.0)
        p = p_cal if p_cal > 0 else float(r.get("p_model", 0.0))
        bid, ask = float(r.get("book_bid", 0.0)), float(r.get("book_ask", 0.0))
        if not (0.0 < p < 1.0) or ask <= 0:
            continue
        mid = (bid + ask) / 2.0 if bid > 0 else ask
        rows.append({"p": p, "m": mid, "y": 1.0 if side == res else 0.0, "weight": float(r.get("filled_contracts", 1))})

    min_needed = int(min_rows) if min_rows is not None else int(MIN_BLEND_ROWS)
    if len(rows) < min_needed:
        _warn(f"Not enough resolved fills to fit blend ({len(rows)}<{min_needed}). Pass --min-rows to override.")
        raise typer.Exit(0)

    w, diag = fit_blend_weight(rows)
    console.print(f"Resolved fills          : {diag['n_rows']}")
    console.print(f"Log-loss market only    : {diag['logloss_market_only']:.4f}  (w=0)")
    console.print(f"Log-loss model only     : {diag['logloss_model_only']:.4f}  (w=1)")
    console.print(f"Log-loss best blend     : {diag['logloss_best']:.4f}  (w={w:.2f})")
    if dry_run:
        _warn("Dry run — blend_meta.json not written.")
        raise typer.Exit(0)
    save_blend_meta(w, diag, start=start_d.strftime("%Y-%m-%d"), end=end_d.strftime("%Y-%m-%d"))
    _success(f"Saved blend weight w={w:.2f} to models/blend_meta.json (scan applies it automatically).")


def fit_blend_segments(
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD (inclusive)."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD (inclusive)."),
    min_rows: int = typer.Option(None, "--min-rows", help="Minimum rows per segment to fit (default: MIN_BLEND_ROWS_SEGMENT)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fit and report, but do not write models/blend_meta_segments.json."),
):
    """
    Fit a market-blend weight ``w`` per |model-market| disagreement bucket.

    Uses full-slate snapshot scoring (like ``model-vs-market``), not fills:
    fills are edge-selected, so low-disagreement buckets would have near-zero
    fill coverage. Writes ``models/blend_meta_segments.json``, which ``scan``
    applies automatically per-bucket (floored at MIN_BLEND_WEIGHT), falling
    back to the global blend_meta.json weight for buckets under
    MIN_BLEND_ROWS_SEGMENT.
    """
    from datetime import timedelta

    from config import MIN_BLEND_ROWS_SEGMENT
    from journal_reader import parse_iso_date
    from market_blend import DISAGREEMENT_BUCKETS, disagreement_bucket, fit_blend_weight, save_segment_blend_meta
    from model_vs_market import evaluate_day

    start_d, end_d = parse_iso_date(start), parse_iso_date(end)
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    _header(f"Fit segment blend weights — {start} → {end}")
    _warn(
        "Saved model may have trained on these dates (look-ahead in the model's favor). "
        "Prefer dates after the model's training cutoff."
    )

    rows = []
    d = start_d
    while d <= end_d:
        ds = d.strftime("%Y-%m-%d")
        day_rows = evaluate_day(ds)
        if day_rows:
            console.print(f"[dim]{ds}: {len(day_rows)} scored markets[/dim]")
        rows.extend(day_rows)
        d += timedelta(days=1)
    if not rows:
        _warn("No scoreable markets (need snapshots + ETL'd boxscores for the range).")
        raise typer.Exit(0)

    min_needed = int(min_rows) if min_rows is not None else int(MIN_BLEND_ROWS_SEGMENT)
    by_bucket: dict[str, list] = {b: [] for b in DISAGREEMENT_BUCKETS}
    for r in rows:
        by_bucket[disagreement_bucket(r.p_model_cal, r.p_market_mid)].append(r)

    t = Table(title="Segment blend fit (by disagreement bucket)", box=box.SIMPLE_HEAD)
    t.add_column("Bucket")
    t.add_column("N", justify="right")
    t.add_column("LL market", justify="right")
    t.add_column("LL model", justify="right")
    t.add_column("w fit", justify="right")
    t.add_column("Status")

    segments: dict[str, tuple[float, dict]] = {}
    for bucket in DISAGREEMENT_BUCKETS:
        bucket_rows = by_bucket[bucket]
        n = len(bucket_rows)
        if n < min_needed:
            t.add_row(bucket, str(n), "-", "-", "-", f"[dim]skipped (<{min_needed})[/dim]")
            continue
        fit_rows = [{"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "weight": 1.0} for r in bucket_rows]
        w, diag = fit_blend_weight(fit_rows)
        segments[bucket] = (w, diag)
        t.add_row(
            bucket,
            str(n),
            f"{diag['logloss_market_only']:.4f}",
            f"{diag['logloss_model_only']:.4f}",
            f"{w:.2f}",
            "[green]fit[/green]",
        )
    console.print(t)

    if not segments:
        _warn(f"No segment cleared the {min_needed}-row minimum. Nothing to save.")
        raise typer.Exit(0)
    if dry_run:
        _warn("Dry run — blend_meta_segments.json not written.")
        raise typer.Exit(0)
    save_segment_blend_meta(segments, start=start_d.strftime("%Y-%m-%d"), end=end_d.strftime("%Y-%m-%d"))
    _success(
        f"Saved {len(segments)} segment weight(s) to models/blend_meta_segments.json "
        "(scan applies them automatically, floored at MIN_BLEND_WEIGHT)."
    )


def refit_blend(
    days: int = typer.Option(None, "--days", help="Lookback window in days (default: BLEND_REFIT_LOOKBACK_DAYS)."),
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Overrides --days."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: today minus BLEND_REFIT_LAG_DAYS."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fit and report, but do not write blend meta files."),
):
    """
    Re-fit the market-blend weight (global + per-disagreement-segment) from full-slate
    scoring over a trailing window, so ``w`` tracks the model's actual recent
    performance vs. the market instead of staying pinned wherever it was last set.

    Uses full-slate snapshot scoring (like ``model-vs-market``), not fills — fills are
    edge-selected and can't validate segments the strategy rarely trades. Meant to run
    on a recurring schedule (see scripts/cron_job.sh `refit-blend` job); each run
    overwrites models/blend_meta.json and models/blend_meta_segments.json with the
    latest fit, so a model that starts beating the market pulls its own weight back up
    over time, without requiring a manual re-fit or a hand-set floor.
    """
    from datetime import timedelta

    from config import BLEND_REFIT_LAG_DAYS, BLEND_REFIT_LOOKBACK_DAYS, MIN_BLEND_ROWS, MIN_BLEND_ROWS_SEGMENT
    from journal_reader import parse_iso_date
    from market_blend import (
        DISAGREEMENT_BUCKETS,
        disagreement_bucket,
        fit_blend_weight,
        save_blend_meta,
        save_segment_blend_meta,
    )
    from model_vs_market import evaluate_day

    end_d = parse_iso_date(end) if end else (datetime.now() - timedelta(days=BLEND_REFIT_LAG_DAYS))
    if start:
        start_d = parse_iso_date(start)
    else:
        lookback = int(days) if days is not None else BLEND_REFIT_LOOKBACK_DAYS
        start_d = end_d - timedelta(days=lookback)
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    range_start, range_end = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    _header(f"Refit market-blend weight — {range_start} → {range_end}")
    _warn(
        "Saved model may have trained on these dates (look-ahead in the model's favor). "
        "Prefer dates after the model's training cutoff."
    )

    rows = []
    d = start_d
    while d <= end_d:
        ds = d.strftime("%Y-%m-%d")
        day_rows = evaluate_day(ds)
        rows.extend(day_rows)
        d += timedelta(days=1)
    if not rows:
        _warn("No scoreable markets in the lookback window (need snapshots + ETL'd boxscores).")
        raise typer.Exit(0)

    t = Table(title="Refit — global + per-bucket", box=box.SIMPLE_HEAD)
    t.add_column("Slice")
    t.add_column("N", justify="right")
    t.add_column("LL market", justify="right")
    t.add_column("LL model", justify="right")
    t.add_column("w fit", justify="right")
    t.add_column("Status")

    global_w, global_diag = None, None
    if len(rows) >= MIN_BLEND_ROWS:
        all_fit_rows = [{"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "weight": 1.0} for r in rows]
        global_w, global_diag = fit_blend_weight(all_fit_rows)
        t.add_row(
            "all (global)",
            str(len(rows)),
            f"{global_diag['logloss_market_only']:.4f}",
            f"{global_diag['logloss_model_only']:.4f}",
            f"{global_w:.2f}",
            "[green]fit[/green]",
        )
    else:
        t.add_row("all (global)", str(len(rows)), "-", "-", "-", f"[dim]skipped (<{MIN_BLEND_ROWS})[/dim]")

    by_bucket: dict[str, list] = {b: [] for b in DISAGREEMENT_BUCKETS}
    for r in rows:
        by_bucket[disagreement_bucket(r.p_model_cal, r.p_market_mid)].append(r)

    segments: dict[str, tuple[float, dict]] = {}
    for bucket in DISAGREEMENT_BUCKETS:
        bucket_rows = by_bucket[bucket]
        n = len(bucket_rows)
        if n < MIN_BLEND_ROWS_SEGMENT:
            t.add_row(bucket, str(n), "-", "-", "-", f"[dim]skipped (<{MIN_BLEND_ROWS_SEGMENT})[/dim]")
            continue
        fit_rows = [{"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "weight": 1.0} for r in bucket_rows]
        w, diag = fit_blend_weight(fit_rows)
        segments[bucket] = (w, diag)
        t.add_row(
            bucket,
            str(n),
            f"{diag['logloss_market_only']:.4f}",
            f"{diag['logloss_model_only']:.4f}",
            f"{w:.2f}",
            "[green]fit[/green]",
        )
    console.print(t)

    if dry_run:
        _warn("Dry run — blend meta files not written.")
        raise typer.Exit(0)
    if global_w is None and not segments:
        _warn("Nothing cleared its row minimum — no files written.")
        raise typer.Exit(0)

    if global_w is not None:
        save_blend_meta(global_w, global_diag, start=range_start, end=range_end)
    if segments:
        save_segment_blend_meta(segments, start=range_start, end=range_end)

    msg = f"global w={global_w:.2f}" if global_w is not None else "global skipped (too few rows)"
    _success(f"Refit complete: {msg}, {len(segments)} segment(s) updated. scan applies both automatically.")


def model_vs_market(
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD (inclusive)."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD (inclusive)."),
    pit_train: bool = typer.Option(False, "--pit-train/--no-pit-train", help="Point-in-time model retraining (slow, no look-ahead)."),
    earliest: bool = typer.Option(False, "--earliest", help="Score at the earliest snapshot per ticker instead of the latest."),
):
    """
    Score model vs market probabilities on FULL snapshot slates (no trade filter).

    For every snapshotted market with a sane book and a boxscore outcome,
    compares calibrated model P(over) and the market yes-mid on log-loss /
    Brier, sliced by line, disagreement, and date — with a fitted blend
    weight per slice showing where (if anywhere) the model earns weight.
    """
    from datetime import timedelta

    from journal_reader import parse_iso_date
    from model_vs_market import evaluate_day, summarize

    start_d, end_d = parse_iso_date(start), parse_iso_date(end)
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    _header(f"Model vs Market — {start} → {end}")
    if not pit_train:
        _warn(
            "Saved model may have trained on these dates (look-ahead in the model's favor). "
            "A market win here is decisive; a model win should be re-checked with --pit-train."
        )

    rows = []
    d = start_d
    while d <= end_d:
        ds = d.strftime("%Y-%m-%d")
        day_rows = evaluate_day(ds, pit_train=pit_train, earliest=earliest)
        if day_rows:
            console.print(f"[dim]{ds}: {len(day_rows)} scored markets[/dim]")
        rows.extend(day_rows)
        d += timedelta(days=1)
    if not rows:
        _warn("No scoreable markets (need snapshots + ETL'd boxscores for the range).")
        raise typer.Exit(0)

    summary = summarize(rows)

    def _print_slice(title: str, data: dict[str, dict]) -> None:
        t = Table(title=title, box=box.SIMPLE_HEAD)
        t.add_column("Group")
        t.add_column("N", justify="right")
        t.add_column("Base", justify="right")
        t.add_column("LL model", justify="right")
        t.add_column("LL market", justify="right")
        t.add_column("ΔLL", justify="right")
        t.add_column("Brier mdl", justify="right")
        t.add_column("Brier mkt", justify="right")
        t.add_column("w fit", justify="right")
        for k, s in data.items():
            delta = s["ll_model"] - s["ll_market"]
            color = "green" if delta < 0 else "red"
            t.add_row(
                k,
                str(s["n"]),
                f"{s['base_rate']:.3f}",
                f"{s['ll_model']:.4f}",
                f"{s['ll_market']:.4f}",
                f"[{color}]{delta:+.4f}[/{color}]",
                f"{s['brier_model']:.4f}",
                f"{s['brier_market']:.4f}",
                f"{s['w_fit']:.2f}",
            )
        console.print(t)

    _print_slice("Overall (ΔLL < 0 ⇒ model beats market)", summary["overall"])
    _print_slice("By Kalshi line", summary["line"])
    _print_slice("By |model − market| disagreement", summary["disagreement"])
    _print_slice("By date", summary["date"])
    console.print(
        "[dim]w fit = blend weight minimizing log-loss within the slice "
        "(0 = market knows best, 1 = model knows best). Fitted in-slice — treat small-N slices as noise.[/dim]"
    )


def segment_report(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD. Default: 14 days ago."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD. Default: today."),
):
    """Segment health report: go/no-go verdict per (side, line, spread, edge) bucket."""
    from datetime import date, timedelta
    from segment_health import segment_health_for_range
    from kalshi_bridge import get_client

    today = date.today()
    end_d = end or today.strftime("%Y-%m-%d")
    start_d = start or (today - timedelta(days=14)).strftime("%Y-%m-%d")

    _header(f"Segment Health — {start_d} → {end_d}")
    try:
        client = get_client()
    except Exception:
        client = None

    report = segment_health_for_range(start_d, end_d, client=client)
    console.print(f"Recommendation: [bold]{'green' if report.recommendation == 'TRADE' else 'red'}]{report.recommendation}[/]")
    for reason in report.summary_reasons:
        console.print(f"  • {reason}")
    console.print()

    if report.segments:
        t = Table(title="Segment Verdicts", box=box.SIMPLE_HEAD)
        t.add_column("Segment", style="cyan")
        t.add_column("Status", justify="center")
        t.add_column("Orders", justify="right")
        t.add_column("Fills", justify="right")
        t.add_column("Fill%", justify="right")
        t.add_column("ROI%", justify="right")
        t.add_column("Avg CLV", justify="right")
        for v in report.segments:
            m = v.metrics
            fill_pct = f"{m.fill_rate * 100:.0f}%" if m.orders > 0 else "—"
            roi = f"{m.roi_pct:.1f}%" if m.fills > 0 else "—"
            clv = f"{m.avg_clv_per_contract:+.4f}" if m.fills > 0 else "—"
            color = "green" if v.status == "PASS" else ("yellow" if v.status == "INSUFFICIENT" else "red")
            t.add_row(m.segment_key, f"[{color}]{v.status}[/{color}]", str(m.orders), str(m.fills), fill_pct, roi, clv)
        console.print(t)


def snapshot(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    series_ticker: str = typer.Option("KXMLBTB", "--series-ticker"),
):
    """Capture a point-in-time snapshot of TB market prices for backtesting / CLV."""
    from datetime import date as _date
    from kalshi_bridge import get_client
    from market_snapshots import append_snapshots

    gd = game_date or _date.today().strftime("%Y-%m-%d")
    _header(f"Snapshot TB markets — {gd}")
    client = get_client()
    lines = client.get_total_bases_lines(gd, series_ticker=series_ticker)
    n = append_snapshots(gd, lines)
    _success(f"Saved {n} market snapshots for {gd}.")


def schedule_snapshots(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    interval: int = typer.Option(30, "--interval-minutes", help="Minutes between snapshots."),
    count: int = typer.Option(4, "--count", help="Number of snapshots to capture."),
    series_ticker: str = typer.Option("KXMLBTB", "--series-ticker"),
):
    """Capture repeated market snapshots at regular intervals (blocking)."""
    import time
    from datetime import date as _date
    from kalshi_bridge import get_client
    from market_snapshots import append_snapshots

    gd = game_date or _date.today().strftime("%Y-%m-%d")
    _header(f"Scheduled snapshots — {gd} × {count} every {interval}m")
    client = get_client()
    for i in range(count):
        lines = client.get_total_bases_lines(gd, series_ticker=series_ticker)
        n = append_snapshots(gd, lines)
        console.print(f"[{i+1}/{count}] Saved {n} snapshots.")
        if i < count - 1:
            time.sleep(interval * 60)
    _success(f"Done. {count} snapshot(s) captured for {gd}.")


def backtest(
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="End date YYYY-MM-DD."),
    bankroll: float = typer.Option(1000.0, "--bankroll"),
    pit_train: bool = typer.Option(False, "--pit-train/--no-pit-train", help="Point-in-time model retraining."),
):
    """Replay market snapshots against actual TB outcomes to estimate P&L."""
    from backtest import run_backtest_range, BacktestReport

    _header(f"Backtest — {start} → {end}")
    reports = run_backtest_range(start, end, bankroll=bankroll, pit_train=pit_train)
    if not reports:
        _warn("No backtest results (missing snapshots for date range).")
        raise typer.Exit(0)

    total_cost = sum(r.total_cost for r in reports)
    total_pnl = sum(r.total_pnl for r in reports)
    total_trades = sum(r.n_trades for r in reports)
    roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    t = Table(title=f"Backtest Summary ({len(reports)} days)", box=box.SIMPLE_HEAD)
    t.add_column("Date", style="cyan")
    t.add_column("Trades", justify="right")
    t.add_column("Cost $", justify="right")
    t.add_column("P&L $", justify="right")
    t.add_column("ROI%", justify="right")
    for r in sorted(reports, key=lambda x: x.game_date):
        day_roi = (r.total_pnl / r.total_cost * 100) if r.total_cost > 0 else 0.0
        color = "green" if r.total_pnl >= 0 else "red"
        t.add_row(r.game_date, str(r.n_trades), f"{r.total_cost:.2f}", f"[{color}]{r.total_pnl:+.2f}[/{color}]", f"{day_roi:.1f}%")
    console.print(t)
    roi_color = "green" if total_pnl >= 0 else "red"
    console.print(f"\nTotal: {total_trades} trades | Cost ${total_cost:.2f} | P&L [{roi_color}]{total_pnl:+.2f}[/{roi_color}] | ROI {roi:.1f}%")


def materialize_features(
    as_of_date: str = typer.Option(None, "--as-of", help="Only include rows strictly before this date (YYYY-MM-DD)."),
):
    """Persist batter_features to SQLite gold table for faster scan and point-in-time backtest replay."""
    from feature_store import materialize_feature_table
    _header("Materialize feature table")
    n = materialize_feature_table(as_of_date=as_of_date)
    _success(f"Materialized {n} feature rows to batter_features table.")

