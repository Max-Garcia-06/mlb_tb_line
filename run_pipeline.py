"""
run_pipeline.py (MLB TB)
-----------------------
CLI orchestrator for the MLB Total Bases pipeline.
"""

import logging
from datetime import datetime

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, help="MLB Total Bases Edge Pipeline")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _header(title: str):
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))


def _warn(msg: str):
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def _success(msg: str):
    console.print(f"[bold green]✓[/bold green] {msg}")


@app.command()
def etl(seasons: str = typer.Option(None, "--seasons", help="Comma-separated seasons (years), e.g. '2023,2024'")):
    _header("Phase 1 — ETL: Pull MLB Batter Games")
    import data_engine as de
    from config import SEASONS, DB_PATH

    season_list = [int(s.strip()) for s in seasons.split(",")] if seasons else SEASONS
    console.print(f"Seasons  : {season_list}")
    console.print(f"Database : {DB_PATH}\n")
    de.build_historical_store(season_list)
    _success("ETL complete.")


@app.command()
def train():
    _header("Phase 2 — Model Training")
    import model as m

    mdl, meta = m.train(save=True)
    _success(f"Model saved. Train rows: {meta['train_rows']:,}")
    console.print(f"  Residual σ  : {meta['residual_std']:.3f}")
    console.print(f"  Residual var: {meta['residual_var']:.3f}\n")

    fi = m.get_feature_importance(mdl)
    t = Table(title="Feature Importance (gain)", box=box.SIMPLE)
    t.add_column("Feature", style="cyan")
    t.add_column("Importance", justify="right")
    for _, row in fi.head(12).iterrows():
        t.add_row(row["feature"], f"{row['importance']:.1f}")
    console.print(t)


@app.command()
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


@app.command()
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


@app.command()
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
    mark_delays: str = typer.Option("30,120", "--mark-delays", help="Comma-separated minutes after scan to record marks, e.g. '30,120'."),
):
    from config import EDGE_THRESHOLD, MAX_YES_LINE, MIN_LIMIT_PRICE, MIN_P, TAIL_P_CUTOFF, TAIL_EDGE_MULT
    from model import load_model, predict_lambda
    from feature_store import build_feature_table, MODEL_FEATURES
    from probability_engine import calculate_probabilities
    from edge_detector import MAX_BID_ASK_SPREAD, _calibrate, dollars_to_contracts, execute_signals, scan_for_edges
    from kalshi_bridge import get_client

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

    client = get_client()
    market_lines = client.get_total_bases_lines(game_date, series_ticker=series_ticker)
    if not market_lines:
        _warn(f"No total bases markets found for date {game_date} (series={series_ticker}).")
        raise typer.Exit(0)
    console.print(f"Found {len(market_lines)} total bases markets on Kalshi\n")

    try:
        feat_df = build_feature_table()
    except Exception as e:
        _warn(f"Could not load feature table ({e}). Using λ fallback.")
        feat_df = None

    predictions = []
    for ml in market_lines:
        if feat_df is not None:
            player_rows = feat_df[feat_df["player_name"].str.lower() == ml.player_name.lower()] if "player_name" in feat_df.columns else feat_df[feat_df["player_name"].astype(str).str.lower() == ml.player_name.lower()]  # noqa
            if not player_rows.empty:
                latest = player_rows.sort_values("game_date").iloc[-1]
                row_features = latest[MODEL_FEATURES].fillna(0).to_dict()
                lam = float(predict_lambda(row_features, trained_model))
            else:
                lam = ml.line * ml.implied_prob / 0.5
        else:
            lam = ml.line * ml.implied_prob / 0.5

        predictions.append(
            {
                "player_id": ml.player_id,
                "player_name": ml.player_name,
                "game_date": ml.game_date,
                "kalshi_line": ml.line,
                "predicted_lambda": float(lam),
            }
        )

    prob_results = calculate_probabilities(predictions, variance)

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
        """Color edge vs ask using the same pre-EV gates as detect_edge (spread/ask/min_p/threshold; not EV)."""
        need = _edge_thr_for_p(p_side)
        spread_ok = spread <= MAX_BID_ASK_SPREAD
        ask_ok = ask >= MIN_LIMIT_PRICE
        line_ok = float(kalshi_line) <= float(MAX_YES_LINE) if side == "yes" else True
        min_ok = p_side >= min_p_eff
        edge_ok = e > need and spread_ok and ask_ok and line_ok
        color = "green" if edge_ok and min_ok else ("yellow" if e > 0 else "red")
        return f"[{color}]{e:+.3f}[/{color}]"

    console.print(
        "[dim]Legend: Pr/Pc/Pu = P(over) raw / calibrated / P(under) cal; eY/eN = cal edge vs YES ask / NO ask "
        f"(same as signals; max spread {MAX_BID_ASK_SPREAD:.2f}); Ymid = YES mid only. "
        "Green eY/eN = passes min_p + edge vs threshold + spread + ask (+ YES line cap for eY), not EV.[/dim]\n"
    )
    t = Table(title=f"TB vs Kalshi — {game_date}", box=box.SIMPLE_HEAD)
    t.add_column("Player", style="cyan", min_width=14, overflow="ellipsis")
    t.add_column("Ln", justify="right", min_width=3)
    t.add_column("lam", justify="right", min_width=4)
    t.add_column("Pr", justify="right", min_width=5)
    t.add_column("Pc", justify="right", min_width=5)
    t.add_column("Pu", justify="right", min_width=5)
    t.add_column("Y\nmid", justify="right", min_width=5)
    t.add_column("Y\nask", justify="right", min_width=5)
    t.add_column("eY", justify="right", min_width=6)
    t.add_column("Ysp", justify="right", min_width=4)
    t.add_column("N\nask", justify="right", min_width=5)
    t.add_column("eN", justify="right", min_width=6)
    t.add_column("Nsp", justify="right", min_width=4)

    for pr, ml in zip(prob_results, market_lines):
        p_or = float(pr.p_over)
        p_ur = float(pr.p_under)
        p_oc = float(_calibrate(p_or))
        p_uc = float(_calibrate(p_ur))
        yes_edge = p_oc - float(ml.yes_ask)
        no_edge = p_uc - float(ml.no_ask)
        eY_cell = _fmt_edge_cell(
            yes_edge, p_oc, float(ml.yes_ask), float(ml.yes_spread), side="yes", kalshi_line=float(ml.line)
        )
        eN_cell = _fmt_edge_cell(
            no_edge, p_uc, float(ml.no_ask), float(ml.no_spread), side="no", kalshi_line=float(ml.line)
        )
        t.add_row(
            pr.player_name,
            f"{ml.line:g}",
            f"{pr.predicted_lambda:.2f}",
            f"{p_or:.3f}",
            f"{p_oc:.3f}",
            f"{p_uc:.3f}",
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

    if max_contracts:
        for s in signals:
            s.recommended_contracts = min(s.recommended_contracts, max_contracts)

    total_raw = sum(s.bet_dollars for s in signals)
    if total_raw > bankroll:
        scale = bankroll / total_raw
        for s in signals:
            s.bet_dollars = round(s.bet_dollars * scale, 2)
            s.recommended_contracts = dollars_to_contracts(s.bet_dollars, s.limit_price)

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
    for s in signals:
        sig_table.add_row(
            s.player_name,
            str(s.kalshi_line),
            f"[green]{s.recommended_side.upper()}[/green]",
            f"[green]{s.edge:+.3f}[/green]",
            f"{s.ev:.3f}",
            str(s.recommended_contracts),
            f"{s.limit_price:.2f}",
            f"${s.bet_dollars:.2f}",
        )
    console.print(sig_table)

    if not dry_run:
        console.print("\n[bold red]LIVE MODE — placing orders...[/bold red]")
        todays_tickers = {ml.ticker for ml in market_lines}
        results = execute_signals(signals, dry_run=False, todays_tickers=todays_tickers, cancel_stale=True)
        if auto_mark and results:
            import subprocess
            import sys

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
                for d in delays:
                    label = f"{d}m"
                    # Spawn a background process that sleeps then records a mark snapshot.
                    cmd = [
                        sys.executable,
                        "-c",
                        (
                            "import time,subprocess,sys;"
                            f"time.sleep({d}*60);"
                            "subprocess.run([sys.executable,'run_pipeline.py','mark','--date',"
                            f"'{game_date}','--label','{label}'], check=False)"
                        ),
                    ]
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                console.print(f"\n[dim]Auto-mark scheduled at: {', '.join(f'{d}m' for d in delays)}[/dim]")
    else:
        _warn("Dry run — no orders placed. Use --live to execute.")


@app.command()
def mark(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    label: str = typer.Option("30m", "--label", help="Mark label, e.g. '30m', '120m'."),
):
    """
    Snapshot current market prices for placed orders and write `note=mark` rows.
    Used for CLV / mark-to-market diagnostics.
    """
    import json

    from kalshi_bridge import get_client
    from trade_journal import TradeRow, append_row, journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Mark snapshot — {game_date} ({label})")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    placed = [r for r in rows if r.get("note") == "post-submit" and r.get("success") is True and r.get("order_id")]
    if not placed:
        _warn("No placed orders found to mark.")
        raise typer.Exit(0)

    existing = {(str(r.get("order_id")), str(r.get("mark_label"))) for r in rows if r.get("note") == "mark" and r.get("order_id")}

    client = get_client()

    def _px(m: dict, dollars_key: str, cents_key: str) -> float:
        if m.get(dollars_key) is not None:
            try:
                return float(m[dollars_key])
            except Exception:
                pass
        if m.get(cents_key) is not None:
            try:
                return float(m[cents_key]) / 100.0
            except Exception:
                pass
        return 0.0

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

        yes_bid = _px(m, "yes_bid_dollars", "yes_bid")
        yes_ask = _px(m, "yes_ask_dollars", "yes_ask")
        no_bid = _px(m, "no_bid_dollars", "no_bid")
        no_ask = _px(m, "no_ask_dollars", "no_ask")
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
                p_market=float(r.get("p_market", 0.0) or 0.0),
                edge=float(r.get("edge", 0.0) or 0.0),
                ev=float(r.get("ev", 0.0) or 0.0),
                expected_pnl=float(r.get("expected_pnl", 0.0) or 0.0),
                book_bid=float(r.get("book_bid", 0.0) or 0.0),
                book_ask=float(r.get("book_ask", 0.0) or 0.0),
                book_spread=float(r.get("book_spread", 0.0) or 0.0),
                mark_label=str(label),
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

@app.command()
def report(game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today.")):
    import json
    from collections import defaultdict

    from kalshi_bridge import get_client
    from trade_journal import journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Report — {game_date}")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    placed = [r for r in rows if r.get("note") == "post-submit" and r.get("success") is True]
    if not placed:
        _warn("No successful placed orders in journal for this date.")
        raise typer.Exit(0)

    # Fill reconciliation rows (note="fill") override cost/position sizing.
    fills: dict[str, dict] = {}
    for r in rows:
        if r.get("note") == "fill" and r.get("order_id"):
            fills[str(r["order_id"])] = r

    # Mark-to-market rows for CLV diagnostics.
    marks: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("note") == "mark" and r.get("order_id"):
            marks[(str(r["order_id"]), str(r.get("mark_label", "")))] = r

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
        return (by_ticker.get(ticker, {}).get("result") or "").lower()

    def pnl_per_contract(side: str, price: float, result: str) -> float | None:
        if result not in {"yes", "no"}:
            return None
        return (1.0 - price) if side == result else (-price)

    total_cost = 0.0
    total_realized = 0.0
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

    def price_bucket(price: float) -> str:
        if price < 0.20:
            return "<0.20"
        if price < 0.50:
            return "0.20-0.49"
        if price < 0.80:
            return "0.50-0.79"
        return ">=0.80"

    def edge_bucket(edge: float) -> str:
        if edge < 0.05:
            return "<0.05"
        if edge < 0.10:
            return "0.05-0.09"
        if edge < 0.15:
            return "0.10-0.14"
        if edge < 0.20:
            return "0.15-0.19"
        return ">=0.20"

    def spread_bucket(spread: float) -> str:
        if spread <= 0:
            return "n/a"
        if spread < 0.05:
            return "<0.05"
        if spread < 0.10:
            return "0.05-0.09"
        if spread < 0.20:
            return "0.10-0.19"
        return ">=0.20"

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


@app.command("report-range")
def report_range(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Default: earliest journal found."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: latest journal found."),
    per_day: bool = typer.Option(True, "--per-day/--no-per-day", help="Print a per-day summary table in addition to combined totals."),
):
    """
    Aggregate trade performance across multiple days by scanning `data/trades_YYYY-MM-DD.jsonl` journals.
    """
    import json
    from collections import defaultdict

    from config import DATA_DIR
    from kalshi_bridge import get_client

    def _parse_date(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    def _date_from_path(p) -> str | None:
        name = p.name
        if not (name.startswith("trades_") and name.endswith(".jsonl")):
            return None
        return name[len("trades_") : -len(".jsonl")]

    journal_paths = sorted(DATA_DIR.glob("trades_*.jsonl"))
    if not journal_paths:
        _warn(f"No trade journals found under {DATA_DIR}")
        raise typer.Exit(0)

    # Determine date window defaults from available journals.
    dates = []
    for p in journal_paths:
        d = _date_from_path(p)
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

    start_d = _parse_date(start) if start else _parse_date(dates[0])
    end_d = _parse_date(end) if end else _parse_date(dates[-1])
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    # Load placed orders across window.
    placed = []
    fill_rows: list[dict] = []
    mark_rows: list[dict] = []
    for p in journal_paths:
        d = _date_from_path(p)
        if not d:
            continue
        try:
            dd = _parse_date(d)
        except Exception:
            continue
        if dd < start_d or dd > end_d:
            continue

        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("note") == "post-submit" and r.get("success") is True:
                r = dict(r)
                r.setdefault("game_date", d)
                placed.append(r)
            if r.get("note") == "fill" and r.get("order_id"):
                rr = dict(r)
                rr.setdefault("game_date", d)
                fill_rows.append(rr)
            if r.get("note") == "mark" and r.get("order_id"):
                rr = dict(r)
                rr.setdefault("game_date", d)
                mark_rows.append(rr)

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
        return (by_ticker.get(ticker, {}).get("result") or "").lower()

    def pnl_per_contract(side: str, price: float, result: str) -> float | None:
        if result not in {"yes", "no"}:
            return None
        return (1.0 - price) if side == result else (-price)

    total_cost = 0.0
    total_realized = 0.0
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

    def price_bucket(price: float) -> str:
        if price < 0.20:
            return "<0.20"
        if price < 0.50:
            return "0.20-0.49"
        if price < 0.80:
            return "0.50-0.79"
        return ">=0.80"

    def edge_bucket(edge: float) -> str:
        if edge < 0.05:
            return "<0.05"
        if edge < 0.10:
            return "0.05-0.09"
        if edge < 0.15:
            return "0.10-0.14"
        if edge < 0.20:
            return "0.15-0.19"
        return ">=0.20"

    def spread_bucket(spread: float) -> str:
        if spread <= 0:
            return "n/a"
        if spread < 0.05:
            return "<0.05"
        if spread < 0.10:
            return "0.05-0.09"
        if spread < 0.20:
            return "0.10-0.19"
        return ">=0.20"

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


@app.command()
def reconcile(
    game_date: str = typer.Option(None, "--date", help="Game date YYYY-MM-DD. Defaults to today."),
    include_resting: bool = typer.Option(False, "--include-resting", help="Also write fill rows for currently resting orders (0 filled)."),
):
    """
    Reconcile journaled orders with Kalshi order status and write `note=fill` rows.

    This prevents `report` from counting resting/unfilled orders as if they were positions.
    """
    import json

    from kalshi_bridge import get_client
    from trade_journal import TradeRow, append_row, journal_path

    game_date = game_date or datetime.today().strftime("%Y-%m-%d")
    path = journal_path(game_date)
    _header(f"Reconcile fills — {game_date}")

    if not path.exists():
        _warn(f"No trade journal found at {path}")
        raise typer.Exit(0)

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    placed = [r for r in rows if r.get("note") == "post-submit" and r.get("success") is True and r.get("order_id")]
    if not placed:
        _warn("No placed orders with order_id found to reconcile.")
        raise typer.Exit(0)

    existing_fill_ids = {str(r.get("order_id")) for r in rows if r.get("note") == "fill" and r.get("order_id")}

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


@app.command()
def calibrate(
    start: str = typer.Option(None, "--start", help="Start date YYYY-MM-DD (inclusive). Default: earliest journal found."),
    end: str = typer.Option(None, "--end", help="End date YYYY-MM-DD (inclusive). Default: latest journal found."),
    min_rows: int = typer.Option(50, "--min-rows", help="Minimum resolved filled trades required to fit calibration."),
):
    """
    Fit an isotonic probability calibrator from reconciled filled trades.

    Uses journal `note=fill` to determine filled size, and Kalshi market result to determine win/loss.
    """
    import json

    import numpy as np

    from calibration import fit_isotonic, save
    from config import DATA_DIR
    from kalshi_bridge import get_client

    def _parse_date(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    def _date_from_path(p) -> str | None:
        name = p.name
        if not (name.startswith("trades_") and name.endswith(".jsonl")):
            return None
        return name[len("trades_") : -len(".jsonl")]

    journal_paths = sorted(DATA_DIR.glob("trades_*.jsonl"))
    if not journal_paths:
        _warn(f"No trade journals found under {DATA_DIR}")
        raise typer.Exit(0)

    dates = sorted({d for p in journal_paths if (d := _date_from_path(p))})
    if not dates:
        _warn(f"No valid journal filenames under {DATA_DIR} (expected trades_YYYY-MM-DD.jsonl)")
        raise typer.Exit(0)

    start_d = _parse_date(start) if start else _parse_date(dates[0])
    end_d = _parse_date(end) if end else _parse_date(dates[-1])
    if end_d < start_d:
        _warn("--end must be >= --start")
        raise typer.Exit(2)

    _header(f"Calibrate p_model — {start_d.strftime('%Y-%m-%d')} → {end_d.strftime('%Y-%m-%d')}")

    # Load fill rows within window
    fill_rows = []
    for p in journal_paths:
        d = _date_from_path(p)
        if not d:
            continue
        try:
            dd = _parse_date(d)
        except Exception:
            continue
        if dd < start_d or dd > end_d:
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
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
        return (by_ticker.get(ticker, {}).get("result") or "").lower()

    ps = []
    ys = []
    w = []
    for r in fill_rows:
        ticker = str(r.get("ticker", ""))
        res = outcome_for(ticker)
        if res not in {"yes", "no"}:
            continue
        side = str(r.get("side", "")).lower()
        y = 1.0 if side == res else 0.0
        p_model = float(r.get("p_model", 0.0))
        filled = int(r.get("filled_contracts", 0) or 0)
        if filled <= 0:
            continue
        ps.append(p_model)
        ys.append(y)
        w.append(float(filled))

    if len(ps) < min_rows:
        _warn(f"Not enough resolved filled trades to calibrate ({len(ps)}<{min_rows}). Try a wider date range or pass --min-rows lower (risk: overfitting).")
        raise typer.Exit(0)

    # Weighted isotonic isn't directly exposed in sklearn's IsotonicRegression via fit args in all versions,
    # so approximate by repeating probabilities (cap to keep it reasonable).
    rep_ps = []
    rep_ys = []
    for p0, y0, w0 in zip(ps, ys, w):
        reps = int(min(20, max(1, round(w0 / 2))))
        rep_ps.extend([p0] * reps)
        rep_ys.extend([y0] * reps)

    cal = fit_isotonic(np.array(rep_ps, dtype=float), np.array(rep_ys, dtype=float))
    save(cal)
    _success(f"Saved calibrator to models/ (isotonic). Rows used: {len(rep_ps)} (from {len(ps)} trades)")


if __name__ == "__main__":
    app()

