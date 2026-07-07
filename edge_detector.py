"""
edge_detector.py (MLB TB)
------------------------
Detect +EV edges on Kalshi total bases markets and size trades.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config import (
    BLOCKED_SEGMENTS,
    EDGE_THRESHOLD,
    ENABLE_VPIN_GUARD,
    KELLY_FRACTION,
    MAKER_MODE,
    MAX_BET_PCT,
    MAX_REALISTIC_ASK,
    MIN_EV,
    MIN_P,
    MIN_LIMIT_PRICE,
    MIN_REALISTIC_ASK,
    MAX_YES_LINE,
    RISKY_BAND_LOW,
    RISKY_BAND_HIGH,
    RISKY_BAND_KELLY_MULT,
    TAIL_P_CUTOFF,
    TAIL_EDGE_MULT,
    USE_ISOTONIC_CALIBRATION,
    USE_OOF_CALIBRATION,
    USE_SEGMENTED_CALIBRATION,
    VPIN_MAX_TOXIC,
)
from fees import fee_per_contract
from market_blend import blend_probability, load_blend_weight
from probability_engine import ProbabilityResult
from kalshi_bridge import MarketLine, OrderResult, get_client
from execution_engine import ExecutionLedger, LedgerKey, suggest_limit_price
from trade_journal import TradeRow, append_row
from calibration import (
    SegmentedCalibratorBundle,
    load as load_calibrator,
    load_oof,
    load_segmented,
)
from identity_bridge import norm_player_name

log = logging.getLogger(__name__)

MAX_BID_ASK_SPREAD = 0.25
_CALIBRATOR = None
_OOF_CALIBRATOR = None
_SEGMENTED: SegmentedCalibratorBundle | None | bool = None


def reset_calibration_cache() -> None:
    """Clear the lazy isotonic calibrator cache (for tests / reload)."""
    global _CALIBRATOR, _OOF_CALIBRATOR, _SEGMENTED
    _CALIBRATOR = None
    _OOF_CALIBRATOR = None
    _SEGMENTED = None


def _get_oof_calibrator():
    global _OOF_CALIBRATOR
    if _OOF_CALIBRATOR is False:
        return None
    if _OOF_CALIBRATOR is None:
        try:
            _OOF_CALIBRATOR = load_oof() if USE_OOF_CALIBRATION else False
        except Exception:
            _OOF_CALIBRATOR = False
    if _OOF_CALIBRATOR is False or _OOF_CALIBRATOR is None:
        return None
    return _OOF_CALIBRATOR


def _get_segmented() -> SegmentedCalibratorBundle | None:
    global _SEGMENTED
    if _SEGMENTED is False:
        return None
    if _SEGMENTED is None:
        try:
            _SEGMENTED = load_segmented() if USE_SEGMENTED_CALIBRATION else False
        except Exception:
            _SEGMENTED = False
    if _SEGMENTED is False or _SEGMENTED is None:
        return None
    return _SEGMENTED


def _calibrate(p: float, *, line: float = 1.5, side: str = "yes", games_played: int = 0) -> float:
    global _CALIBRATOR
    if not USE_ISOTONIC_CALIBRATION:
        return float(p)
    # Live fill-based calibration is checked before OOF: OOF is fit on all
    # historical games, but live trades are a filtered subset (only where
    # edge clears threshold), so its calibration curve doesn't transfer to
    # the subset actually traded. OOF remains the fallback for cold starts.
    seg = _get_segmented()
    if seg is not None:
        try:
            return float(seg.transform(float(p), line=line, side=side, games_played=games_played))
        except Exception:
            pass
    oof = _get_oof_calibrator()
    if oof is not None:
        try:
            return float(oof.transform(float(p)))
        except Exception:
            pass
    if _CALIBRATOR is None:
        try:
            _CALIBRATOR = load_calibrator()
        except Exception:
            _CALIBRATOR = False  # sentinel: don't try again
    if not _CALIBRATOR:
        return float(p)
    try:
        return float(_CALIBRATOR.transform(float(p)))
    except Exception:
        return float(p)


def calibrate_over_under(
    p_over_raw: float,
    *,
    line: float = 1.5,
    games_played: int = 0,
) -> tuple[float, float]:
    """
    Calibrate P(over) only; P(under) = 1 - P(over) so YES/NO views stay mutually consistent.
    """
    p = float(p_over_raw)
    p = min(1.0 - 1e-6, max(1e-6, p))
    p_over_c = float(_calibrate(p, line=line, side="yes", games_played=games_played))
    p_over_c = min(1.0 - 1e-6, max(1e-6, p_over_c))
    return p_over_c, 1.0 - p_over_c


def fill_calibrated_probabilities(
    results: list[ProbabilityResult],
    *,
    games_played_by_player: dict[str, int] | None = None,
) -> None:
    """Set ``p_over_calibrated`` / ``p_under_calibrated`` once per row (avoids repeated isotonic work)."""
    gp_map = games_played_by_player or {}
    for r in results:
        gp = int(r.games_played if r.games_played is not None else gp_map.get(r.player_name, 0) or 0)
        r.games_played = gp
        oc, uc = calibrate_over_under(r.p_over, line=float(r.kalshi_line), games_played=gp)
        r.p_over_calibrated = oc
        r.p_under_calibrated = uc


@dataclass
class EdgeSignal:
    player_name: str
    player_id: int
    game_date: str
    ticker: str
    kalshi_line: float
    predicted_lambda: float
    p_model: float
    p_model_raw: float
    p_market: float
    edge: float
    ev: float
    kelly_f: float
    recommended_contracts: int
    recommended_side: str
    bet_dollars: float
    limit_price: float
    book_bid: float
    book_ask: float
    book_spread: float
    games_played: int = 0
    fee_per_contract: float = 0.0
    p_model_cal: float = 0.0  # calibrated, pre-blend side prob (input for fit-blend)


def is_blocked_segment(line: float, side: str) -> bool:
    """True when (line, side) is disabled via BLOCKED_SEGMENTS (e.g. '1.5:no')."""
    return (f"{float(line):g}", str(side).lower()) in BLOCKED_SEGMENTS


def quote_side_edge(
    p_side_cal: float,
    *,
    bid: float,
    ask: float,
    side: str,
) -> tuple[float, float, float, float]:
    """
    Blend the calibrated side prob toward the side's market mid, pick the limit
    price, and return the fee-adjusted edge vs that limit.

    Returns ``(p_blend, limit_price, fee_per_contract, edge)``.
    """
    bid, ask = float(bid), float(ask)
    # Empty book side ⇒ mid = ask/2 is noise; shrink toward the ask instead (conservative).
    mid = (bid + ask) / 2.0 if bid > 0 else ask
    p_blend = blend_probability(float(p_side_cal), mid, load_blend_weight())
    limit = suggest_limit_price(side=side, bid=bid, ask=ask, model_fair=p_blend, maker=MAKER_MODE)
    is_taker = limit >= ask - 1e-9
    fee = fee_per_contract(limit, is_taker=is_taker)
    edge = p_blend - limit - fee
    return p_blend, limit, fee, edge


def expected_pnl_per_contract_mean_var(p: float, limit_price: float, fee: float = 0.0) -> tuple[float, float]:
    """
    Mean and variance of profit (USD) for one long contract at limit L, win prob p.

    Win payoff (1-L-fee), lose payoff (-L-fee); the fee is a constant shift so
    the variance is unchanged from the fee-free Bernoulli two-point distribution.
    """
    p = float(min(1.0 - 1e-9, max(1e-9, p)))
    L = float(min(0.9999, max(1e-6, limit_price)))
    win = 1.0 - L
    lose = -L
    mean_gross = p * win + (1.0 - p) * lose
    ex2 = p * win * win + (1.0 - p) * lose * lose
    var = max(0.0, ex2 - mean_gross * mean_gross)
    return mean_gross - float(fee), var


def expected_pnl_usd(signal: EdgeSignal) -> float:
    """
    Expected profit in USD for the sized position: long N binary contracts at limit_price.

    Per contract E[profit] = p*(1-L) - (1-p)*L - fee = p - L - fee with p = P(win) for the chosen side.
    """
    n = int(signal.recommended_contracts)
    if n <= 0:
        return 0.0
    mu, _ = expected_pnl_per_contract_mean_var(
        float(signal.p_model), float(signal.limit_price), float(signal.fee_per_contract)
    )
    return float(n) * mu


def expected_pnl_usd_std(signal: EdgeSignal) -> float:
    """Std dev of total leg PnL (USD) assuming i.i.d. contracts at the same (p, limit)."""
    n = int(signal.recommended_contracts)
    if n <= 0:
        return 0.0
    _, v = expected_pnl_per_contract_mean_var(float(signal.p_model), float(signal.limit_price))
    return float(math.sqrt(max(0.0, n * v)))


def portfolio_expected_pnl_std(signals: list[EdgeSignal]) -> float:
    """
    Std dev of sum of leg PnLs if legs were independent (sqrt of sum of variances).

    Same-slate / player correlation typically widens the true distribution vs this.
    """
    tot_var = 0.0
    for s in signals:
        n = int(s.recommended_contracts)
        if n <= 0:
            continue
        _, v = expected_pnl_per_contract_mean_var(float(s.p_model), float(s.limit_price))
        tot_var += n * v
    return float(math.sqrt(max(0.0, tot_var)))


def fractional_kelly(p: float, b: float, fraction: float = KELLY_FRACTION) -> float:
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return round(max(0.0, f_full * fraction), 4)


def dollars_to_contracts(dollars: float, contract_price: float) -> int:
    """Whole contracts only; no minimum — sub-one-contract dollars map to 0."""
    if contract_price <= 0:
        return 0
    d = float(dollars)
    if d <= 0:
        return 0
    return int(d / contract_price)


def detect_edge(
    prob_result: ProbabilityResult,
    market_line: MarketLine,
    bankroll: float,
    edge_threshold: float = EDGE_THRESHOLD,
    max_spread: float = MAX_BID_ASK_SPREAD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
) -> Optional[EdgeSignal]:
    p_over_raw = float(prob_result.p_over)
    p_under_raw = float(prob_result.p_under)
    if prob_result.p_over_calibrated is not None and prob_result.p_under_calibrated is not None:
        p_over, p_under = float(prob_result.p_over_calibrated), float(prob_result.p_under_calibrated)
    else:
        gp = int(prob_result.games_played or 0)
        p_over, p_under = calibrate_over_under(
            prob_result.p_over, line=float(prob_result.kalshi_line), games_played=gp
        )

    # Blend toward the side's market mid, pick a limit, and measure the edge
    # net of the expected Kalshi fee at that limit (taker fee if it crosses).
    if not (MIN_REALISTIC_ASK <= float(market_line.yes_ask) <= MAX_REALISTIC_ASK):
        p_over_b, yes_limit, yes_fee, yes_edge = p_over, float(market_line.yes_ask), 0.0, -1.0
    else:
        p_over_b, yes_limit, yes_fee, yes_edge = quote_side_edge(
            p_over, bid=market_line.yes_bid, ask=market_line.yes_ask, side="yes"
        )
    if not (MIN_REALISTIC_ASK <= float(market_line.no_ask) <= MAX_REALISTIC_ASK):
        p_under_b, no_limit, no_fee, no_edge = p_under, float(market_line.no_ask), 0.0, -1.0
    else:
        p_under_b, no_limit, no_fee, no_edge = quote_side_edge(
            p_under, bid=market_line.no_bid, ask=market_line.no_ask, side="no"
        )

    def _thr(p: float) -> float:
        return edge_threshold * (tail_edge_mult if p < tail_p_cutoff else 1.0)

    best = None
    if (
        p_over_b >= min_p
        and yes_edge > _thr(p_over_b)
        and market_line.yes_spread <= max_spread
        and market_line.yes_ask >= MIN_LIMIT_PRICE
        and market_line.line <= MAX_YES_LINE
        and not is_blocked_segment(market_line.line, "yes")
    ):
        best = ("yes", p_over_b, p_over, yes_limit, yes_fee, yes_edge)
    if (
        p_under_b >= min_p
        and no_edge > _thr(p_under_b)
        and market_line.no_spread <= max_spread
        and market_line.no_ask >= MIN_LIMIT_PRICE
        and not is_blocked_segment(market_line.line, "no")
    ):
        cand = ("no", p_under_b, p_under, no_limit, no_fee, no_edge)
        if best is None or cand[5] > best[5]:
            best = cand
    if best is None:
        return None

    side, p, p_cal, limit_price, fee, edge = best
    p_raw_side = p_over_raw if side == "yes" else p_under_raw
    b = (1.0 - limit_price - fee) / limit_price if limit_price > 0 else 0.0
    ev = b * p - (1.0 - p)
    if ev <= float(MIN_EV) or ev > 2.0:
        return None

    kf = fractional_kelly(p, b)
    if RISKY_BAND_LOW <= p < RISKY_BAND_HIGH:
        kf *= RISKY_BAND_KELLY_MULT
    bet_dollars = min(kf * bankroll, MAX_BET_PCT * bankroll)
    contracts = dollars_to_contracts(bet_dollars, limit_price)
    if contracts <= 0:
        return None

    if side == "yes":
        c = float(market_line.yes_ask)
        book_bid, book_ask, book_spread = float(market_line.yes_bid), float(market_line.yes_ask), float(market_line.yes_spread)
    else:
        c = float(market_line.no_ask)
        book_bid, book_ask, book_spread = float(market_line.no_bid), float(market_line.no_ask), float(market_line.no_spread)

    return EdgeSignal(
        player_name=prob_result.player_name,
        player_id=prob_result.player_id,
        game_date=prob_result.game_date,
        ticker=market_line.ticker,
        kalshi_line=market_line.line,
        predicted_lambda=prob_result.predicted_lambda,
        p_model=p,
        p_model_raw=float(p_raw_side),
        p_market=c,
        edge=edge,
        ev=ev,
        kelly_f=kf,
        recommended_contracts=contracts,
        recommended_side=side,
        bet_dollars=round(bet_dollars, 2),
        limit_price=limit_price,
        book_bid=book_bid,
        book_ask=book_ask,
        book_spread=book_spread,
        games_played=int(prob_result.games_played or 0),
        fee_per_contract=float(fee),
        p_model_cal=float(p_cal),
    )


def scan_for_edges(
    prob_results: list[ProbabilityResult],
    market_lines: list[MarketLine],
    bankroll: float,
    edge_threshold: float = EDGE_THRESHOLD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
) -> list[EdgeSignal]:
    market_map = {(norm_player_name(ml.player_name), ml.line): ml for ml in market_lines}
    out: list[EdgeSignal] = []
    for pr in prob_results:
        ml = market_map.get((norm_player_name(pr.player_name), pr.kalshi_line))
        if not ml:
            continue
        sig = detect_edge(
            pr,
            ml,
            bankroll,
            edge_threshold=edge_threshold,
            min_p=min_p,
            tail_p_cutoff=tail_p_cutoff,
            tail_edge_mult=tail_edge_mult,
        )
        if sig:
            out.append(sig)
    out.sort(key=lambda s: s.ev, reverse=True)
    return out


def apply_flow_guard(signals: list[EdgeSignal], market_lines: list[MarketLine]) -> list[EdgeSignal]:
    """
    Drop edges on either side when VPIN proxy indicates toxic / informed flow.
    """
    if not signals:
        return []
    by_ticker = {ml.ticker: ml for ml in market_lines}
    out: list[EdgeSignal] = []
    for s in signals:
        ml = by_ticker.get(s.ticker)
        tox = float(ml.vpin_toxicity) if ml else 0.0
        if ENABLE_VPIN_GUARD and tox >= VPIN_MAX_TOXIC:
            log.info("Skipping %s edge on %s (VPIN toxicity=%.3f)", s.recommended_side.upper(), s.ticker, tox)
            continue
        out.append(s)
    return out


def execute_signals(
    signals: list[EdgeSignal],
    dry_run: bool = True,
    todays_tickers: Optional[set[str]] = None,
    cancel_stale: bool = True,
    replace_if_price_diff: float = 0.02,
) -> list[OrderResult]:
    client = get_client(force_mock=dry_run)
    ledger = ExecutionLedger()
    results: list[OrderResult] = []

    desired = {(s.ticker, s.recommended_side): s for s in signals}

    if not dry_run and cancel_stale and hasattr(client, "get_orders") and hasattr(client, "cancel_order"):
        try:
            open_orders = client.get_orders(status="resting", limit=200)
        except Exception as e:
            log.warning(f"Could not fetch open orders ({e}); skipping cancel/replace.")
            open_orders = []
        todays_tickers = todays_tickers or set()
        for o in open_orders:
            if todays_tickers and o.ticker not in todays_tickers:
                continue
            if (o.ticker, o.side) not in desired:
                client.cancel_order(o.order_id)

    for sig in signals:
        key = LedgerKey(game_date=sig.game_date, ticker=sig.ticker, side=sig.recommended_side)
        if dry_run and ledger.has(key):
            continue
        if not dry_run and ledger.has_successful_submit(key):
            continue

        book_bid, book_ask, book_spread = float(sig.book_bid), float(sig.book_ask), float(sig.book_spread)
        p_market = float(sig.p_market)

        expected_pnl = float(sig.edge) * int(sig.recommended_contracts)

        ledger.add_attempt(key, price=sig.limit_price, contracts=sig.recommended_contracts, dollars=sig.bet_dollars, note="pre-submit")
        append_row(
            sig.game_date,
            TradeRow(
                game_date=sig.game_date,
                ticker=sig.ticker,
                side=sig.recommended_side,
                action="buy",
                contracts=sig.recommended_contracts,
                limit_price=sig.limit_price,
                order_id="",
                player_name=sig.player_name,
                kalshi_line=sig.kalshi_line,
                games_played=int(sig.games_played),
                predicted_lambda=sig.predicted_lambda,
                p_model=sig.p_model,
                p_model_raw=float(sig.p_model_raw),
                p_model_cal=float(sig.p_model_cal),
                p_market=p_market,
                fee_per_contract=float(sig.fee_per_contract),
                edge=sig.edge,
                ev=sig.ev,
                expected_pnl=expected_pnl,
                book_bid=book_bid,
                book_ask=book_ask,
                book_spread=book_spread,
                note="pre-submit",
                success=None,
            ).to_dict(),
        )

        if not dry_run and hasattr(client, "get_orders") and hasattr(client, "cancel_order"):
            try:
                existing = [o for o in client.get_orders(status="resting", ticker=sig.ticker, limit=50) if o.side == sig.recommended_side]
            except Exception:
                existing = []
            for o in existing:
                if abs(o.price - sig.limit_price) >= replace_if_price_diff:
                    client.cancel_order(o.order_id)

        result = client.place_order(sig.ticker, sig.recommended_side, sig.recommended_contracts, sig.limit_price)
        if not result.success:
            log.warning("Order FAILED %s %s %s@%.2f: %s", sig.ticker, sig.recommended_side, sig.recommended_contracts, sig.limit_price, result.message)
        ledger.add_attempt(
            key,
            price=sig.limit_price,
            contracts=sig.recommended_contracts,
            dollars=sig.bet_dollars,
            note="post-submit",
            order_id=result.order_id,
            success=result.success,
        )
        append_row(
            sig.game_date,
            TradeRow(
                game_date=sig.game_date,
                ticker=sig.ticker,
                side=sig.recommended_side,
                action="buy",
                contracts=sig.recommended_contracts,
                limit_price=sig.limit_price,
                order_id=result.order_id,
                player_name=sig.player_name,
                kalshi_line=sig.kalshi_line,
                games_played=int(sig.games_played),
                predicted_lambda=sig.predicted_lambda,
                p_model=sig.p_model,
                p_model_raw=float(sig.p_model_raw),
                p_model_cal=float(sig.p_model_cal),
                p_market=p_market,
                fee_per_contract=float(sig.fee_per_contract),
                edge=sig.edge,
                ev=sig.ev,
                expected_pnl=expected_pnl,
                book_bid=book_bid,
                book_ask=book_ask,
                book_spread=book_spread,
                note="post-submit",
                success=result.success,
            ).to_dict(),
        )
        results.append(result)

    return results

