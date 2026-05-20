"""
Simultaneous portfolio sizing for correlated binary contracts.

Starts from per-signal fractional-Kelly dollars (as in edge_detector), then solves a
concave mean–variance program that penalizes correlated exposure via a QP (cvxpy).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from config import (
    MAX_PORTFOLIO_PCT,
    PORTFOLIO_RISK_AVERSION,
    SAME_SLATE_CORR,
    CROSS_SLATE_CORR,
    USE_CVX_PORTFOLIO,
)

if TYPE_CHECKING:
    from edge_detector import EdgeSignal

log = logging.getLogger(__name__)

try:
    import cvxpy as cp

    _HAS_CVX = True
except ImportError:
    cp = None  # type: ignore
    _HAS_CVX = False


def _return_std_per_dollar(p: float, price: float, side: str) -> float:
    side = (side or "").lower()
    c = float(np.clip(price, 1e-4, 0.9999))
    if side == "yes":
        win = (1.0 - c) / c
        lose = -1.0
    elif side == "no":
        win = (1.0 - c) / c
        lose = -1.0
    else:
        return 0.01
    mu = float(p * win + (1.0 - p) * lose)
    m2 = float(p * win * win + (1.0 - p) * lose * lose)
    return float(max(np.sqrt(max(m2 - mu * mu, 1e-10)), 1e-6))


def build_dollar_pnl_covariance(
    dollars: np.ndarray,
    p: np.ndarray,
    prices: np.ndarray,
    sides: list[str],
    game_dates: list[str],
    rho_same: float = SAME_SLATE_CORR,
    rho_cross: float = CROSS_SLATE_CORR,
) -> np.ndarray:
    n = len(dollars)
    sig = np.array(
        [_return_std_per_dollar(float(p[i]), float(prices[i]), str(sides[i])) for i in range(n)],
        dtype=float,
    )
    corr = np.full((n, n), rho_cross, dtype=float)
    np.fill_diagonal(corr, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            if game_dates[i] == game_dates[j]:
                corr[i, j] = corr[j, i] = rho_same
    scale = dollars * sig
    cov = corr * np.outer(scale, scale)
    cov = 0.5 * (cov + cov.T)
    cov.flat[:: n + 1] += 1e-8
    return cov


def resize_correlated_dollars(
    raw_dollars: list[float],
    p: list[float],
    prices: list[float],
    sides: list[str],
    game_dates: list[str],
    bankroll: float,
    max_portfolio_pct: float = MAX_PORTFOLIO_PCT,
    risk_aversion: float = PORTFOLIO_RISK_AVERSION,
    force_independent: bool = False,
) -> np.ndarray:
    """
    Input nonnegative raw dollar targets (e.g. independent Kelly). Returns resized dollars.
    """
    n = len(raw_dollars)
    if n == 0:
        return np.array([], dtype=float)
    w0 = np.maximum(np.asarray(raw_dollars, dtype=float), 0.0)
    cap = max(0.0, float(max_portfolio_pct) * float(bankroll))
    if w0.sum() <= 1e-9:
        return w0
    if w0.sum() > cap and cap > 0:
        w0 = w0 * (cap / w0.sum())

    if force_independent or not USE_CVX_PORTFOLIO or not _HAS_CVX or n == 1:
        return w0

    Sigma = build_dollar_pnl_covariance(w0, np.asarray(p), np.asarray(prices), sides, list(game_dates))
    w = cp.Variable(n)
    lam = float(max(risk_aversion, 1e-6))
    # Pull toward independent targets while penalizing correlated variance of PnL.
    obj = cp.Maximize(cp.sum(w) - (lam / 2.0) * cp.quad_form(w, Sigma))
    cons = [w >= 0, w <= w0, cp.sum(w) <= min(cap, float(np.sum(w0)))]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            log.warning("cvxpy solve failed (%s); keeping independent Kelly dollars.", e)
            return resize_correlated_dollars(
                raw_dollars=list(w0),
                p=p,
                prices=prices,
                sides=sides,
                game_dates=game_dates,
                bankroll=bankroll,
                max_portfolio_pct=max_portfolio_pct,
                risk_aversion=risk_aversion,
                force_independent=True,
            )
    if w.value is None or prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        log.warning("cvxpy status=%s; keeping independent Kelly dollars.", prob.status)
        return w0
    return np.maximum(np.asarray(w.value, dtype=float).ravel(), 0.0)


def resize_signals_portfolio(signals: list["EdgeSignal"], bankroll: float) -> None:
    """Mutate signals' bet_dollars and recommended_contracts in place."""
    from edge_detector import dollars_to_contracts

    if not signals:
        return
    raw = [float(s.bet_dollars) for s in signals]
    p = [float(s.p_model) for s in signals]
    prices = [float(s.limit_price) for s in signals]
    sides = [s.recommended_side for s in signals]
    dates = [s.game_date for s in signals]
    out = resize_correlated_dollars(raw, p, prices, sides, dates, bankroll)
    for s, d in zip(signals, out):
        s.bet_dollars = round(float(d), 2)
        s.recommended_contracts = dollars_to_contracts(s.bet_dollars, s.limit_price)
