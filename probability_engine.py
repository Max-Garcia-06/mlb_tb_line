"""
probability_engine.py (MLB TB)
------------------------------
Converts predicted mean TB (λ) into P(TB > k).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, Sequence, TypedDict

import numpy as np
from scipy.stats import poisson, nbinom

from config import DISTRIBUTION
from ordinal_core import prob_over_line_from_pmf


def _nb_params(mu: float, var: float) -> tuple[float, float]:
    var = max(var, mu + 1e-6)
    p = mu / var
    p = min(max(p, 1e-6), 1 - 1e-6)
    r = mu * p / (1 - p)
    r = max(r, 1e-3)
    return r, p


def prob_exceed_poisson(lam: float, k: float) -> float:
    floor_k = int(math.floor(k))
    return float(1.0 - poisson.cdf(floor_k, mu=lam))


def prob_exceed_nbinom(lam: float, k: float, variance: float) -> float:
    r, p = _nb_params(lam, variance)
    floor_k = int(math.floor(k))
    return float(1.0 - nbinom.cdf(floor_k, n=r, p=p))


class PredictionInput(TypedDict, total=False):
    """One row passed to ``calculate_probabilities`` (live scan or diagnostics)."""

    player_id: int
    player_name: str
    game_date: str
    kalshi_line: float
    predicted_lambda: float
    tb_pmf: NotRequired[list[float]]


def prob_exceed(
    lam: float,
    k: float,
    variance: float,
    distribution: Literal["poisson", "nbinom"] = DISTRIBUTION,
) -> float:
    lam = max(lam, 0.01)
    if distribution == "poisson":
        return prob_exceed_poisson(lam, k)
    return prob_exceed_nbinom(lam, k, variance)


@dataclass
class ProbabilityResult:
    player_id: int
    player_name: str
    game_date: str
    kalshi_line: float
    predicted_lambda: float
    p_over: float
    p_under: float
    distribution: str
    variance: float
    pmf: tuple[float, ...] | None = None
    p_over_calibrated: float | None = None
    p_under_calibrated: float | None = None


def calculate_probabilities(
    predictions: Sequence[PredictionInput | dict[str, Any]],
    variance: float,
    distribution: str = DISTRIBUTION,
) -> list[ProbabilityResult]:
    results = []
    for pred in predictions:
        lam = float(pred["predicted_lambda"])
        k = float(pred["kalshi_line"])
        pmf_arr = pred.get("tb_pmf")
        if pmf_arr is not None:
            p_over = prob_over_line_from_pmf(np.asarray(pmf_arr, dtype=float), k)
            dist = "ordinal"
            var_eff = float(variance)
        else:
            p_over = prob_exceed(lam, k, variance, distribution)
            dist = distribution
            var_eff = float(variance)
        results.append(
            ProbabilityResult(
                player_id=int(pred.get("player_id", 0)),
                player_name=str(pred.get("player_name", "")),
                game_date=str(pred.get("game_date", "")),
                kalshi_line=k,
                predicted_lambda=lam,
                p_over=float(p_over),
                p_under=float(1.0 - p_over),
                distribution=dist,
                variance=var_eff,
                pmf=tuple(float(x) for x in pmf_arr) if pmf_arr is not None else None,
            )
        )
    return results


def brier_score(actuals: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - actuals) ** 2))

