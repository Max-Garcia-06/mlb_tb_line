"""
Ordinal (proportional-odds) logistic regression for discrete total bases.

Full probability mass over TB categories improves Under-side pricing vs mean + Poisson/NB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


# TB levels: 0,1,...,11 represent exact counts; 12 is the top bucket (TB >= 12).
NUM_TB_LEVELS = 13


def tb_values_to_levels(tb: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(tb, dtype=float).round().clip(0, 1e9)
    return np.minimum(arr.astype(int), NUM_TB_LEVELS - 1)


def prob_over_line_from_pmf(pmf: np.ndarray, line: float) -> float:
    """
    pmf[j] = P(TB = j) for j = 0..10, pmf[11] = P(TB = 11), pmf[12] = P(TB >= 12).
    Kalshi-style over on strike k: P(TB > k) with half-integer lines.
    """
    pmf = np.asarray(pmf, dtype=float).reshape(-1)
    if pmf.shape[0] != NUM_TB_LEVELS:
        raise ValueError(f"pmf must have length {NUM_TB_LEVELS}, got {pmf.shape[0]}")
    pmf = pmf / max(pmf.sum(), 1e-12)
    floor_k = int(math.floor(float(line)))
    # P(TB > k) = sum_{tb > k} — for integer TB, strict exceed of 1.5 means TB >= 2.
    start = floor_k + 1
    if start >= NUM_TB_LEVELS:
        return 0.0
    return float(pmf[start:].sum())


def fit_ordinal_logit(X: pd.DataFrame, y_tb: np.ndarray, disp: bool = False) -> Any:
    # statsmodels requires endog and exog indices to match; X often keeps non-contiguous df index.
    X_fit = X.astype(float).reset_index(drop=True)
    y_levels = tb_values_to_levels(y_tb)
    y_series = pd.Series(np.asarray(y_levels, dtype=int)).reset_index(drop=True)
    model = OrderedModel(y_series, X_fit, distr="logit")
    return model.fit(method="bfgs", disp=disp, maxiter=200)


def predict_pmf(result: Any, X: pd.DataFrame) -> np.ndarray:
    """Rows of shape (n, NUM_TB_LEVELS)."""
    Xp = X.astype(float).reset_index(drop=True)
    probs = result.model.predict(result.params, exog=Xp, which="prob")
    return np.asarray(probs, dtype=float)


def expected_tb_from_pmf(pmf: np.ndarray) -> float:
    pmf = np.asarray(pmf, dtype=float).reshape(-1)
    pmf = pmf / max(pmf.sum(), 1e-12)
    # Levels 0..11 = TB equals that count; level 12 = TB >= 12
    ev = float(sum(j * pmf[j] for j in range(12)) + 13.0 * pmf[12])
    return ev


@dataclass
class OrdinalModelBundle:
    result: Any
    feature_names: list[str]

    def predict_pmf_row(self, row: dict | pd.Series) -> np.ndarray:
        if isinstance(row, dict):
            X = pd.DataFrame([row])
        else:
            X = pd.DataFrame([row.to_dict()])
        X = X[self.feature_names].astype(float)
        return predict_pmf(self.result, X)[0]
