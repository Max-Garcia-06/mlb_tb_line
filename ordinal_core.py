"""
Ordinal (proportional-odds) logistic regression for discrete total bases.

Full probability mass over TB categories improves Under-side pricing vs mean + Poisson/NB.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


log = logging.getLogger(__name__)


# TB levels: 0,1,...,11 represent exact counts; 12 is the top bucket (TB >= 12).
NUM_TB_LEVELS = 13

# Variance below this counts as "constant" for statsmodels' k_constant check.
_CONSTANT_COL_TOL = 1e-12

# Marker attribute on a fitted OrderedModelResults that records the surviving
# exog columns (after constant-column pruning).  Inference uses this to keep
# the prediction matrix column-aligned with the fit-time matrix.
_KEPT_COLS_ATTR = "_kept_feature_cols"


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


def _drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop zero-variance columns that would otherwise trip statsmodels'
    ``OrderedModel`` k_constant guard ("There should not be a constant in the
    model").  Returns the trimmed frame and the list of dropped column names.

    Robust to NaN-filled columns: NaNs are treated as a distinct value via
    ``nunique(dropna=False)`` so a column that is entirely NaN is also dropped.
    """
    if X.shape[1] == 0:
        return X, []
    # ptp() is fast and matches statsmodels' detection for purely numeric exog;
    # fall back to nunique for columns with NaNs that ptp can't summarise.
    nunique = X.nunique(dropna=False)
    candidate_const = [str(c) for c, n in nunique.items() if int(n) <= 1]
    if not candidate_const:
        # Tighter variance check guards against pathological floats that compare
        # unique-but-numerically-identical (e.g. 0.0 vs -0.0).
        try:
            variances = X.var(axis=0, ddof=0, numeric_only=True)
            near_const = [str(c) for c, v in variances.items() if float(v) <= _CONSTANT_COL_TOL]
        except (TypeError, ValueError):
            near_const = []
        candidate_const = near_const
    if not candidate_const:
        return X, []
    keep = [c for c in X.columns if c not in candidate_const]
    return X[keep].copy(), candidate_const


def fit_ordinal_logit(X: pd.DataFrame, y_tb: np.ndarray, disp: bool = False) -> Any:
    """
    Fit a proportional-odds ordinal logit on engineered features.

    Constant (zero-variance) columns are silently dropped before the fit, with
    the surviving column list stamped on the returned result as
    ``_kept_feature_cols``.  Callers should pass the *same* DataFrame schema at
    prediction time; ``predict_pmf`` will re-subset to ``_kept_feature_cols``.
    """
    # statsmodels requires endog and exog indices to match; X often keeps non-contiguous df index.
    X_fit = X.astype(float).reset_index(drop=True)
    X_fit, dropped = _drop_constant_columns(X_fit)
    if dropped:
        log.warning(
            "fit_ordinal_logit: dropping %d zero-variance feature(s) before OrderedModel fit: %s",
            len(dropped),
            dropped,
        )
    if X_fit.shape[1] == 0:
        raise ValueError(
            "All feature columns are constant; cannot fit ordinal logit. "
            "Re-run ETL so matchup/environment fields are populated."
        )
    y_levels = tb_values_to_levels(y_tb)
    y_series = pd.Series(np.asarray(y_levels, dtype=int)).reset_index(drop=True)
    model = OrderedModel(y_series, X_fit, distr="logit")
    res = model.fit(method="lbfgs", disp=disp, maxiter=200)
    # Stash the kept column list so predict_pmf can stay column-aligned even
    # when callers pass the original (untrimmed) feature DataFrame.
    try:
        setattr(res, _KEPT_COLS_ATTR, list(X_fit.columns))
    except Exception:  # pragma: no cover - statsmodels results are mutable but guard anyway
        pass
    return res


def predict_pmf(result: Any, X: pd.DataFrame) -> np.ndarray:
    """Rows of shape (n, NUM_TB_LEVELS)."""
    Xp = X.astype(float).reset_index(drop=True)
    kept = getattr(result, _KEPT_COLS_ATTR, None)
    if kept is None:
        # Fall back to the underlying OrderedModel's exog_names so legacy
        # pickles (fit before this guard) still predict correctly.
        try:
            kept = list(getattr(result.model, "exog_names", []) or [])
        except Exception:
            kept = []
    if kept:
        missing = [c for c in kept if c not in Xp.columns]
        if missing:
            raise KeyError(
                f"predict_pmf: feature DataFrame is missing columns required by the fitted "
                f"ordinal model: {missing}"
            )
        Xp = Xp[kept]
    probs = result.model.predict(result.params, exog=Xp, which="prob")
    return np.asarray(probs, dtype=float)


def expected_tb_from_pmf(pmf: np.ndarray) -> float:
    pmf = np.asarray(pmf, dtype=float).reshape(-1)
    pmf = pmf / max(pmf.sum(), 1e-12)
    # Levels 0..11 = TB equals that count; level 12 = TB >= 12
    ev = float(sum(j * pmf[j] for j in range(12)) + 12.70 * pmf[12])
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
