"""
model.py (MLB TB)
-----------------
Train/load/predict a model for batter total bases (TB).

Approach:
- Predict expected TB (λ) via XGBoost regression on trailing features.
- Convert λ -> probability via Poisson/NB in probability_engine.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import MODEL_DIR, CV_GAP_DATES, EVAL_LINES
from feature_store import build_feature_table, MODEL_FEATURES
from probability_engine import prob_exceed


MODEL_PATH = Path(MODEL_DIR) / "tb_xgb.pkl"
META_PATH = Path(MODEL_DIR) / "model_meta.pkl"
BEST_PARAMS_PATH = Path(MODEL_DIR) / "best_params.json"


def load_best_params(path: Path = BEST_PARAMS_PATH) -> dict:
    """
    Load tuned hyperparameters (if present).

    This file is written by `tune_hyperparameters()` and is optional.
    """
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date")
    X = df[MODEL_FEATURES].copy()
    y = df["tb"].astype(float).values
    return X, y, df


def _make_model(random_state: int = 42, params_override: dict | None = None) -> XGBRegressor:
    # Tweedie works well for non-negative, overdispersed counts like TB.
    base_params = dict(
        # Use a large cap on trees; early stopping will pick the effective number.
        n_estimators=4000,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2.0,
        objective="reg:tweedie",
        tweedie_variance_power=1.25,
        # Set eval metric here for older/newer XGBoost sklearn APIs.
        eval_metric="mae",
        random_state=random_state,
        n_jobs=4,
    )
    tuned = load_best_params()
    if tuned:
        base_params.update({k: tuned[k] for k in tuned.keys()})
    if params_override:
        base_params.update({k: params_override[k] for k in params_override.keys()})
    return XGBRegressor(**base_params)


def train(save: bool = True) -> tuple[XGBRegressor, dict]:
    X, y, df = prepare_data()

    # Time-based train/validation split for early stopping.
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq_dates = np.array(sorted(dates.unique()))
    if len(uniq_dates) < 4:
        # Not enough history to do a meaningful split; fall back to fitting on all data.
        model = _make_model(random_state=42)
        model.fit(X, y)
    else:
        # Use oldest ~80% of dates for training, newest ~20% for validation.
        cutoff_idx = max(1, int(len(uniq_dates) * 0.8))
        if cutoff_idx >= len(uniq_dates):
            cutoff_idx = len(uniq_dates) - 1
        train_dates = uniq_dates[:cutoff_idx]
        val_dates = uniq_dates[cutoff_idx:]

        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Some XGBoost sklearn versions don't support early stopping kwargs in fit().
        # Implement a simple "early stopping"-like selection by choosing n_estimators
        # based on validation MAE over an increasing tree count grid.
        candidate_ns = [200, 400, 700, 1000, 1400, 1800, 2400, 3200, 4000]
        best = {"n": None, "mae": float("inf")}
        for n in candidate_ns:
            m = _make_model(random_state=42)
            m.set_params(n_estimators=int(n))
            m.fit(X_train, y_train, verbose=False)
            pred_val = m.predict(X_val)
            mae = mean_absolute_error(y_val, pred_val)
            if mae < best["mae"]:
                best = {"n": int(n), "mae": float(mae)}

        # Retrain on the full dataset with the chosen tree count.
        model = _make_model(random_state=42)
        model.set_params(n_estimators=int(best["n"] or 900))
        model.fit(X, y, verbose=False)

    preds = model.predict(X)
    resid = y - preds
    meta = {
        "train_rows": int(len(X)),
        "residual_std": float(np.std(resid)),
        "residual_var": float(np.var(resid)),
        "feature_names": list(X.columns),
        "trained_on": str(df["game_date"].max()),
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)

    return model, meta


def load_model() -> tuple[XGBRegressor, dict]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python run_pipeline.py train")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return model, meta


def predict_lambda(row_features: dict | pd.DataFrame, model: XGBRegressor) -> float | np.ndarray:
    if isinstance(row_features, dict):
        X = pd.DataFrame([row_features])
        return float(model.predict(X)[0])
    X = row_features[MODEL_FEATURES] if isinstance(row_features, pd.DataFrame) else row_features
    return model.predict(X)


def walk_forward_cv(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> dict:
    # Backwards-compatible wrapper: run date-based CV if we can infer dates.
    # This retains the old signature used by `run_pipeline.py`.
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date").reset_index(drop=True)
    X2 = df[MODEL_FEATURES].copy()
    y2 = df["tb"].astype(float).values
    return walk_forward_cv_by_date(df, X2, y2, n_splits=n_splits, gap_dates=CV_GAP_DATES)


def walk_forward_cv_by_date(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    gap_dates: int = 1,
) -> dict:
    """
    Walk-forward CV split by unique game_date (with an embargo gap in date units).

    This prevents subtle leakage where row-based splits can interleave the same day.
    """
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < (n_splits + 2):
        raise ValueError(f"Not enough unique dates ({len(uniq)}) for n_splits={n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    fold_rows: list[int] = []
    oof_pred = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)

    for fold, (train_d_idx, test_d_idx) in enumerate(tscv.split(uniq), start=1):
        train_dates = uniq[train_d_idx]
        test_dates = uniq[test_d_idx]

        if gap_dates and gap_dates > 0:
            cutoff = train_dates[-1]
            embargo_start = cutoff
            # Exclude the last `gap_dates` training dates and any dates between train and test.
            if len(train_dates) > gap_dates:
                train_dates = train_dates[: -gap_dates]
                embargo_start = train_dates[-1]
            # Ensure we don't accidentally overlap by date
            test_dates = test_dates[test_dates > embargo_start]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        m = _make_model(random_state=42 + fold)
        m.fit(X.iloc[train_idx], y[train_idx])
        pred = m.predict(X.iloc[test_idx])
        oof_pred[test_idx] = pred

        maes.append(mean_absolute_error(y[test_idx], pred))
        fold_rows.append(int(len(test_idx)))

    valid = np.isfinite(oof_pred)
    if not np.any(valid):
        raise RuntimeError("CV produced no out-of-fold predictions. Check split settings.")

    # Probability scoring for common TB lines (aligned to your 'over' markets).
    # We use a fixed NB variance estimated from OOF residuals.
    resid = y[valid] - oof_pred[valid]
    variance = float(np.var(resid))

    prob_metrics: dict[str, float] = {}
    eps = 1e-6
    for line in EVAL_LINES:
        p_over = np.array([prob_exceed(float(l), float(line), variance) for l in oof_pred[valid]], dtype=float)
        p_over = np.clip(p_over, eps, 1 - eps)
        y_over = (y[valid] > float(line)).astype(float)

        brier = float(np.mean((p_over - y_over) ** 2))
        logloss = float(-np.mean(y_over * np.log(p_over) + (1.0 - y_over) * np.log(1.0 - p_over)))
        prob_metrics[f"brier@{line:g}"] = brier
        prob_metrics[f"logloss@{line:g}"] = logloss

    return {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "std_mae": float(np.std(maes)) if maes else float("nan"),
        "fold_rows_mean": float(np.mean(fold_rows)) if fold_rows else 0.0,
        "oof_residual_var": variance,
        **prob_metrics,
    }


def get_feature_importance(model: XGBRegressor) -> pd.DataFrame:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    df = pd.DataFrame({"feature": list(score.keys()), "importance": list(score.values())})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _score_cv_for_params(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int,
    gap_dates: int,
) -> dict:
    """
    Score a hyperparameter set using the same date-based CV logic and probability metrics.
    """
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < (n_splits + 2):
        raise ValueError(f"Not enough unique dates ({len(uniq)}) for n_splits={n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    oof_pred = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)

    for fold, (train_d_idx, test_d_idx) in enumerate(tscv.split(uniq), start=1):
        train_dates = uniq[train_d_idx]
        test_dates = uniq[test_d_idx]

        if gap_dates and gap_dates > 0:
            embargo_start = train_dates[-1]
            if len(train_dates) > gap_dates:
                train_dates = train_dates[: -gap_dates]
                embargo_start = train_dates[-1]
            test_dates = test_dates[test_dates > embargo_start]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        m = _make_model(random_state=4242 + fold, params_override=params)
        m.fit(X.iloc[train_idx], y[train_idx], verbose=False)
        pred = m.predict(X.iloc[test_idx])
        oof_pred[test_idx] = pred
        maes.append(mean_absolute_error(y[test_idx], pred))

    valid = np.isfinite(oof_pred)
    if not np.any(valid):
        raise RuntimeError("CV produced no out-of-fold predictions. Check split settings.")

    resid = y[valid] - oof_pred[valid]
    variance = float(np.var(resid))

    eps = 1e-6
    prob_metrics: dict[str, float] = {}
    loglosses: list[float] = []
    for line in EVAL_LINES:
        p_over = np.array([prob_exceed(float(l), float(line), variance) for l in oof_pred[valid]], dtype=float)
        p_over = np.clip(p_over, eps, 1 - eps)
        y_over = (y[valid] > float(line)).astype(float)
        brier = float(np.mean((p_over - y_over) ** 2))
        logloss = float(-np.mean(y_over * np.log(p_over) + (1.0 - y_over) * np.log(1.0 - p_over)))
        prob_metrics[f"brier@{line:g}"] = brier
        prob_metrics[f"logloss@{line:g}"] = logloss
        loglosses.append(logloss)

    return {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "oof_residual_var": variance,
        "mean_logloss": float(np.mean(loglosses)) if loglosses else float("nan"),
        **prob_metrics,
    }


def tune_hyperparameters(
    *,
    trials: int = 25,
    n_splits: int = 4,
    gap_dates: int = CV_GAP_DATES,
    random_seed: int = 42,
    save_path: Path = BEST_PARAMS_PATH,
) -> tuple[dict, dict]:
    """
    Hyperparameter search optimized for probability quality (mean logloss across EVAL_LINES).

    Returns: (best_params, best_score_dict)
    """
    rng = np.random.default_rng(int(random_seed))
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date").reset_index(drop=True)
    X = df[MODEL_FEATURES].copy()
    y = df["tb"].astype(float).values

    # Search space: keep it modest so it finishes on a laptop.
    # (We avoid early_stopping_rounds since your XGBoost sklearn wrapper doesn't support it.)
    n_estimators_choices = np.array([300, 500, 800, 1200, 1800, 2400, 3200, 4000], dtype=int)
    max_depth_choices = np.array([3, 4, 5, 6], dtype=int)

    best_params: dict | None = None
    best_score: dict | None = None

    for t in range(int(trials)):
        params = {
            "n_estimators": int(rng.choice(n_estimators_choices)),
            "learning_rate": float(rng.uniform(0.02, 0.08)),
            "max_depth": int(rng.choice(max_depth_choices)),
            "subsample": float(rng.uniform(0.7, 1.0)),
            "colsample_bytree": float(rng.uniform(0.7, 1.0)),
            "min_child_weight": float(rng.uniform(1.0, 6.0)),
            "reg_lambda": float(rng.uniform(0.5, 4.0)),
            "tweedie_variance_power": float(rng.uniform(1.1, 1.55)),
        }

        score = _score_cv_for_params(df=df, X=X, y=y, params=params, n_splits=int(n_splits), gap_dates=int(gap_dates))
        if best_score is None or float(score["mean_logloss"]) < float(best_score["mean_logloss"]):
            best_params = params
            best_score = score

    best_params = best_params or {}
    best_score = best_score or {}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(best_params, indent=2, sort_keys=True))
    return best_params, best_score

