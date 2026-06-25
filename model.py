"""
model.py (MLB TB)
-----------------
Train/load/predict batter total bases (TB).

Default: proportional-odds ordinal logistic regression on engineered features,
yielding a full TB PMF (better Under / tail pricing than mean + Poisson/NB).

Legacy: set USE_LEGACY_XGB=1 to restore XGBoost Tweedie + count distribution head.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from config import CV_GAP_DATES, CV_PRIMARY_METRIC, EVAL_LINES, MODEL_DIR, USE_LEGACY_XGB
from feature_store import MODEL_FEATURES, build_feature_table
from ordinal_core import (
    NUM_TB_LEVELS,
    expected_tb_from_pmf,
    fit_ordinal_logit,
    predict_pmf,
    prob_over_line_from_pmf,
)
from probability_engine import prob_exceed


def _use_xgb_head(meta: dict | None = None) -> bool:
    if USE_LEGACY_XGB:
        return True
    if meta and meta.get("model_kind") == "xgb_tweedie":
        return True
    return False


MODEL_PATH = Path(MODEL_DIR) / "tb_model.pkl"
META_PATH = Path(MODEL_DIR) / "model_meta.pkl"
BEST_PARAMS_PATH = Path(MODEL_DIR) / "best_params.json"
LEGACY_XGB_PATH = Path(MODEL_DIR) / "tb_xgb.pkl"


def load_best_params(path: Path = BEST_PARAMS_PATH) -> dict:
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


def prepare_data_as_of(as_of_date: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Training rows strictly before ``as_of_date`` (point-in-time, no lookahead)."""
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date")
    cutoff = pd.Timestamp(as_of_date)
    df = df[pd.to_datetime(df["game_date"]) < cutoff].copy()
    if df.empty:
        raise ValueError(f"No training rows before {as_of_date}")
    X = df[MODEL_FEATURES].copy()
    y = df["tb"].astype(float).values
    return X, y, df


def train_as_of(as_of_date: str) -> tuple[object, dict]:
    """Train in-memory model using only data before ``as_of_date``."""
    X, y, df = prepare_data_as_of(as_of_date)
    if _use_xgb_head(None):
        return _train_xgb_legacy(X, y, df, save=False)
    return _train_ordinal(X, y, df, save=False)


def collect_oof_calibration_rows(
    n_splits: int = 5,
    gap_dates: int | None = None,
) -> list[dict]:
    """Build OOF (p, y, line) rows from walk-forward CV for isotonic calibration."""
    from config import CV_GAP_DATES, EVAL_LINES

    gap = CV_GAP_DATES if gap_dates is None else int(gap_dates)
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date").reset_index(drop=True)
    X = df[MODEL_FEATURES].copy()
    y = df["tb"].astype(float).values
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < (n_splits + 2):
        return []

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict] = []

    for fold, (train_d_idx, test_d_idx) in enumerate(tscv.split(uniq), start=1):
        train_dates = uniq[train_d_idx]
        test_dates = uniq[test_d_idx]
        if gap and gap > 0:
            embargo_start = train_dates[-1]
            if len(train_dates) > gap:
                train_dates = train_dates[: -gap]
                embargo_start = train_dates[-1]
            test_dates = test_dates[test_dates > embargo_start]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        train_idx = np.flatnonzero(train_mask.values)
        test_idx = np.flatnonzero(test_mask.values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        if _use_xgb_head(None):
            m = _make_model(random_state=42 + fold)
            m.fit(X.iloc[train_idx], y[train_idx])
            oof_pred = m.predict(X.iloc[test_idx])
            resid_var = float(np.var(y[train_idx] - m.predict(X.iloc[train_idx])))
            for ii, ti in enumerate(test_idx):
                lam = float(oof_pred[ii])
                tb_actual = float(y[ti])
                for line in EVAL_LINES:
                    p_over = float(prob_exceed(lam, float(line), resid_var))
                    rows.append({"p": p_over, "y": 1.0 if tb_actual > float(line) else 0.0, "line": float(line)})
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_ordinal_logit(X.iloc[train_idx], y[train_idx], disp=False)
            pmf_te = predict_pmf(res, X.iloc[test_idx])
            for ii, ti in enumerate(test_idx):
                pmf = pmf_te[ii]
                tb_actual = float(y[ti])
                for line in EVAL_LINES:
                    p_over = float(prob_over_line_from_pmf(pmf, float(line)))
                    rows.append({"p": p_over, "y": 1.0 if tb_actual > float(line) else 0.0, "line": float(line)})
    return rows


def fit_and_save_oof_calibrator(n_splits: int = 5) -> bool:
    """Run CV, fit OOF isotonic calibrator, persist to models/."""
    from calibration import fit_oof_from_rows, save_oof

    rows = collect_oof_calibration_rows(n_splits=n_splits)
    cal = fit_oof_from_rows(rows)
    if cal is None:
        return False
    save_oof(cal)
    return True


def _make_model(random_state: int = 42, params_override: dict | None = None) -> XGBRegressor:
    base_params = dict(
        n_estimators=4000,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2.0,
        objective="reg:tweedie",
        tweedie_variance_power=1.25,
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


def _train_xgb_legacy(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, save: bool) -> tuple[XGBRegressor, dict]:
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq_dates = np.array(sorted(dates.unique()))
    if len(uniq_dates) < 4:
        model = _make_model(random_state=42)
        model.fit(X, y)
    else:
        cutoff_idx = max(1, int(len(uniq_dates) * 0.8))
        if cutoff_idx >= len(uniq_dates):
            cutoff_idx = len(uniq_dates) - 1
        train_dates = uniq_dates[:cutoff_idx]
        val_dates = uniq_dates[cutoff_idx:]
        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
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
        model = _make_model(random_state=42)
        model.set_params(n_estimators=int(best["n"] or 900))
        model.fit(X, y, verbose=False)

    preds = model.predict(X)
    resid = y - preds
    meta = {
        "model_kind": "xgb_tweedie",
        "train_rows": int(len(X)),
        "residual_std": float(np.std(resid)),
        "residual_var": float(np.var(resid)),
        "feature_names": list(X.columns),
        "trained_on": str(df["game_date"].max()),
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"kind": "xgb", "model": model}, f)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
        with open(LEGACY_XGB_PATH, "wb") as f:
            pickle.dump(model, f)
    return model, meta


def _train_ordinal(X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, save: bool) -> tuple[object, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_ordinal_logit(X, y, disp=False)
    # fit_ordinal_logit silently drops zero-variance columns; persist only the
    # columns that actually contributed to the fit so live inference stays
    # column-aligned with the underlying OrderedModel exog.
    kept_cols = list(getattr(res, "_kept_feature_cols", X.columns))
    dropped_cols = [c for c in X.columns if c not in kept_cols]
    pmf_all = predict_pmf(res, X)
    lam_hat = np.array([expected_tb_from_pmf(pmf_all[i]) for i in range(len(X))])
    resid = y - lam_hat
    meta = {
        "model_kind": "ordinal_logit",
        "train_rows": int(len(X)),
        "residual_std": float(np.std(resid)),
        "residual_var": float(np.var(resid)),
        "feature_names": kept_cols,
        "all_input_features": list(X.columns),
        "dropped_constant_features": dropped_cols,
        "trained_on": str(df["game_date"].max()),
        "num_tb_levels": int(NUM_TB_LEVELS),
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {
                    "kind": "ordinal",
                    "result": res,
                    "feature_names": kept_cols,
                    "kept_feature_cols": kept_cols,
                },
                f,
            )
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
    return res, meta


def train(save: bool = True) -> tuple[object, dict]:
    X, y, df = prepare_data()
    if _use_xgb_head(None):
        return _train_xgb_legacy(X, y, df, save)
    return _train_ordinal(X, y, df, save)


def load_model() -> tuple[object, dict]:
    if not META_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python run_pipeline.py train")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    blob_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_XGB_PATH
    if not blob_path.exists():
        raise FileNotFoundError("Model weights not found. Run: python run_pipeline.py train")
    with open(blob_path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and blob.get("kind") == "ordinal":
        meta.setdefault("model_kind", "ordinal_logit")
        res = blob["result"]
        # Re-attach the kept-column marker so predict_pmf can keep the
        # prediction matrix column-aligned with the fitted exog.  Older pickles
        # written before the constant-column guard won't have this key and will
        # fall back to result.model.exog_names inside predict_pmf.
        kept = blob.get("kept_feature_cols") or blob.get("feature_names")
        if kept and not hasattr(res, "_kept_feature_cols"):
            try:
                setattr(res, "_kept_feature_cols", list(kept))
            except Exception:
                pass
        return res, meta
    if isinstance(blob, dict) and blob.get("kind") == "xgb":
        meta.setdefault("model_kind", "xgb_tweedie")
        return blob["model"], meta
    # Legacy pickle was raw XGBRegressor
    meta.setdefault("model_kind", "xgb_tweedie")
    return blob, meta


def predict_lambda(
    row_features: dict | pd.DataFrame,
    model: object,
    feature_names: list[str] | None = None,
    meta: dict | None = None,
) -> float | np.ndarray:
    fn = feature_names or list(MODEL_FEATURES)
    if _use_xgb_head(meta):
        if isinstance(row_features, dict):
            X = pd.DataFrame([row_features])
            return float(model.predict(X)[0])
        X = row_features[fn] if isinstance(row_features, pd.DataFrame) else row_features
        return model.predict(X)
    from ordinal_core import OrdinalModelBundle

    if isinstance(model, OrdinalModelBundle):
        bundle = model
    else:
        bundle = OrdinalModelBundle(result=model, feature_names=fn)
    if isinstance(row_features, dict):
        pmf = bundle.predict_pmf_row({k: row_features[k] for k in fn})
        return float(expected_tb_from_pmf(pmf))
    X = row_features[fn] if isinstance(row_features, pd.DataFrame) else row_features
    pmfs = predict_pmf(bundle.result, X.astype(float))
    return np.array([expected_tb_from_pmf(pmfs[i]) for i in range(len(pmfs))])


def predict_tb_pmf_row(
    row_features: dict,
    model: object,
    feature_names: list[str] | None = None,
    meta: dict | None = None,
) -> np.ndarray:
    """Return length-NUM_TB_LEVELS PMF for one feature row (dict keys must include features)."""
    fn = feature_names or list(MODEL_FEATURES)
    if _use_xgb_head(meta):
        lam = float(predict_lambda(row_features, model, feature_names=fn, meta=meta))
        from scipy.stats import poisson

        p = np.array([poisson.pmf(k, max(lam, 0.05)) for k in range(NUM_TB_LEVELS - 1)], dtype=float)
        tail = 1.0 - p.sum()
        out = np.zeros(NUM_TB_LEVELS, dtype=float)
        out[: NUM_TB_LEVELS - 1] = p
        out[-1] = max(tail, 0.0)
        out /= out.sum()
        return out
    X = pd.DataFrame([{k: row_features[k] for k in fn}]).astype(float)
    return predict_pmf(model, X)[0]


def walk_forward_cv(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> dict:
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
    dates = pd.to_datetime(df["game_date"]).dt.normalize()
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < (n_splits + 2):
        raise ValueError(f"Not enough unique dates ({len(uniq)}) for n_splits={n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    fold_rows: list[int] = []
    oof_pred = np.full(shape=(len(X),), fill_value=np.nan, dtype=float)
    oof_pmf: list[np.ndarray | None] = [None] * len(X)

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

        if _use_xgb_head(None):
            m = _make_model(random_state=42 + fold)
            m.fit(X.iloc[train_idx], y[train_idx])
            pred = m.predict(X.iloc[test_idx])
            oof_pred[test_idx] = pred
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_ordinal_logit(X.iloc[train_idx], y[train_idx], disp=False)
            pmf_te = predict_pmf(res, X.iloc[test_idx])
            pred = np.array([expected_tb_from_pmf(pmf_te[i]) for i in range(len(test_idx))])
            oof_pred[test_idx] = pred
            for j, ti in enumerate(test_idx):
                oof_pmf[ti] = pmf_te[j]

        maes.append(mean_absolute_error(y[test_idx], pred))
        fold_rows.append(int(len(test_idx)))

    valid = np.isfinite(oof_pred)
    if not np.any(valid):
        raise RuntimeError("CV produced no out-of-fold predictions. Check split settings.")

    resid = y[valid] - oof_pred[valid]
    variance = float(np.var(resid))

    prob_metrics: dict[str, float] = {}
    eps = 1e-6
    for line in EVAL_LINES:
        if _use_xgb_head(None):
            p_over = np.array([prob_exceed(float(l), float(line), variance) for l in oof_pred[valid]], dtype=float)
        else:
            p_over = np.zeros(int(np.sum(valid)), dtype=float)
            vi = np.flatnonzero(valid)
            for ii, idx in enumerate(vi):
                pmf = oof_pmf[idx]
                if pmf is None:
                    continue
                p_over[ii] = prob_over_line_from_pmf(pmf, float(line))
        p_over = np.clip(p_over, eps, 1 - eps)
        y_over = (y[valid] > float(line)).astype(float)
        brier = float(np.mean((p_over - y_over) ** 2))
        logloss = float(-np.mean(y_over * np.log(p_over) + (1.0 - y_over) * np.log(1.0 - p_over)))
        prob_metrics[f"brier@{line:g}"] = brier
        prob_metrics[f"logloss@{line:g}"] = logloss

    brier_vals = [v for k, v in prob_metrics.items() if k.startswith("brier@")]
    logloss_vals = [v for k, v in prob_metrics.items() if k.startswith("logloss@")]

    return {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "std_mae": float(np.std(maes)) if maes else float("nan"),
        "fold_rows_mean": float(np.mean(fold_rows)) if fold_rows else 0.0,
        "oof_residual_var": variance,
        "mean_brier": float(np.mean(brier_vals)) if brier_vals else float("nan"),
        "mean_logloss": float(np.mean(logloss_vals)) if logloss_vals else float("nan"),
        **prob_metrics,
    }


def get_feature_importance(model: object, meta: dict | None = None) -> pd.DataFrame:
    if _use_xgb_head(meta) or hasattr(model, "get_booster"):
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        df = pd.DataFrame({"feature": list(score.keys()), "importance": list(score.values())})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    try:
        names = list(getattr(model.model, "exog_names", []) or MODEL_FEATURES)
        params = np.asarray(model.params, dtype=float)
        n_feat = min(len(names), len(params) - max(int(getattr(model.model, "k_levels", NUM_TB_LEVELS - 1)), 1))
        imp = np.abs(params[:n_feat])
        df = pd.DataFrame({"feature": names[:n_feat], "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"feature": [], "importance": []})


def _score_cv_for_params(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int,
    gap_dates: int,
) -> dict:
    if not _use_xgb_head(None):
        return walk_forward_cv_by_date(df, X, y, n_splits=n_splits, gap_dates=int(gap_dates))

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

    brier_vals = [v for k, v in prob_metrics.items() if k.startswith("brier@")]
    return {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "oof_residual_var": variance,
        "mean_logloss": float(np.mean(loglosses)) if loglosses else float("nan"),
        "mean_brier": float(np.mean(brier_vals)) if brier_vals else float("nan"),
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
    if not _use_xgb_head(None):
        df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date").reset_index(drop=True)
        X = df[MODEL_FEATURES].copy()
        y = df["tb"].astype(float).values
        feature_subsets = [list(MODEL_FEATURES)]
        # Lightweight ordinal feature ablation trials
        base_matchup = [f for f in MODEL_FEATURES if f in (
            "is_home", "lineup_slot_norm", "expected_pa_proxy", "opp_tb_allowed_roll", "opp_sp_hand_L", "platoon_tb_adj"
        )]
        if base_matchup:
            feature_subsets.append([f for f in MODEL_FEATURES if f not in base_matchup])
        best_score: dict | None = None
        best_feats: list[str] = MODEL_FEATURES
        for feats in feature_subsets:
            Xsub = df[feats].copy()
            score = walk_forward_cv_by_date(df, Xsub, y, n_splits=int(n_splits), gap_dates=int(gap_dates))
            if _cv_score_better(score, best_score):
                best_score = score
                best_feats = feats
        meta_path = MODEL_DIR / "ordinal_feature_subset.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"feature_names": best_feats}, indent=2))
        return {"feature_names": best_feats}, best_score or {}

    rng = np.random.default_rng(int(random_seed))
    df = build_feature_table().dropna(subset=["tb"] + MODEL_FEATURES).sort_values("game_date").reset_index(drop=True)
    X = df[MODEL_FEATURES].copy()
    y = df["tb"].astype(float).values

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
        if _cv_score_better(score, best_score):
            best_params = params
            best_score = score

    best_params = best_params or {}
    best_score = best_score or {}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(best_params, indent=2, sort_keys=True))
    return best_params, best_score


def _cv_primary_value(score: dict) -> float:
    key = "mean_logloss" if CV_PRIMARY_METRIC == "mean_logloss" else "mean_brier"
    return float(score.get(key, float("nan")))


def _cv_score_better(score: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return _cv_primary_value(score) < _cv_primary_value(best)
