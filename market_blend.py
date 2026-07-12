"""
market_blend.py
---------------
Shrink model probabilities toward the market price (logit-space blend).

Motivation (fills 2026-04-27 → 2026-07-05): every p_model bucket won less
often than predicted (gaps −0.05 to −0.21) and ROI decayed monotonically with
stated edge — the winner's-curse signature of trading unshrunk model
probabilities against a better-informed book. The blend weight ``w`` is fit on
resolved fills by weighted log-loss:

    p_final = sigmoid(w * logit(p_model) + (1 - w) * logit(p_market_mid))

``w = 1`` trusts the model fully (old behavior); ``w = 0`` trusts the market.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from config import (
    BLEND_META_PATH,
    BLEND_WEIGHT_OVERRIDE,
    DEFAULT_BLEND_WEIGHT,
    MIN_BLEND_ROWS_SEGMENT,
    MIN_BLEND_WEIGHT,
    SEGMENT_BLEND_META_PATH,
    USE_MARKET_BLEND,
)

log = logging.getLogger(__name__)

_P_EPS = 1e-4
# Logit inputs are clamped to [0.01, 0.99]: isotonic calibrators can saturate at
# exactly 0/1, and logit(1-1e-6) ≈ 13.8 lets even a tiny w manufacture edges.
_LOGIT_CLAMP = 0.01
_CACHED_WEIGHT: Optional[float] = None
_CACHED_SEGMENT_WEIGHTS: Optional[dict[str, float]] = None

DISAGREEMENT_BUCKETS = ("<0.05", "0.05-0.10", "0.10-0.15", ">=0.15")


def disagreement_bucket(p_model: float, p_market: float) -> str:
    """Bucket label for |p_model - p_market|; see DISAGREEMENT_BUCKETS."""
    d = abs(float(p_model) - float(p_market))
    if d < 0.05:
        return "<0.05"
    if d < 0.10:
        return "0.05-0.10"
    if d < 0.15:
        return "0.10-0.15"
    return ">=0.15"


def _logit(p: float) -> float:
    p = min(1.0 - _LOGIT_CLAMP, max(_LOGIT_CLAMP, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend_probability(p_model: float, p_market: float, w: float) -> float:
    """Logit-space convex blend of model and market probabilities."""
    w = min(1.0, max(0.0, float(w)))
    return _sigmoid(w * _logit(p_model) + (1.0 - w) * _logit(p_market))


def reset_blend_cache() -> None:
    global _CACHED_WEIGHT, _CACHED_SEGMENT_WEIGHTS
    _CACHED_WEIGHT = None
    _CACHED_SEGMENT_WEIGHTS = None


def _load_segment_weights() -> dict[str, float]:
    """Fitted per-bucket weights meeting MIN_BLEND_ROWS_SEGMENT, floored at MIN_BLEND_WEIGHT."""
    global _CACHED_SEGMENT_WEIGHTS
    if _CACHED_SEGMENT_WEIGHTS is not None:
        return _CACHED_SEGMENT_WEIGHTS
    weights: dict[str, float] = {}
    try:
        meta = json.loads(SEGMENT_BLEND_META_PATH.read_text())
        for bucket, seg in (meta.get("segments") or {}).items():
            if int(seg.get("n_rows", 0)) < MIN_BLEND_ROWS_SEGMENT:
                continue
            w = float(seg["w"])
            weights[bucket] = min(1.0, max(MIN_BLEND_WEIGHT, w))
    except FileNotFoundError:
        pass
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        log.warning("Could not read segment blend weights from %s (%s)", SEGMENT_BLEND_META_PATH, e)
    _CACHED_SEGMENT_WEIGHTS = weights
    return weights


def load_blend_weight(p_model: Optional[float] = None, p_market: Optional[float] = None) -> float:
    """
    Effective blend weight: env override > per-segment fit (needs p_model/p_market)
    > fitted models/blend_meta.json > DEFAULT_BLEND_WEIGHT. Returns 1.0 (no shrink)
    when USE_MARKET_BLEND is off.
    """
    global _CACHED_WEIGHT
    if not USE_MARKET_BLEND:
        return 1.0
    if BLEND_WEIGHT_OVERRIDE is not None:
        return float(BLEND_WEIGHT_OVERRIDE)
    if p_model is not None and p_market is not None:
        bucket = disagreement_bucket(p_model, p_market)
        seg_w = _load_segment_weights().get(bucket)
        if seg_w is not None:
            return seg_w
    if _CACHED_WEIGHT is not None:
        return _CACHED_WEIGHT
    w = float(DEFAULT_BLEND_WEIGHT)
    try:
        meta = json.loads(BLEND_META_PATH.read_text())
        w = float(meta["w"])
        if w < MIN_BLEND_WEIGHT:
            log.info("Fitted blend weight w=%.4f below floor; using floor w=%.2f", w, MIN_BLEND_WEIGHT)
            w = MIN_BLEND_WEIGHT
    except FileNotFoundError:
        log.info("No fitted blend weight (%s missing); using default w=%.2f", BLEND_META_PATH, w)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        log.warning("Could not read blend weight from %s (%s); using default w=%.2f", BLEND_META_PATH, e, w)
    _CACHED_WEIGHT = min(1.0, max(0.0, w))
    return _CACHED_WEIGHT


def _weighted_logloss(rows: list[dict], w: float) -> float:
    tot, wsum = 0.0, 0.0
    for r in rows:
        pb = blend_probability(r["p"], r["m"], w)
        pb = min(1.0 - _P_EPS, max(_P_EPS, pb))
        wt = float(r.get("weight", 1.0) or 1.0)
        y = float(r["y"])
        tot += wt * -(y * math.log(pb) + (1.0 - y) * math.log(1.0 - pb))
        wsum += wt
    return tot / wsum if wsum > 0 else float("inf")


def fit_blend_weight(rows: list[dict], step: float = 0.01) -> tuple[float, dict]:
    """
    Grid-search ``w`` minimizing weighted log-loss on resolved fills.

    Args:
        rows: dicts with keys ``p`` (model prob, pre-blend), ``m`` (market mid
            for the traded side at entry), ``y`` (1.0 win / 0.0 loss), and
            optional ``weight`` (contracts filled).
        step: grid resolution over [0, 1].

    Returns:
        (w_best, diagnostics) where diagnostics holds log-losses at w=0
        (market only), w=1 (model only), and w_best.
    """
    if not rows:
        raise ValueError("fit_blend_weight requires at least one row")
    best_w, best_ll = 0.0, float("inf")
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        w = i * step
        ll = _weighted_logloss(rows, w)
        if ll < best_ll:
            best_w, best_ll = w, ll
    diag = {
        "logloss_market_only": _weighted_logloss(rows, 0.0),
        "logloss_model_only": _weighted_logloss(rows, 1.0),
        "logloss_best": best_ll,
        "n_rows": len(rows),
    }
    return round(best_w, 4), diag


def save_blend_meta(w: float, diag: dict, *, start: str, end: str) -> None:
    meta = {
        "w": float(w),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in diag.items()},
    }
    BLEND_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLEND_META_PATH.write_text(json.dumps(meta, indent=2))
    reset_blend_cache()


def save_segment_blend_meta(segments: dict[str, tuple[float, dict]], *, start: str, end: str) -> None:
    """``segments``: bucket label -> (w_fit, diag) from fit_blend_weight, one per disagreement bucket."""
    meta = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "segments": {
            bucket: {
                "w": float(w),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in diag.items()},
            }
            for bucket, (w, diag) in segments.items()
        },
    }
    SEGMENT_BLEND_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEGMENT_BLEND_META_PATH.write_text(json.dumps(meta, indent=2))
    reset_blend_cache()
