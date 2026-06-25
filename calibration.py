from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import MAX_CALIB_P_DELTA, MIN_CALIB_ROWS_GLOBAL, MIN_CALIB_ROWS_SEGMENT, MODEL_DIR


CALIB_PATH = Path(MODEL_DIR) / "p_calibrator_isotonic.pkl"
SEGMENTED_CALIB_PATH = Path(MODEL_DIR) / "p_calibrator_segmented.pkl"
OOF_CALIB_PATH = Path(MODEL_DIR) / "p_calibrator_oof.pkl"


@dataclass
class ProbabilityCalibrator:
    iso: IsotonicRegression

    def transform(self, p: float) -> float:
        p = float(p)
        p = min(1.0 - 1e-6, max(1e-6, p))
        raw_iso = float(self.iso.predict([p])[0])
        delta = raw_iso - p
        cap = float(MAX_CALIB_P_DELTA)
        if cap > 0:
            delta = max(-cap, min(cap, delta))
        out = p + delta
        return min(1.0 - 1e-6, max(1e-6, out))


def fit_isotonic(ps: np.ndarray, ys: np.ndarray) -> ProbabilityCalibrator:
    ps = np.asarray(ps, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ps, ys)
    return ProbabilityCalibrator(iso=iso)


def save(cal: ProbabilityCalibrator, path: Path = CALIB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cal, f)


def load(path: Path = CALIB_PATH) -> ProbabilityCalibrator | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, SegmentedCalibratorBundle):
        return obj.global_cal
    return obj


def line_bucket(line: float) -> str:
    return f"{float(line):g}"


def games_played_bucket(n: int) -> str:
    return "gp50plus" if int(n) >= 50 else "gp_lt50"


def segment_key(*, line: float, side: str, games_played: int) -> str:
    return f"{line_bucket(line)}|{str(side).lower()}|{games_played_bucket(games_played)}"


@dataclass
class SegmentedCalibratorBundle:
    """Per (line, side, games_played) isotonic maps with global fallback."""

    global_cal: ProbabilityCalibrator | None
    segments: dict[str, ProbabilityCalibrator] = field(default_factory=dict)
    segment_counts: dict[str, int] = field(default_factory=dict)
    min_segment_rows: int = MIN_CALIB_ROWS_SEGMENT

    def transform(
        self,
        p: float,
        *,
        line: float,
        side: str,
        games_played: int = 0,
    ) -> float:
        key = segment_key(line=line, side=side, games_played=games_played)
        seg = self.segments.get(key)
        if seg is not None and self.segment_counts.get(key, 0) >= self.min_segment_rows:
            return seg.transform(p)
        if self.global_cal is not None:
            return self.global_cal.transform(p)
        return float(p)


def fit_segmented(
    rows: list[dict],
    *,
    min_global: int = MIN_CALIB_ROWS_GLOBAL,
    min_segment: int = MIN_CALIB_ROWS_SEGMENT,
) -> SegmentedCalibratorBundle | None:
    """
    rows: dicts with keys p, y, line, side, games_played, weight (optional)
    """
    if len(rows) < min_global:
        return None

    def _fit_group(sub: list[dict]) -> ProbabilityCalibrator | None:
        if len(sub) < min_segment:
            return None
        ps, ys, ws = [], [], []
        for r in sub:
            w = float(r.get("weight", 1) or 1)
            reps = int(min(20, max(1, round(w / 2))))
            ps.extend([float(r["p"])] * reps)
            ys.extend([float(r["y"])] * reps)
        return fit_isotonic(np.array(ps), np.array(ys))

    global_cal = _fit_group(rows)
    segments: dict[str, ProbabilityCalibrator] = {}
    counts: dict[str, int] = {}
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        k = segment_key(line=float(r["line"]), side=str(r["side"]), games_played=int(r.get("games_played", 0)))
        buckets.setdefault(k, []).append(r)
        counts[k] = counts.get(k, 0) + 1
    for k, sub in buckets.items():
        cal = _fit_group(sub)
        if cal is not None:
            segments[k] = cal
    return SegmentedCalibratorBundle(
        global_cal=global_cal,
        segments=segments,
        segment_counts=counts,
        min_segment_rows=min_segment,
    )


def save_segmented(bundle: SegmentedCalibratorBundle, path: Path = SEGMENTED_CALIB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    if bundle.global_cal is not None:
        save(bundle.global_cal, CALIB_PATH)


def save_oof(cal: ProbabilityCalibrator, path: Path = OOF_CALIB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cal, f)


def load_oof(path: Path = OOF_CALIB_PATH) -> ProbabilityCalibrator | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def fit_oof_from_rows(rows: list[dict], *, min_rows: int = 200) -> ProbabilityCalibrator | None:
    """
    Fit global isotonic on OOF rows: dicts with keys ``p`` (raw P over) and ``y`` (1 if TB > line).
    """
    if len(rows) < min_rows:
        return None
    ps = np.array([float(r["p"]) for r in rows], dtype=float)
    ys = np.array([float(r["y"]) for r in rows], dtype=float)
    return fit_isotonic(ps, ys)


def load_segmented(path: Path = SEGMENTED_CALIB_PATH) -> SegmentedCalibratorBundle | None:
    if not path.exists():
        g = load(CALIB_PATH)
        if g is None:
            return None
        return SegmentedCalibratorBundle(global_cal=g, segments={})
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, SegmentedCalibratorBundle):
        return obj
    return None
