from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import MODEL_DIR


CALIB_PATH = Path(MODEL_DIR) / "p_calibrator_isotonic.pkl"


@dataclass
class ProbabilityCalibrator:
    iso: IsotonicRegression

    def transform(self, p: float) -> float:
        p = float(p)
        p = min(1.0 - 1e-6, max(1e-6, p))
        out = float(self.iso.predict([p])[0])
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
        return pickle.load(f)

