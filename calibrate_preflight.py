"""
Preflight checks for probability calibrators before live scan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CALIBRATE_MAX_AGE_DAYS,
    MODEL_DIR,
    REQUIRE_FILL_CALIB_FOR_LIVE,
    USE_OOF_CALIBRATION,
    USE_SEGMENTED_CALIBRATION,
)
from calibration import OOF_CALIB_PATH, SEGMENTED_CALIB_PATH

log = logging.getLogger(__name__)

CALIBRATOR_META_PATH = Path(MODEL_DIR) / "calibrator_meta.json"


@dataclass
class CalibratePreflightResult:
    ok: bool
    warnings: list[str]
    errors: list[str]


def write_calibrator_meta(
    *,
    n_rows: int,
    n_segments: int,
    start: str,
    end: str,
    path: Path = CALIBRATOR_META_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(n_rows),
        "n_segments": int(n_segments),
        "start": str(start),
        "end": str(end),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibrator_meta(path: Path = CALIBRATOR_META_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_days(trained_at: str) -> float | None:
    try:
        s = trained_at.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0
    except Exception:
        return None


def check_calibrate_preflight(*, live: bool = False) -> CalibratePreflightResult:
    warnings: list[str] = []
    errors: list[str] = []

    if USE_OOF_CALIBRATION and not OOF_CALIB_PATH.exists():
        warnings.append(
            f"OOF calibrator missing ({OOF_CALIB_PATH.name}); run train with --fit-oof or disable USE_OOF_CALIBRATION"
        )

    seg_missing = USE_SEGMENTED_CALIBRATION and not SEGMENTED_CALIB_PATH.exists()
    seg_stale = False
    meta = load_calibrator_meta()
    if meta and CALIBRATE_MAX_AGE_DAYS > 0:
        age = _age_days(str(meta.get("trained_at", "") or ""))
        if age is not None and age > float(CALIBRATE_MAX_AGE_DAYS):
            seg_stale = True

    if seg_missing:
        msg = (
            f"Fill-based segmented calibrator missing ({SEGMENTED_CALIB_PATH.name}); "
            "run reconcile then calibrate on your fills"
        )
        if live and REQUIRE_FILL_CALIB_FOR_LIVE:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif seg_stale:
        msg = (
            f"Fill calibrator older than {CALIBRATE_MAX_AGE_DAYS}d "
            f"(trained {meta.get('trained_at')}); run reconcile + calibrate"
        )
        if live and REQUIRE_FILL_CALIB_FOR_LIVE:
            errors.append(msg)
        else:
            warnings.append(msg)

    ok = len(errors) == 0
    return CalibratePreflightResult(ok=ok, warnings=warnings, errors=errors)
