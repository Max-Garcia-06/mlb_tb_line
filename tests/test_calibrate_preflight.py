from pathlib import Path

from calibrate_preflight import check_calibrate_preflight, write_calibrator_meta, load_calibrator_meta


def test_write_and_load_meta(tmp_path, monkeypatch):
    p = tmp_path / "calibrator_meta.json"
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", p)
    write_calibrator_meta(n_rows=50, n_segments=3, start="2026-05-01", end="2026-05-20")
    meta = load_calibrator_meta()
    assert meta is not None
    assert meta["n_rows"] == 50


def test_preflight_missing_segmented(tmp_path, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr("calibrate_preflight.OOF_CALIB_PATH", tmp_path / "oof.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", False)
    r = check_calibrate_preflight(live=True)
    assert r.ok
    assert any("missing" in w.lower() for w in r.warnings)


def test_preflight_blocks_when_required(tmp_path, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    r = check_calibrate_preflight(live=True)
    assert not r.ok
