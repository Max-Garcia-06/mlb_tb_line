import numpy as np

from calibration import fit_segmented, segment_key


def test_fit_segmented_global_and_segment():
    rows = []
    for i in range(60):
        rows.append(
            {
                "p": 0.55 + (i % 5) * 0.02,
                "y": 1.0 if i % 3 == 0 else 0.0,
                "line": 1.5,
                "side": "yes",
                "games_played": 60,
                "weight": 2,
            }
        )
    for i in range(35):
        rows.append(
            {
                "p": 0.4,
                "y": 0.0,
                "line": 2.5,
                "side": "no",
                "games_played": 10,
                "weight": 1,
            }
        )
    bundle = fit_segmented(rows, min_global=50, min_segment=30)
    assert bundle is not None
    assert bundle.global_cal is not None
    assert segment_key(line=1.5, side="yes", games_played=60) in bundle.segments
    out = bundle.transform(0.55, line=1.5, side="yes", games_played=60)
    assert 0.0 < out < 1.0
