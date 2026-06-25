"""Point-in-time training and snapshot loading."""

from __future__ import annotations

import pandas as pd

from market_snapshots import MarketSnapshot, load_snapshots


def test_load_snapshots_earliest_vs_latest(tmp_path, monkeypatch):
    from market_snapshots import SNAPSHOT_DIR, snapshot_path

    monkeypatch.setattr("market_snapshots.SNAPSHOT_DIR", tmp_path)
    gd = "2024-06-01"
    path = snapshot_path(gd)
    rows = [
        MarketSnapshot(
            captured_at="2024-06-01T10:00:00+00:00",
            game_date=gd,
            ticker="T1",
            event_ticker="E1",
            player_name="A",
            line=1.5,
            yes_ask=0.5,
            yes_bid=0.48,
            no_ask=0.52,
            no_bid=0.5,
        ),
        MarketSnapshot(
            captured_at="2024-06-01T18:00:00+00:00",
            game_date=gd,
            ticker="T1",
            event_ticker="E1",
            player_name="A",
            line=1.5,
            yes_ask=0.6,
            yes_bid=0.58,
            no_ask=0.42,
            no_bid=0.4,
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in rows:
            import json
            from dataclasses import asdict

            f.write(json.dumps(asdict(s)) + "\n")

    early = load_snapshots(gd, earliest_only=True)
    late = load_snapshots(gd, latest_only=True)
    assert len(early) == 1
    assert early[0].yes_ask == 0.5
    assert late[0].yes_ask == 0.6


def test_prepare_data_as_of_excludes_future(monkeypatch):
    import model as m

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "game_date": dates,
            "tb": 1.0,
            **{c: 0.1 for c in m.MODEL_FEATURES},
        }
    )
    monkeypatch.setattr(m, "build_feature_table", lambda: df)
    X, y, dff = m.prepare_data_as_of("2024-01-15")
    assert len(dff) == 14
    assert pd.to_datetime(dff["game_date"]).max() < pd.Timestamp("2024-01-15")
