from datetime import datetime, timedelta, timezone

from market_snapshots import MarketSnapshot, load_snapshots_within_hours_of_start


def test_load_snapshots_within_hours_of_start(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    path = tmp_path / "snapshots" / f"tb_markets_{game_date}.jsonl"
    path.parent.mkdir(parents=True)

    start = datetime(2026, 5, 20, 19, 0, 0, tzinfo=timezone.utc)
    in_window = (start - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    too_early = (start - timedelta(hours=5)).isoformat().replace("+00:00", "Z")

    rows = [
        MarketSnapshot(
            captured_at=too_early,
            game_date=game_date,
            ticker="KXMLBTB-26MAY201940BOSKC-P-2",
            event_ticker="KXMLBTB-26MAY201940BOSKC",
            player_name="A",
            line=1.5,
            yes_ask=0.5,
            yes_bid=0.48,
            no_ask=0.52,
            no_bid=0.5,
        ),
        MarketSnapshot(
            captured_at=in_window,
            game_date=game_date,
            ticker="KXMLBTB-26MAY201940BOSKC-P-2",
            event_ticker="KXMLBTB-26MAY201940BOSKC",
            player_name="A",
            line=1.5,
            yes_ask=0.55,
            yes_bid=0.53,
            no_ask=0.47,
            no_bid=0.45,
        ),
    ]
    import json
    from dataclasses import asdict

    path.write_text("\n".join(json.dumps(asdict(r)) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr("market_snapshots.SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(
        "market_snapshots.slate_schedule_index",
        lambda _d: {
            "BOSKC": {
                "status": "Pre-Game",
                "start_utc": start,
            }
        },
    )

    snaps = load_snapshots_within_hours_of_start(game_date, within_hours=3.0)
    assert len(snaps) == 1
    assert snaps[0].yes_ask == 0.55
