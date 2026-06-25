from kalshi_bridge import MarketLine
from market_snapshots import MarketSnapshot, append_snapshots, load_snapshots, snapshot_path


def test_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("market_snapshots.SNAPSHOT_DIR", tmp_path)
    ml = MarketLine(
        ticker="KXMLBTB-26MAY201940BOSKC-P-2",
        player_name="Test Player",
        player_id=1,
        game_date="2026-05-20",
        line=1.5,
        yes_ask=0.52,
        yes_bid=0.50,
        no_ask=0.52,
        no_bid=0.50,
        event_ticker="KXMLBTB-26MAY201940BOSKC",
    )
    append_snapshots("2026-05-20", [ml])
    snaps = load_snapshots("2026-05-20")
    assert len(snaps) == 1
    assert snaps[0].player_name == "Test Player"
    assert snapshot_path("2026-05-20").exists()
