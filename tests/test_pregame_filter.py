from dataclasses import dataclass

from data_engine import (
    filter_market_lines_pregame,
    game_status_allows_scan,
    matchup_slug,
    parse_kalshi_event_matchup,
)


def test_parse_kalshi_event_matchup():
    assert parse_kalshi_event_matchup("KXMLBTB-26MAY201940BOSKC") == ("BOS", "KC")
    assert parse_kalshi_event_matchup("KXMLBTB-26MAY201540SFAZ") == ("SF", "AZ")
    assert parse_kalshi_event_matchup("invalid") is None


def test_game_status_allows_scan():
    assert game_status_allows_scan("Pre-Game")
    assert game_status_allows_scan("Scheduled")
    assert not game_status_allows_scan("In Progress")
    assert not game_status_allows_scan("Final")


def test_matchup_slug():
    assert matchup_slug("TEX", "COL") == "TEXCOL"


@dataclass
class _Line:
    ticker: str
    event_ticker: str = ""


def test_filter_market_lines_pregame(monkeypatch):
    monkeypatch.setattr(
        "data_engine.matchup_status_map",
        lambda _date: {"TEXCOL": "In Progress", "BOSKC": "Pre-Game"},
    )
    lines = [
        _Line("KXMLBTB-26MAY201510TEXCOL-PLAYER-2", "KXMLBTB-26MAY201510TEXCOL"),
        _Line("KXMLBTB-26MAY201940BOSKC-PLAYER-2", "KXMLBTB-26MAY201940BOSKC"),
        _Line("MOCK-TICKER"),
    ]
    kept, excluded = filter_market_lines_pregame(lines, "2026-05-20")
    assert len(kept) == 2
    assert len(excluded) == 1
    assert excluded[0][1] == "TEXCOL"
    assert excluded[0][2] == "In Progress"
