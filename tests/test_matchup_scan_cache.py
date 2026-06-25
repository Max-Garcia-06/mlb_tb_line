"""Matchup helpers used by scan should not repeat heavy work per player."""

from unittest.mock import patch

from matchup_features import build_opp_tb_allowed_lookup, live_matchup_overrides, slate_teams


def test_live_matchup_overrides_uses_passed_slate():
    slate = {"BOSNYY": {"away": "BOS", "home": "NYY", "away_sp_hand": "R", "home_sp_hand": "L"}}
    opp = {"NYY": 1.42}
    with patch("matchup_features.build_slate_matchup_index") as mock_build:
        out = live_matchup_overrides(
            game_date="2026-05-23",
            player_team="BOS",
            event_ticker="KXMLBTB-26MAY201940BOSNYY",
            tb_roll=1.1,
            bats_hand="L",
            slate=slate,
            opp_tb_allowed_by_team=opp,
        )
        mock_build.assert_not_called()
    assert out["is_home"] == 0.0
    assert out["opp_sp_hand_L"] == 1.0
    assert out["opp_tb_allowed_roll"] == 1.42


def test_slate_teams_collects_abbreviations():
    slate = {"BOSNYY": {"away": "BOS", "home": "NYY", "away_sp_hand": "R", "home_sp_hand": "L"}}
    assert slate_teams(slate) == frozenset({"BOS", "NYY"})


def test_build_opp_tb_allowed_lookup_empty_teams():
    assert build_opp_tb_allowed_lookup("2099-01-01", frozenset()) == {}
