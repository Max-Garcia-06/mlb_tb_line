"""ID-based pitcher-hand resolution and the decorrelated platoon feature."""

from unittest.mock import patch

import pandas as pd
import pytest

from matchup_features import (
    MATCHUP_FEATURE_NAMES,
    _pitcher_hand_from_id,
    attach_platoon_features,
    finalize_matchup_columns,
    live_matchup_overrides,
)


def test_pitcher_hand_from_id_resolves_left():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get") as mock_get:
        mock_get.return_value = {"people": [{"pitchHand": {"code": "L"}}]}
        assert _pitcher_hand_from_id(477132) == "L"
    mock_get.assert_called_once_with("person", {"personId": 477132})


def test_pitcher_hand_from_id_defaults_to_right_on_missing_data():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get") as mock_get:
        mock_get.return_value = {"people": []}
        assert _pitcher_hand_from_id(1) == "R"


def test_pitcher_hand_from_id_defaults_to_right_on_exception():
    _pitcher_hand_from_id.cache_clear()
    with patch("matchup_features.statsapi.get", side_effect=Exception("boom")):
        assert _pitcher_hand_from_id(2) == "R"


def test_matchup_feature_names_drops_expected_pa_proxy_and_renames_platoon():
    assert "expected_pa_proxy" not in MATCHUP_FEATURE_NAMES
    assert "platoon_tb_adj" not in MATCHUP_FEATURE_NAMES
    assert "platoon_edge" in MATCHUP_FEATURE_NAMES


def test_attach_platoon_features_produces_decorrelated_edge():
    df = pd.DataFrame(
        {
            "opp_sp_hand_L": [1.0, 0.0, 0.0],
            "bats_hand": ["L", "R", "L"],
            "tb_roll": [2.0, 1.0, 0.5],
        }
    )
    out = attach_platoon_features(df)
    # LHB vs LHP (opp_sp_hand_L=1.0) -> no platoon edge
    assert out["platoon_edge"].iloc[0] == pytest.approx(0.0)
    # RHB vs RHP (opp_sp_hand_L=0.0) -> no platoon edge
    assert out["platoon_edge"].iloc[1] == pytest.approx(0.0)
    # LHB vs RHP (opp_sp_hand_L=0.0) -> platoon edge, independent of tb_roll
    assert out["platoon_edge"].iloc[2] == pytest.approx(0.08)


def test_finalize_matchup_columns_defaults_platoon_edge_to_zero():
    df = pd.DataFrame({"lineup_slot": [5]})
    out = finalize_matchup_columns(df)
    assert "expected_pa_proxy" not in out.columns
    assert out["platoon_edge"].iloc[0] == 0.0


def test_live_matchup_overrides_returns_platoon_edge_not_pa_proxy():
    # BOS (away) batter is L, facing NYY's (home) starter who is also L per the
    # slate ("home_sp_hand": "L") -> opp_sp_hand_L=1.0 -> same-handed matchup
    # (LHB vs LHP), no platoon edge.
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
    assert "expected_pa_proxy" not in out
    assert out["opp_sp_hand_L"] == 1.0
    assert out["platoon_edge"] == 0.0

    # Same slate, but a right-handed batter facing that same left-handed NYY
    # starter -> RHB vs LHP is the opposite-handed (platoon-advantage) matchup,
    # boost=1.06 -> platoon_edge=0.06.
    out_r = live_matchup_overrides(
        game_date="2026-05-23",
        player_team="BOS",
        event_ticker="KXMLBTB-26MAY201940BOSNYY",
        tb_roll=1.1,
        bats_hand="R",
        slate=slate,
        opp_tb_allowed_by_team=opp,
    )
    assert out_r["platoon_edge"] == pytest.approx(0.06)
