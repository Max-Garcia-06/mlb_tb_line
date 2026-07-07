"""ETL-path pitcher-hand resolution must use pitcher ids, not name search."""

from unittest.mock import patch

from data_engine import _opp_sp_hand_L_by_game_id


def test_opp_sp_hand_l_uses_id_based_lookup_not_name_search():
    _opp_sp_hand_L_by_game_id.cache_clear()
    game = {
        "game_id": 824089,
        "away_id": 143,
        "home_id": 118,
        "away_probable_pitcher": "Cristopher Sánchez",
        "home_probable_pitcher": "Noah Cameron",
    }
    with patch("data_engine.schedule_games_by_date", return_value=[game]), patch(
        "data_engine._team_id_to_abbr", return_value={143: "PHI", 118: "KC"}
    ), patch(
        "data_engine.get_probable_starters", return_value={"PHI": 111, "KC": 222}
    ), patch(
        "data_engine._pitcher_hand_from_id", side_effect=lambda pid: "L" if pid == 111 else "R"
    ) as mock_hand, patch(
        "data_engine.statsapi.lookup_player"
    ) as mock_lookup:
        out = _opp_sp_hand_L_by_game_id("2026-07-06")

    mock_lookup.assert_not_called()
    mock_hand.assert_any_call(111)
    mock_hand.assert_any_call(222)
    assert out[824089] == (1.0, 0.0)  # away (PHI, pid 111) = L, home (KC, pid 222) = R
