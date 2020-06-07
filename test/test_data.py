from datetime import datetime
from armchair_analysis.game_data import game_data


def test_game_counts():
    """
    Each team should play 16 regular season games per year,
    and there should be 11 post season games total.

    """
    now = datetime.now()

    # data is only complete for previous seasons
    all_games = game_data.dataframe
    all_games = all_games.loc[all_games.season < now.year]

    # each team plays 16 regular season games
    games = all_games[(all_games.week <= 17)]
    away_count = games.groupby(by=['team_away', 'season']).size()
    home_count = games.groupby(by=['team_home', 'season']).size()
    total_count = away_count + home_count
    assert all(count == 16 for count in total_count)

    # each season has 11 playoff games
    games = all_games[(all_games.week > 17)]
    total_count = games.groupby(by=['season']).size()
    assert all(count == 11 for count in total_count)


def test_value_ranges():
    """
    All box score numbers should have reasonable values

    """
    now = datetime.now()

    feature_lim = {
        'season': (2000, now.year),
        'week': (1, 21),
        'over/under': (30, 65),
        'spread': (-27, 27),
        'gm_temperature': (-10, 110),
        'gm_humidity': (0, 100),
        'gm_wind_speed': (0, 40),
        'tm_pts_home': (0, 65),
        'tm_rush_yds_home': (-20, 425),
        'tm_rush_att_home': (0, 60),
        'tm_pass_yds_home': (-10, 550),
        'tm_pass_att_home': (0, 70),
        'tm_pass_comp_home': (0, 45),
        'tm_sacks_home': (0, 12),
        'tm_sack_yds_home': (0, 100),
        'tm_ints_home': (0, 7),
        'tm_int_yds_home': (-25, 200),
        'tm_fumbles_home': (0, 7),
        'tm_punts_home': (0, 15),
        'tm_punt_yds_home': (0, 575),
        'tm_touchdowns_home': (0, 8),
        'tm_field_goals_home': (0, 8),
        'tm_field_goal_att_home': (0, 8),
        'tm_penalty_yds_home': (0, 200),
        'tm_possess_time_home': (15, 48),
        'tm_pts_away': (0, 65),
        'tm_rush_yds_away': (-20, 425),
        'tm_rush_att_away': (0, 60),
        'tm_pass_yds_away': (-10, 550),
        'tm_pass_att_away': (0, 70),
        'tm_pass_comp_away': (0, 45),
        'tm_sacks_away': (0, 12),
        'tm_sack_yds_away': (0, 100),
        'tm_ints_away': (0, 7),
        'tm_int_yds_away': (-25, 200),
        'tm_fumbles_away': (0, 7),
        'tm_punts_away': (0, 15),
        'tm_punt_yds_away': (0, 575),
        'tm_touchdowns_away': (0, 8),
        'tm_field_goals_away': (0, 8),
        'tm_field_goal_att_away': (0, 8),
        'tm_penalty_yds_away': (0, 200),
        'tm_possess_time_away': (15, 48),
    }

    for feature, (vmin, vmax) in feature_lim.items():
        vals = game_data.dataframe[feature]
        assert (vmin <= vals.min()) and (vals.max() <= vmax)
