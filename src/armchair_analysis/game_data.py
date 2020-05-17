import pandas as pd
from . import datadir


class SpreadData:
    def __init__(self):

        self.dataframe = self.game_info.merge(
            self.team_info,
            left_on=['game_id', 'team_home'],
            right_on=['game_id', 'team'],
        ).merge(
            self.team_info,
            left_on=['game_id', 'team_away'],
            right_on=['game_id', 'team'],
            suffixes=('_home', '_away')
        )

        dup_cols = self.dataframe.columns.duplicated()
        self.dataframe = self.dataframe.loc[:, ~dup_cols]

        order_cols = [
            'game_id',
            'season',
            'week',
            'day',
            'over/under',
            'away_spread',
            'temperature_game',
            'humidity_game',
            'wind_speed_game',
            'condition_game',
            *(c for c in self.dataframe.columns if c.endswith('_home')),
            *(c for c in self.dataframe.columns if c.endswith('_away'))
        ]

        self.dataframe = self.dataframe[order_cols]

    @property
    def game_info(self):
        """
        Load game info, subset to relevant columns, and remap column names

        """
        df_game = pd.read_csv(datadir / 'GAME.csv')

        game_cols = {
            'gid': 'game_id',
            'seas': 'season',
            'wk': 'week',
            'day': 'day',
            'h': 'team_home',
            'v': 'team_away',
            'ou': 'over/under',
            'sprv': 'away_spread',
            'temp': 'temperature_game',
            'humd': 'humidity_game',
            'wspd': 'wind_speed_game',
            'cond': 'condition_game',
        }

        df_game = df_game[game_cols.keys()].rename(game_cols, axis=1)

        return df_game

    @property
    def team_info(self):
        """
        Load team info, subset to relevant columns, and remap column names

        """
        df_team = pd.read_csv(datadir / 'TEAM.csv')

        team_cols = {
            'gid': 'game_id',
            'tname': 'team',
            'pts': 'pts',
            'ry': 'rush_yds',
            'ra': 'rush_att',
            'py': 'pass_yds',
            'pa': 'pass_att',
            'pc': 'pass_comp',
            'sk': 'sacks',
            'sky': 'sack_yds',
            'ints': 'ints',
            'iry': 'int_yds',
            'fum': 'fumbles',
            'pu': 'punts',
            'gpy': 'punt_yds',
            'td': 'touchdowns',
            'fgm': 'field_goals',
            'fgat': 'field_goal_att',
            'pen': 'penalty_yds',
            'top': 'possess_time'
        }

        df_team = df_team[team_cols.keys()].rename(team_cols, axis=1)

        return df_team
