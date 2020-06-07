import pandas as pd
from armchair_analysis import datadir


class GameData:
    """
    Load armchairanalysis.com NFL game data and preprocess
    it into a form that is useful for calibrating a point spread
    forecasting model.

    """
    def __init__(self):

        self.dataframe = self.game_info.merge(
            self.schedule_info,
            on='game_id',
        ).merge(
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

        self.dataframe['week_id'] = (
            100*self.dataframe.season + self.dataframe.week)

        order_cols = [
            'game_id',
            'week_id',
            'date',
            'season',
            'week',
            'day',
            'team_home',
            'team_away',
            'over/under',
            'spread',
            *(c for c in self.dataframe.columns if c.startswith('gm_')),
            *(c for c in self.dataframe.columns if c.startswith('tm_')),
        ]

        self.dataframe = self.dataframe[order_cols]

        self.dataframe["outcome"] = (
            self.dataframe.tm_pts_home -
            self.dataframe.tm_pts_away -
            self.dataframe.spread
        )

    @property
    def game_info(self):
        """
        Load game info, subset to relevant columns, and remap column names

        """
        df_game = pd.read_csv(datadir / 'GAME.csv.gz')

        game_cols = {
            'gid': 'game_id',
            'seas': 'season',
            'wk': 'week',
            'day': 'day',
            'h': 'team_home',
            'v': 'team_away',
            'ou': 'over/under',
            'sprv': 'spread',
            'temp': 'gm_temperature',
            'humd': 'gm_humidity',
            'wspd': 'gm_wind_speed',
        }

        df_game = df_game[game_cols.keys()].rename(game_cols, axis=1)

        return df_game

    @property
    def schedule_info(self):
        """
        Load game date from schedule

        """
        df_schedule = pd.read_csv(datadir / 'SCHEDULE.csv.gz')

        sched_cols = {
            'gid': 'game_id',
            'date': 'date',
        }

        df_schedule = df_schedule[sched_cols.keys()].rename(sched_cols, axis=1)

        return df_schedule

    @property
    def team_info(self):
        """
        Load team info, subset to relevant columns, and remap column names

        """
        df_team = pd.read_csv(datadir / 'TEAM.csv.gz')

        team_cols = {
            'gid': 'game_id',
            'tname': 'team',
            'pts': 'tm_pts',
            'ry': 'tm_rush_yds',
            'ra': 'tm_rush_att',
            'py': 'tm_pass_yds',
            'pa': 'tm_pass_att',
            'pc': 'tm_pass_comp',
            'sk': 'tm_sacks',
            'sky': 'tm_sack_yds',
            'ints': 'tm_ints',
            'iry': 'tm_int_yds',
            'fum': 'tm_fumbles',
            'pu': 'tm_punts',
            'gpy': 'tm_punt_yds',
            'td': 'tm_touchdowns',
            'fgm': 'tm_field_goals',
            'fgat': 'tm_field_goal_att',
            'pen': 'tm_penalty_yds',
            'top': 'tm_possess_time'
        }

        df_team = df_team[team_cols.keys()].rename(team_cols, axis=1)

        return df_team


game_data = GameData()
