import numpy as np
import pandas as pd

from . import datadir


class GameData:
    """
    Load armchairanalysis.com NFL game data and preprocess
    it into a form that is useful for calibrating a point spread
    forecasting model.

    """
    def __init__(self):

        # merge team info with starting qb
        team_info = self.team_info.merge(
                self.quarterback_info,
                on=['game_id', 'team'])

        # merge game stats tables
        self.dataframe = self.game_info.merge(
            self.schedule_info,
            on='game_id',
        ).merge(
            team_info,
            left_on=['game_id', 'team_home'],
            right_on=['game_id', 'team'],
        ).merge(
            team_info,
            left_on=['game_id', 'team_away'],
            right_on=['game_id', 'team'],
            suffixes=('_home', '_away')
        )

        # remove duplicate columns
        dup_cols = self.dataframe.columns.duplicated()
        self.dataframe = self.dataframe.loc[:, ~dup_cols]

        # create week identifier column
        self.dataframe['week_id'] = (
            100*self.dataframe.season + self.dataframe.week)

        # create point-total outcome column
        self.dataframe["total_outcome"] = (
            self.dataframe.tm_pts_home + self.dataframe.tm_pts_away)

        # create point-spread outcome column
        self.dataframe["spread_outcome"] = (
            self.dataframe.tm_pts_home - self.dataframe.tm_pts_away)

        # rearrange columns
        order_cols = [
            'game_id',
            'week_id',
            'date',
            'season',
            'week',
            'day',
            'team_home',
            'team_away',
            'qb_home',
            'qb_away',
            'total_vegas',
            'total_outcome',
            'spread_vegas',
            'spread_outcome',
            *(c for c in self.dataframe.columns if c.startswith('gm_')),
            *(c for c in self.dataframe.columns if c.startswith('tm_'))]

        # preprocess raw game data for modelling
        self.dataframe = self.preprocess(self.dataframe[order_cols])

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
            'ou': 'total_vegas',
            'sprv': 'spread_vegas',
            'temp': 'gm_temperature',
            'humd': 'gm_humidity',
            'wspd': 'gm_wind_speed'}

        df_game = df_game[game_cols.keys()].rename(game_cols, axis=1)

        return df_game

    @property
    def quarterback_info(self):
        """
        QB specific attributes

        """
        df_play = pd.read_csv(datadir / 'PLAY.csv.gz')
        df_play = df_play[['gid', 'pid', 'off', 'def']]

        df_pass = pd.read_csv(datadir / 'PASS.csv.gz')
        df_pass = df_pass[['pid', 'psr']]

        df_qb = df_play.merge(df_pass, on='pid')

        df_qb = df_qb.groupby(
            by=['gid', 'off']
        ).agg({'psr': 'first'}).reset_index()

        qb_cols = {
            'gid': 'game_id',
            'off': 'team',
            'psr': 'qb'}

        df_qb = df_qb[qb_cols.keys()].rename(qb_cols, axis=1)

        return df_qb

    @property
    def schedule_info(self):
        """
        Load game date from schedule

        """
        df_schedule = pd.read_csv(datadir / 'SCHEDULE.csv.gz')

        sched_cols = {'gid': 'game_id', 'date': 'date'}

        df_schedule = df_schedule[sched_cols.keys()].rename(sched_cols, axis=1)
        df_schedule.date = pd.to_datetime(df_schedule.date)

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
            'fgm': 'tm_field_goals',
            'fgat': 'tm_field_goal_att',
            'pen': 'tm_penalty_yds',
            'top': 'tm_possess_time',
            'tdp': 'tm_pass_tds',
            'tdr': 'tm_rush_tds',
            'td': 'tm_tds',
            'qba': 'tm_qb_rush_att',
            'qby': 'tm_qb_rush_yds'}

        df_team = df_team[team_cols.keys()].rename(team_cols, axis=1)

        return df_team

    def preprocess(self, games):
        """
        Preprocesses raw game data, returning a model input table.

        This function calculates some new columns and adds them to the
        games table:

                column  description
                  home  home team name joined to home quarterback name
                  away  away team name joined to away quarterback name
        rest_days_home  home team days rested
        rest_days_away  away team days rested
              exp_home  games played by the home quarterback
              exp_away  games played by the away quarterback

        """
        # sort games by date
        games = games.sort_values(by=["date", "team_home"])

        # give jacksonville jaguars a single name
        games.replace("JAC", "JAX", inplace=True)

        # give teams which haved moved cities their current name
        games.replace("SD", "LAC", inplace=True)
        games.replace("STL", "LA", inplace=True)

        # game dates for every team
        game_dates = pd.concat([
            games[["date", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["date", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("date")

        # game dates for every team
        game_dates = pd.concat([
            games[["date", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["date", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("date")

        # compute days rested
        for team in ["home", "away"]:
            games_prev = game_dates.rename(
                columns={"team": "team_{}".format(team)})

            games_prev["date_{}_prev".format(team)] = games.date

            games = pd.merge_asof(
                games, games_prev,
                on="date", by="team_{}".format(team),
                allow_exact_matches=False
            )

        # days rested since last game
        one_day = pd.Timedelta("1 days")
        games["rest_days_home"] = np.clip(
            (games.date - games.date_home_prev) / one_day, 3, 16).fillna(7)
        games["rest_days_away"] = np.clip(
            (games.date - games.date_away_prev) / one_day, 3, 16).fillna(7)

        return games


game_data = GameData()
