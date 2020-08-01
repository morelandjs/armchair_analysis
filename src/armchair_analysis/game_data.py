import numpy as np
import pandas as pd

from . import datadir
import matplotlib.pyplot as plt

class GameData:
    """
    Load armchairanalysis.com NFL game data and preprocess
    it into a form that is suitable for calibrating a point spread
    forecasting model.

    """
    def __init__(self):

        # merge team and quarterback info
        team_qb_info = self.team_info.merge(
            self.qb_points, on=['game_id', 'team', 'qb']
        ).drop_duplicates(
            subset=['game_id', 'team']).reset_index(drop=True)

        # merge game stats tables
        self.dataframe = self.game_info.merge(
            team_qb_info,
            left_on=['game_id', 'team_home'],
            right_on=['game_id', 'team'],
        ).drop(columns='team').merge(
            team_qb_info,
            left_on=['game_id', 'team_away'],
            right_on=['game_id', 'team'],
            suffixes=('_home', '_away'),
            how='left')

        # create week identifier column
        self.dataframe['week_id'] = (
            100*self.dataframe.season + self.dataframe.week)

        # add calculated columns
        self.dataframe = self.calculated_columns(self.dataframe)
        columns = self.dataframe.columns

        # rearrange columns
        order_cols = [
            'game_id',
            'week_id',
            'date',
            'season',
            'week',
            'day',
            'team_away',
            'team_home',
            'qb_away',
            'qb_home',
            'qb_prev_away',
            'qb_prev_home',
            'spread_vegas',
            'total_vegas',
            *(c for c in sorted(columns) if c.startswith('gm_')),
            *(c for c in sorted(columns) if c.startswith('tm_'))]

        # arrange columns
        self.dataframe = self.dataframe[order_cols]

    @property
    def game_info(self):
        """
        Load game info, subset to relevant columns, and remap column names

        """
        df_game = pd.read_csv(datadir / 'GAME.csv.gz')
        df_schedule = pd.read_csv(datadir / 'SCHEDULE.csv.gz')
        df_schedule.date = pd.to_datetime(df_schedule.date)

        # pull game date from schedule table
        df_game = df_game.merge(df_schedule[['gid', 'date']], on='gid')

        game_cols = {
            'gid': 'game_id',
            'date': 'date',
            'seas': 'season',
            'wk': 'week',
            'day': 'day',
            'h': 'team_home',
            'v': 'team_away',
            'ou': 'total_vegas',
            'sprv': 'spread_vegas',
            'temp': 'gm_temperature',
            'humd': 'gm_humidity',
            'wspd': 'gm_wind_speed',
            'ptsv': 'tm_pts_away',
            'ptsh': 'tm_pts_home'}

        df_game = df_game[game_cols.keys()].rename(game_cols, axis=1)

        return df_game

    @property
    def quarterback_info(self):
        """
        QB specific attributes

        """
        df_pass = pd.read_csv(datadir / 'PASS.csv.gz')
        df_play = pd.read_csv(datadir / 'PLAY.csv.gz')
        df_player = pd.read_csv(datadir / 'PLAYER.csv.gz')

        df_pass_plays = df_pass[['pid', 'psr']].merge(
            df_play[['gid', 'pid', 'off', 'def']], on='pid')

        df_qbs = df_player[(df_player.pos1 == 'QB') | (df_player.pos2 == 'QB')]
        df_qb_pass_plays = df_pass_plays[df_pass_plays.psr.isin(df_qbs.player)]

        df_qb = df_qb_pass_plays.groupby(
            by=['gid', 'off']
        ).agg({'psr': 'first'}).reset_index()

        qb_cols = {
            'gid': 'game_id',
            'off': 'team',
            'psr': 'qb'}

        df_qb = df_qb[qb_cols.keys()].rename(qb_cols, axis=1)

        return df_qb

    @property
    def player_name(self):
        """
        Return dictionay with keys equal to player id, and values equal
        to player name

        """
        df_player = pd.read_csv(datadir / 'PLAYER.csv.gz')

        player_name = dict(zip(df_player.player, df_player.pname))

        return player_name

    @property
    def team_info(self):
        """
        Load team info, subset to relevant columns, and remap column names

        """
        df_team = pd.read_csv(datadir / 'TEAM.csv.gz')

        team_cols = {
            'gid': 'game_id',
            'tname': 'team',
            #'pts': 'tm_pts',
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

        df_team = df_team.merge(self.quarterback_info, on=['game_id', 'team'])

        return df_team

    @property
    def qb_points(self):
        """
        Compute expected points added by each QB per play

        """
        # load required datasets
        df_play = pd.read_csv(datadir / 'PLAY.csv.gz')
        df_sched = pd.read_csv(datadir / 'SCHEDULE.csv.gz')
        df_rush = pd.read_csv(datadir / 'RUSH.csv.gz')
        df_drive = pd.read_csv(datadir / 'DRIVE.csv.gz')

        # Merge schedule and play data
        df_play = df_sched[['date', 'gid', 'v', 'h']].merge(
            df_play[['gid', 'pid', 'off', 'def', 'epa', 'eps']], on='gid')

        # Merge starting quarterback and rush info
        df_play = df_play.merge(
            self.quarterback_info,
            left_on=['gid', 'off'], right_on=['game_id', 'team']
        ).merge(df_rush[['pid', 'bc']], on='pid', how='left')

        # Drop non-qb rushing plays
        non_qb_rusher = ~df_play.bc.isna() & (df_play.qb != df_play.bc)
        df_play_filtered = df_play[~non_qb_rusher]

        # Aggregate qb expected points added (EPA)
        df_qb_epa = df_play_filtered.groupby(
            by=['gid', 'date', 'qb', 'off', 'def', 'h', 'v']
        ).agg({'epa': 'sum'}).reset_index()

        # Subtract expected points starting (EPS) yielded to opponent
        df_first_play = df_play.merge(
            df_drive[['gid', 'fpid']],
            left_on=['gid', 'pid'], right_on=['gid', 'fpid'])

        # Aggregate each team's expected points starting (EPS)
        df_qb_eps = df_first_play.groupby(
            by=['gid', 'team']
        ).agg({'eps': 'sum'}).reset_index()
        df_qb_eps['eps'] -= df_qb_eps.eps.mean()

        # Merge EPA and EPS values
        df_qb = df_qb_epa.merge(
            df_qb_eps, left_on=['gid', 'def'], right_on=['gid', 'team'])

        # Compute "QB points"
        df_qb['tm_qb_pts'] = df_qb.epa #- df_qb.eps

        qb_cols = {
            'gid': 'game_id',
            'off': 'team',
            'qb': 'qb',
            'tm_qb_pts': 'tm_qb_pts'}

        df_qb = df_qb[qb_cols.keys()].rename(qb_cols, axis=1)

        return df_qb

    def rest_days(self, games):
        """
        Compute home and away teams days rested

        """
        game_dates = pd.concat([
            games[["date", "team_home"]].rename(
                columns={"team_home": "team"}),
            games[["date", "team_away"]].rename(
                columns={"team_away": "team"}),
        ]).sort_values("date")

        game_dates['date_prev'] = game_dates.date

        game_dates = pd.merge_asof(
            game_dates[['team', 'date']],
            game_dates[['team', 'date', 'date_prev']],
            on='date', by='team', allow_exact_matches=False)

        for team in ["home", "away"]:

            game_dates_team = game_dates.rename(
                columns={
                    'date_prev': f'date_{team}_prev',
                    'team': f'team_{team}'})

            games = games.merge(game_dates_team, on=['date', f'team_{team}'])

        one_day = pd.Timedelta("1 days")
        games["tm_rest_days_home"] = np.clip(
            (games.date - games.date_home_prev) / one_day, 3, 16).fillna(7)
        games["tm_rest_days_away"] = np.clip(
            (games.date - games.date_away_prev) / one_day, 3, 16).fillna(7)

        return games

    def previous_quarterback(self, games):
        """
        Keep track of previous quarterback for each game and each team.

        """
        game_dates = pd.concat([
            games[["date", "team_home", "qb_home"]].rename(
                columns={"team_home": "team", "qb_home": "qb"}),
            games[["date", "team_away", "qb_away"]].rename(
                columns={"team_away": "team", "qb_away": "qb"}),
        ]).sort_values("date")

        game_dates['qb_prev'] = game_dates.qb

        game_dates = pd.merge_asof(
            game_dates[['team', 'date']],
            game_dates[['team', 'date', 'qb_prev']],
            on='date', by='team', allow_exact_matches=False)

        for team in ["home", "away"]:

            game_dates_team = game_dates.rename(
                columns={'qb_prev': f'qb_prev_{team}',
                         'team': f'team_{team}'})

            games = games.merge(game_dates_team, on=['date', f'team_{team}'])

        return games

    def calculated_columns(self, games):
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

        # compute rest days for each team
        games = self.rest_days(games)

        # record previous game's qb
        games = self.previous_quarterback(games)

        return games


game_data = GameData()
