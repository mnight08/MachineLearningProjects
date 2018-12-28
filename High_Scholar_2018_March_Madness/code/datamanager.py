# -*- coding: utf-8 -*-
"""Manages access to the underlying Marchmadness data.
This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
A datamanager is responsible for all access to data.
The methods are of the form:
load_{what to load}(args)
get_{what to get}(args)
make_{what to make}(args)
load methods pull data into memory.  get methods return copies of tables
loaded from disk,
or loads tables if they are missing.  make methods generate content that is
saved into the build folder.
takes a data set, loads it from disk, caches the data, and
returns dat  data frame for the given year from the events data.
Created on Fri May 18 16:32:47 2018
This file will be responsible for transorming the file system data into
appropriate dataframes. Any code that neeeds to access the data from disk will
should do so from data manager.  A DataManager object will have a local copy of
each table stored in the directory.
@author: vpx365
"""
import os
import functools
import itertools
import pandas as pd
import numpy as np
class DataManager:
    """Manages access to the underlying Marchmadness data.
    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.
    Example:
    A datamanager is responsible for all access to data.
    The methods are of the form:
    load_{what to load}(args)
    get_{what to get}(args)
    make_{what to make}(args)
    load methods pull data into memory.  get methods return copies of tables
    loaded from disk,
    or loads tables if they are missing.  make methods generate content that is
    saved into the build folder.
    takes a data set, loads it from disk, caches the data, and
    returns dat  data frame for the given year from the events data.
    """
    def __init__(self):
        self.load_path = "../../Learning_Data/Mens-March-Madness_2018/"
        self.make_path = "../build/"

    def load_team_data(self):
        '''Returns data file for teams.csv'''
        path = self.load_path+"DataFiles/"+"/Teams.csv"
        return pd.read_csv(path)
    def load_coach_data(self):
        path = self.load_path+"DataFiles/"+"/TeamCoaches.csv"
        return pd.read_csv(path)

    @functools.lru_cache(maxsize=32)
    def load_pbp_players(self, year):
        """
        This will create data frame for the given year from the events data.
        Caching is used to speed up access to this data.
        """
        path = self.load_path+"PlayByPlay_"+str(year)+"/Players_"+str(year)+".csv"
        return pd.read_csv(path)

    @functools.lru_cache(maxsize=32)
    def load_pbp_events(self, year):
        """
        This will create data frame for the given year from the events data.
        Caching leads to significant speedup here.
        """
        path = self.load_path+"PlayByPlay_"+str(year)+"/Events_"+str(year)+".csv"
        #print(path)
        return pd.read_csv(path)

    @functools.lru_cache(maxsize=32)
    def load_massey_ordinals(self, through2018=False):
        """
        Open the masseyordinals.  Use the 2018 data if the parameter is true.
        Default to use the older data.
        """
        if not through2018:
            path = self.load_path+"MasseyOrdinals/"+"MasseyOrdinals.csv"
        elif through2018:
            path = self.load_path+"MasseyOrdinals_thruSeason2018_Day128/"\
                 +"MasseyOrdinals_thruSeason2018_Day128.csv"
        return pd.read_csv(path)

    #@functools.lru_cache(maxsize=32)
    def load_ncaa_tourney_results(self, compact=False):
        """
        Load march madness tournament results.  There are two files, one that has
        detailed data about the tournament, and one that has compact data.
        The compact data goes back to 1985.  The detailed data goes back to 2003
        """
        if not compact:
            path = self.load_path+"DataFiles/"\
                 +"NCAATourneyDetailedResults.csv"
        elif compact:
            path = self.load_path+"DataFiles/"\
                 +"NCAATourneyCompactResults.csv"
        return pd.read_csv(path)

        #@functools.lru_cache(maxsize=32)
    def load_reg_season_results(self, compact=False):
        """
        Load march madness tournament results.  There are two files, one that has
        detailed data about the tournament, and one that has compact data.
        The compact data goes back to 1985.  The detailed data goes back to 2003
        """
        if not compact:
            path = self.load_path+"DataFiles/"\
                 +"RegularSeasonDetailedResults.csv"
        elif compact:
            path = self.load_path+"DataFiles/"\
                 +"RegularSeasonCompactResults.csv"
        return pd.read_csv(path)

    #@functools.lru_cache(maxsize=32)
    def load_secondary_tourney_results(self):
        """
        Load march madness tournament results.  There are two files, one that has
        detailed data about the tournament, and one that has compact data.
        The compact data goes back to 1985.  The detailed data goes back to 2003
        """
        path = self.load_path+"DataFiles/SecondaryTourneyCompactResults.csv"
        return pd.read_csv(path)


    #@functools.lru_cache(maxsize=32)
    def load_conference_tourney_games(self, stage2=False):
        """ Returns a dataframe containing conference tourney games.
        """
        path = self.load_path
        if not stage2:
            path = path+"DataFiles/"
        else:
            path = path+"Stage2UpdatedDataFiles/"

        path = path+"ConferenceTourneyGames.csv"
        return pd.read_csv(path)



    #@functools.lru_cache(maxsize=32)
    def load_cities(self, stage2=False):
        """load city data from datafiles folder.
           Caching is not used since test did better without.
        """
        path = self.load_path
        if not stage2:
            path = path+"DataFiles/"
        else:
            path = path+"Stage2UpdatedDataFiles/"

        path = path+"Cities.csv"
        return pd.read_csv(path)




    #@functools.lru_cache(maxsize=32)
    def load_conferences(self):
        '''Return conference data.
        '''
        path = self.load_path+"DataFiles/Conferences.csv"
        return pd.read_csv(path)



    def load_variables(self, blacklisted=None):
        '''Return an list of file paths and column names as latex.
        Below is the code to create a enumeratede list of the of files for latex.
        '''
        tex = "\begin{enumerate}\n"
        variables = []
        for name in os.listdir(self.load_path):
            if os.path.isdir(self.load_path+name):
                for filename in os.listdir(self.load_path+name):
                    if not blacklisted is None and filename not in blacklisted \
                        and filename.endswith(".csv"):
                        variables.append((name+"/"+filename,
                                          pd.read_csv(self.load_path+name+"/"+
                                                      filename,
                                                      nrows=1).columns))
            elif name not in blacklisted and name.endswith(".csv"):
                variables.append((name, pd.read_csv(self.load_path+"/"+name,
                                                    nrows=1).columns))
            else:
                print("File was not opened because it is  black listed or not a csv file. "+name)

            for line in ["\item "+file.replace("_", "\_") for file, cols in variables]:
                tex = tex+line
        tex = tex+"\n\end{enumerate}"
        return  tex

    def get_player_event_data_by_id(self, player_ids=None, years=None):
        '''
        Get the event data for the given player id
        Data is only for that one id, which is valid only for one season.
        '''
        tables = [self.load_pbp_events(year) for year in years]
        table = pd.concat([tables.query("EventPlayerID == @player_id")
                           for player_id in player_ids])

        return table



    def get_num_games(self, teams, years=range(2010, 2019)):
        '''Return a list of team id and numbers of games played pairs by
        each team in the
        years given.
        '''
        games_played = []
        if isinstance(teams, int):
            teams = [teams]

        if isinstance(years, int):
            years = [years]

        year_data = self.get_pbp_events_teams(teams, years)
        for team in teams:
            win_mask = (year_data["WTeamID"] == team) & (year_data["Season"].isin(years))
            lose_mask = (year_data["LTeamID"] == team)&(year_data["Season"].isin(years))
            games_played.append((team, len(year_data[
                win_mask|lose_mask].groupby(['Season', 'DayNum', 'WTeamID',
                                             'LTeamID']))))

        return games_played

    def get_num_div1_years_coach(self, games):
        '''Takes in a dataframe of games played, and return a copy with the number of division 1
        years the coaches had the year of that game as appended columns

        '''
        coach_data = self.load_coach_data()
        coach_data['Div1'] = ((coach_data['LastDayNum']
                               -coach_data['FirstDayNum'])/154)
        coaches = self.get_coaches(games)
        cumm = coach_data[['CoachName', 'Season', 'Div1', 'TeamID']].groupby([
            'CoachName', 'Season', 'TeamID']).sum().groupby(
                level=[0]).cumsum()
        div1 = coaches.merge(cumm, left_on=['Season', 'TeamID'], right_on=[
            'Season', 'TeamID'])
        games = games.merge(div1, left_on=['Season', 'Team1'], right_on=[
            'Season', 'TeamID'], how="inner")
        games = games.drop('TeamID', axis=1).rename(columns={"Div1":"CDiv1"})
        games = games.merge(div1, left_on=['Season', 'Team2'], 
                            right_on=['Season', 'TeamID'], how="inner")
        games = games.drop('TeamID', axis=1).rename(columns={"Div1":"CDiv2"})
        return games.drop(["CoachName_x", "CoachName_y"], axis=1)

    def get_coaches(self, games):
        '''Returns the coaches that participated in the input games.
        Each row of the returned data frame is of the form
        'Season', 'TeamID', 'CoachName'
        Coaches will repeat for every season they played
        '''
        teams = self.get_teams(games)
        coach_data = self.load_coach_data()
        mask = (coach_data['TeamID'].isin(teams['TeamID']))& (
            coach_data['Season'].isin(teams['Season']))
        relevant_coaches = coach_data.loc[mask,['Season', 'TeamID', 'CoachName']]
        return relevant_coaches



    def get_win_rate_coach(self, games, period="All"):
        coaches = self.get_coaches(games)


    def get_num_div1_years_team(self, td):
        '''Takes in a dataframe of games played, and returns the data frame 
        with the number of division 1 years the each team had at the begining of the year
        appended as columns 'T1Div1', and 'T2Div1'.
        
        Example:
            td=get_num_div1_years_team(td)
        '''
        team_data = self.load_team_data()
        teams = self.get_teams(td)
        team_data = team_data[team_data['TeamID'].isin(teams['TeamID'])]
        td=td.merge(team_data[['TeamID', 'FirstD1Season']], left_on='Team1', right_on='TeamID', how='left')
        td=td.merge(team_data[['TeamID', 'FirstD1Season']], left_on='Team2', right_on='TeamID', how='left')
        td['TDiv1'] = td['Season']-td['FirstD1Season_x']
        td['TDiv2'] = td['Season']-td['FirstD1Season_y']        
        td=td.drop(['FirstD1Season_x','FirstD1Season_y', 'TeamID_x', 'TeamID_y'], axis=1)      
        return td

    def get_avg_ranking(self, td, n=10):
        '''
        Take a dataframe of games, and append ranking columns for the two teams
        that played each game. Ranking data for the n most popular systems is
        averaged for each team over each year.  The average from a previous
        year will be the ranking
        for a given game.
        '''
        rankings = self.load_massey_ordinals()
        teams = self.get_teams(td)
        seasons = [year for year in td['Season'].value_counts().index.values]
        seasons.append(min(seasons)-1)
        teams = teams['TeamID'].value_counts().index.values

        team_rankings = rankings[rankings["TeamID"].isin(teams)]
        team_rankings = rankings[rankings["Season"].isin(seasons)]

        system_use_count = team_rankings.groupby(['SystemName']).count()
        systems = system_use_count['TeamID'].nlargest(n).keys()
        systems_mask = (team_rankings['SystemName'].isin(systems))

        popular_rankings = team_rankings[systems_mask]

        #Average all ranking data for a given year.
        avg_rankings = popular_rankings.drop("RankingDayNum", axis=1).groupby([
            'TeamID', 'Season']).mean()


        #use the previous season ranking data as predictor of current season performance.

        td['MergeKey'] = td['Season']-1

        #system_avg_ranking=time_avg_rankings.groupby(['TeamID']).mean()

        #system_avg_ranking=system_avg_ranking['OrdinalRank'].to_frame()

        td = td.merge(avg_rankings, left_on=['Team1', 'MergeKey'], right_on=[
            'TeamID', 'Season'], how="left")


        td = td.rename(index=str, columns={'OrdinalRank':'Rank1'})


        td = td.merge(avg_rankings, left_on=['Team2', 'Season'], 
                      right_on=['TeamID', 'Season'], how="left")

        td = td.rename(index=str, columns={'OrdinalRank':'Rank2'})
        td = td.drop('MergeKey', axis=1)


        return td




    def get_game_types(self, games):
        '''Deterimine the type of a game.
        Returns: 'Regular', 'March Madness', 'Conference Tourney' 'Secondary Tourney',
        'Invalid'. Game is a tuple of the form (year,  day,  WTeamID,  LTeamID)
        '''
        pass
#        conf_tourney=self.dm.load_conference_tourney_games()[['Season','DayNum','WTeamID','LTeamID']]
#        regular_season=self.dm.load_regular_season_games()[['Season','DayNum','WTeamID','LTeamID']]
#        secoundary_tourney=self.dm.load_secondary_tourney_games()[['Season','DayNum','WTeamID','LTeamID']]
#        ncaa_tourney=self.dm.load_ncaa_tourney_results()[['Season','DayNum','WTeamID','LTeamID']]

        #Chekc if the game is a conference game.
#        if (np.array(game) == conf_tourney.drop_duplicates()).all(1).any():
#            return 'Conference Tourney'
 #       elif (np.array(game) == regular_season.drop_duplicates()).all(1).any():
  #          return 'Regular'
  #      elif (np.array(game) == ncaa_tourney.drop_duplicates()).all(1).any():
  #          return 'March Madness'
  #      elif (np.array(game) == secoundary_tourney.drop_duplicates()).all(1).any():
  #          return 'Secondary Tourney'
#        else:
             #print("Game not found.  Check tuple")
#             return "Invalid"



    def get_player_names(self, years=None):
        '''Returns a datatframe containing all player names for the given years.
        '''
        table = pd.concat([self.load_pbp_players(year)[
            'PlayerName'] for year in years])
        #print(table)
        return table.drop_duplicates()

    def get_winning_rate(self, td, who=None, when=None):
        '''appends wins/num games played for each team/coach 
        in each each game in td'''
        if who == "Coach":
            coaches = self.get_coaches(td)
            
        elif who == "Team":

            teams= self.get_teams(td)
            
        return td
    def get_teams(self, games):
        '''Return a list of team ids and year that played the given games'''
        team1 = games.loc[:, ['Team1', 'Season']]
        team1 = team1.rename(index=str, columns={"Team1": "TeamID"})
        team2 = games.loc[:, ['Team2', 'Season']]
        team2 = team2.rename(index=str, columns={"Team2": "TeamID"})
        return pd.concat([team1, team2], ignore_index=True).drop_duplicates()


    def get_games(self, years, period="All"):
        '''Returns a dataframe for the games played during the period given.
        The games will always have Team1 be the team with the lower Id.
        This data is pulled from "RegularSeasonCompactResults",
        "SecondaryTourneyCompactResults", "ConferenceTourneyGames", and
        "NCAATourneyCompactResults"
        The games in the files are not necessarily unique Duplicate rows will
        be dropped.

        Parameters:
            years - The years we want the data for.
            period - "All", "March Madness","Tourney", "Regular", "Secondary",
                    "Conference"
        Returns a dataframe for the games played during the period given with
        columns=['Season', 'DayNum', 'Team1', 'Team2', 'Team1W']
        '''
        if isinstance(years, int):
            years = [years]
        cols = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
        mm = self.load_ncaa_tourney_results(compact=True)[cols]
        mm = mm[mm['Season'].isin(years)]
        conf = self.load_conference_tourney_games()[cols]
        conf = conf[conf['Season'].isin(years)]
        reg = self.load_reg_season_results()[cols]
        reg = reg[reg['Season'].isin(years)]
        sec = self.load_secondary_tourney_results()[cols]
        sec = sec[sec['Season'].isin(years)]

        games = pd.DataFrame()
        if period == "All":
            games = pd.concat([mm, conf, reg, sec]).drop_duplicates()
        elif period == "March Madness":
            games = mm
        #Take reg, and remove any mm, conf, sec, game.
        elif period == "Regular":
            games = reg
            #tourney=reg[(reg[cols].isin(mm[cols]))|(reg[cols].isin(
            #        conf[cols]))|(reg[cols].isin(sec[cols]))]
            #games=reg.drop(tourney)
        #return only the secondary tourney games
        elif period == "Secondary":
            games = sec
        #return mm, conf, and sec concatenated together.
        elif period == "Tourney":
            games = pd.concat([mm, sec, conf]).drop_duplicates()
        elif period == "Conference":
            games = conf
        games.rename(columns={'WTeamID':'Team1',
                              'LTeamID':'Team2'}, inplace=True)
        games['Team1W'] = 1
        mask = games['Team1']>games['Team2']
        swap = games[mask].copy()
        games.loc[mask, 'Team1'] = games.loc[mask, 'Team2']
        games.loc[mask, 'Team2'] = swap.loc[:, 'Team1']
        games.loc[mask, 'Team1W'] = 0
        return games.reset_index(drop=True)


    def get_player_data_by_name(self, names=None, years=None):
        '''Returns the rows of the player.csv files in the play by play data
        sets that coorespond to the cartesian product of names and years provided.
        '''
        if isinstance(names, str):
            names = [names]
        if isinstance(years, int):
            years = [years]
        elif isinstance(years, list):
            print("No years specified.")
            return pd.DataFrame()

        year_tables = [self.load_pbp_players(year)[
            self.load_pbp_players(year)['PlayerName'].isin(names)]
                       for year in years]
        #Collect non empty tables and return concatenated dataframe.
        tables = list()
        for year_table in year_tables:
            if not year_table.empty:
                tables.append(year_table)
        return pd.concat(tables)

    def get_pbp_events_player_names(self, player_name, team=None, 
                                    years=range(2010, 2019)):
        '''Returns all pbp data for players that are named player_name.  
        More than one player may be contained in the data in the event of 
        name collisions.
        '''
        player_info = self.load_pbp_players(years)
        ids = player_info.query("PlayerName == @player_name")['PlayerID']
        print("Player Ids: "+ str(ids))
        #print(data)
        player_data = [[self.load_pbp_events(year).query(
            "EventPlayerID == @player_id")
            for player_id in ids] for year in years]
        print(player_data)
        return pd.concat([pd.concat(year_data_list)
              for year_data_list in player_data])

    def get_player_event_data_by_name(self, names=None, years=None):
        '''Returns the rows of the player.csv files in the play by play data
        sets that
        coorespond to the names and years provided.
        There are two modes supported for list input:
        return the data for the cartesian product of names and years.
        '''
        #get player dataframe for the given names and years.
        player_data = self.get_player_data_by_name(names, years)
        player_ids = player_data['PlayerID'].values

        print(player_ids)
        year_tables = list()
        for year in years:
            year_tables.append(self.load_pbp_events(year)[
                    self.load_pbp_events(year)['EventPlayerID'].isin(
                            player_ids)])
        #year_tables=[self.load_pbp_players(year).query("PlayerName in @names") for year in years]
        #create an empty list of tables.  Add only the non empty lists to it.
        print(year_tables)
        #tables=list()

        for year_table in year_tables:
            if not year_table.empty:
                year_tables.append(year_table)
        return pd.concat(year_tables)


    #@functools.lru_cache(maxsize=32)
    def get_pbp_events_teams(self, team_ids=None, years=range(2010, 2019)):
        '''Returns the play by play event data for the years and teams ids given.
        '''
        if team_ids is None:
            print("No teams provided")
            return pd.DataFrame()
        elif isinstance(team_ids, int):
            team_ids = [team_ids]
        if isinstance(years, int):
            years = [years]
        data = []
        for year in years:
            year_data = self.load_pbp_events(year)
            data.append(year_data[year_data['EventTeamID'].isin(team_ids)])
        return pd.concat(data)




    def make_test_data_years(self, years, period="All"):
        '''Makes the data to run our stage 1 and stage 2 tests.  If period is
        none, it will return alll the training data split up into x_test any
        y_test for the given years.  If period = "March Madness, then only the
        march madness games are returned. This is used to estimate our true
        score on kaggle.

        Returns a prediction set x_test(df),  and true values y_test(df).  These will
        Parameters:
                year - the year the test data should be made for
                period - the session during the year that we will test against
                            None - all games in the yaer
                            "March Madness" - Only the March Madness Games
                            "Tourney" - All primary and secondary tournament games
                            "Regular" - All regular season games.
        Parameters:

                period: What part of the season we are testing on.
                        Current values are: None, "March Madness"

        Return:
                x_test:  Data to use to predict y_test.
                y_test   The true outcome we will test against'''

        x_test, y_test = self.make_training_data_years(years, period)
        return x_test, y_test


    def make_training_data_years(self, years, period="All", teams = None):
        training_data = pd.concat([
            self.make_training_data_year(year, period, teams) 
            for year in years])

        x_train = training_data.drop('Team1W', axis=1)
        y_train = training_data['Team1W']
        return x_train, y_train

    '''
    Create the training data for our model
    '''
    @functools.lru_cache(maxsize=32)
    def make_training_data_year(self, year, period="All", teams = None):
        '''
        Make x_train and y_train consisting of game features, and labels of whether
        team 1 won the game or not.  To fit the format of kaggle submission,
        the smaller id will always be team 1.  Currently the games are built from
        the playbyplay data.  It is likely more efficient to use one of the game summary files.
        That said, if we want to extract event specefic features, then we would still need to
        load the event data.
        '''
        #generate the training data that does not depend on pbp results.
        td = self.get_games(year, period)
    
        #make team specific columns.        
        #Number of division 1 years prior to the current season a game is playedin
        td=self.get_num_div1_years_team(td)

       
        #(time  outs  +  dead  rebounds  prior  to  the  current  game  in  the  last  2years)/number of games in last 2 years
        #td=self.get_team_type_events(td, years=2)
        
        # Tourney win ratio (wins/games played )
        td=self.get_winning_rate(td, who="Team", when ="Tourney")
        #Regular season win ratio(wins/games played)
        #td=self.get_winning_rate(td, who="Team", when ="Regular")

        #turnovers/game in last 2 years.
        #td=self.get_tournovers_team(td, who="Team", when ="Tourney")
 


        #make coach columns.
        td = self.get_num_div1_years_coach(td)
        #wins/total played in previous three seasons.
        
        #td=self.get_win_rate_coach(td, period="All")
        #td=self.get_win_rate_coach(td, period="Tourney")



        #make player derived columns.
        
        #Average player points per attempt in last 2 years
        #td=self.get_player_points(td)
        #Average years a player has played.
        #td=self.get_num_div1_years_player(td)

        
        #Average player dynamic
        #td=self.get_dynamic(td)

        
        #Average  player  unreliability
        #td=self.get_unreliability(td)

        
        #Average reliability
        #td=self.get_reliability(td)


        #make ranking columns
        td = self.get_avg_ranking(td)

        #Make the game type data:
        #td=self.get_game_type(td)

        return td
