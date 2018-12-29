# -*- coding: utf-8 -*-
"""Manages access to the underlying data.
Created on Fri December 28 9:07 pm
@author: vpx365
"""
import os
import functools
import itertools
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import signal
import matplotlib.pyplot as plt

class DataManager:
    """Manages access to the underlying data.
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
        self.load_path = "../../Learning_Data/VSB_Partial_Discharge_Detection_Due_March_2018/all/"
        self.make_path = "../../Machine_Learning_Artifact/VSB/"
        self.train=None
        self.test=None
    
    def load_signals(self, ids=None):
        if isinstance(ids, int):
            self.train=pq.read_pandas(self.load_path+"train.parquet",columns=[str(ids)]).to_pandas()
            
        elif isinstance(ids, list):
            self.train=pq.read_pandas(self.load_path+"train.parquet",columns=[str(i) for i in ids]).to_pandas()
        else:
            self.train=pq.read_pandas(self.load_path+"train.parquet").to_pandas()
        
                   
        
    
    
    def plot_spectogram(self, id):
        x = np.linspace(0,1/50,80000)
        f, t, Sxx = signal.spectrogram(x, self.train.iloc[:,id])
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    
    
    def plot_signal(self, id):
         x = np.linspace(0,1/50,80000)
         plt.plot(x,self.train.iloc[:,3])         
         plt.ylabel('Voltage')
         plt.xlabel('Time [sec]')
         plt.show()
    
    
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
