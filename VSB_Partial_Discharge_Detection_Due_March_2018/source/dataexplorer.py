# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:14:17 2018
This file will take a data frame and create various visualizations of the 
variables and combinations.
@author: vpx365
"""

'''
Create a 

'''
from datamanager import DataManager
import matplotlib as plt
import pandas as pd
import seaborn as sns
import numpy as np
import functools

class DataExplorer():
    
    make_path="../fig/"
    dm=DataManager()


    '''Needs to be finished.  '''
    def visualize_events_year(self,year=2010):
        play_by_play=self.dm.load_play_by_play_events(year)
        print(play_by_play['EventType'].value_counts())
        grouped=play_by_play.groupby('EventType')
        for group in grouped.groups:
            pass
    
    

    '''
    call the visualize events for each year from 2010-2017
    '''
    def visualize_events_all(self):
        play_by_play=self.dm.load_play_by_play_events(year)
        
        yearly_event_frequency=play_by_play['EventType'].value_counts()
        print(yearly_event_frequency)
        yearly_event_frequency.hist()
    
    
    
    
    '''
    Create the body of a latex table giving the frequency and relative frequecny
    for the event types.  
    '''
    def make_frequency_table_events(self):
        table=self.dm.get_all_pbp_event_data()        
        counts=table['EventType'].value_counts()
        total=sum([counts[key] for key in counts.keys()])
        tex=[key+"&"+str(counts[key])+"&{:.2%} \\\\\hline".format(counts[key]/total) for key in counts.keys()]
        for x in tex:
            print(x.replace('%','\%').replace('_','\_'))
    
 
    '''
    Create a pie chart for event types.  The pie chart is exploded since there
    are several small categories.
    '''
    def make_event_pie_chart(self):
        table=self.dm.get_pbp_event_data()
        colors=[]
        explode=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.4, 0.4, 0.4, 0.4]
        ax=table['EventType'].value_counts().plot(kind='pie',figsize=(8,8),labels=None,explode=explode)
        ax.legend(loc="best", labels=table['EventType'].value_counts().keys())



    '''
    This will create a box plot for each event type.
    '''
    def make_event_box_plot(self):
        pass



    def visualize_colesium_results(self, coliseum_results, stages=['Imputer', 'Model'], save=False):
        targets=['test_score', 'fit_time', 'score_time']
        for target in targets:
            for stage in stages:
                ax=self.make_colesium_results_box_plot(coliseum_results, stage, target)
                if save:
                    fig=ax.get_figure()
                    fig.savefig(self.make_path+stage+"-"+target+".png")

    '''
    This will create a box plot for test score for the given colesium results
    '''
    def make_colesium_results_box_plot(self, coliseum_results, stage='Model', target= "test_score"):
       
        return coliseum_results.boxplot(column=target, figsize=(6,6), by=stage)        
        

    

    '''
    Returns a data frame containing the points scored by a team in given years
    add up all the scored points in a given year and team.
    
    The columns of the data frame are
    TeamID Season Points
    For playbyplay data.  This includes all basketball games from 2010-2018.
    '''
    def get_points_year(self, years=range(2010,2019), teams=None):     
        
        #slice out the team if it is specified
        if teams is not None:
            #slice out for the given year.
            play_by_play=self.dm.get_pbp_event_data_by_year(years).query("EventTeamID in @teams")
        
            #slice out the play data for the given team .
            #play_by_play=play_by_play[play_by_play['EventTeamID']==team]
           
        
            event_types=play_by_play['EventType']
        
            return play_by_play
       
    ''''
        def points_map(event_type):
            if event_type.startswith('made2'):
                return 2
            elif event_type.startswith('made3'):
                return 3
            elif event_type.startswith('made1'):
                return 1
            else:
                return 0    
        
        return event_types.map(points_map).sum()
    '''
    '''
    Try to get this to return computation time.
    timeit.timeit(get_points_cummmulative, number=1)
    '''
    def get_points_cummmulative(self, year=2018,team=None):
        total=0
           
        for x in range(2010,year):
            print("Now working on year: "+str(x))
            total=total + self.get_points_year(x,team)
        return total
    
    
    
    '''Needs to be finished. 
    columns:
        season teamid points in season
    
    
    place the data in build/data 
    '''
    def make_avg_points_table(self):
        points_in_season=pd.DataFrame(columns=['Season','TeamID','Points'])
        
        
        #Loop through season and team ids.  Then call get_points_year
        
        teamid=1410
        season=2010
        points=self.get_points_year(teamid,season)
        
        #insert the row here!!
        #points_in_season.loc[]
        
        return points_in_season
        #play_by_play.to_csv("avg_points.csv")
    
    '''
    Computes a statistic over regular season, March Madness, or both
    for a collection of players and for a fixed event type.
    
    '''
    def get_player_event_counts(self, player_ids, scale='PerGame'):
        tables=[self.get_player_event_data(player_id) for player_id in player_ids]
 
        counts=[table['EventType'].value_counts() for table in tables]
        
        return counts
    
    
    
    def get_event_statistic(self,event_type):
        pass    
        
    def get_team_statistic(self):
        pass