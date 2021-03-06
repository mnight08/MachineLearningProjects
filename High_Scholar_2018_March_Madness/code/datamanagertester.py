# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 08:54:30 2018
Performs unit test for data manager class. 
@author: vpx365
"""



import unittest
import timeit

import datamanager

def timer(n=1):
    '''
    This is a decorator to time a test.  add @timer to a test to time it.
    '''
    def decorator(method):
        '''Returns a function that wraps method'''
        def wrapper(self):
            '''The new function that makes a call to method'''
            print('Average time for ' + str(n) + ' Calls of method: ' +
                  method.__name__)
            def stmt():
                '''A function that evaluates to the method supplied.'''
                return method(self)

            avg_time = timeit.timeit(stmt=stmt,
                                     setup='pass', number=n)/n
            print(avg_time)
            return avg_time
        return wrapper
    return decorator



class DataManagerTester(unittest.TestCase):
    '''Tests that verify the DataManager Class is running as it should.'''

    #set this to true to run all test.  To only run a specific test, comment out
    #the @unittest.skipunless above the relevant function and leave this as false.
    testall = False
    @classmethod
    def setUpClass(cls):
        '''Setup datamanager to run tests on.'''
        cls.dm = datamanager.DataManager()


    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_cities(self):
        '''Test load cities'''
        #check if the data loaded is empty
        self.assertFalse(self.dm.load_cities(stage2=False).empty)
        self.assertFalse(self.dm.load_cities(stage2=True).empty)


    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_pbp_players(self):
        '''test_load_pbp_players'''
        #check if the data loaded is empty
        for year in range(2010, 2019):
            self.assertFalse(self.dm.load_pbp_players(year).empty)

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_pbp_events(self):
        '''test_load_pbp_events'''
        #check if the data loaded is empty
        for year in range(2010, 2019):
            print("Testing load pbp events for year "+str(year))
            self.assertFalse(self.dm.load_pbp_events(year).empty)

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_conference_tourney_games(self):
        '''test_load_conference_tourney_games'''
        self.assertFalse(self.dm.load_conference_tourney_games(stage2=False).empty)
        self.assertFalse(self.dm.load_conference_tourney_games(stage2=True).empty)


    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_conferences(self):
        '''test_load_conferences'''
        self.assertFalse(self.dm.load_conferences().empty)


    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_tourney_results(self):
        '''test_load_tourney_results'''
        self.assertFalse(self.dm.load_tourney_results(compact=False).empty)
        self.assertFalse(self.dm.load_tourney_results(compact=True).empty)


    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_massey_ordinals(self):
        '''test_load_massey_ordinals'''
        print("testing massey ordinals")
        self.assertFalse(self.dm.load_massey_ordinals(through2018=False).empty)
        self.assertFalse(self.dm.load_massey_ordinals(through2018=True).empty)





    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_get_player_names(self):
        '''test_get_player_names'''
        years = range(2010, 2019)
        player_names = self.dm.get_player_names(years)
        print(type(player_names))
        self.assertFalse(player_names.empty)

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_get_player_event_data_by_name(self):
        '''test_get_player_event_data_by_name'''
        years = range(2010, 2011)

        player_names = self.dm.get_player_names(years)
        #print(player_names.value_counts())
        player_event_data = self.dm.get_player_event_data_by_name(player_names, years)

        self.assertFalse(player_event_data.empty)
        #david_browns=pd.concat(player_names).query('PlayerName == "BROWN_DAVID"')
        #print(david_browns.to_latex())




    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_get_player_data_by_name(self):
        '''test_get_player_data_by_name'''
        #print("test_get_player_data_by_name")
        years = range(2010, 2011)
        #print(years)
        player_names = "BROWN_DAVID"#self.dm.get_player_names(years)
        #print(type(player_names))
        player_data = self.dm.get_player_data_by_name(player_names, years)
        #print(player_data)
        print("data loaded")
        self.assertFalse(player_data.empty)

        #for player in pd.concat(player_names)['PlayerName'].value_counts().keys():
        #      self.assertFalse(self.dm.get_player_event_data_by_name())

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_get_pbp_event_data(self, years=range(2010, 2019)):
        '''test_get_pbp_event_data'''
        pass

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_load_variables(self, blacklisted=None):
        '''test_load_variables'''
        pass

    @unittest.skipUnless(testall, reason="Developing other Method")
    @timer()
    def test_get_player_event_data_by_id(self, player_id=None, years=None):
        '''test_get_player_event_data_by_id'''
        
        
        
    @unittest.skipUnless(testall, reason="Developing other Method")
    def test_get_games(self):
        '''Returns a dataframe for the games played during the period given.
        The games will always have Team1 be the team with the lower Id. 
        This data is pulled from "RegularSeasonCompactResults", 
        "SecondaryTourneyCompactResults", "ConferenceTourneyGames", and 
        "NCAATourneyCompactResults"
        The games in the files are not necessarily unique Duplicate rowss will 
        be dropped.
        
        Parameters:
            years - The years we want the data for.
            period - "All", "March Madness","Tourney", "Regular", "Secondary", 
                    "Conference"
        Returns a dataframe for the games played during the period given with
        columns=['Season', 'DayNum', 'Team1', 'Team2', 'Team1W']
        '''
        years=[2010]
        self.assertTrue(self.dm.get_games(years, 
                    period ="All").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
                
        self.assertTrue(self.dm.get_games(years, 
                    period ="March Madness").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
                
    
        self.assertTrue(self.dm.get_games(years, 
                    period ="Tourney").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
                
        self.assertTrue(self.dm.get_games(years, 
                    period ="Regular").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
                
        self.assertTrue(self.dm.get_games(years, 
                    period ="Secondary").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
                
        self.assertTrue(self.dm.get_games(years, 
                    period ="Conference").columns.isin( ['Season', 
                   'DayNum', 'Team1', 'Team2', 'Team1W']).all())
    
    @unittest.skipUnless(testall, reason="Developing other Method")
    def test_get_num_div1_years_team(self):
        td = self.dm.get_games([2010], 'All')

        td=self.dm.get_num_div1_years_team(td)
        self.assertFalse(td.empty)



    #@unittest.skipUnless(testall, reason="Developing other Method")
    def test_get_winning_rate(self):
        td = self.dm.get_games([2010, 2011], 'All')
        
        
        self.assertFalse(self.dm.get_winning_rate(td,
                                               who="Team",
                                               when ="Regular").empty)
        self.assertFalse(self.dm.get_winning_rate(td,
                                               who="Team",
                                               when ="Tourney").empty)
        
        self.assertFalse(self.dm.get_winning_rate(td,
                                               who="Coach",
                                               when ="Regular").empty)
        self.assertFalse(self.dm.get_winning_rate(td,
                                               who="Coach",
                                               when ="Tourney").empty)
        

if __name__ == '__main__':
    unittest.main()
    
