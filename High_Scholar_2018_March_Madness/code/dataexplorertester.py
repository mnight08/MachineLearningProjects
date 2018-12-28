# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:31:16 2018

@author: vpx365
"""

import dataexplorer
import unittest
import timeit
'''
This is a decorator to time a test.  add @timer to a test to time it.
'''
def timer(n=1):
    def decorator(method):
        def wrapper(self):
            print('Average time for '+str(n)+' Calls of method: '+method.__name__)
          
            def stmt():     
                return method(self)
                 
            avg_time=timeit.timeit(stmt=stmt,
                               setup='pass', number=n)/n               
            
            print(avg_time)
            return avg_time
        
        return wrapper
    return decorator

testall=False

class DataManagerTester(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.de=dataexplorer.DataExplorer()
        
        
    @unittest.skipUnless(testall==True,reason="Developing other Method")
    @timer()
    def test_make_event_pie_chart(self):
        
        self.de.make_event_pie_chart()
        #check if the data loaded is empty
        
        
        #self.assertFalse(self.dm.load_cities(stage2=False).empty)
        #self.assertFalse(self.dm.load_cities(stage2=True).empty)
        

   #@unittest.skipUnless(testall==True,reason="Developing other Method")
    @timer()
    def test_get_points_year(self):
        years=range(2010,2019)
        teamid=1410
        print(self.de.get_points_year(years,teamid))
        



if __name__ == '__main__':
    unittest.main()