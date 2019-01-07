# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 03:20:55 2019

@author: vpx365
"""

from datamanager import DataManager
from dataexplorer import DataExplorer

import unittest
import timeit



testall=False

class DataExplorerTester(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.dm = DataManager()
        cls.de = DataExplorer(cls.dm)
        
        
    @unittest.skipUnless(testall==True,reason="Developing other Method")
    def test_plot_signal(self):
        #check to see if signal is automatically loaded into memory and plotted 
        #for a single id.
        self.de.plot_signal(0)
        
        
        self.de.plot_signal([0,1,2,3,4])
        
        
        
        
        #try to print a random collection of 100 signals.
        signal_ids=self.dm.train_meta['signal_id'].sample(100).values
        self.de.plot_signal(signal_ids)
        
        #check if the data loaded is empty
        
        
        #self.assertFalse(self.dm.load_cities(stage2=False).empty)
        #self.assertFalse(self.dm.load_cities(stage2=True).empty)
        

    #@unittest.skipUnless(testall==True,reason="Developing other Method")
    def test_plot_triple(self):
        #check to see if signal is automatically loaded into memory and plotted 
        #for a single id.
        self.de.plot_triple(0)
        
        
        self.de.plot_triple([1,4])
        
        
        
        
        #try to print a random collection of 100 signals.
        triple_ids=self.dm.train_meta['id_measurement'].drop_duplicates().sample(100).values
        self.de.plot_triple(triple_ids)
        
        #check if the data loaded is empty
        
        
        #self.assertFalse(self.dm.load_cities(stage2=False).empty)
        #self.assertFalse(self.dm.load_cities(stage2=True).empty)
    
    #@unittest.skipUnless(testall==True,reason="Developing other Method")
    def test_get_index_signals(self):
        
        pass
    
    
    
    #@unittest.skipUnless(testall==True,reason="Developing other Method")
    def test_get_index_triples(self):
        
        pass
    
if __name__ == '__main__':
    unittest.main()