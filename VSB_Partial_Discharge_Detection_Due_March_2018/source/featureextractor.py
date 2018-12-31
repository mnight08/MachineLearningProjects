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

class FeatureExtractor:
    """Responsible for taking the a data manager object, and extracting 
    features that will be used to train a classification model.
    Example:
        Each feature cooresponds to a single function that takes the datamanager, 
        requests appropriate signals, and returns a dataframe containing the 
        features.  Each feature data frame should include at minimum a signal id with
        relevant feature data.  Here a signal refers the the time 
        
    """
    def __init__(self, data_manager):
        self.dm=data_manager
    def extract_power(self, ids):
        ''''Returns the root mean square power of a signal'''    
        pass
    
    @functools.lru_cache(maxsize=32)
    def make_training_data(self, mode=None, ids=None):
        '''
        Make x_train feature and y_train class for each measurement triple of signals.
        '''
        
        x_train = self.make_x_train_data(mode,ids)
        y_train = None
        return x_train, y_train

    
    @functools.lru_cache(maxsize=32)
    def make_x_train_data(self, mode=None, ids=None):
        '''
        Make x_train features for each measurement triple of signals.
        '''
        x_train=None
        
        return x_train
