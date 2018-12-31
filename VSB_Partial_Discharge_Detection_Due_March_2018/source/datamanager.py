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
        self.train_meta = pd.read_csv(self.load_path+"metadata_train.csv")
        self.test_meta = pd.read_csv(self.load_path+"metadata_train.csv")
        self.test = None
    
    def load_signals(self, ids=None):
        if isinstance(ids, int):
            self.train=pq.read_pandas(self.load_path+"train.parquet",columns=[str(ids)]).to_pandas()
            
        elif isinstance(ids, list):
            self.train=pq.read_pandas(self.load_path+"train.parquet",columns=[str(i) for i in ids]).to_pandas()
        else:
            self.train=pq.read_pandas(self.load_path+"train.parquet").to_pandas()
        
    def get_measurements(self, measurement_id):
        '''A measurement consist of three signals at appropriate phases taken 
        over the sample period. Returns the triples for the given measurement ids.'''
        
        
        
    def load_test_data(self, ids=None):
        
