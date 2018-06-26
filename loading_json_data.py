# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:02:50 2018

@author: vpx365
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:27:32 2017

@author: vpx365
"""



 # Load libraries
import pandas

 
# Load dataset
data='{"names": ["John","jim","sohn","wim"],"ages":[2,3,4,3]}'
dataset = pandas.read_json(data)

 # shape
