

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
A class of methods that return a subset of 
features of the given dataset that are the most meaningful.  

X_train=dataset[featureSelectorNaive(dataset)]

@author: vpx365
"""

import pandas

class FeatureSelector:
    #return all of the features
    @classmethod
    def naive(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
    
                return ['Pclass',	'Sex',	'Age',	'SibSp',	'Parch',	'Fare']
        except TypeError as error:
            print("invalid type for feature selection"+ str(error))
