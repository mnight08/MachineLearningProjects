

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
Remove the rows in the training data that have missing age.
Not really imputaion.
Should function like below.
imputed_date=imputer_naive(dataset)
@author: vpx365
"""

import pandas

class Imputer:
    #replace male with 0 and female with 1.
    @classmethod
    def dropna(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                return dataframe.dropna(axis=0,subset=['Age'])
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")

