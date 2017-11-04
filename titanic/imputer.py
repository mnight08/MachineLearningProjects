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
    @classmethod
    def naive(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                return dataframe.dropna(axis=0,subset=['Age'])
        except TypeError as error:
            print("invalid type for imputation")



    @classmethod
    def replace_with_mean_of_pop(self,dataframe):
        mean=dataframe['Age'].mean()
        return dataframe.fillna(mean)
