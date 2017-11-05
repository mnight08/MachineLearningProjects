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
    def naive(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                #code male and female data. Males are 0, females are 1.
                return dataframe.replace(to_replace={'Sex':{'male':0,'female':1}},inplace=True)

        except TypeError as error:
            print("invalid type for imputation")

