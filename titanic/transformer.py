
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
Apply a transform to the data to get a better distribution(e. g. normal)
Should happen before imputing since it will effect the results of impution. 

transformed_data=Transformer.transformer_naive(coded_data)
@author: vpx365
"""

import pandas

class Transformer:
    @classmethod
    def normalizeAge(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                #code male and female data. Males are 0, females are 1.
                return dataframe['Age'].applymap(func=lambda x:(x-dataframe['Age'].mean())/dataframe['Age'].std())
        except TypeError as error:
            print("invalid type for transforming")

