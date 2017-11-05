# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
Replace qualitative quantities by numerical ones so they can be used in modeling.

coded_data=coder_naive(dataset)
@author: vpx365
"""

import pandas

class Coder:
    @classmethod
    def naive(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
                return dataframe.dropna(axis=0,subset=['Age'])
        except TypeError as error:
            print("invalid type for imputation")



    @classmethod
    def replace_with_mean_of_pop(self,dataframe):
        mean=dataframe['Age'].mean()
        return dataframe.fillna(mean)

