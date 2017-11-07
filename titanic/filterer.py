# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:28:13 2017

@author: vpx365
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
A class of different methods to remove observations that may be detrimental to 
the model
Should function like below.
filtered_data=filter_naive(dataset)
@author: vpx365
"""

import pandas

class Filterer:
    @classmethod
    #Dont Remove anything
    def naive(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                return dataframe
        except TypeError as error:
            print("invalid type for filtering")


