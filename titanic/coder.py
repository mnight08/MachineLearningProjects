
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
Replace qualitative quantities by numerical ones so they can be used in modeling.

coded_data=coder_naive(dataframe)
@author: vpx365
"""

import pandas

class Coder:
    @classmethod
    def gender(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                #code male and female data. Males are 0, females are 1.
                return dataframe.replace(to_replace={'Sex':{'male':0,'female':1}},inplace=False)
        except TypeError as error:
            print("invalid type for coding")

