# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:22:30 2017
Remove the rows in the training data that have missing age.
Not really imputaion.
Should function like below.
imputed_date=imputer_naive(dataset)
@author: vpx365
"""
def imputer_naive(dataframe):
    return dataframe.dropna(axis=0,subset=['Age'])
