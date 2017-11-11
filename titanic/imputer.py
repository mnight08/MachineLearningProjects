

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
    #fill in data with 
    @classmethod
    def dropna(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                return dataframe.dropna(axis=0,subset=['Age'])
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")

    #fill in data with 
    @classmethod
    def replace_age_by_pop_mean(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                mean=dataframe['Age'].mean()
                return dataframe.fillna(mean)
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")

    #fill in data with 
    @classmethod
    def replace_age_by_pop_median(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                median=dataframe['Age'].median()
                return dataframe.fillna(median)
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")
            
            
    #TODO: complete this regression method
    #use regression to find the misssing values for age 
    #based on 
    @classmethod
    def replace_regression_age(self, dataframe):
        try:        
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:
                
                regression_model=1
                return dataframe#dataframe.replace({'Age' : regression_model(dataframe['Age'])})
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")
            
            
