

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
from sklearn import linear_model


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
    def generalized_linear_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.LinearRegression()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")



   #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def ridge_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.Ridge()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")




   #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def lasso_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.Lasso()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")


#TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def elastic_net_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.ElasticNet()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")


#TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def OMP_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.OrthogonalMatchingPursuit()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")


            #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def bayseian_ridge_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.BayesianRidge()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")


    #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def ARD_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.ARDRegression()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")


    #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def sgd_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.SGDRegressor()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")

#TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def ransac_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.RANSACRegressor()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")



  #TODO: complete this regression method
    #use regression to find the misssing values for age
    #based on
    @classmethod
    def ths_regression_age(self, dataframe):
        try:
            if type(dataframe)!=pandas.core.frame.DataFrame:
                raise TypeError()
            else:

               #remove na to build regression model
               reduced_dataframe=dataframe.dropna(axis=0,subset=['Age'])
               regression_data_X=reduced_dataframe[['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']]
               regression_data_Y=reduced_dataframe[['Age']]


               clf = linear_model.TheilSenRegressor()
               clf.fit(regression_data_X,regression_data_Y)

               #Initialize the datframe to hold the final imputed data.
               imputed_data=pandas.DataFrame(dataframe)

               #slice out the null values for age and set them equal to the predicted value.
               mask=dataframe['Age'].isnull()
               imputed_data.loc[mask,'Age']\
               =clf.predict(dataframe[mask][['Pclass',    'Sex',        'SibSp',    'Parch',    'Fare']])


               return imputed_data
        except TypeError as error:
            print("In imputer, the type is " +str(type(dataframe))+"invalid type for imputation")
