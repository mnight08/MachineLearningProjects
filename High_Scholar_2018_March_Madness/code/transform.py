# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 03:10:02 2018

@author: vpx365
"""
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest


class Transform:
    '''
    A Transform object has a fit and transform method.  The fit method
    should return a transform object that is ready to apply its transform.
    The transform method should actually apply the transform to a dataframe
    and return the transformed data frame.
    '''
    def __init__(self, pipe):
        '''A pipe should be a method in a class the inherits from
        Transform. Each pipe should take a dataframe X and return a
        transformation of that dataframe.
        '''
        self.transform = pipe

    def fit(self, X, y=None):
        '''Takes a training set X, and optional y, and prepares the
        transformation to transform data. This should be a step in a  Pipeline
        object. Pipeline requires that this return a reference to the fitted object.'''

        return self
#
#    def transform(self, X, y=None):
#        pass

class Imputer(Transform):
    '''Collection of static methods that take a data frame and remove na
    values by dropping or filling in.'''
#    @staticmethod
#    def drop_imp(df):
#        '''
#        Pipelines are picky about chaning the number of rows in the data.
#        This does not work right now.  I am not sure if we can actually drop
#        rows using a pipeline.  In any case, the dropping methods seem to
#        perform poorly compared to the mean.'''
#        return df.dropna()

    @staticmethod
    def fill_zero(df):
        return df.fillna(0)


#    @staticmethod
#    def fill_mean(df):
#        return df.fillna(df.mean())
#
#    @staticmethod
#    def fill_median(df):
#        return df.fillna(df.median())
#
#
    @staticmethod
    def linear_regression(df):
        '''Takes the columns that have missing values and uses other
        variables without nan to create linear model to predict missing
        values'''
        columns = df.columns
        df = df.copy()

        target_columns = columns[df.isnull().any().values]


        model = LinearRegression(n_jobs=1)

        for column in target_columns:
            target_rows = df[df[column].isnull()].index
            train_rows = df.loc[df.index.difference(target_rows)].index
            x_train = df.loc[train_rows].drop(target_columns, axis=1)
            y_train = df.loc[train_rows][column]
            x_pred = df.loc[target_rows].drop(target_columns, axis=1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_pred)

            df.loc[target_rows, column] = y_pred

        return df


class Filterer(Transform):
    '''Collection of static methods that take a data frame and removes outliers.'''
    @staticmethod
    def identity_filt(df):
        return df

class Scaler(Transform):
    '''Collection of static methods that take a data frame and removes outliers.'''
    @staticmethod
    def identity_scale(df):
        return df
    @staticmethod
    def z_score(df):

        scaler=StandardScaler().fit(df)
        return scaler.transform(df)


class FeatureSelector(Transform):
    '''Collection of static methods that take a data frame and return a
    subset of the columns.'''
#    def __init__(self):
#        pass
    @staticmethod
    def identity_feat(df):
        return df

    @staticmethod
    def __select(df, sel):
        drop_these = [col for col in df.columns[~sel] if col not in ['Season', 'DayNum', 'Team1', 'Team2']]
        return df.drop(drop_these, axis=1)
    @staticmethod
    def only_ordinal(df):
        return df[['Rank1', 'Rank1']]

#currently broken
#    @staticmethod
#    def var_threshhold(df):
#        sel = df.columns[VarianceThreshold(threshold = 2).fit(df).get_support()]
#        return FeatureSelector.__select(df,sel)

#    @staticmethod
#    def k_best(df):
#        sel = SelectKBest(k=3).fit(df.drop('Team1W'), df['Team1W']).get_support()
#        return FeatureSelector.__select(df,sel)
#
    @staticmethod
    def remove_teamids(df):
        return df.drop(['Team1', 'Team2'], axis=1)

class Model(Transform):
    '''Collection of static methods that can be fit to a data frame
    and returns a model with a predict method.
    Takes a function and creates a new function whose fit method is the
    input. A Pipeline object expects a Model to have a fit and predict method'''

    def fit(self, X, y):
        '''For a model, a pipe should have a model object'''
        self.model=self.transform()
        return self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict_proba(X)
#    @staticmethod
#    def svc():
#        return SVC(probability=True)


    @staticmethod
    def lr():
        return LogisticRegression(n_jobs=1)

#    @staticmethod
#    def lda():
#        return LinearDiscriminantAnalysis()
#
#    @staticmethod
#    def knn():
#        return KNeighborsClassifier(n_jobs=1)
#
#    @staticmethod
#    def cart():
#        return DecisionTreeClassifier()
#
#    @staticmethod
#    def naive_bayes():
#        return GaussianNB()
