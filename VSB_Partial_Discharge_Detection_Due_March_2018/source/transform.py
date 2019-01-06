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
