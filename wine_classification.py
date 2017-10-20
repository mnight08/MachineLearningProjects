# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:51:21 2017

@author: ez pawn
"""

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['class',
'Alcohol',
'Malicacid',
'Ash',
'Alcalinity of ash',
'Magnesium',
'Totalphenols',
'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity',
'Hue',
'OD280/OD315',
'Proline']
dataset = pandas.read_csv(url, names=names,index_col=False)


print(dataset.describe())



