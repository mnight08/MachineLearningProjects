# -*- coding: utf-8 -*-
"""
Perform exhaustive feature selection using cross validation using svm model.
Compare accuracy.
@author: ez pawn
"""

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

#Load data into data fram.
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

pritn(dataset.corr())




print(clf.coef_)
