# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:57:35 2017
This is our first submission of the titanic classifcation problem 
@author: vpx365
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


seed=7

#Load data into data fram.
filename="train.csv"
names = ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	
         'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
dataset = pandas.read_csv(filename, names=names)

