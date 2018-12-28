# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:27:32 2017

@author: vpx365
"""



 # Load libraries
import pandas
from pandas.plotting import scatter_matrix
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

import seaborn as sns


# Load dataset
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']


 # shape
print(dataset.shape)

# head
#print(dataset.head(20))
 
# descriptions
print(dataset.describe())



# histograms of the variables
#dataset.hist()
#plt.show()
