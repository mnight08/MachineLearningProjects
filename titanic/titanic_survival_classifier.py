# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:57:35 2017
This is our first submission of the titanic classifcation problem.  

Goal is to classify the test data as survived or died 
based on  labeled training data. The columns are.
'PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	
'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked'

Categorical Variables:
'PassengerId', 'Survived', 'Ticket', 'Cabin',	'Embarked'


Quantitative Variables:
'Age',	'SibSp',	'Parch', 'Fare',	

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
#column names are auto filled. Columns are  ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	
#        'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
dataset = pandas.read_csv(filename)

#Numerical summary of the data.  Some ages are missing.  Need to decide how to fill in.
print(dataset.describe())

#get histogram for the entire dataset.  This ignores
#the categorical variables that are not 
#coded.
dataset[['Age','Fare','Parch','Pclass','SibSp']].hist()

#Analyze relationship between survival and other variables
grouped_survivors=dataset.groupby('Survived');


#Return Series with number of non-NA/null observations over requested axis
print(grouped_survivors.count())


grouped_sex=dataset.groupby('Sex');

print(grouped_sex.count())





#Numerical summary of the groups.
#categorical variables
#print(grouped_survivors['Sex'].count())
#print(grouped_survivors['Pclass'].count())

#print(grouped_survivors['Name'].count())

#print(grouped_survivors['Ticket'].count())

#print(grouped_survivors['Cabin'].count())

#print(grouped_survivors['Embarked'].count())

#quantitative variables
#print(grouped_survivors['SibSp'].mean())
#print(grouped_survivors['Parch'].mean())




#Visualization for the groups.



#Fill in the data from the group






