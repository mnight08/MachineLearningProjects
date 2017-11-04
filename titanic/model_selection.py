# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:32:33 2017
Perform cross validation using several out of the box models and imputation methods
Should return the model and imputation method that maximizes

(avg acc)/(1+std acc).
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

from imputer import Imputer

#returns a statistic that measures the cross validation results.
def objective(results):
    return results.mean()/(1+results.std()**2)




seed=7

#Load data into data fram.
filename="train.csv"
#column names are auto filled. Columns are  ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	
#        'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
dataset = pandas.read_csv(filename)

#code male and female data. Males are 0, females are 1.
dataset.replace(to_replace={'Sex':{'male':0,'female':1}},inplace=True)



max_obj=0;
best_imputer='';
best_model=''
best_model_name=''
objective_score=0

#generate a list of the imputation methods in the imputer class
imputer_list = [getattr(Imputer, method) for method in dir(Imputer) 
                if callable(getattr(Imputer, method)) and not method.startswith("__")]
model_list={'SVM': SVC(),
            'LR': LogisticRegression(),
            'LDA': LinearDiscriminantAnalysis(),
            'KNN': KNeighborsClassifier(),
            'CART': DecisionTreeClassifier(),
            'NB': GaussianNB()}

#cross validation setup.
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring='accuracy'

#Pick the best imputer model combination based on cv.
for imputer in imputer_list:
    print("working on imputer method---------"+imputer.__name__)
    imputed_data=imputer(dataset)
    X_train=imputed_data[['Pclass',	'Sex',	'Age',	'SibSp',	'Parch',	'Fare']]
    Y_train=imputed_data['Survived']

    for model_name, model in model_list.items():
        print("working on model*********"+model_name)
        kfold.random_state=seed
          
        cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)

        objective_score=objective(cv_results)
        print(objective_score)
        if max_obj<objective_score:
            best_imputer=imputer
            
            best_model=model
            best_model_name=model_name
        

print("The best model imputer pair were: " + best_model_name +" / "+ best_imputer.__name__)


#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))



#prepare submission for kaggle.




