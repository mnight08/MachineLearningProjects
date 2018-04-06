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

import itertools

import time


from coder import Coder
from imputer import Imputer
from filterer import Filterer
from feature_selector import FeatureSelector


#returns a statistic that measures the cross validation results.
def objective(results):
    return results.mean()#/(1+results.std())

#return cv results for a given workflow
#return list of the form 'ModName', 'Coder', 'Cleaner','ImpName','CVResults','Score'
def evaluateWorkflow(dataframe,code, impu, filt,feat,model):
    log="working on dataframe/coder/imputer/filter/featureselector/model: \n"\
    +str(type(dataframe)) \
    +"/"+code.__name__ \
    +"/"+impu.__name__ \
    +"/"+filt.__name__  +"/"+feat.__name__+"/"+model[0]+"\n"

    #print("dataframe type is " +str(type(dataframe)))
    #code the data.
    coded_data=code(dataframe)
    imputed_data=impu(coded_data)

    filtered_data=filt(imputed_data)

    log=log+"selected features are :" +str(feat(filtered_data))

    #slice the relevant features
    X_train=filtered_data[feat(filtered_data)]

    #validation data
    Y_train=filtered_data['Survived']

    seed=7

    #cross validation setup.
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    scoring='accuracy'




    kfold.random_state=seed


    cv_results=model_selection.cross_val_score(model[1], X_train,Y_train, cv=kfold, scoring=scoring)
    objective_score=objective(cv_results)
    log=log+"objective score is " +str(objective_score)
    print(log)
    return [code,impu,filt,feat,model,objective_score]


# do stuff with imports and functions defined about

start = time.time()


# what are your inputs, and what operation do you want to
# perform on each input.
#generate a list of the coder methods in the imputer class

#generate a list of the imputation methods in the imputer class
imputer_list = [getattr(Imputer, method) for method in dir(Imputer)
                if callable(getattr(Imputer, method)) and not method.startswith("__")]
model_list={'SVM': SVC(),
        'LR': LogisticRegression(),
        'LDA': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'CART': DecisionTreeClassifier(),
        'NB': GaussianNB()}

#generate a list of the coder methods in the coder class
coder_list=[getattr(Coder, method) for method in dir(Coder)
            if callable(getattr(Coder, method)) and not method.startswith("__")]



filter_list = [getattr(Filterer, method) for method in dir(Filterer) if callable(getattr(Filterer, method)) and not method.startswith("__")]
feature_selector_list= [getattr(FeatureSelector, method) for method in dir(FeatureSelector) if callable(getattr(FeatureSelector, method)) and not method.startswith("__")]

print("Generated Lists")

#Load data into data fram.
filename="train.csv"
#column names are auto filled. Columns are  ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',
#        'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
dataframe = pandas.read_csv(filename)

print("loaded Data")


colleseum_results = []
for code,impu,filt,fea,model in itertools.product(coder_list,imputer_list,filter_list,feature_selector_list,model_list.items()):
    colleseum_results.append(evaluateWorkflow(dataframe,code,impu,filt,fea,model))

colleseum_results=pandas.DataFrame(colleseum_results)

colleseum_results.columns=['code','impu','filt','feat','model0','model1','objective_score'



#    print(accuracy_score(Y_validation, predictions))
#    print(confusion_matrix(Y_validation, predictions))
#    print(classification_report(Y_validation, predictions))



#prepare submission for kaggle.


end = time.time()
print("Computation Time: "+ str(end - start))
    print("Best workflows are:")
    colleseum_results[colleseum_results['objective_score']==
                      colleseum_results['objective_score'].max()]
    