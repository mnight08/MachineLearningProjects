# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:32:33 2017
Perform cross validation using several out of the box models and imputation methods
Creates a data frame with columns 'ModName', 'Coder', 'Cleaner','ImpName','CVResults','Score'

must specify imputer, coder, feature_selector module names with imputer, coder, and feature
selector methods 

The workflow of model selection is 
raw data->coder->imputer->filter->feature_selector->colleseum->cv dataframe with objective score
followed by a message indicating the tuple of methods that achieved the highest results.

Evaluates each tuple in parallel.

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
from joblib import Parallel, delayed
import multiprocessing

import time


from coder import Coder
from imputer import Imputer
from filterer import Filterer
from feature_selector import FeatureSelector



start = time.time()


num_cores = multiprocessing.cpu_count()

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

#generate a list of the coder methods in the imputer class
coder_list=[getattr(Coder, method) for method in dir(Coder) 
                if callable(getattr(Coder, method)) and not method.startswith("__")]



filter_list = [getattr(Filterer, method) for method in dir(Filterer) if callable(getattr(Filterer, method)) and not method.startswith("__")]
feature_selector_list= [getattr(FeatureSelector, method) for method in dir(FeatureSelector) if callable(getattr(FeatureSelector, method)) and not method.startswith("__")]


#Load data into data fram.
filename="train.csv"
#column names are auto filled. Columns are  ['PassengerId',	'Survived',	'Pclass',	'Name',	'Sex',	
#        'Age',	'SibSp',	'Parch',	'Ticket',	'Fare',	'Cabin',	'Embarked']
dataset = pandas.read_csv(filename)

#returns a statistic that measures the cross validation results.
def objective(results):
    return results.mean()/(1+results.std())


seed=7




#code male and female data. Males are 0, females are 1.
dataset.replace(to_replace={'Sex':{'male':0,'female':1}},inplace=True)



#cross validation setup.
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring='accuracy'



#return cv results for a given workflow
#return list of the form 'ModName', 'Coder', 'Cleaner','ImpName','CVResults','Score'

def evaluateWorkflow(code, impu, filt,feat,model):
    
    print("working on coder method---------"+code.__name__)
    
    print("working on imputer method---------"+impu.__name__)
    print("working on filter method---------"+code.__name__)
    print("working on feature method---------"+code.__name__)
    
    print("working on model---------"+model._name)
    imputed_data=impu(dataset)
    filtered_data=filt(imputed_data)
    results=[]
    for features in feat(filtered_data):
        X_train=imputed_data[['Pclass',	'Sex',	'Age',	'SibSp',	'Parch',	'Fare']]
        Y_train=imputed_data['Survived']
        
        kfold.random_state=seed          
        cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
        objective_score=objective(cv_results)
        results.append([code,impu,filt,feat,model,objective_score])
        print(objective_score)
    return results


        
    

 

#Collect the results of cv.
colleseum_results = pandas.DataFrame(Parallel(n_jobs=num_cores)
    (delayed(evaluateWorkflow)(code,impu,filt,fea) 
    for code,impu,filt,fea in itertools.product(coder_list,imputer_list,filter_list,feature_selector_list,model_list)))



  



#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))



#prepare submission for kaggle.


end = time.time()
print("Computation Time: "+ str(end - start))



