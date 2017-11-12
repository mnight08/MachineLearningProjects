


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


#returns a statistic that measures the cross validation results.
def objective(results):
    return results.mean()#/(1+results.std())

#return cv results for a given workflow
#return list of the form 'ModName', 'Coder', 'Cleaner','ImpName','CVResults','Score'
def evaluateWorkflow(dataset,code, impu, filt,feat,model):
    log="working on :" \
    +"/"+code.__name__ \
    +"/"+impu.__name__ \
    +"/"+filt.__name__  +"/"+feat.__name__+"/"+model[0]+"\n"

    #print("dataset type is " +str(type(dataset)))
    #code the data.
    coded_data=code(dataset)
    imputed_data=impu(coded_data)

    filtered_data=filt(imputed_data)






    #log=log+"selected features are :" +str(feat(filtered_data))

    #slice the relevant features
    X_train=filtered_data[feat(filtered_data)]

    #validation data
    Y_train=filtered_data['Survived']

    seed=7

    #cross validation setup.
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    scoring='accuracy'




    kfold.random_state=seed


    try:

        cv_results=model_selection.cross_val_score(model[1], X_train,Y_train, cv=kfold, scoring=scoring)
        objective_score=objective(cv_results)#log=log+"objective score is " +str(objective_score)
        print(log)
        return [code,impu,filt,feat,model[0],model[1],objective_score]
    except ValueError as error:
        print(log+"cross validation failed.")
        return []







if __name__ == '__main__':
    # do stuff with imports and functions defined about

    start = time.time()

    print(start)
    num_cores = multiprocessing.cpu_count()
    #coder/imputer/filter/featureselector/model:
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
    dataset = pandas.read_csv(filename)

    print("loaded Data")


    colleseum_results = pandas.DataFrame(
           Parallel(n_jobs=num_cores,backend="threading")    (delayed(evaluateWorkflow)(dataset,code,impu,filt,fea,model)
       for code,impu,filt,fea,model in itertools.product(coder_list,imputer_list,filter_list,feature_selector_list,model_list.items())))







#    print(accuracy_score(Y_validation, predictions))
#    print(confusion_matrix(Y_validation, predictions))
#    print(classification_report(Y_validation, predictions))



    #prepare submission for kaggle.


    end = time.time()
    print("Computation Time: "+ str(end - start))



