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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import itertools


import time


def findsubsets(S):
    subsets=[]
    for m in range(1,len(S)+1):
        for subset in [list(comb) for comb in itertools.combinations(S, m)]:
            subsets.append(subset) 
    return subsets


seed=7

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
dataset = pandas.read_csv(url, names=names)



#print(dataset.corr())


#feature columns
features=dataset.columns[1:14]


#split-out dataset for sanity check
array=dataset.values

#features columns
X=array[:,1:14]

#class column
Y=array[:,0]


#save data for sanity check after comparing
validation_size=0.20
X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X,Y, test_size=validation_size,random_state=seed)


scoring='accuracy'

#A list of the classes of models we will use.
models = []
models.append(('SVM', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model with each subset of features.


subsets =findsubsets(features)

max_mean_results = []
names = []


start = time.time()



max_score=0;
max_feature_set=''
max_model=''
max_name=''

X_train_frame=pandas.DataFrame(X_train,columns=features)
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#feature selection.  Goes through every model in the list and decide on what 
#features produce the best cv mean score with what model. This is not practical
#for large feature sets.   
for name, model in models:
    i=1
    print("working on model"+name)
    for selected in subsets:
        kfold.random_state=seed
  
            
        cv_results=model_selection.cross_val_score(model, X_train_frame[selected],Y_train, cv=kfold, scoring=scoring)
        if cv_results.mean()>max_score:
            max_score=cv_results.mean()
            max_results=cv_results
            max_model=model
            max_feature_set=selected
            max_name=name
        #results.append(cv_results)
        #names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
        i=i+1
        if i%1000==0: 
            print(str(i)+"subsets testsed")
print('Scoring is done! The feature set with highest mean accuracy is ')
print(max_feature_set)
print('With mean accuracy:'+str(max_score))
print('Using the model' +max_name)


# Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()


X_validation_frame=pandas.DataFrame(X_validation,columns=features)

#sanity check: use the sanity data to validate model.
max_model.fit(X_train_frame[max_feature_set], Y_train)
predictions=max_model.predict(X_validation_frame[max_feature_set])

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


end = time.time()
print("Computation Time: "+ str(end - start))


