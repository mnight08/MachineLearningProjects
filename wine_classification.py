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

import itertools


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



print(dataset.corr())


#split-out dataset for sanity
array=dataset.values

#features columns
X=array[:,1:13]

#class column
Y=array[:,0]

seed=7


#save data for sanity check after comparing
validation_size=0.20
X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X,Y, test_size=validation_size,random_state=seed)


scoring='accuracy'

#A list of the classes of models we will use.
models = []
models.append(('SVM', SVC()))

# evaluate each model with each subset of features.
results = [1]
names = [1]

features=['Alcohol',
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



def findsubsets(S):
    return [subset for subset in [list(itertools.combinations(S, m)) for m in range(1,len(S)+1)]]




for name, model in models:
    for selected in subsets:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




#Fix this.
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



