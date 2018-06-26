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

def iris_box_plot(dataset):
    sns.set_style("whitegrid")
    irises=sns.boxplot(x="class", y="cm", data=dataset)


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

 # shape
print(dataset.shape)

# head
#print(dataset.head(20))
 
# descriptions
print(dataset.describe())



# histograms of the variables
#dataset.hist()
#plt.show()


 # box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

 

 # scatter plot matrix
scatter_matrix(dataset)
plt.show()



# group the data by type of iris and makd scatter matrix.
groupby=dataset.groupby('class')
print(groupby.size())
for name, group in groupby:
    print(name + " data visualization")
    scatter_matrix(group)
    group.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

 
    
    
    
print("Setting up Cross validation:")
#split-out dataset for sanity
array=dataset.values

X=array[:,0:4]
Y=array[:,4]

validation_size=0.20

seed=7

X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X,Y, test_size=validation_size,random_state=seed)



scoring='accuracy'

 # Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# evaluate each model in turn
results = []
names = []


print("Training models and cross validating with 80 percent of data:")
print("Name: mean cv accuracy (standard deviation)" )
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
print("visualizing cross validation results")
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#sanity check compare to reserved 20% of data.
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print("Sanity Check Accuracy of best model:"+ str(accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
