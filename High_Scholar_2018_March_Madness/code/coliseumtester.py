# -*- coding: utf-8 -*-
"""A coliseum is an object that will compare a list of pipelines over some
    parameter space will compare different models, and data cleaning methods,
    subset of the training data using cross validation and logloss.
    Sets up pipeline for the models.  Defines the transforms and runs cross
    validation with various parameters. Each method will return a
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline  import Pipeline
from sklearn import model_selection
from sklearn.metrics import make_scorer, log_loss

from sklearn.feature_selection import VarianceThreshold


from dataexplorer import DataExplorer
from datamanager import DataManager


from joblib import Memory, Parallel, delayed




#import traceback

import pandas as pd
import itertools
import timeit
import time
import datetime

#def timer(n=1):
#    '''
#    This is a decorator to time a test.  add @timer to a test to time it.
#    '''
#    print(locals())
#    def decorator(method,*args):
#        print(locals())
#
#        '''Returns a function that wraps method'''
#        def wrapper(self, args):
#            print(locals())
#            '''The new function that makes a call to method'''
#            print('Average time for ' + str(n) + ' Calls of method: ' +
#                  method.__name__)
#            def stmt():
#                '''A function that evaluates to the method supplied.'''
#                return method(self, *args)
#
#            avg_time = timeit.timeit(stmt=stmt,
#                                     setup='pass', number=n)/n
#            print(avg_time)
#            return avg_time
#        print(wrapper)
#        return wrapper
#    return decorator
class Transform:
    '''
    A Transform object has a fit and transform method.  The fit method
    should return a transform object that is ready to apply its transform.
    The transform method should actually apply the transform to a dataframe
    and return the transformed data frame.
    '''
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.pipe(X)

class Imputer:
    '''Collection of static methods that take a data frame and remove na
    values by dropping or filling in.'''
    '''@staticmethod
    def drop_imp(df):
        Pipelines are picky about chaning the number of rows in the data.
        This does not work right now.  I am not sure if we can actually drop
        rows using a pipeline.  In any case, the dropping methods seem to
        perform poorly compared to the mean.
        return df.dropna()
    '''

    @staticmethod
    def fill_zero(df):
        return df.fillna(0)


    @staticmethod
    def fill_mean(df):
        return df.fillna(df.mean())

    @staticmethod
    def fill_median(df):
        return df.fillna(df.median())


    @staticmethod
    def linear_regression(df):
        '''Takes the columns that have missing values and uses other
        variables without nan to create linear model to predict missing
        values'''
        columns = df.columns
        df = df.copy()

        target_columns = columns[df.isnull().any().values]


        model = LinearRegression(n_jobs=1)

        for column in target_columns:
            target_rows = df[df[column].isnull()].index
            train_rows = df.loc[df.index.difference(target_rows)].index
            x_train = df.loc[train_rows].drop(target_columns, axis=1)
            y_train = df.loc[train_rows][column]
            x_pred = df.loc[target_rows].drop(target_columns, axis=1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_pred)

            df.loc[target_rows, column] = y_pred

        return df


class Filterer:
    '''Collection of static methods that take a data frame and removes outliers.'''
    def identity_filt(df):
        return df


class FeatureSelector:
    '''Collection of static methods that take a data frame and return a
    subset of the columns.'''
    
    @staticmethod
    def identity_feat(df):
        return df
    
#    @staticmethod
#    def only_ordinal(df):
#        return df[['Rank1', 'Rank1']]
#    
    @staticmethod
    def var_threshhold(df):
        sel = VarianceThreshold(threshold = 1).fit(df).get_support()
        columns = df.columns[sel]
        
        return pd.DataFrame(VarianceThreshold(threshold=1).fit_transform(df), columns=columns)
    

    @staticmethod
    def remove_teamids(df):
        return df.drop(['Team1', 'Team2'], axis=1)

class Model:
    '''Collection of static methods that can be fit to a data frame
    and returns a model with a predict method.
    Takes a function and creates a new function whose fit method is the
    input.'''
    def __init__(self, pipe):
        self.pipe=pipe

    def fit(self, X, y):
        self.predict = self.predict_proba
        return self.pipe(X, y.iloc[X.index])

    def transform(self, X, y = None):
        return self.pipe(X, y.iloc[X.index])


#    @staticmethod
#    def svc():
#        return SVC()
#

    @staticmethod
    def lr():
        return LogisticRegression(n_jobs=1)

    @staticmethod
    def lda():
        return LinearDiscriminantAnalysis()

#    @staticmethod
#    def knn():
#        return KNeighborsClassifier(n_jobs=1)
#
#    @staticmethod
#    def cart():
#        return DecisionTreeClassifier()
#
#    @staticmethod
#    def naive_bayes():
#        return GaussianNB()


def make_pipelines(transform_class_list, batches=1):
    '''Takes a list of classes of transforms.  Extracts the callable methods,
       creates all possibles tuples in the order given of the methods, creates
       pipeline object for each and returns list of lists of the pipelines.
    '''
    methods_list = []
    for transform in transform_class_list:
        attributes = [getattr(transform, method) for method in dir(transform)]
        methods = [attribute for attribute in attributes
                   if callable(attribute) and
                   not attribute.__name__.startswith("__") and
                   not attribute.__name__ == 'fit' and
                   not attribute.__name__ == 'transform' and
                   not attribute.__name__ == 'type']
        methods_list.append(methods)

    print("Generated Lists")
    print("Now building pipelines.")

    product = itertools.product(*methods_list)
    pipelines = []
    try:
        for prod in product:
            pipes = []
            for pipe in prod[:-1]:
                pipes.append((pipe.__name__, Transform(pipe)))

            pipes.append((prod[-1].__name__, prod[-1]()))
            pipelines.append(Pipeline(pipes))
    except TypeError as error:
        print("Pipeline Creation Failed!")
        print(error)
        #print(methods_list)
        #print(transform_class_list)

        print(pipes)

    pipelines = chunk_pipelines(pipelines, n_chunks=batches)
    return pipelines

def chunk_pipelines(pipelines, n_chunks = 1):
    '''Take a list of pipeline objects, and return a list of n_chunks sublists
    of pipelines.  This should try to evenly distribute the workload required
    for each batch but does not in this iteration'''

    if n_chunks == 1:
        return [pipelines]
    else:
        chunk_size = int(len(pipelines)/n_chunks)
        #create first equally size chunks
        chunks = [pipelines[chunk_size*i:chunk_size*(i+1)] for i in range(0,
                            n_chunks-1)]
        
        #make the last chunk
        chunks.append(pipelines[n_chunks*chunk_size:len(pipelines)])
        chunks = [chunk for chunk in chunks if chunk]
        return chunks


def make_pipeline_df(pipelines, transform_class_names):
    rows=[]
    for pipeline in pipelines:
        rows.append([pipeline]+[step[0] for step in pipeline.steps])
    df=pd.DataFrame(rows, columns=['pipeline']+transform_class_names)

    return df



def objective_score(results):
    return pd.DataFrame(results).mean().values[0]
def test_parallel():#pipelines):
    print('hello')#for pipeline in pipelines:
    #    print(pipeline)
        
    
#@timer()
def run_coliseum(x_train, y_train, transform_class_list, scorer=None, grids=None, n_jobs=1):
    '''
    Takes a training set, and a sequence of transform classes, and compares the 
    possible pipelines in the given order. 
    Parameters:
        x_train - Predictors in the training data.  They should be compatible 
                    with the given models.
        y_train - The true values we are training with.  We want to predict 
                    these.
        tranform_class_list - This is a list of classes that we will build the 
                    pipelines from.  Each step of the pipeline is pulled from 
                    the class in the respective position in this list.
        scorer - This is a Scorer object that is compatible with cross_validate
        grids - Not currenty used.  This represents input for hyper parameter 
                    search.
        n_jobs - The number of processes to run this under.  
                    In future n_jobs will allow parallel computing.  This is 
                    failing right now. Using n_jobs in cv step turned out to 
                    slow things down more than anything.  Too much memory 
                    moving around. This current iteration batches the pipelines
                    into chunks that are processed in order.  This was done to 
                    allow parallel processing of the different pipelines. In 
                    the current parallel design, each batch  of 
                    pipelines(a list) together with copies of the traiing data
                    are is passed to the evauate pipelines function. There is 
                    a bug associated with the pickling of pipelines in the 
                    parallelization process. Parallelizing at the pipeline 
                    level seems like the route to go since processes are only spawned at the first step, and there is minimal inter process communication.
                    This design seems like it would be scalable for large numbers of pipelines. Joblib seems to perform poorly with its Parallel the dataframe as input.
                
                    A simple test 
                        Bug Example: Here pipelines is a series of Pipeline objects. A Pipeline is effevitively a list of tranformation objects followed by a model. 
                            Each transformation has a fit and transform method. To create the different pipelines, a tranform object rebinds its 
                            transform method to a given function(df->df).  The fit method returns the transform object that is ready to transform.
                            The model needs to implement fit(x_train,y_train->model) for the Pipeline to work.   
                            
        
                    pipelines=coliseum_results['pipeline']

                    Parallel(n_jobs=5, backend = "multiprocessing")(delayed(evaluate_pipelines)(chunk) for chunk in pipelines)
                    
        Return:
            A data frame with columns ['pipeline']+transform_names+ ['average fit_time',
                                                         'average score_time', 
                                                         'average test_score'])
            Currently the first column is a series of Pipelines that have steps bound to functions in the tranform classes.  This seems to give us some trouble with parallelizing.
            
    '''
    seed = 7
    
    if scorer == None:
        scorer = make_scorer(log_loss, labels=[0, 1])
    
    
    pipelines = make_pipelines(transform_class_list, batches = n_jobs)
    transform_names=[transform.__name__ for transform in transform_class_list]
    #cross validation setup.
    #When n_splits was set to 22, there was an error introduced when doing feature selection.  
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    
    
    #mode='serial'
    if n_jobs ==1:
        cv_results = [evaluate_pipelines(x_train, y_train,
                              transform_names, chunk, scorer,
                              kfold) for chunk in pipelines]
    #mode='parallel'
    elif n_jobs != 1:
        cv_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
                delayed(evaluate_pipelines)(
                x_train, y_train, transform_names, chunk, scorer,
                kfold) for chunk in pipelines)
    return pd.concat(cv_results)


def evaluate_pipelines(x_train, y_train, transform_names, pipelines, scorer, kfold):
    cv_results = []
    n=1
    for pipeline in pipelines:
        print("Running pipeline: "+str(n)+" Out of "+ str(len(pipelines)))
        cv = evaluate_pipeline(x_train, y_train, pipeline, scorer, kfold)
        cv = pd.DataFrame(cv).mean().values
        cv_results.append(cv)
        n = n+1
    pipeline_df=make_pipeline_df(pipelines, transform_names)
    #data frame containing cv results for all the pipelines.
    coliseum_results = pd.DataFrame(cv_results, columns=['fit_time',
                                                         'score_time', 
                                                         'test_score'])

    results = pd.concat([pipeline_df, coliseum_results], axis=1)
    #results.columns = ['pipeline']+transform_names+
    return results

def evaluate_pipeline(x_train, y_train, pipeline, scorer, kfold):
    '''Takes training data, and a list of pipelines, fits the pipelines to
    the data, scores each pipeline, and returns a dataframe containing
        #return list of the form "ModName", "Coder", "Cleaner","ImpName",
        "CVResults","Score"
    '''
    log = "working on :" +"/"+str(pipeline)+"\n"
    try:
        cv_results = model_selection.cross_validate(pipeline, x_train, 
                                                    y_train, cv=kfold,
                                                    scoring=scorer,
                                                    n_jobs=1, 
                                                    return_train_score=False)
        return cv_results
    except ValueError as error:
        print("cross validation failed.")
        print(error)
        return float('nan')
        #print(traceback.print_tb(error.__traceback__))

def run_tests(dm, stage=1, n_jobs=1):
    '''Run the stage one and stage 2 tests for the kaggle comp.
    Stage 1 consist of predicting the marchmadness outcome for every possible
    team matchup of the seeded teams each year.
    '''
    transform_class_list = [Imputer, Filterer, FeatureSelector, Model]

    #compute log loss scores using 2010-2013 training data.  Predict on 2014,
    #2015,2016,2017
    if stage == 'sanity':
        training_years=[2010]
        test_years=[2010]

    elif stage == 1:
        training_years=[2010, 2011, 2012, 2013]
        test_years=[2014,2015,2016,2017]


    #compute log loss scores using 2010-2017 training data.  Predict on 2018
    elif stage ==2:
        training_years=[2010,2011,2012,2013,2014,2015,2016,2017]
        test_years=[2018]

    x_train, y_train = dm.make_training_data_years(training_years)
    x_test, y_test = dm.make_test_data_years(test_years)

    print("loaded Data")
    print("Running Coliseum")

    #compare the pipelines using cv.
    coliseum_results = run_coliseum(x_train, y_train, transform_class_list, 
                                    n_jobs = n_jobs)

    #print("Best Pipelines are:")
    #slice out the pipelines that had the minimum (mean) cv score on the test
    #data.
    best=coliseum_results[coliseum_results['test_score']==
                     coliseum_results['test_score'].min()]
    #train best models on all training data, and compute total log loss
    # of the predictions on the

    y_pred_list=[]
    for b in best['pipeline']:
        y_pred=b.fit(x_train,y_train).predict(x_test)
        y_pred_list.append(y_pred)
        print("total log loss on training data for best model is :" +str(
                log_loss(y_test, y_pred)))
    #prepare submission for kaggle.
    coliseum_results.to_csv("colleseum_results_stage_"+str(stage)+str(datetime.datetime.today()).split()[0]+".csv")


    end = time.time()
    print("Computation Time: "+ str(end - start))

    return coliseum_results, best, y_pred_list


def create_submission_file(results, stage=None):
    if stage==None:
        print("Not Needed now")

#A sanity check for the program working.
if __name__ == '__main__':
    start = time.time()
    if 'dm' not in locals():
        dm = DataManager()
    print("Now loading training data.")
    #coliseum_results, best, y_pred_list = run_tests(dm, stage='sanity',n_jobs=1)
    #coliseum_results, best, y_pred_list = run_tests(dm, stage=1,n_jobs=1)
    coliseum_results, best, y_pred_list = run_tests(dm, stage=2,n_jobs=1)

#   print(accuracy_score(Y_validation, predictions))
#   print(confusion_matrix(Y_validation, predictions))
#   print(classification_report(Y_validation, predictions))

    #prepare submission for kaggle.
    end = time.time()
    print("Computation Time: "+ str(end - start))
    print("Best Pipelines are:")
    best=coliseum_results[coliseum_results['test_score']==
                     coliseum_results['test_score'].min()]


