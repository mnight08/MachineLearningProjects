# -*- coding: utf-8 -*-
"""A coliseum is an object that will compare a list of pipelines over some
    parameter space will compare different models, and data cleaning methods,
    subset of the training data using cross validation and logloss.
    Sets up pipeline for the models.  Defines the transforms and runs cross
    validation with various parameters. Each method will return a
"""
from functools import partial
import multiprocessing
import timeit
import time
import datetime
import itertools


from sklearn.pipeline  import Pipeline
from sklearn import model_selection
from sklearn.metrics import make_scorer, log_loss


import pandas as pd




from dataexplorer import DataExplorer
from datamanager import DataManager
from featureextractor import FeatureExtractor
from transform import *

def make_pipelines(transform_class_list, batches=1):
    '''Takes a list of classes of transforms.  Extracts the callable methods,
       creates all possibles tuples in the order given of the methods, creates
       a list of data frames representing the possible pipelines.
    '''
    methods_list = []
    for transform in transform_class_list:
        attributes = [getattr(transform, method) for method in dir(transform)]
        methods = [attribute for attribute in attributes
                   if callable(attribute) and
                   not attribute.__name__.startswith("__") and
                   not attribute.__name__ == 'fit' and
                   not attribute.__name__ == 'transform' and
                   not attribute.__name__ == 'predict' and
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
                pipes.append(pipe.__name__)

            pipes.append(prod[-1].__name__)
            pipelines.append(pipes)
    except TypeError as error:
        print("Pipeline Creation Failed!")
        print(error)
        print(pipes)
    pipelines = chunk_pipelines(pipelines, n_chunks=batches)
    transform_names = [transform.__name__ for transform in transform_class_list]
    return [make_pipeline_df(chunk, transform_names) for chunk in pipelines]

def chunk_pipelines(pipelines, n_chunks=1):
    '''Take a list of pipelines, and return a list of n_chunks
    of pipelines.  This should try to evenly distribute the workload required
    for each batch but does not in this iteration'''
    if n_chunks == 1:
        return [pipelines]
    else:
        chunk_size = int(len(pipelines)/n_chunks)
        #create first equally size chunks
        chunks = [pipelines[chunk_size*i
                            :chunk_size*(i+1)] for i in range(0, n_chunks)]

        #make the last chunk
        chunks.append(pipelines[n_chunks*chunk_size:len(pipelines)])
        chunks = [chunk for chunk in chunks if chunk]
        return chunks

def make_pipeline_df(pipelines, transform_class_names):
    ''''''
    return pd.DataFrame(pipelines, columns=transform_class_names)

def objective_score(results):
    ''''''
    return pd.DataFrame(results).mean().values[0]

def run_coliseum(x_train, y_train, transform_class_list, scorer=None, grids=None,
                 n_jobs=1):
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
                    Parallelizing at the pipeline level.  does not balance load 
                    right now.

        Return:
            A data frame with columns transform_names+ ['average fit_time',
                                                         'average score_time',
                                                         'average test_score'])

    '''
    seed = 7

    if scorer is None:
        scorer = make_scorer(log_loss, labels=[0, 1])

    #create a data frame of the pipelines that we need to test.
    chunks = make_pipelines(transform_class_list, batches=n_jobs)

    #get the transform types names for the columns of the data frame.
    transform_names = [transform.__name__ for transform in transform_class_list]
    #cross validation setup.
    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    #mode='serial'
    if n_jobs == 1:
        cv_results = [evaluate_pipelines(x_train, y_train,
                                         transform_names, scorer,
                                         kfold, chunk) for chunk in chunks]
    #mode='parallel'
    elif n_jobs != 1:
        pool = multiprocessing.Pool(processes=n_jobs)
        func = partial(evaluate_pipelines, x_train, y_train,
                       transform_names, scorer, kfold)
        cv_results = pool.map(func, chunks)
    return pd.concat(cv_results, ignore_index=True)


def evaluate_pipelines(x_train, y_train, transform_names, scorer,
                       kfold, pipeline_df):
    '''
    Takes a pipeline data frame and cross validates each corresponding
    pipelines on the training data.
    '''
    cv_results = []
    n = 1
    pipelines = get_pipelines(pipeline_df)
    for pipeline in pipelines:
        print("Running pipeline: "+str(n)+" Out of "+ str(len(pipelines)))
        cv = evaluate_pipeline(x_train, y_train, pipeline, scorer, kfold)
        cv = pd.DataFrame(cv).mean().values
        cv_results.append(cv)
        n = n+1
    #data frame containing cv results for all the pipelines.
    coliseum_results = pd.DataFrame(cv_results, columns=['fit_time',
                                                         'score_time',
                                                         'test_score'])

    results = pd.concat([pipeline_df, coliseum_results], axis=1)
    return results

def get_pipelines(pipeline_df):
    '''
    Takes a pipeline dataframe and returns a list of Pipeline objects
    corresponding the string function names.  Each step in the df is a column
    of strings.
    '''
    pipelines = []
    print(pipeline_df)
    stages = [globals()[class_name] for class_name  in pipeline_df.columns]
    for index, pipeline in pipeline_df.iterrows():
        steps = []
        for stage in stages:
            stage_index = stages.index(stage)
            func_name = pipeline[stage_index]
            transform = stage(getattr(stage, func_name))
            steps.append((func_name, transform))

        pipelines.append(Pipeline(steps))
    print("got pipeline")
    return pipelines

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


def get_best_model(dm, coliseum_results):
    '''Return a single pipeline that performed the best on the training data and 
    train it with the full training set.'''
    model = None
    
    return model



def make_submission(dm, coliseum_results):
    '''Choose the best model based on coliseum results, take the test data, 
    extract the features of each measurement, predict the class of each 
    measurement, then format for kaggle. The format is signal_id, class.  
    The testing data set is much larger than the training set at over 8 gb. 
    The features will need to be extracted in chunks.'''
    fe = FeatureExtractor(dm)
    best_model = get_best_model(coliseum_results)
    submission_chunks = []
    
    pd.DataFrame(columns = ['signal_id', 'target'])
    for chunk in dm.get_test_data_chunks(num_chunks=20):  
        x_test = fe.make_x_train_data()
        y_pred=best_model.predict(x_test)
        submission_chunk=pd.DataFrame(columns = ['signal_id', 'target'])
        submission_chunks.append(submission_chunk)
        
        
        
    submission = pd.concat(submission_chunks)
    return submission.to_csv()

    
    
def run_tests(dm, n_jobs=1):
    '''Run the stage one and stage 2 tests for the kaggle comp.
    Stage 1 consist of predicting the marchmadness outcome for every possible
    team matchup of the seeded teams each year.
    '''
    transform_class_list = [Imputer, Filterer, FeatureSelector, Scaler, Model]
    test_period = "March Madness"

    #compute log loss scores using 2010-2013 training data.  Predict on 2014,
    #2015,2016,2017
    if stage == 'sanity':
        training_years = [2010]
        test_years = [2010]

    elif stage == 1:
        training_years = [2010, 2011, 2012, 2013]

        test_years = [2014, 2015, 2016, 2017]
        test_period = "March Madness"


    #compute log loss scores using 2010-2017 training data.  Predict on 2018
    elif stage == 2:
        training_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        #The MM data for this stage is not available on computer.  This will fail
        #to run.

        training_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
        test_years = [2018]
        test_period = "March Madness"

    x_train, y_train = dm.make_training_data_years(training_years, period="All")
    x_test, y_test = dm.make_test_data_years(test_years, period=test_period)

    print("loaded Data")
    print("Running Coliseum")

    #compare the pipelines using cv.
    coliseum_results = run_coliseum(x_train, y_train, transform_class_list,
                                    n_jobs=n_jobs)
    coliseum_results.to_csv("coliseum_results_stage_"+str(stage)
                            +str(datetime.datetime.today()).split()[0]+".csv")
    #print("Best Pipelines are:")
    #slice out the pipelines that had the minimum (mean) cv score on the test
    #data.
    best = coliseum_results[
        coliseum_results['test_score'] == coliseum_results['test_score'].min()]
    #train best models on all training data, and compute total log loss
    # of the predictions on the

    y_pred_list = []
    for pipeline in get_pipelines(pd.DataFrame(
            best[0:len(transform_class_list)],
            columns=[t.__name__ for t in transform_class_list])):
        y_pred = pipeline.fit(x_train, y_train).predict(x_test)
        y_pred_list.append(y_pred)
        print("total log loss on test data for best model is :" +str(
            log_loss(y_test, y_pred)))
    #prepare submission for kaggle.

    end = time.time()
    print("Computation Time: "+ str(end - start))
    return x_train, y_train, coliseum_results, best, y_pred_list




#A sanity check for the program working.
if __name__ == '__main__':
    start = time.time()
    if 'dm' not in locals():
        dm = DataManager()
    print("Now loading training data.")
    #x_train, y_train, coliseum_results, best, y_pred_list = run_tests(
    #    dm, stage='sanity', n_jobs=8)
    x_train, y_train, coliseum_results, best, y_pred_list = run_tests(dm, stage=1, n_jobs=1)
    #x_train, y_train, coliseum_results, best, y_pred_list = run_tests(dm, stage=2, n_jobs=1)

#   print(accuracy_score(Y_validation, predictions))
#   print(confusion_matrix(Y_validation, predictions))
#   print(classification_report(Y_validation, predictions))

    #prepare submission for kaggle.
    end = time.time()
#    print("Computation Time: "+ str(end - start))
#    print("Best Pipelines are:")
    
