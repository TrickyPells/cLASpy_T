#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################
#  #################_###############_###############_###############  #
#  ################/\ \############/ /\############/\ \#############  #
#  ###############/  \ \##########/ /  \##########/  \ \############  #
#  ##############/ /\ \ \########/ / /\ \_#######/ /\ \_\###########  #
#  #############/ / /\ \_\######/ / /\ \__\#####/ / /\/_/###########  #
#  ############/ / /_/ / /######\ \ \#\/__/####/ / /#______#########  #
#  ###########/ / /__\/ /########\ \ \########/ / /#/\_____\########  #
#  ##########/ / /_____/##########\ \ \######/ / /##\/____ /########  #
#  #########/ / /\ \ \######/_/\__/ / /#####/ / /_____/ / /#########  #
#  ########/ / /##\ \ \#####\ \/___/ /#####/ / /______\/ /##########  #
#  ########\/_/####\_\/######\_____\/######\/___________/###########  #
#  ---------- REMOTE -------- SENSING --------- GROUP --------------  #
#  #################################################################  #
#       'training.py' from cLASpy_T library to train dataset          #
#                    By Xavier PELLERIN LE BAS                        #
#                         November 2019                               #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description:                                                       #
#                                                                     #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------

import joblib
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def split_dataset(data_values, target_values, train_ratio=0.5, test_ratio=0.5, samples=0.5):
    """
    Split the input data and target in data_train, data_test, target_train and target_test.
    :param data_values: the np.ndarray with the data features.
    :param target_values: the np.ndarray with the target.
    :param train_ratio: (optional) Ratio of the size of training dataset.
    :param test_ratio: (optional) Ratio of the size of testing dataset.
    :param samples: (optional) Number of samples beyond which the dataset
    is splitted with two integers, for train_size and test_size.
    The samples is paired with train_ratio and test_ratio.
    :return: data_train, data_test, target_train and target_test as np.ndarray.
    """
    # Set the train_size and test_size according sample size
    train_size = int(samples * train_ratio)
    test_size = int(samples * test_ratio)

    # Perform the train test split
    data_train, data_test, target_train, target_test = train_test_split(data_values, target_values,
                                                                        random_state=0,
                                                                        train_size=train_size,
                                                                        test_size=test_size,
                                                                        stratify=target_values)
    # Convert target_train and target_test column-vectors as 1d array
    target_train = target_train.reshape(train_size)
    target_test = target_test.reshape(test_size)

    print("\tNumber of used points: {} pts".format(train_size + test_size))
    print("\tSize of train|test datasets: {} pts | {} pts".format(train_size, test_size))

    return data_train, data_test, target_train, target_test


def check_parameters(classifier, fit_params):
    """
    Check if the given parameters match with the given classifier
    and set the classifier with the well defined parameters.
    :param classifier: The given classifier.
    :param fit_params: Parameters to check in dict.
    :return: classifier: The setting up classifier.
    """
    # Get the type of classifier
    clf_name = str(type(classifier)).split('.')[-1][:-2]

    # Check if the parameters are valid for the given classifier
    for key in fit_params.keys():
        try:
            temp_dict = {key: fit_params[key]}
            classifier.set_params(**temp_dict)
        except ValueError:
            print("ValueError: Invalid parameter '{}' for {}, "
                  "it was skipped!".format(str(key), clf_name))

    return classifier


def check_grid_params(pipeline, grid_params):
    """
    Check if the given grid_params match with the given classifier.
    :param pipeline: Pipeline set with 'scaler' and 'classifier' (at least).
    :param grid_params: Grid parameters to check in dict.
    :return: pipe_params: Grid parameters with only the well defined parameters for the pipeline.
    """
    # Get the type of classifier
    clf_name = str(type(pipeline.named_steps['classifier'])).split('.')[-1][:-2]

    # Get the list of valid parameters from the classifier
    param_names = pipeline.named_steps['classifier']._get_param_names()

    # Check if grid_params is None or empty
    well_params = dict()  # dictionary with only good parameters
    if grid_params:
        for key in grid_params.keys():  # Check if the keys of grid_params are in list of valid parameters
            if key in param_names:
                well_params[key] = grid_params[key]
            else:
                print("GridSearchCV: Invalid parameter '{}' for {}, it was skipped!".format(str(key), clf_name))

    # Check if well_params is empty dict and set predefined parameters
    if not well_params:
        if clf_name == 'RandomForestClassifier' or clf_name == 'GradientBoostingClassifier':
            well_params = {'n_estimators': [50, 100, 500],
                           'max_depth': [8, 11, 14],
                           'min_samples_leaf': [100, 500, 1000]}
        elif clf_name == 'LinearSVC':
            well_params = {'penalty': ('l1', 'l2'),
                           'C': [0.01, 1.0, 100]}
        elif clf_name == 'MLPClassifier':
            well_params = {'hidden_layer_sizes': [(25, 25), (25, 25, 25), (25, 50, 25)],
                           'activation': ('tanh', 'relu'),
                           'alpha': [0.0001, 0.01, 1.0]}

    # Create parameter dictionary for Pipeline
    pipe_params = dict()
    for key in well_params.keys():
        new_key = 'classifier__' + key
        pipe_params[new_key] = well_params[key]

    return pipe_params


def set_random_forest(fit_params=None, n_jobs=-1):
    """
    Set the learning algorithm as RandomForestClassier.
    :param fit_params: A dict with the parameters to set up.
    :param n_jobs: The number of CPU used.
    :return: classifier: the desired classifier with the required parameters
    """
    # Set the classifier
    if isinstance(fit_params, dict):
        fit_params['random_state'] = 0
        classifier = RandomForestClassifier()
        classifier = check_parameters(classifier, fit_params)  # Check and set parameters

    else:
        classifier = RandomForestClassifier(n_estimators=100,
                                            max_depth=8,
                                            min_samples_leaf=500,
                                            n_jobs=n_jobs,
                                            random_state=0)

    return classifier


def set_gradient_boosting(fit_params=None):
    """
    Set the learning algorithm as GradientBoostingClassifier.
    :param fit_params: A dict with the parameters to set up.
    :return: classifier: the desired classifier with the required parameters
    """
    # Set the classifier
    if isinstance(fit_params, dict):
        fit_params['random_state'] = 0
        classifier = GradientBoostingClassifier()
        classifier = check_parameters(classifier, fit_params)
    else:
        classifier = GradientBoostingClassifier(loss='deviance',
                                                n_estimators=100,
                                                max_depth=3,
                                                min_samples_leaf=1000,
                                                random_state=0)

    return classifier


def set_linear_svc(fit_params=None):
    """
    Set the learning algorithm as LinearSVC.
    :param fit_params: A dict with the parameters to set up.
    :return: classifier: the desired classifier with the required parameters
    """
    # Set the classifier
    if isinstance(fit_params, dict):
        fit_params['random_state'] = 0
        fit_params['dual'] = False
        classifier = LinearSVC()
        classifier = check_parameters(classifier, fit_params)
    else:
        classifier = LinearSVC(penalty='l2',
                               loss='hinge',
                               dual=False,
                               C=1.0,
                               multi_class='ovr',
                               max_iter=1000,
                               random_state=0)

    return classifier


def set_mlp_classifier(fit_params=None):
    """
    Set the learning algorithm as MLPClassifier.
    :param fit_params: A dict with the parameters to set up.
    :return: classifier: the desired classifier with the required parameters
    """
    # Set the classifier
    if isinstance(fit_params, dict):
        fit_params['random_state'] = 0
        fit_params['max_iter'] = 10000
        classifier = MLPClassifier()
        classifier = check_parameters(classifier, fit_params)

    else:
        classifier = MLPClassifier(hidden_layer_sizes=(50, 50),
                                   activation='relu',
                                   solver='adam',
                                   alpha=0.0001,
                                   max_iter=10000,
                                   random_state=0)

    return classifier


def training_gridsearch(pipeline, training_data, training_target, grid_params=None, n_jobs=-1, scoring='accuracy'):
    """
    Train model with GridSearchCV meta-estimator according the chosen learning algorithm.
    :param pipeline: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param grid_params: The parameters for the GridSearchCV.
    :param n_jobs: Number of CPU used.
    :param scoring: Set the scorer according scikit-learn documentation.
    :return: classifier, results: Classifier with best parameters and results from grid search.
    """

    # Set cross_validation method with train_size 80% and validation_size 20%
    cross_val = StratifiedShuffleSplit(n_splits=5,
                                       train_size=0.8,
                                       test_size=0.2,
                                       random_state=0)

    # Set the GridSearchCV
    print("\tSearching best parameters...")
    grid = GridSearchCV(pipeline,
                        param_grid=grid_params,
                        n_jobs=n_jobs,
                        cv=cross_val,
                        scoring=scoring,
                        verbose=1,
                        error_score=np.nan)

    # Training the model to find the best parameters
    grid.fit(training_data, training_target)
    results = pd.DataFrame(grid.cv_results_)
    print("\tBest score: {0:.4f}".format(grid.best_score_))
    print("\tBest parameters: {}".format(grid.best_params_))
    print("\tModel trained!")

    return grid, results


def training_nogridsearch(pipeline, training_data, training_target, n_jobs=-1, scoring='accuracy'):
    """
    Train model with cross-validation according the chosen classifier.
    :param pipeline: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param n_jobs: Number of CPU used.
    :param scoring: Set the scorer according scikit-learn documentation.
    :return: model, training_scores: The training model and the scores of n_splits training.
    """
    # Set cross_validation method with train_size 80% and validation_size 20%
    cross_val = StratifiedShuffleSplit(n_splits=5,
                                       train_size=0.8,
                                       test_size=0.2,
                                       random_state=0)

    # Get the training scores
    results = cross_validate(pipeline,
                             training_data,
                             training_target,
                             cv=cross_val,
                             n_jobs=n_jobs,
                             scoring=scoring,
                             verbose=2,
                             return_estimator=True)

    print("\n\tTraining model scores with cross-validation:\n\t{}\n".format(results["test_score"]))

    # Set the classifier with training_data and target
    time.sleep(1.)  # Delay to print the previous message
    pipeline = results["estimator"][1]  # Get the second estimator from the cross_validation

    print(" Model trained!")

    return pipeline, results["test_score"]


def save_model(model_to_save, file_name):
    """
    Save the passing model into a file.
    :param model_to_save: The model to save.
    :param file_name: The path and name of the file.
    """
    joblib.dump(model_to_save, file_name)
    print("Model path: {}/".format('/'.join(file_name.split('/')[:-1])))
    print("Model file: {}".format(file_name.split('/')[-1]))

# -------------------------
# --------- MAIN ----------
# -------------------------

# if __name__ == '__main__':
