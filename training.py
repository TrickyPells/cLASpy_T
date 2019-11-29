#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################
#  #################_###############_###############_###############  #
#  ################/\ \############/ /\############/\ \#############  #
#  ###############/  \ \##########/ /  \##########/  \ \############  #
#  ##############/ /\ \ \########/ / /\ \_#######/ /\ \_\###########  #
#  #############/ / /\ \_\######/ / /\ \__\#####/ / /\/_/###########  #
#  ############/ / /_/ / /######\ \ \#\/__/####/ / /#______#########  #
#  ###########/ / /__\/ /#########\ \ \#######/ / /#/\_____\########  #
#  ##########/ / /_____/_##########\ \ \#####/ / /##\/____ /########  #
#  #########/ / /\ \ \######/_/\__/ / /#####/ / /_____/ / /#########  #
#  ########/ / /##\ \ \#####\ \/___/ /#####/ / /______\/ /##########  #
#  ########\/_/####\_\/######\_____\/######\/___________/###########  #
#  --------- REMOTE --------- SENSING --------- GROUP --------------  #
#  #################################################################  #
#       'training.py' from classer library to train dataset           #
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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def split_dataset(data_values, target_values, train_ratio=0.8, test_ratio=0.2, threshold=0.5):
    """
    Split the input data and target in data_train, data_test, target_train and target_test.
    Check the length of the dataset. If length > threshold, train_size = train_ratio * threshold pts
    and test_size = test_ratio * threshold pts.
    :param data_values: the np.ndarray with the data features.
    :param target_values: the np.ndarray with the target.
    :param train_ratio: (optional) Ratio of the size of training dataset.
    :param test_ratio: (optional) Ratio of the size of testing dataset.
    :param threshold: (optional) Number of samples beyond which the dataset is splitted with two integers,
    for train_size and test_size. The threshold is paired with train_ratio and test_ratio.
    :return: data_train, data_test, target_train and target_test as np.ndarray.
    """

    print("3. Splitting the data...", end='')
    # Rescale threshold
    threshold = int(threshold * 1000000)

    # Check if train_ratio + test_ratio =< 1.0
    if train_ratio > 1.0:
        train_ratio = 0.8

    if train_ratio + test_ratio > 1.0:
        test_ratio = 1.0 - train_ratio

    # Check if dataset > 500 kpts
    n_samples = len(data_values[:, 0])
    if n_samples > threshold:
        train_size = int(train_ratio * threshold)
        test_size = int(test_ratio * threshold)
    else:
        train_size = int(train_ratio * n_samples)
        test_size = int(test_ratio * n_samples)

    data_train, data_test, target_train, target_test = train_test_split(data_values, target_values,
                                                                        random_state=0,
                                                                        train_size=train_size,
                                                                        test_size=test_size,
                                                                        stratify=target_values)
    # Convert target_train and target_test column-vectors as 1d array
    target_train = target_train.reshape(train_size)
    target_test = target_test.reshape(test_size)

    print(" Done.")
    print("\tNumber of used points: {} pts".format(train_size+test_size))
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


def check_grid_params(classifier, grid_params):
    """
    Check if the given grid_params match with the given classifier.
    :param classifier: The given classifier.
    :param grid_params: Grid parameters to check in dict.
    :return: well_params: Grid parameters with only the well defined parameters.
    """
    # Get the type of classifier
    clf_name = str(type(classifier)).split('.')[-1][:-2]

    # Get the list of valid parameters from the classifier
    param_names = classifier._get_param_names()

    # Check if grid_params is None or empty
    well_params = dict()  # dictionary with only good parameters
    if grid_params is None or not bool(grid_params):  # If grid_params is None or empty
        params_isfull = False
        grid_params = well_params
    else:
        params_isfull = True

    # Check if the keys of grid_params are in list of valid parameters
    if params_isfull:
        for key in grid_params.keys():
            if key in param_names:
                well_params[key] = grid_params[key]
            else:
                print("GridSearchCV: Invalid parameter '{}' for {}, it was skipped!".format(str(key), clf_name))

    # Check if well_params is None or empty dict and set predefined parameters
    if well_params is None or not bool(well_params):  # If well_params is None or empty
        if clf_name == 'RandomForestClassifier':
            well_params = {'n_estimators': [10, 50, 100, 500],
                           'max_depth': [5, 8, 11, 14],
                           'min_samples_leaf': [100, 500, 1000]}
        elif clf_name == 'GradientBoostingClassifier':
            well_params = {'loss': ('deviance', 'exponential'),
                           'n_estimators': [10, 50, 100, 500],
                           'max_depth': [5, 8, 11, 14],
                           'min_samples_leaf': [100, 500, 1000]}
        elif clf_name == 'LinearSVC':
            well_params = {'penalty': ('l1', 'l2'),
                           'C': [0.01, 1.0, 100, 1000]}
        elif clf_name == 'MLPClassifier':
            well_params = {'hidden_layer_sizes': [(25, 25), (25, 25, 25), (25, 50, 25)],
                           'activation': ('identity', 'logistic', 'tanh', 'relu'),
                           'alpha': [0.0001, 0.01, 1.0]}

    return well_params


def set_random_forest(fit_params=None):
    """
    Set the learning algorithm as RandomForestClassier.
    :param fit_params: A dict with the parameters to set up.
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
                                            n_jobs=-1,
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
        classifier = MLPClassifier()
        classifier = check_parameters(classifier, fit_params)

    else:
        classifier = MLPClassifier(hidden_layer_sizes=(50, 50),
                                   activation='relu',
                                   solver='adam',
                                   alpha=0.0001,
                                   max_iter=200,
                                   random_state=0)

    return classifier


def training_gridsearch(classifier, training_data, training_target, grid_params=None, n_jobs=-1, scoring='accuracy'):
    """
    Train model with GridSearchCV meta-estimator according the chosen learning algorithm.
    :param classifier: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param grid_params: The parameters for the GridSearchCV.
    :param n_jobs: Number of CPU used.
    :param scoring: Set the scorer according scikit-learn documentation.
    :return: classifier, results: Classifier with best parameters and results from grid search.
    """

    print('4. Training the model with GridSearchCV...')
    # Set cross_validation method with train_size 80% and validation_size 20%
    cross_val = StratifiedShuffleSplit(n_splits=5,
                                       train_size=0.8,
                                       test_size=0.2,
                                       random_state=0)

    # Set the GridSearchCV
    print("\tSearching best parameters...")
    classifier = GridSearchCV(classifier,
                              param_grid=grid_params,
                              n_jobs=n_jobs,
                              cv=cross_val,
                              scoring=scoring,
                              verbose=1,
                              error_score=np.nan)

    # Training the model to find the best parameters
    classifier.fit(training_data, training_target)
    results = pd.DataFrame(classifier.cv_results_)
    print("\tThe best score: {0:.4f}".format(classifier.best_score_))
    print("\tThe best parameters: {}".format(classifier.best_params_))
    print("\tModel trained!")

    return classifier, results


def training_nogridsearch(classifier, training_data, training_target, n_jobs=-1, scoring='accuracy'):
    """
    Train model with cross-validation according the chosen classifier.
    :param classifier: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param n_jobs: Number of CPU used.
    :param scoring: Set the scorer according scikit-learn documentation.
    :return: model, training_scores: The training model and the scores of n_splits training.
    """

    print("4. Training the model with cross validation...")
    # Set cross_validation method with train_size 80% and validation_size 20%
    cross_val = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2, random_state=0)

    # Get the training scores
    training_scores = cross_val_score(classifier, training_data, training_target,
                                      cv=cross_val,
                                      n_jobs=n_jobs,
                                      scoring=scoring,
                                      verbose=1)

    print("\n\tTraining model scores with cross-validation:\n\t{}\n".format(training_scores))

    # Set the classifier with training_data and target
    print("\tRefitting the model with all given data...", end='')
    classifier.fit(training_data, training_target)

    print(" Model trained!\n")

    return classifier, training_scores


def save_model(model_to_save, file_name):
    """
    Save the passing model into a file.
    :param model_to_save: The model to save.
    :param file_name: The path and name of the file.
    """
    joblib.dump(model_to_save, file_name)
    print("Model path: {}".format('/'.join(file_name.split('/')[:-1])))
    print("Model file: {}".format(file_name.split('/')[-1]))

# -------------------------
# --------- MAIN ----------
# -------------------------

# if __name__ == '__main__':
