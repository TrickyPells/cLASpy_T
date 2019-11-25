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
#               'classer.py' from classer library                     #
#                By Xavier PELLERIN LE BAS and                        #
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

import os
import subprocess
import sys
import yaml
import argparse

from training import *
from datetime import datetime


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

parser = argparse.ArgumentParser(description="This library performs machine learning algorithms to classify\n"
                                             "points of 3D point cloud. The input data has to be in CSV format.\n\n"
                                             "For training, csv_data_file must contain:\n"
                                             "    --> target field named 'target' AND data fields\n"
                                             "For prediction, csv_data_file must contain:\n"
                                             "    --> data fields\n\n"
                                             "If X, Y and/or Z fields are present, they are excluded.\n"
                                             "If a field_name is 'classif', 'raw_classification'...\n"
                                             "the field is discarded.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("algorithm",
                    help="the learning algorithm to use:\n"
                         "    'rf': RandomForestClassifier\n"
                         "    'gb': GradientBoostingClassifier\n"
                         "    'svm': LinearSVC\n"
                         "    'ann': MLPClassifier\n",
                    type=str, choices=['rf', 'gb', 'svm', 'ann'])

parser.add_argument("csv_data_file",
                    help="the CSV file with needed data:\n"
                         "    [WINDOWS]: 'C:/path/to/the/data.file'\n"
                         "    [UNIX]: '/path/to/the/data.file'",
                    metavar="[\"/path/to.file\"]")

parser.add_argument("-g", "--grid_search",
                    help="perform the training with GridSearchCV",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help="pass the grid parameters as list(sep=' ') in dict.\n"
                         "    If empty, GridSearchCV uses presets.\n"
                         "    Example: \"{'n_estimators': [50, 100, 500] , 'loss': ['deviance', 'exponential']}\"\n",
                    type=str, metavar="[\"dict\"]")

parser.add_argument("-m", "--model_to_import",
                    help="the model file to import to make predictions:\n"
                         "    [WINDOWS]: 'C:/path/to/the/training/model.file'\n"
                         "    [UNIX]: '/path/to/the/training/model.file'",
                    type=str, metavar="[\"/path/to.file\"]")

parser.add_argument("-p", "--parameters",
                    help="the parameters to set up the classifier as dict.\n"
                         "    Example: \"{'n_estimators': 50, 'max_depth': 5, 'max_iter': 500}\"",
                    type=str, metavar="[\"dict\"]")

parser.add_argument("-s", "--scaler",
                    help="method to scale the data before training.\n"
                         "    See the preprocessing documentation of scikit-learn.",
                    type=str, choices=['Standard', 'Robust', 'MinMax'])

parser.add_argument("--test_ratio",
                    help="set the test ratio as float [0.0-1.0] to split into train and test data.\n"
                         "    If train_ratio + test_ratio > 1\n"
                         "    then test_ratio = 1 - train_ratio",
                    type=float, default=0.2, metavar="[0.0-1.0]")

parser.add_argument("--threshold",
                    help="set the threshold, in Million points, for large dataset.\n"
                         "    If data length > threshold:\n"
                         "    then train + test length = threshold",
                    type=float, default=0.5, metavar="[in M points]")

parser.add_argument("--train_ratio",
                    help="set the train ratio as float number to split into train and test data.\n"
                         "    If train_ratio + test_ratio > 1:\n"
                         "    then test_ratio = 1 - train_ratio",
                    type=float, default=0.8, metavar="[0.0-1.0]")

args = parser.parse_args()

# -------------------------
# --------- MAIN ----------
# -------------------------

# path to the CSV file
raw_data = args.csv_data_file

# Path to the saved model and results
save_path = '.'.join(raw_data.split('.')[:-1])

# Learning algorithm 'rf', 'gb', 'svm', 'ann'
algo = args.algorithm

# Check parameters exists
parameters = None
if args.parameters:
    parameters = yaml.safe_load(args.parameters)

# Set the chosen learning classifier
if algo is 'rf':
    clf = set_random_forest(parameters)
elif algo is 'gb':
    clf = set_gradient_boosting(parameters)
elif algo is 'svm':
    clf = set_linear_svc(parameters)
elif algo is 'ann':
    clf = set_mlp_classifier(parameters)
else:
    raise ValueError("Any valid classifier was selected !")

# Is it TRAINING or PREDICTIONS
if not args.model_to_import:

    # With or without GridSearchCV
    grid = False
    if args.grid_search:
        grid = True

    # Check param_grid exists
    param_grid = None
    if args.param_grid:
        param_grid = yaml.safe_load(args.param_grid)

    param_grid = check_grid_params(clf, grid_params=param_grid)

    # Format the data as data / XY / Z / target DataFrames and remove raw_classification from some LAS files.
    print("1. Format data as pandas.Dataframe...", end='')
    data, xy_coord, z_height, target = format_dataset(raw_data, raw_classif='lassif')
    print(" Done.")

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    print("2. Scaling and splitting the data...", end='')
    data = scale_dataset(data, method='Standard')

    # Create samples for training and testing
    X_train_val, X_test, y_train_val, y_test = split_dataset(data.values, target.values,
                                                             train_ratio=args.train_ratio,
                                                             test_ratio=args.test_ratio,
                                                             threshold=args.threshold)

    # kernel approximation for SVM
    if algo is 'svm':
        feature_map_nystroem = Nystroem(gamma=.2, n_components=100, random_state=0)
        X_train_val = feature_map_nystroem.fit_transform(X_train_val)
        X_test = feature_map_nystroem.fit_transform(X_test)

    # What type of training
    if grid:
        print('3. Training the model with GridSearchCV...')
        model, grid_results = training_gridsearch(clf, X_train_val, y_train_val, grid_params=param_grid)
        grid_results.to_csv(str(save_path + '_results.csv'), index=False)
        print("\tModel trained!")

    else:
        print("3. Training the model with cross validation...")
        model, cv_results = training_nogridsearch(clf, X_train_val, y_train_val)
        print("\tModel trained!")

    print("4. Score model with test_dataset: {0:.4f}".format(model.score(X_test, y_test)))

    joblib.dump(model, str(save_path + '_model.joblib'))

# else