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
import yaml
import argparse

from common import *
from training import *
from predict import *

from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.kernel_approximation import Nystroem

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
                    type=str, metavar="/path/to.file")

parser.add_argument("-g", "--grid_search",
                    help="perform the training with GridSearchCV",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help="pass the grid parameters as list(sep=' ') in dict.\n"
                         "    If empty, GridSearchCV uses presets.\n"
                         "    Example: \"{'n_estimators': [50, 100, 500] , 'loss': ['deviance', 'exponential']}\"\n",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-m", "--model_to_import",
                    help="the model file to import to make predictions:\n"
                         "    [WINDOWS]: 'C:/path/to/the/training/model.file'\n"
                         "    [UNIX]: '/path/to/the/training/model.file'",
                    type=str, metavar="[=\"/path/to.file\"]")

parser.add_argument("-n", "--n_jobs",
                    help="set the number of CPU used, '-1' means all CPU available.",
                    type=int, metavar="[1, 2,..., -1]")

parser.add_argument("-p", "--parameters",
                    help="the parameters to set up the classifier as dict.\n"
                         "    Example: \"{'n_estimators': 50, 'max_depth': 5, 'max_iter': 500}\"",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-s", "--scaler",
                    help="method to scale the data before training.\n"
                         "    See the preprocessing documentation of scikit-learn.",
                    type=str, choices=['Standard', 'Robust', 'MinMax'])

parser.add_argument("--scoring",
                    help="set scorer to GridSearchCV or cross_val_score according\n"
                         "    to sckikit-learn documentation.",
                    type=str, default='accuracy',
                    metavar="[='accuracy', balanced_accuracy', 'average_precision',"
                            " 'precision', 'recall', ...]")

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

# Create a folder to store model, results, confusion matrix and grid results
print("Create a new folder to store the results files...", end='')
raw_data = '/'.join(raw_data.split('\\'))  # Change '\' in '/'
folder_path = '.'.join(raw_data.split('.')[:-1])  # remove extension so give folder path
try:
    os.mkdir(folder_path)  # Using file path to make new folder
    print(" Done.")
except (TypeError, FileExistsError):
    print(" Folder already exists.")

# Learning algorithm 'rf', 'gb', 'svm', 'ann'
algo = args.algorithm

# Check parameters exists
parameters = None
if args.parameters:
    parameters = yaml.safe_load(args.parameters)

# Set the chosen learning classifier
if algo == 'rf':
    clf = set_random_forest(parameters)
elif algo == 'gb':
    clf = set_gradient_boosting(parameters)
elif algo == 'svm':
    clf = set_linear_svc(parameters)
elif algo == 'ann':
    clf = set_mlp_classifier(parameters)
else:
    raise ValueError("Any valid classifier was selected !")

# Timestamp for files created
create_time = datetime.now().strftime("%y%m%d_%H%M%S")  # Timestamp for file creation

# Is it TRAINING or PREDICTIONS ?
if not args.model_to_import:
    mod = 'training'
    # With or without GridSearchCV
    grid = False
    if args.grid_search:
        grid = True

    # Check param_grid exists
    param_grid = None
    if args.param_grid:
        param_grid = yaml.safe_load(args.param_grid)

    param_grid = check_grid_params(clf, grid_params=param_grid)

    # Format the data XY & Z & target DataFrames and remove raw_classification from some LAS files.
    data, xy_coord, z_height, target = format_dataset(raw_data,
                                                      mode=mod,
                                                      raw_classif='lassif')

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    data_trans = scale_dataset(data, method='Standard')

    # Create samples for training and testing
    X_train_val, X_test, y_train_val, y_test = split_dataset(data_trans.values, target.values,
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
        model, grid_results = training_gridsearch(clf, X_train_val, y_train_val,
                                                  grid_params=param_grid,
                                                  scoring=args.scoring)

        grid_results_file = str(folder_path + '/' + str(algo) + '_grd_srch_' + create_time + '.csv')
        grid_results.to_csv(grid_results_file, index=True)

    else:
        model, cv_results = training_nogridsearch(clf, X_train_val, y_train_val,
                                                  scoring=args.scoring)

    # Save model
    print("5. Score model with the test dataset: {0:.4f}".format(model.score(X_test, y_test)))
    model_file = str(folder_path + '/' + str(algo))
    model_filename = str(model_file + "_" + create_time + ".model")
    save_model(model, model_filename)

    # Save and give confusion matrix
    y_test_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_mat_name = str(model_file + "_cnf_mt_" + create_time + ".csv")
    save_conf_mat(conf_mat, conf_mat_name)

    # Save and give classification report
    report_class = classification_report(y_test, y_test_pred)
    f = open(str(model_file + '_classif_report_' + create_time + '.csv'), "w")
    f.write(report_class)
    f.close()
    print("\n{}".format(report_class))

    # Save classifaction result as point cloud file with all data
    print("6. Make predictions for the entire dataset...", end='')
    y_pred = model.predict(data_trans.values)
    classif_filename = str(model_file + '_classification_' + create_time + '.csv')
    save_classification(y_pred, classif_filename,
                        xy_fields=xy_coord,
                        z_field=z_height,
                        data_fields=data,
                        target_field=target)
    print(" Done and save.")
    print("   Overall accuracy with entire dataset: "
          "{}".format(model.score(data_trans.values, target.values)))

else:
    mod = 'predict'

    # Format the data XY & Z as DataFrames and remove raw_classification from some LAS files.
    data, xy_coord, z_height, target = format_dataset(raw_data,
                                                      mode=mod,
                                                      raw_classif='lassif')

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    data_trans = scale_dataset(data, method='Standard')

    # Load the trained model
    model = load_model(args.model_to_import)
