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

import argparse
import numpy as np
import os
import yaml
from datetime import datetime
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import confusion_matrix, classification_report

from common import *
from predict import *
from training import *

# -------------------------
# ---- ARGUMENT_PARSER ----
# -------------------------

parser = argparse.ArgumentParser(description="\nThis library based on Sci-kit Learn library performs machine learning\n"
                                             "to classify points of 3D point cloud. The input data must be in CSV.\n\n"
                                             "For training, csv_data_file must contain:\n"
                                             "    --> target field named 'target' AND data fields\n"
                                             "For prediction, csv_data_file must contain:\n"
                                             "    --> data fields\n\n"
                                             "If X, Y and/or Z fields are present, they are excluded.\n"
                                             "If a field_name is 'classif', 'raw_classification'...\n"
                                             "the field is discarded.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("algorithm",
                    help="The learning algorithm to use:\n"
                         "    'rf': RandomForestClassifier\n"
                         "    'gb': GradientBoostingClassifier\n"
                         "    'ann': MLPClassifier\n",
                    type=str, choices=['rf', 'gb', 'ann'])

parser.add_argument("csv_data_file",
                    help="The CSV file with needed data:\n"
                         "    [WINDOWS]: 'C:/path/to/the/data.file'\n"
                         "    [UNIX]: '/path/to/the/data.file'",
                    type=str, metavar="/path/to/file.csv")

parser.add_argument("-g", "--grid_search",
                    help="Perform the training with GridSearchCV",
                    action="store_true")

parser.add_argument("-i", "--importance",
                    help="Export feature importance from randomForest and gradientBoosting model as a PNG image file.",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help="Set the parameters to pass at the GridSearch as list(sep=',') in dict. NO SPACE\n"
                         "If empty, GridSearchCV uses presets.\n"
                         "Example: -k=\"{'n_estimators':[50,100,500],'loss':['deviance','exponential'],"
                         "'hidden_layer_sizes':[[100,100],[50,100,50]]}\"\n"
                         "Wrong parameters will be ignored\n",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-m", "--model_to_import",
                    help="The model file to import to make predictions:\n"
                         "    [WINDOWS]: 'C:/path/to/the/training/model.file'\n"
                         "    [UNIX]: '/path/to/the/training/model.file'",
                    type=str, metavar="[=\"/path/to.file\"]")

parser.add_argument("-n", "--n_jobs",
                    help="Set the number of CPU used, '-1' means all CPU available.",
                    type=int, metavar="[1,2,...,-1]", default=-1)

parser.add_argument("-p", "--parameters",
                    help="Set the parameters to pass at the classifier for training, as dict.\n"
                         "Example: -p=\"{'n_estimators':50,'max_depth':5,'max_iter':500}\"",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-s", "--samples",
                    help="Set the number of samples, in Million points, for large dataset.\n"
                         "If data length > samples:\n"
                         "then train + test length = samples",
                    type=float, metavar="[in Mpts]")

parser.add_argument("--scaler",
                    help="Set method to scale the data before training.\n"
                         "See the preprocessing documentation of scikit-learn.",
                    type=str, choices=['Standard', 'Robust', 'MinMax'], default='Standard')

parser.add_argument("--scoring",
                    help="Set scorer to GridSearchCV or cross_val_score according\n"
                         "to sckikit-learn documentation.",
                    type=str, default='accuracy',
                    metavar="[='accuracy','balanced_accuracy','average_precision',"
                            "'precision','recall',...]")

parser.add_argument("--test_ratio",
                    help="Set the test ratio as float [0.0-1.0] to split into train and test data.\n"
                         "If train_ratio + test_ratio > 1\n"
                         "then test_ratio = 1 - train_ratio",
                    type=float, default=0.5, metavar="[0.0-1.0]")

parser.add_argument("--train_ratio",
                    help="Set the train ratio as float number to split into train and test data.\n"
                         "If train_ratio + test_ratio > 1:\n"
                         "then test_ratio = 1 - train_ratio",
                    type=float, default=0.5, metavar="[0.0-1.0]")

args = parser.parse_args()

# -------------------------
# --------- MAIN ----------
# -------------------------

# Set the chosen learning classifier
if args.algorithm == 'rf':
    algo = 'RandomForestClassifier'
    classifier = set_random_forest(fit_params=parameters,
                                   n_jobs=args.n_jobs)
elif args.algorithm == 'gb':
    algo = 'GradientBoostingClassifier'
    classifier = set_gradient_boosting(fit_params=parameters)
elif args.algorithm == 'ann':
    algo = 'MLPClassifier'
    classifier = set_mlp_classifier(fit_params=parameters)
else:
    raise ValueError("No valid classifier!")

# Introduction
print("\n####### POINT CLOUD CLASSIFICATION #######\n"
      "Algorithm used: {}\n"
      "Path to CSV file: {}\n".format(algo, args.csv_data_file))

# Create a folder to store model, results, confusion matrix and grid results
print("Create a new folder to store the result files...", end='')
raw_data = args.csv_data_file
raw_data = '/'.join(raw_data.split('\\'))  # Change '\' in '/'
folder_path = '.'.join(raw_data.split('.')[:-1])  # remove extension so give folder path
try:
    os.mkdir(folder_path)  # Using file path to make new folder
    print(" Done.")
except (TypeError, FileExistsError):
    print(" Folder already exists.")

# Check parameters exists
parameters = None
if args.parameters:
    parameters = yaml.safe_load(args.parameters)

# Timestamp for created files
create_time = datetime.now()
timestamp = create_time.strftime("%m%d_%H%M")  # Timestamp for file creation MD_HM

# Set the mode as 'training' or 'prediction'
if args.model_to_import is None:
    mod = 'training'
else:
    mod = 'prediction'

# FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
print("\n1. Formatting data as pandas.Dataframe...", end='')
data, xy_coord, z_height, target = format_dataset(raw_data,
                                                  mode=mod,
                                                  raw_classif='lassif')

# Get the number of points
nbr_pts = nbr_pts(data_length=len(z_height), samples_size=args.samples)
str_nbr_pts = format_nbr_pts(nbr_pts)  # Format in string for filename

# Give the report filename
report_filename = str(folder_path + '/' + mod[0:6] + '_' +
                      args.algorithm + str_nbr_pts + str(timestamp))

# Get the feature names
feature_names = data.columns.values.tolist()

# TRAINING or PREDICTION
if mod == 'training':  # Training mode
    # Set useless parameter as None
    model_to_load = None

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\n2. Scaling data...", end='')
    scale_method = args.scaler
    data_trans = scale_dataset(data, method=scale_method)

    # Create samples for training and testing
    print("\n3. Splitting data in train and test sets...", end='')
    X_train_val, X_test, y_train_val, y_test = split_dataset(
        data_trans.values,
        target.values,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        samples=nbr_pts)

    # Get the train and test sizes
    train_size = len(y_train_val)
    test_size = len(y_test)

    # TYPE OF TRAINING
    if args.grid_search:  # Training with GridSearchCV
        cv_results = None  # So non cross validation results

        # Check param_grid exists
        if args.param_grid is not None:
            param_grid = yaml.safe_load(args.param_grid)
        else:
            param_grid = None
        param_grid = check_grid_params(classifier,
                                       grid_params=param_grid)

        model, grid_results = training_gridsearch(classifier,
                                                  X_train_val,
                                                  y_train_val,
                                                  grid_params=param_grid,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

    else:  # Training with Cross Validation
        grid_results = None  # So no GridSearchCV results
        model, cv_results = training_nogridsearch(classifier,
                                                  X_train_val,
                                                  y_train_val,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

    print("\n5. Score model with the test dataset: {0:.4f}".format(
        model.score(X_test, y_test)))

    # Save model
    model_filename = str(report_filename + '_' + args.scaler + '.model')
    save_model(model, model_filename)

    # Get the model parameters to print them in report
    applied_parameters = ["{}: {}".format(
        param, model.get_params()[param]) for param in model.get_params()]

    # Importance of each feature in RF and GB
    if args.grid_search or args.algorithm == 'ann' or args.algorithm == 'svm':
        args.importance = False  # Overwrite 'False' if '-i' option set with grid, ann or svm
    if args.importance:
        feature_filename = str(report_filename + '_feat_importance.png')
        save_feature_importance(model, feature_names, feature_filename)

    # Save confusion matrix
    print("\n6. Creating confusion matrix:")
    y_test_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_mat = precision_recall(conf_mat)  # return Dataframe
    print("\n{}".format(conf_mat))

    # Get classification report
    report_class = classification_report(y_test, y_test_pred)


else:  # Prediction mode
    # Set useless parameters as None
    train_size = None
    test_size = None
    grid_results = None
    cv_results = None

    # Get model and scaling parameter
    scale_method = '.'.join(args.model_to_import.split('.')[:-1]).split('_')[-1]

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    data_trans = scale_dataset(data, method=scale_method)

    # Load trained model
    model_to_load = args.model_to_import
    model = load_model(model_to_load)

    # Predic target of input data
    print("\n4. Making predictions for the entire dataset...")
    y_pred = model.predict(data_trans.values)

    # Get the model parameters to print them in report
    applied_parameters = ["{}: {}".format(
        param, model.get_params()[param]) for param in model.get_params()]

    if target is not None:
        # Save confusion matrix
        print("\n5. Creating confusion matrix:")
        conf_mat = confusion_matrix(target.values, y_pred)
        conf_mat = precision_recall(conf_mat)  # return Dataframe
        print("\n{}".format(conf_mat))

        # Save classification report
        print("\n6. Creating classification report:")
        report_class = classification_report(target.values, y_pred)
        print("\n{}\n".format(report_class))

    else:
        conf_mat = None
        report_class = None

    # Save classifaction result as point cloud file with all data
    print("\n7. Saving classified point cloud as CSV file:")
    predic_filename = str(report_filename + '.csv')
    save_predictions(y_pred,
                     predic_filename,
                     xy_fields=xy_coord,
                     z_field=z_height,
                     data_fields=data,
                     target_field=target)

# Create and save prediction report
print("\n8. Creating classification report:")
spent_time = datetime.now() - create_time

# Write the entire report
write_report(report_filename,
             mode=mod,
             algo=algo,
             data_file=args.csv_data_file,
             start_time=create_time,
             elapsed_time=spent_time,
             feat_names=feature_names,
             scale_method=scale_method,
             data_len=nbr_pts,
             train_len=train_size,
             test_len=test_size,
             applied_param=applied_parameters,
             model=model_to_load,
             grid_results=grid_results,
             cv_results=cv_results,
             conf_mat=conf_mat,
             score_report=report_class)

if mod == 'training':
    print("\n{}\n\nModel trained in {}".format(report_class, spent_time))
else:
    print("\nPredictions done in {}".format(spent_time))
