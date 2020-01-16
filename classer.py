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
                    type=str, choices=['rf','gb','svm','ann'])

parser.add_argument("csv_data_file",
                    help="the CSV file with needed data:\n"
                         "    [WINDOWS]: 'C:/path/to/the/data.file'\n"
                         "    [UNIX]: '/path/to/the/data.file'",
                    type=str, metavar="/path/to.file")

parser.add_argument("-g", "--grid_search",
                    help="perform the training with GridSearchCV",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help="pass the grid parameters as list(sep=',') in dict. NO SPACE\n"
                         "    If empty, GridSearchCV uses presets.\n"
                         "    Example: -k=\"{'n_estimators':[50,100,500],'loss':['deviance','exponential'],"
                         "    'hidden_layer_sizes':[[100,100],[50,100,50]]}\"\n"
                         "    Wrong pameters will be ignored\n",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-m", "--model_to_import",
                    help="the model file to import to make predictions:\n"
                         "    [WINDOWS]: 'C:/path/to/the/training/model.file'\n"
                         "    [UNIX]: '/path/to/the/training/model.file'",
                    type=str, metavar="[=\"/path/to.file\"]")

parser.add_argument("-n", "--n_jobs",
                    help="set the number of CPU used, '-1' means all CPU available.",
                    type=int, metavar="[1,2,...,-1]", default=-1)

parser.add_argument("-p", "--parameters",
                    help="the parameters to set up the classifier as dict.\n"
                         "    Example: -p=\"{'n_estimators':50,'max_depth':5,'max_iter':500}\"",
                    type=str, metavar="[=\"dict\"]")

parser.add_argument("-s", "--scaler",
                    help="method to scale the data before training.\n"
                         "    See the preprocessing documentation of scikit-learn.",
                    type=str, choices=['Standard','Robust','MinMax'], default='Standard')

parser.add_argument("--scoring",
                    help="set scorer to GridSearchCV or cross_val_score according\n"
                         "    to sckikit-learn documentation.",
                    type=str, default='accuracy',
                    metavar="[='accuracy','balanced_accuracy','average_precision',"
                            "'precision','recall',...]")

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

# Introduction
print("\n####### POINT CLOUD CLASSIFICATION #######\n"
      "Algorithm used: {}\n"
      "Path to CSV file: {}\n".format(args.algorithm, args.csv_data_file))

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
    clf = set_random_forest(fit_params=parameters, n_jobs=args.n_jobs)
elif algo == 'gb':
    clf = set_gradient_boosting(fit_params=parameters)
elif algo == 'svm':
    clf = set_linear_svc(fit_params=parameters)
elif algo == 'ann':
    clf = set_mlp_classifier(fit_params=parameters)
else:
    raise ValueError("Any valid classifier was selected !")

# Timestamp for files created
create_time = datetime.now().strftime("%y%m%d_%H%M%S")  # Timestamp for file creation

# Prefix of the model_filename
training_model_file = str(folder_path + '/training_' + str(algo) + '_' + str(create_time))
predict_model_file = str(folder_path + '/predict_' + str(algo) + '_' + str(create_time))

# Is it TRAINING or PREDICTIONS ?
if not args.model_to_import:
    mod = 'training'

    report_filename = str(training_model_file + '_report.txt')

    # Format the data XY & Z & target DataFrames and remove raw_classification from some LAS files.
    data, xy_coord, z_height, target = format_dataset(raw_data,
                                                      mode=mod,
                                                      raw_classif='lassif')

    # Get the feature names
    feature_names = data.columns.values.tolist()

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    data_trans = scale_dataset(data, method=args.scaler)

    # Create samples for training and testing
    X_train_val, X_test, y_train_val, y_test = split_dataset(data_trans.values, target.values,
                                                             train_ratio=args.train_ratio,
                                                             test_ratio=args.test_ratio,
                                                             threshold=args.threshold)

    # Create the report file
    with open(report_filename, 'w', encoding='utf-8') as report:
        report.write('Report of ' + str(algo) + ' training\nDatetime: ' + str(create_time) + '\n')
        report.write('\nFeatures:\n' + '\n'.join(feature_names) + '\n')
        report.write('\nScaling method: ' + str(args.scaler) + '\n')
        report.write('\nSize of train and test dataset:'
                     '\nTrain size: ' + str(len(y_train_val)) + ' pts'
                     '\nTest size: ' + str(len(y_test)) + ' pts\n')

    # kernel approximation for SVM
    if algo is 'svm':
        feature_map_nystroem = Nystroem(gamma=.2, n_components=100, random_state=0)
        X_train_val = feature_map_nystroem.fit_transform(X_train_val)
        X_test = feature_map_nystroem.fit_transform(X_test)

    # What type of training
    if args.grid_search:  # Training with GridSearchCV

        # Check param_grid exists
        param_grid = None
        if args.param_grid:
            param_grid = yaml.safe_load(args.param_grid)
        param_grid = check_grid_params(clf, grid_params=param_grid)

        model, grid_results = training_gridsearch(clf, X_train_val, y_train_val,
                                                  grid_params=param_grid,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

        with open(report_filename, 'a', encoding='utf-8') as report:
            report.write('\nResults of the GridSearchCV:\n')
            report.write(grid_results.to_string(index=False))
            report.write('\n')

    else:  # Training with Cross Validation
        model, cv_results = training_nogridsearch(clf, X_train_val, y_train_val,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

        with open(report_filename, 'a', encoding='utf-8') as report:
            report.write('Results of the Cross-Validation:\n')
            report.write(pd.DataFrame(cv_results).to_string(index=False, header=False))
            report.write('\n')
    print("\n5. Score model with the test dataset: {0:.4f}".format(model.score(X_test, y_test)))

    # Save model
    model_filename = str(training_model_file + '_' + args.scaler + '.model')
    save_model(model, model_filename)

    # Importance of each feature in RF and GB
    # if not args.grid_search:
    #    if algo == 'rf' or algo == 'gb':
    #        feature_filename = str(training_model_file + '_feat_importance.png')
    #        save_feature_importance(model, feature_names, feature_filename)

    # Save confusion matrix
    print("\n6. Creating confusion matrix:")
    y_test_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_mat = precision_recall(conf_mat)  # return Dataframe
    with open(report_filename, 'a', encoding='utf-8') as report:
        report.write('\nConfusion Matrix:\n')
        report.write(conf_mat.to_string())
        report.write('\n')
    print("\n{}".format(conf_mat))

    # Save classification report
    print("\n7. Creating classification report:")
    report_class = classification_report(y_test, y_test_pred)
    with open(report_filename, 'a', encoding='utf-8') as report:
        report.write('\nClassification Report:\n')
        report.write(report_class)
    print("\n{}\nFinished!".format(report_class))

else:
    mod = 'predict'

    report_filename = str(predict_model_file + '_report.txt')

    # Format data XY & Z as DataFrames and remove raw_classification from some LAS files.
    data, xy_coord, z_height, target = format_dataset(raw_data,
                                                      mode=mod,
                                                      raw_classif='lassif')

    # Get the feature names
    feature_names = data.columns.values.tolist()

    # Get model and scaling parameters
    scaler_method = '.'.join(args.model_to_import.split('.')[:-1]).split('_')[-1]

    # Scale the dataset 'Standard', 'Robust', 'MinMaxScaler'
    data_trans = scale_dataset(data, method=scaler_method)

    # Create report file for predictions
    with open(report_filename, 'w', encoding='utf-8') as report:
        report.write('Report of ' + str(algo) + ' predictions\nDatetime: ' + str(create_time) + '\n')
        report.write('\nFeatures:\n' + '\n'.join(feature_names))
        report.write('\nScaling method: ' + str(scaler_method))
        report.write('\nNumber of points to classify: ' + str(len(z_height)) + ' pts\n')

    # Load trained model
    model = load_model(args.model_to_import)

    # Predic target of input data
    print("4. Make predictions for the entire dataset...")
    y_pred = model.predict(data_trans.values)

    if target is not None:
        # Save confusion matrix
        print("\n5. Creating confusion matrix:")
        conf_mat = confusion_matrix(target.values, y_pred)
        conf_mat = precision_recall(conf_mat)  # return Dataframe
        with open(report_filename, 'a', encoding='utf-8') as report:
            report.write('\nConfusion Matrix:\n')
            report.write(conf_mat.to_string())
            report.write('\n')
        print("\n{}".format(conf_mat))

        # Save classification report
        print("\n6. Creating classification report:")
        report_class = classification_report(target.values, y_pred)
        with open(report_filename, 'a', encoding='utf-8') as report:
            report.write('\nClassification Report:\n')
            report.write(report_class)
        print("\n{}".format(report_class))

    # Save classifaction result as point cloud file with all data
    print("\n7. Save classified point cloud as CSV file:")
    predic_filename = str(predict_model_file + '_predictions.csv')
    save_predictions(y_pred, predic_filename,
                     xy_fields=xy_coord,
                     z_field=z_height,
                     data_fields=data,
                     target_field=target)
    print("\nPredictions done and saved.")
