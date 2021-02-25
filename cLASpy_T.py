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
#               'cLASpy_T.py' from cLASpy_T program                   #
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
import textwrap

import yaml
import json

from common import *
from predict import *
from training import *
from sklearn.metrics import confusion_matrix, classification_report


# -------------------------
# ---- ARGUMENT_PARSER ----
# -------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=textwrap.dedent('''\
                                 -------------------------------------------------------------------------------
                                                             cLASpy_T
                                                     ------------------------
                                 This program classifies 3D point clouds using machine learning algorithms based
                                 on the Scikit-Learn library. Three supervised classifiers are available
                                 (RandomForestClassifier, GradientBoostingClassifier and MLPClassifier)
                                 and one unsupervised clustering algorithm (KMeans).
                                 The input data must be CSV or LAS file.
                                 
                                 For training, input_data file must contain:
                                    --> target field named 'target'
                                    --> data fields
                                 
                                 For prediction, input_data file must contain:
                                    --> data fields
                                 
                                 For CSV files:
                                    If X, Y and/or Z fields are present, they are excluded.
                                    To use them, rename them!
                                
                                 For LAS files:
                                    All standard dimensions like 'intensity' or 'scan_angle_rank' are discarded.
                                    To use them, rename them!
                                    
                                 -------------------------------------------------------------------------------
                                 '''))

parser.add_argument("mode",
                    help="set the mode for the algorithm, among training, prediction\n"
                         "    and segmentation.",
                    type=str, choices=['train', 'pred', 'seg'])

parser.add_argument("-a", "--algo",
                    help="set the algorithm:  ['rf', 'gb', 'ann', 'kmeans']\n"
                         "    'rf': RandomForestClassifier\n"
                         "    'gb': GradientBoostingClassifier\n"
                         "    'ann': MLPClassifier\n"
                         "    'kmeans': KMeans\n\n",
                    type=str, choices=['rf', 'gb', 'ann', 'kmeans'], metavar='')

parser.add_argument("-c", "--config",
                    help="give the configuration file with all parameters\n"
                         "    and selected scalar fields.\n"
                         "    [WINDOWS]: 'X:/path/to/the/config.json'\n"
                         "    [UNIX]: '/path/to/the/config.json'\n\n",
                    type=str, metavar='')

parser.add_argument("-i", "--input_data",
                    help="set the input data file:\n"
                         "    [WINDOWS]: 'X:/path/to/the/input_data.file'\n"
                         "    [UNIX]: '/path/to/the/input_data.file'\n\n",
                    type=str, metavar='')

parser.add_argument("-o", "--output",
                    help="set the output folder to save all result files:\n"
                         "    [WINDOWS]: 'X:/path/to/the/output/folder'\n"
                         "    [UNIX]: '/path/to/the/output/folder'\n"
                         "    Default: '/path/to/the/input_data/'\n\n",
                    type=str, metavar='')

parser.add_argument("-g", "--grid_search",
                    help="perform the training with GridSearchCV.\n\n",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help="set the parameters to pass to the GridSearch as list\n"
                         "    in dictionary. NO WHITESPACES!\n"
                         "    If empty, GridSearchCV uses presets.\n"
                         "    Wrong parameters will be ignored.\n\n"
                         "Example:\n"
                         "    -k=\"{'n_estimators':[50,100,500],'loss':['deviance',\n"
                         "    'exponential'],'hidden_layer_sizes':[[100,100],[50,100,50]]}\"\n\n",
                    type=str, metavar='')

parser.add_argument("-m", "--model",
                    help="import the model file to make predictions:\n"
                         "    '/path/to/the/training/file.model'\n\n",
                    type=str, metavar='')

parser.add_argument("-n", "--n_jobs",
                    help="set the number of CPU used, '-1' means all CPU available.\n"
                         "    Default: '-n=-1'\n\n",
                    type=int, metavar='', default=-1)

parser.add_argument("-p", "--parameters",
                    help="set the parameters to pass to the classifier for training,\n"
                         "    as dictionary. NO WHITESPACES!\n\n"
                         "Example:\n"
                         "    -p=\"{'n_estimators':50,'max_depth':5,'max_iter':500}\"\n\n",
                    type=str, metavar='')

parser.add_argument("--pca",
                    help="set the Principal Component Analysis and the number of\n"
                         "    principal components.\n"
                         "    Default: '--pca=8'\n\n",
                    type=int, metavar='')

parser.add_argument("--png_features",
                    help="export the feature importance from RandomForest and\n"
                         "    GradientBoosting algorithms as a PNG image.\n\n",
                    action="store_true")

parser.add_argument("-s", "--samples",
                    help="set the number of samples for large dataset.\n"
                         "    (float number in million points)\n"
                         "    samples = train set + test set\n\n",
                    type=float, metavar='')

parser.add_argument("--scaler",
                    help="set method to scale the data before training.\n"
                         "    ['Standard', 'Robust', 'MinMax']\n"
                         "    See the preprocessing documentation of scikit-learn.\n"
                         "    Default: '--scaler=Standard'\n\n",
                    type=str, choices=['Standard', 'Robust', 'MinMax'],
                    default='Standard', metavar='')

parser.add_argument("--scoring",
                    help="set scorer for GridSearchCV or cross_val_score:\n"
                         "    ['accuracy','balanced_accuracy','precision','recall',...]\n"
                         "    See the scikit-learn documentation.\n"
                         "    Default: '--scoring=accuracy'\n\n",
                    type=str, default='accuracy', metavar="")

parser.add_argument("--test_r",
                    help="set the test ratio as float [0.0 - 1.0] to split into\n"
                         "    train and test data.\n"
                         "    If train_ratio + test_ratio > 1:\n"
                         "        test_ratio = 1 - train_ratio\n"
                         "    Default: '--test_r=0.5'\n\n",
                    type=float, default=0.5, metavar="")

parser.add_argument("--train_r",
                    help="set the train ratio as float [0.0 - 1.0] to split into\n"
                         "    train and test data.\n"
                         "    If train_ratio + test_ratio > 1:\n"
                         "        test_ratio = 1 - train_ratio\n"
                         "    Default: '--train_r=0.5'\n\n",
                    type=float, default=0.5, metavar="")

args = parser.parse_args()

# -------------------------
# --------- MAIN ----------
# -------------------------

# Set non-common parameters as None
train_size = None
test_size = None
pca = None
pca_compo = None
grid_results = None
cv_results = None
model_to_load = None
conf_mat = None
test_report = None
parameters = None


# INTRODUCTION
data_path, folder_path, start_time = introduction(algo, args.data_file)  # Start prompt
timestamp = start_time.strftime("%m%d_%H%M")  # Timestamp for file creation MonthDay_HourMinute

# FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
print("\n1. Formatting data as pandas.Dataframe...")
data, target = format_dataset(data_path, mode=mode)

# Get the number of points
nbr_pts = number_of_points(data.shape[0], sample_size=args.samples)
str_nbr_pts = format_nbr_pts(nbr_pts)  # Format nbr_pts as string for filename

# Set the report filename
report_filename = str(folder_path + '/' + mode[0:5] + '_' +
                      args.algorithm + str_nbr_pts + str(timestamp))

# Get the feature names
feature_names = data.columns.values.tolist()


elif mode == 'unsupervised':


else:  # Prediction mode

    # Get model, scaler and pca
    print("\n2. Loading model...")
    model_to_load = args.model_to_import  # Set variable for the report
    model, scaler, pca = load_model(model_to_load)

    # Apply scaler to data
    print("\n3. Scaling data...")
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=feature_names)

    # Apply pca to data if exists
    if pca:
        data_scaled = apply_pca(pca, data_scaled)
        pca_compo = np.array2string(pca.components_)

    # Predic target of input data
    print("\n4. Making predictions for entire dataset...")
    # y_pred = model.predict(data_scaled.values)
    y_pred = predict_with_proba(model, data_scaled.values)

    if target is not None:
        # Save confusion matrix
        print("\nCreating confusion matrix...")
        conf_mat = confusion_matrix(target.values, y_pred.transpose()[0])
        conf_mat = precision_recall(conf_mat)  # return Dataframe
        test_report = classification_report(target.values, y_pred.transpose()[0])  # Get classification report
        print("\n{}\n".format(test_report))

    # Save classifaction result as point cloud file with all data
    print("\n5. Saving classified point cloud:")
    predic_filename = report_filename
    print(predic_filename)
    save_predictions(y_pred, predic_filename, data_path)

# Create and save prediction report
print("\n6. Creating classification report:")
print(report_filename + '.txt')

# Get the model parameters to print in the report
applied_parameters = ["{}: {}".format(param, model.get_params()[param])
                      for param in model.get_params()]

# Compute elapsed time
spent_time = datetime.now() - start_time

# Write the entire report
write_report(report_filename,
             mode=mode,
             algo=algo,
             data_file=args.data_file,
             start_time=start_time,
             elapsed_time=spent_time,
             feat_names=feature_names,
             scaler=scaler,
             data_len=nbr_pts,
             train_len=train_size,
             test_len=test_size,
             applied_param=applied_parameters,
             pca_compo=pca_compo,
             model=model_to_load,
             grid_results=grid_results,
             cv_results=cv_results,
             conf_mat=conf_mat,
             score_report=test_report)

if mode == 'train':
    print("\nModel trained in {}".format(spent_time))
else:
    print("\nPredictions done in {}".format(spent_time))
