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

from predict import *
from training import *

# -------------------------
# ---- ARGUMENT_PARSER ----
# -------------------------

# Create global parser
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

# Add subparsers
subparsers = parser.add_subparsers(help="cLASpy_T modes:\n\n", metavar='')

# Create sub-command for training
parser_train = subparsers.add_parser('train', help="training mode",
                                     description=textwrap.dedent('''\
                                     -------------------------------------------------------------------------------
                                                             cLASpy_T
                                                        train sub-command
                                                     ------------------------
                                     'train' allows to perform training according the selected supervised algorithm,
                                     among Random Forest ('rf'), Gradient Boosting ('gb') and Neural Network ('ann').
                                 
                                     For training, input_data file must contain:
                                        --> target field named 'target'
                                        --> data fields
                                     
                                     For CSV files:
                                        If X, Y and/or Z fields are present, they are excluded.
                                        To use them, rename them!
                                    
                                     For LAS files:
                                        All standard dimensions like 'intensity' or 'scan_angle_rank' are discarded.
                                        To use them, rename them!
                                    
                                     -------------------------------------------------------------------------------
                                         
                                     '''),
                                     formatter_class=argparse.RawTextHelpFormatter)

parser_train.add_argument("-a", "--algo",
                          help="set the algorithm:  ['rf', 'gb', 'ann']\n"
                               "    'rf': RandomForestClassifier\n"
                               "    'gb': GradientBoostingClassifier\n"
                               "    'ann': MLPClassifier\n\n",
                          type=str, choices=['rf', 'gb', 'ann'], metavar='')

parser_train.add_argument("-c", "--config",
                          help="give the configuration file with all parameters\n"
                               "    and selected scalar fields.\n"
                               "    [WINDOWS]: 'X:/path/to/the/config.json'\n"
                               "    [UNIX]: '/path/to/the/config.json'\n\n",
                          type=str, metavar='')

parser_train.add_argument("-i", "--input_data",
                          help="set the input data file:\n"
                               "    [WINDOWS]: 'X:/path/to/the/input_data.file'\n"
                               "    [UNIX]: '/path/to/the/input_data.file'\n\n",
                          type=str, metavar='')

parser_train.add_argument("-o", "--output",
                          help="set the output folder to save all result files:\n"
                               "    [WINDOWS]: 'X:/path/to/the/output/folder'\n"
                               "    [UNIX]: '/path/to/the/output/folder'\n"
                               "    Default: '/path/to/the/input_data'\n\n",
                          type=str, metavar='')

parser_train.add_argument("-f", "--features",
                          help="select the features to used to train the model.\n"
                               "    Give a list of feature names. Whitespaces"
                               "    will be replaced by underscore '_'."
                               "Example: f=['Anisotropy_5m', 'R', 'G', 'B', ...]",
                          type=str, default=None, metavar='')

parser_train.add_argument("-g", "--grid_search",
                          help="perform the training with GridSearchCV.\n\n",
                          action="store_true")

parser_train.add_argument("-k", "--param_grid",
                          help="set the parameters to pass to the GridSearch as list\n"
                               "    in dictionary. NO WHITESPACES!\n"
                               "    If empty, GridSearchCV uses presets.\n"
                               "    Wrong parameters will be ignored.\n\n"
                               "Example:\n"
                               "    -k=\"{'n_estimators':[50,100,500],'loss':['deviance',\n"
                               "    'exponential'],'hidden_layer_sizes':[[100,100],[50,100,50]]}\"\n\n",
                          type=str, metavar='')

parser_train.add_argument("-n", "--n_jobs",
                          help="set the number of CPU used, '-1' means all CPU available.\n"
                               "    Default: '-n=-1'\n\n",
                          type=int, metavar='', default=-1)

parser_train.add_argument("-p", "--parameters",
                          help="set the parameters to pass to the classifier for training,\n"
                               "    as dictionary. NO WHITESPACES!\n\n"
                               "Example:\n"
                               "    -p=\"{'n_estimators':50,'max_depth':5,'max_iter':500}\"\n\n",
                          type=str, metavar='')

parser_train.add_argument("--pca",
                          help="set the Principal Component Analysis and the number of\n"
                               "    principal components. If it enabled, default\n"
                               "    number of principal components is 8 ('--pca=8')\n\n",
                          type=int, metavar='')

parser_train.add_argument("--png_features",
                          help="export the feature importance from RandomForest and\n"
                               "    GradientBoosting algorithms as a PNG image.\n\n",
                          action="store_true")

parser_train.add_argument("--random_state",
                          help="set the random_state for data split and\n"
                               "    the GridSearchCV or the cross-validation.\n\n",
                          type=int, default=0, metavar='')

parser_train.add_argument("-s", "--samples",
                          help="set the number of samples for large dataset.\n"
                               "    (float number in million points)\n"
                               "    samples = train set + test set\n\n",
                          type=float, metavar='')

parser_train.add_argument("--scaler",
                          help="set method to scale the data before training.\n"
                               "    ['Standard', 'Robust', 'MinMax']\n"
                               "    See the preprocessing documentation of scikit-learn.\n"
                               "    Default: '--scaler=Standard'\n\n",
                          type=str, choices=['Standard', 'Robust', 'MinMax'],
                          default='Standard', metavar='')

parser_train.add_argument("--scoring",
                          help="set scorer for GridSearchCV or cross_val_score:\n"
                               "    ['accuracy','balanced_accuracy','precision','recall',...]\n"
                               "    See the scikit-learn documentation.\n"
                               "    Default: '--scoring=accuracy'\n\n",
                          type=str, default='accuracy', metavar="")

parser_train.add_argument("--test_r",
                          help="set the test ratio as float [0.0 - 1.0] to split into\n"
                               "    train and test data.\n"
                               "    If train_ratio + test_ratio > 1:\n"
                               "        test_ratio = 1 - train_ratio\n"
                               "    Default: '--test_r=0.5'\n\n",
                          type=float, default=0.5, metavar="")

parser_train.add_argument("--train_r",
                          help="set the train ratio as float [0.0 - 1.0] to split into\n"
                               "    train and test data.\n"
                               "    If train_ratio + test_ratio > 1:\n"
                               "        test_ratio = 1 - train_ratio\n"
                               "    Default: '--train_r=0.5'\n\n",
                          type=float, default=0.5, metavar="")

parser_train.set_defaults(func=train)  # Use training function

# Create sub-command for predictions
parser_predict = subparsers.add_parser('predict', help="prediction mode",
                                       formatter_class=argparse.RawTextHelpFormatter)

parser_predict.add_argument("-c", "--config",
                            help="give the configuration file with all parameters\n"
                                 "    and selected scalar fields.\n"
                                 "    [WINDOWS]: 'X:/path/to/the/config.json'\n"
                                 "    [UNIX]: '/path/to/the/config.json'\n\n",
                            type=str, metavar='')

parser_predict.add_argument("-i", "--input_data",
                            help="set the input data file:\n"
                                 "    [WINDOWS]: 'X:/path/to/the/input_data.file'\n"
                                 "    [UNIX]: '/path/to/the/input_data.file'\n\n",
                            type=str, metavar='')

parser_predict.add_argument("-o", "--output",
                            help="set the output folder to save all result files:\n"
                                 "    [WINDOWS]: 'X:/path/to/the/output/folder'\n"
                                 "    [UNIX]: '/path/to/the/output/folder'\n"
                                 "    Default: '/path/to/the/input_data/'\n\n",
                            type=str, metavar='')

parser_predict.add_argument("-m", "--model",
                            help="import the model file to make predictions:\n"
                                 "    '/path/to/the/training/file.model'\n\n",
                            type=str, metavar='')

parser_predict.add_argument("-s", "--samples",
                            help="set the number of samples for large dataset.\n"
                                 "    (float number in million points)\n"
                                 "    samples = train set + test set\n\n",
                            type=float, metavar='')

parser_predict.add_argument("-n", "--n_jobs",
                            help="set the number of CPU used, '-1' means all CPU available.\n"
                                 "    Default: '-n=-1'\n\n",
                            type=int, metavar='', default=-1)

parser_predict.set_defaults(func=predict)  # Use predict function

# Create sub-command for segmentation
parser_segment = subparsers.add_parser('segment', help="segmentation mode",
                                       formatter_class=argparse.RawTextHelpFormatter)


parser_segment.add_argument("-c", "--config",
                            help="give the configuration file with all parameters\n"
                                 "    and selected scalar fields.\n"
                                 "    [WINDOWS]: 'X:/path/to/the/config.json'\n"
                                 "    [UNIX]: '/path/to/the/config.json'\n\n",
                            type=str, metavar='')

parser_segment.add_argument("-i", "--input_data",
                            help="set the input data file:\n"
                                 "    [WINDOWS]: 'X:/path/to/the/input_data.file'\n"
                                 "    [UNIX]: '/path/to/the/input_data.file'\n\n",
                            type=str, metavar='')

parser_segment.add_argument("-o", "--output",
                            help="set the output folder to save all result files:\n"
                                 "    [WINDOWS]: 'X:/path/to/the/output/folder'\n"
                                 "    [UNIX]: '/path/to/the/output/folder'\n"
                                 "    Default: '/path/to/the/input_data/'\n\n",
                            type=str, metavar='')

parser_segment.add_argument("-f", "--features",
                            help="select the features to used to train the model.\n"
                                 "    Give a list of feature names. Whitespaces"
                                 "    will be replaced by underscore '_'."
                                 "Example: f=['Anisotropy_5m', 'R', 'G', 'B', ...]",
                            type=str, default=None, metavar='')

parser_segment.add_argument("-n", "--n_jobs",
                            help="set the number of CPU used, '-1' means all CPU available.\n"
                                 "    Default: '-n=-1'\n\n",
                            type=int, metavar='', default=-1)

parser_segment.add_argument("-p", "--parameters",
                            help="set the parameters to pass to the classifier for training,\n"
                                 "    as dictionary. NO WHITESPACES!\n\n"
                                 "Example:\n"
                                 "    -p={'n_estimators':50,'max_depth':5,'max_iter':500}\n\n",
                            type=str, metavar='')

parser_segment.add_argument("--pca",
                            help="set the Principal Component Analysis and the number of\n"
                                 "    principal components. If it enabled, default\n"
                                 "    number of principal components is 8 ('--pca=8')\n\n",
                            type=int, metavar='')

parser_segment.set_defaults(func=segment)  # Use segment function

# parse the args and call whatever function was selected
args = parser.parse_args()
args.func(args)
