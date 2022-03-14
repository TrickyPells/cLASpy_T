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

import os
import yaml
import joblib
import json
import argparse
import textwrap
from cLASpy_Classes import ClaspyTrainer, ClaspyPredicter, ClaspySegmenter


# -------------------------
# ------ VARIABLES --------
# -------------------------

# Define version of cLASpy_T
cLASpy_T_version = '0.2.0'  # 0.2.0 : Version of cLASpy_T with classes

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
                                 on scikit-learn. Three supervised classifiers (RandomForestClassifier, Gradient
                                 BoostingClassifier and MLPClassifier) and one clustering algorithm (KMeans) are
                                 available.
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

# parser_train.add_argument("-h", "--help", action='help', default=argparse.SUPPRESS,
#                           help="Show this nice message and exit")

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
                          type=str, default=None, metavar='')

parser_train.add_argument("-f", "--features",
                          help="select the features used to train the model.\n"
                               "    Give a list of feature names.\n"
                               "    Whitespaces will be replaced by underscore '_'.\n"
                               "    Example:\n"
                               "    -f=['Anisotropy_5m', 'R', 'G', 'B', ...]\n\n",
                          type=str, default=None, metavar='')

parser_train.add_argument("-g", "--grid_search",
                          help="perform the training with GridSearchCV.\n\n",
                          action="store_true")

parser_train.add_argument("-k", "--param_grid",
                          help="set the parameters to pass to the GridSearch\n"
                               "    as list in dictionary. NO WHITESPACES!\n"
                               "    If empty, GridSearchCV uses presets.\n"
                               "    Wrong parameters will be ignored.\n"
                               "    Example:\n"
                               "    -k=\"{'n_estimators':[50,100,500],'loss':['deviance','exponential'],\n"
                               "    'hidden_layer_sizes':[[100,100],[50,100,50]]}\"\n\n",
                          type=str, metavar='')

parser_train.add_argument("-n", "--n_jobs",
                          help="set the number of CPU used, '-1' means all available CPU.\n"
                               "    Default: '-n=-1'\n\n",
                          type=int, metavar='', default=-1)

parser_train.add_argument("-p", "--parameters",
                          help="set the parameters to pass to the classifier for training,\n"
                               "    as dictionary. NO WHITESPACES!\n"
                               "    Example:\n"
                               "    -p=\"{'n_estimators':50,'max_depth':5,'max_iter':500}\"\n\n",
                          type=str, metavar='')

parser_train.add_argument("--pca",
                          help="set the Principal Component Analysis and\n"
                               "     the number of principal components.\n\n",
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

parser_train.add_argument("--train_r",
                          help="set the train ratio as float [0.0 - 1.0]\n"
                               "     to split into train and test data.\n"
                               "    Default: '--train_r=0.5'\n\n",
                          type=float, default=0.5, metavar="")

parser_train.set_defaults(func='train')  # Use training function

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
                            type=str, default=None, metavar='')

parser_predict.add_argument("-m", "--model",
                            help="import the model file to make predictions:\n"
                                 "    '/path/to/the/training/file.model'\n\n",
                            type=str, metavar='')

# parser_predict.add_argument("-s", "--samples",
#                             help="set the number of samples for large dataset.\n"
#                                  "    (float number in million points)\n"
#                                  "    samples = train set + test set\n\n",
#                             type=float, metavar='')

parser_predict.set_defaults(func='predict')  # Use predict function

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
                            type=str, default=None, metavar='')

parser_segment.add_argument("-f", "--features",
                            help="select the features to used to train the model.\n"
                                 "    Give a list of feature names. Whitespaces\n"
                                 "    will be replaced by underscore '_'.\n"
                                 "    Example:\n"
                                 "    -f=['Anisotropy_5m', 'R', 'G', 'B', ...]\n\n",
                            type=str, default=None, metavar='')

parser_segment.add_argument("-p", "--parameters",
                            help="set the parameters to pass to the classifier\n"
                                 "     for segmentation, as dictionary. NO WHITESPACES!\n\n"
                                 "    Example:\n"
                                 "    -p={'n_clusters':8,'init':'k-means++','max_iter':500}\n\n",
                            type=str, metavar='')

# parser_segment.add_argument("--pca",
#                             help="set the Principal Component Analysis and the number of\n"
#                                  "    principal components. If it enabled, default\n"
#                                  "    number of principal components is 8 ('--pca=8')\n\n",
#                             type=int, metavar='')

# parser_segment.add_argument("-s", "--samples",
#                             help="set the number of samples for large dataset.\n"
#                                  "    (float number in million points)\n\n",
#                             type=float, metavar='')

parser_segment.set_defaults(func='segment')  # Use segment function

# parse the args and call whatever function was selected
args = parser.parse_args()


def shortname_algo(algorithm):
    """
    Give the short name of selected algorithm
    :param algorithm: the selected algo.
    :return: the fullname of the algorithm
    """
    # Get short name of algorithm
    if algorithm == 'RandomForestClassifier':
        algo = 'rf'
    elif algorithm == 'GradientBoostingClassifier':
        algo = 'gb'
    elif algorithm == 'MLPClassifier':
        algo = 'ann'
    elif algorithm == 'KMeans':
        algo = 'kmeans'
    else:
        raise ValueError("Choose a machine learning algorithm ('--algo')!")

    return algo


def arguments_from_config():
    """
    Update the arguments from the config file given in args.config.
    """
    # Open the config file
    args.config = os.path.normpath(args.config)
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Get the version and mode of config_file
    # version = config['version'].split('_')[0]
    mode = config['version'].split('_')[-1]

    # Global arguments (all modes)
    if args.input_data is None:
        args.input_data = os.path.normpath(config['input_file'])
    if args.output is None:
        args.output = os.path.normpath(config['output_folder'])

    # Arguments for training or segment mode
    if mode == 'train' or mode == 'segme':
        # if argument not set with argparser take value from config file
        if args.algo is None:
            args.algo = shortname_algo(config['algorithm'])
        if args.features is None:
            args.features = config['feature_names']
        if args.scaler is None:
            args.scaler = config['scaler']
        if args.pca is None:
            # Check if PCA is present
            try:
                args.pca = config['pca']
            except KeyError:
                pass

    # Arguments for training mode
    if mode == 'train':
        # if argument not set with argparser take value from config file
        if args.png_features is False:
            args.png_features = config['png_features']
        if args.samples is None:
            args.samples = config['samples']
        if args.scoring is None:
            args.scoring = config['scorer']
        if args.train_r is None:
            args.train_r = config['training_ratio']
        if args.random_state == 0:
            args.random_state = config['random_state']
        if args.n_jobs == -1:
            args.n_jobs = config['n_jobs_cv']
        # grid_search flag: always take value from config file (to avoid incompatibility)
        args.grid_search = config['grid_search']
        # Set grid parameters or classifier parameters (GridSearchCV or not)
        if args.grid_search:
            args.param_grid = config['param_grid']
        else:
            args.parameters = config['parameters']

    # Arguments for predict mode
    if mode == 'predi':
        args.model = config['model']

    # Arguments for segment mode
    if mode == 'segme':
        # if argument not set with argparser take value from config file
        if args.parameters is None:
            args.parameters = config['parameters']


def train(arguments):
    """
    Perform training according the passed arguments.
    :param arguments: parser arguments
    """
    # cLASpy_T starts
    print("\n# # # # # # # # # #  cLASpy_T  # # # # # # # # # # # #"
          "\n - - - - - - - -    TRAIN MODE    - - - - - - - - - -"
          "\n * * * *    Point Cloud Classification    * * * * * *\n")

    # Config file exists ?
    if arguments.config:
        arguments_from_config()  # Get the arguments from the config file

    trainer = ClaspyTrainer(input_data=arguments.input_data,
                            output_data=arguments.output,
                            algo=arguments.algo,
                            algorithm=None,
                            parameters=arguments.parameters,
                            features=arguments.features,
                            grid_search=arguments.grid_search,
                            grid_param=arguments.param_grid,
                            pca=arguments.pca,
                            n_jobs=arguments.n_jobs,
                            random_state=arguments.random_state,
                            samples=arguments.samples,
                            scaler=arguments.scaler,
                            scoring=arguments.scoring,
                            train_ratio=arguments.train_r,
                            png_features=arguments.png_features)

    # Set the classifier according parameters
    trainer.set_classifier()

    # Introduction
    intro = trainer.introduction(verbose=True)
    print(intro)

    # Format dataset
    print("\nStep 1/7: Formatting data as pandas.DataFrame...")
    step1 = trainer.format_dataset(verbose=True)
    print(step1)

    # Split data into training and testing sets
    print("\nStep 2/7: Splitting data in train and test sets...")
    step2 = trainer.split_dataset(verbose=True)
    print(step2)

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\nStep 3/7: Scaling data...")
    step3 = trainer.set_scaler_pca(verbose=True)
    print(step3)

    # Train model
    if trainer.grid_search:  # Training with GridSearchCV
        print('\nStep 4/7: Training model with GridSearchCV...\n')
    else:  # Training with Cross Validation
        print("\nStep 4/7: Training model with cross validation...\n")

    step4 = trainer.train_model(verbose=True)  # Perform both training
    print(step4)

    # Create confusion matrix
    print("\nStep 5/7: Creating confusion matrix...")
    step5 = trainer.confusion_matrix(verbose=True)
    print(step5)

    # Save algorithm, model, scaler, pca and feature_names
    print("\nStep 6/7: Saving model and scaler in file:")
    step6 = trainer.save_model(verbose=True)
    print(step6)

    # Create and save prediction report
    print("\nStep 7/7: Creating classification report:")
    print(trainer.report_filename + '.txt')
    step7 = trainer.classification_report(verbose=True)
    print(step7)


def predict(arguments):
    """
    Perform prediction according the passed arguments.
    :param arguments: parser arguments
    """
    # cLASpy_T starts
    print("\n# # # # # # # # # #  cLASpy_T  # # # # # # # # # # # #"
          "\n - - - - - - - -   PREDICT MODE   - - - - - - - - - -"
          "\n * * * *    Point Cloud Classification    * * * * * *\n")

    # Config file exists ?
    if arguments.config:
        arguments_from_config()  # Get the arguments from the config file

    predicter = ClaspyPredicter(model=arguments.model,
                                input_data=arguments.input_data,
                                output_data=arguments.output)

    # Load model
    print("\nStep 1/6: Loading model...")
    step1 = predicter.load_model(verbose=True)
    print(step1)

    # Introduction
    intro = predicter.introduction(verbose=True)
    print(intro)

    # Format dataset
    print("\nStep 2/6: Formatting data as pandas.DataFrame...")
    step2 = predicter.format_dataset(verbose=True)
    print(step2)

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\nStep 3/6: Scaling data...")
    step3 = predicter.scale_dataset(verbose=True)
    print(step3)

    # Predic target of input data
    print("\nStep 4/6: Making predictions for entire dataset...")
    step4 = predicter.predict(verbose=True)
    print(step4)

    # Save classification result as point cloud file with all data
    print("\nStep 5/6: Saving classified point cloud:")
    step5 = predicter.save_predictions(verbose=True)
    print(step5)

    # Create and save prediction report
    print("\nStep 6/6: Creating classification report:")
    print(predicter.report_filename + '.txt')
    step6 = predicter.classification_report(verbose=True)
    print(step6)


def segment(arguments):
    """
    Perform segmentation according the passed arguments.
    :param arguments: parser arguments
    """
    # cLASpy_T starts
    print("\n# # # # # # # # # #  cLASpy_T  # # # # # # # # # # # #"
          "\n - - - - - - - -   SEGMENT MODE   - - - - - - - - - -"
          "\n * * * *    Point Cloud Classification    * * * * * *\n")

    # Config file exists ?
    if arguments.config:
        arguments_from_config()  # Get the arguments from the config file

    segmenter = ClaspySegmenter(input_data=arguments.input_data,
                                output_data=arguments.output,
                                parameters=arguments.parameters,
                                features=arguments.features)

    # Set the classifier according parameters
    segmenter.set_classifier()

    # Introduction
    intro = segmenter.introduction(verbose=True)
    print(intro)

    # Format dataset
    print("\nStep 1/5: Formatting data as pandas.DataFrame...")
    step1 = segmenter.format_dataset(verbose=True)
    print(step1)

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\nStep 2/5: Scaling data...")
    step2 = segmenter.set_scaler_pca(verbose=True)
    print(step2)

    # Split data into training and testing sets
    print("\nStep 3/5: Clustering the dataset...")
    step3 = segmenter.segment(verbose=True)
    print(step3)

    # Save algorithm, model, scaler, pca and feature_names
    print("\nStep 4/5: Saving segmented point cloud in file...")
    step4 = segmenter.save_clusters(verbose=True)
    print(step4)

    # Create and save prediction report
    print("\nStep 5/5: Creating classification report:")
    print(segmenter.report_filename + '.txt')
    step5 = segmenter.classification_report(verbose=True)
    print(step5)


if args.func == 'train':
    train(args)
elif args.func == 'predict':
    predict(args)
elif args.func == 'segment':
    segment(args)
else:
    raise KeyError("No valid function selected!")

