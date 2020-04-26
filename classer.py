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

import yaml
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

parser.add_argument("--pca",
                    help="Set the PCA analysis and the number of principal components",
                    type=int, metavar="--pca=10")

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

# Set the mode as 'training' or 'prediction'
if args.model_to_import is None:
    mod = 'training'
else:
    mod = 'prediction'

# Set non-common parameters as None
train_size = None
test_size = None
pca = None
pca_compo = None
grid_results = None
cv_results = None
model_to_load = None
conf_mat = None
report_class = None

# Check parameters exists
parameters = None
if args.parameters:
    parameters = yaml.safe_load(args.parameters)

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

# INTRODUCTION
raw_data, folder_path, start_time = introduction(algo, args.csv_data_file)
timestamp = start_time.strftime("%m%d_%H%M")  # Timestamp for file creation MD_HM

# FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
print("\n1. Formatting data as pandas.Dataframe...", end='')
data, xy_coord, z_height, target = format_dataset(raw_data,
                                                  mode=mod,
                                                  raw_classif='lassif')

# Get the number of points
nbr_pts = nbr_pts(data_length=len(z_height), samples_size=args.samples)
str_nbr_pts = format_nbr_pts(nbr_pts)  # Format in string for filename

# Give the report filename
report_filename = str(folder_path + '/' + mod[0:5] + '_' +
                      args.algorithm + str_nbr_pts + str(timestamp))

# Get the feature names
feature_names = data.columns.values.tolist()

# TRAINING or PREDICTION
if mod == 'training':  # Training mode

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\n2. Scaling data...", end='')
    scale_method = args.scaler
    scaler = set_scaler(data, method=scale_method)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame.from_records(
        data_scaled, columns=data.columns.values.tolist())

    # Split data into training and testing sets
    print("\n3. Splitting data in train and test sets...", end='')
    X_train_val, X_test, y_train_val, y_test = split_dataset(
        data_scaled.values,
        target.values,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        samples=nbr_pts)

    # Get the train and test sizes
    train_size = len(y_train_val)
    test_size = len(y_test)

    # Apply PCA if it's not None
    if args.pca:
        pca_filename = str(report_filename + '_pca.png')
        pca = set_pca(pca_filename,
                      X_train_val,
                      feature_names,
                      n_components=args.pca)

        X_train_val = pca.transform(X_train_val)
        X_test = pca.transform(X_test)
        pca_compo = str(pca.components_)
        print("\nPCA Applied !\n")

    # TYPE OF TRAINING
    if args.grid_search:  # Training with GridSearchCV
        print('\n4. Training model with GridSearchCV...')

        # Check param_grid exists
        if args.param_grid:
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
        print("\n4. Training model with cross validation...")
        model, cv_results = training_nogridsearch(classifier,
                                                  X_train_val,
                                                  y_train_val,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

    print("\tScore model with the test dataset: {0:.4f}\n".format(
        model.score(X_test, y_test)))

    # Importance of each feature in RF and GB
    if args.grid_search or args.algorithm == 'ann':
        args.importance = False  # Overwrite 'False' if '-i' option set with grid, ann or svm
    if args.importance:
        feat_imp_filename = str(report_filename + '_feat_importance.png')
        save_feature_importance(model, feature_names, feat_imp_filename)

    # Save confusion matrix
    print("\n5. Creating confusion matrix:")
    y_test_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_mat = precision_recall(conf_mat)  # return Dataframe
    print("\n{}".format(conf_mat))

    # Get classification report
    report_class = classification_report(y_test, y_test_pred)
    print("\n{}\n".format(report_class))

    # Save model and scaler and pca
    print("\n6. Saving model and scaler in file:")
    model_filename = str(report_filename + '_' + args.scaler + '.model')
    model_to_save = (model, scaler, pca)
    save_model(model_to_save, model_filename)

else:  # Prediction mode

    # Get model and scaling parameter
    print("\n2. Loading model...", end='')
    model_to_load = args.model_to_import  # Set variable for the report
    loaded_model = load_model(model_to_load)
    # Get the model
    model = loaded_model[0]

    # Get the scaler and scale data
    print("\n3. Scaling data...", end='')
    scaler = loaded_model[1]
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=data.columns.values.tolist())

    # Predic target of input data
    print("\n4. Making predictions for entire dataset...")
    y_pred = model.predict(data_scaled.values)

    if target is not None:
        # Save confusion matrix
        print("\n5 Creating confusion matrix:")
        conf_mat = confusion_matrix(target.values, y_pred)
        conf_mat = precision_recall(conf_mat)  # return Dataframe
        print("\n{}".format(conf_mat))

        # Get classification report
        report_class = classification_report(target.values, y_pred)
        print("\n{}\n".format(report_class))

    # Save classifaction result as point cloud file with all data
    print("\n6. Saving classified point cloud as CSV file:")
    predic_filename = str(report_filename + '.csv')
    print(predic_filename)
    save_predictions(y_pred,
                     predic_filename,
                     xy_fields=xy_coord,
                     z_field=z_height,
                     data_fields=data,
                     target_field=target)

# Create and save prediction report
print("\n7. Creating classification report:")
print(report_filename + '.txt')

# Get the model parameters to print them in report
applied_parameters = ["{}: {}".format(
    param, model.get_params()[param]) for param in model.get_params()]

# Compute elapsed time
spent_time = datetime.now() - start_time

# Write the entire report
write_report(report_filename,
             mode=mod,
             algo=algo,
             data_file=args.csv_data_file,
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
             score_report=report_class)

if mod == 'training':
    print("\n\nModel trained in {}".format(spent_time))
else:
    print("\nPredictions done in {}".format(spent_time))
