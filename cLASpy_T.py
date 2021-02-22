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
import yaml
import json

from common import *
from predict import *
from training import *
from sklearn.metrics import confusion_matrix, classification_report


# -------------------------
# ---- ARGUMENT_PARSER ----
# -------------------------

parser = argparse.ArgumentParser(description="\nThis library based on Sci-kit Learn library classify 3D point cloud\n"
                                             "using machine learning algorithms. The input data must be CSV or LAS "
                                             "file. Three supervised classifiers are available from the sklearn library"
                                             "(RandomForestClassifier, GradientBoostingClassifier and MLPClassifier) "
                                             "and one unsupervised clustering algorithm (KMeans).\n"
                                             "\nFor training, data_file must contain:\n"
                                             "    --> target field named 'target' AND data fields\n"
                                             "For prediction, data_file must contain:\n"
                                             "    --> data fields\n\n"
                                             "If X, Y and/or Z fields are present, they are excluded.\n"
                                             "All standard dimensions of LAS like 'intensity', 'scan_angle_rank'\n"
                                             "are discarded.",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("algorithm",
                    help="The available algorithms:\n"
                         "    'rf': RandomForestClassifier\n"
                         "    'gb': GradientBoostingClassifier\n"
                         "    'ann': MLPClassifier\n"
                         "    'kmeans': KMeans\n",
                    type=str, choices=['rf', 'gb', 'ann', 'kmeans'])

parser.add_argument("data_file",
                    help="The data file:\n"
                         "    [WINDOWS]: 'C:/path/to/the/data.file'\n"
                         "    [UNIX]: '/path/to/the/data.file'",
                    type=str, metavar="/path/to/data.file")

parser.add_argument("-g", "--grid_search",
                    help="Perform the training with GridSearchCV",
                    action="store_true")

parser.add_argument("-i", "--importance",
                    help="Export feature importance from randomForest and gradientBoosting model as a PNG image file.",
                    action="store_true")

parser.add_argument("-k", "--param_grid",
                    help='Set the parameters to pass at the GridSearch as list(sep=",") in dict. NO SPACE\n'
                         'If empty, GridSearchCV uses presets.\n'
                         'Example: -k=\'{"n_estimators":[50,100,500],"loss":["deviance","exponential"],'
                         '"hidden_layer_sizes":[[100,100],[50,100,50]]}\'\n'
                         'Wrong parameters will be ignored\n',
                    type=str, metavar="[=\'dict\']")

parser.add_argument("-m", "--model_to_import",
                    help="The model file to import to make predictions:\n"
                         "'/path/to/the/training/file.model'",
                    type=str, metavar="[=\"/path/to/file.model\"]")

parser.add_argument("-n", "--n_jobs",
                    help="Set the number of CPU used, '-1' means all CPU available.",
                    type=int, metavar="[1,2,...,-1]", default=-1)

parser.add_argument("-p", "--parameters",
                    help='Set the parameters to pass at the classifier for training, as dict.\n'
                         'Example: -p=\'{"n_estimators":50,"max_depth":5,"max_iter":500}\'',
                    type=str, metavar="[=\'dict\']")

parser.add_argument("--pca",
                    help="Set the PCA analysis and the number of principal components",
                    type=int, metavar="--pca=8")

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
                         "to scikit-learn documentation.",
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

# Check parameters exists
if args.parameters:
    parameters = yaml.safe_load(args.parameters)

# Set the chosen learning classifier
if args.algorithm == 'rf' or args.algorithm == 'RandomForestClassifier':
    algo = 'RandomForestClassifier'
    classifier = set_random_forest(fit_params=parameters, n_jobs=args.n_jobs)

elif args.algorithm == 'gb' or args.algorithm == 'GradientBoostingClassifier':
    algo = 'GradientBoostingClassifier'
    classifier = set_gradient_boosting(fit_params=parameters)

elif args.algorithm == 'ann' or args.algorithm == 'MLPClassifier':
    algo = 'MLPClassifier'
    classifier = set_mlp_classifier(fit_params=parameters)

elif args.algorithm == 'kmeans' or args.algorithm == 'KMeans':
    algo = 'KMeans'
    classifier = set_kmeans_cluster(fit_params=parameters)

else:
    raise ValueError("No valid classifier!")

# Set mode as 'training', 'prediction' or 'unsupervised'
if args.model_to_import is None:
    if any(args.algorithm in s for s in ['rf', 'gb', 'ann']):
        mode = 'training'
    else:
        mode = 'unsupervised'
else:
    mode = 'prediction'

# INTRODUCTION
data_path, folder_path, start_time = introduction(algo, args.data_file)  # Start prompt
timestamp = start_time.strftime("%m%d_%H%M")  # Timestamp for file creation MonthDay_HourMinute

# FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
print("\n1. Formatting data as pandas.Dataframe...")
data, target = format_dataset(data_path, mode=mode)

# Get the number of points
nbr_pts = nbr_pts(data_length=data.shape[0], sample_size=args.samples, mode=mode)
str_nbr_pts = format_nbr_pts(nbr_pts)  # Format nbr_pts as string for filename

# Set the report filename
report_filename = str(folder_path + '/' + mode[0:5] + '_' +
                      args.algorithm + str_nbr_pts + str(timestamp))

# Get the feature names
feature_names = data.columns.values.tolist()

# TRAINING or PREDICTION
if mode == 'training':  # Training mode

    # Split data into training and testing sets
    print("\n2. Splitting data in train and test sets...")
    X_train_val, X_test, y_train_val, y_test = split_dataset(
        data.values,
        target.values,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        samples=nbr_pts)

    # Get the train and test sizes
    train_size = len(y_train_val)
    test_size = len(y_test)

    # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
    print("\n3. Scaling data...")
    scaler = set_scaler(method=args.scaler)

    # Create PCA if it's called
    if args.pca:
        pca_filename = str(report_filename + '_pca.png')
        pca = set_pca(n_components=args.pca)

    # Create Pipeline for GridSearchCV or CV
    pipeline = set_pipeline(scaler, classifier, pca=pca)

    # TYPE OF TRAINING
    if args.grid_search:  # Training with GridSearchCV
        print('\n4.1 Training model with GridSearchCV...\n')

        # Check param_grid exists
        if args.param_grid:
            param_grid = yaml.safe_load(args.param_grid)

        else:
            param_grid = None

        param_grid = check_grid_params(pipeline,
                                       grid_params=param_grid)

        model, grid_results = training_gridsearch(pipeline,
                                                  X_train_val,
                                                  y_train_val,
                                                  grid_params=param_grid,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

    else:  # Training with Cross Validation
        print("\n4.1 Training model with cross validation...\n")
        model, cv_results = training_nogridsearch(pipeline,
                                                  X_train_val,
                                                  y_train_val,
                                                  scoring=args.scoring,
                                                  n_jobs=args.n_jobs)

    # Importance of each feature in RF and GB
    if args.grid_search or args.algorithm == 'ann':
        args.importance = False  # Overwrite 'False' if '-i' option set with grid, ann
    if args.importance:
        feat_imp_filename = str(report_filename + '_feat_imp.png')
        save_feature_importance(model, feature_names, feat_imp_filename)

    # Create confusion matrix
    print("\n4.2. Creating confusion matrix...")
    y_test_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_test_pred)
    conf_mat = precision_recall(conf_mat)  # return Dataframe
    test_report = classification_report(y_test, y_test_pred)  # Get classification report
    print("\n{}".format(test_report))

    # Save model and scaler and pca
    print("\n5. Saving model and scaler in file:")
    model_filename = str(report_filename + '.model')
    save_model(model, model_filename)

elif mode == 'unsupervised':
    print("\n2. Clustering the entire dataset...")
    y_pred = classifier.fit_predict(data)

    # Save clustering result as point cloud file with all data
    print("\n3. Saving segmented point cloud as CSV file:")
    predic_filename = str(report_filename + '.csv')
    print(predic_filename)
    save_predictions(y_pred, predic_filename, data_path)
    model = classifier
    scaler = None

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

if mode == 'training':
    print("\nModel trained in {}".format(spent_time))
else:
    print("\nPredictions done in {}".format(spent_time))
