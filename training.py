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

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def format_dataset(path_raw_data, raw_classif=None):
    """
    To format the input data as panda DataFrame. Exclude XYZ fields.
    :param path_raw_data: Path of the input data as text file (.CSV).
    :param raw_classif: (optional): set the field name of the raw_classification of some LiDAR point clouds.
    :return: features_data, coord, height and target as DataFrames.
    """
    frame = pd.read_csv(path_raw_data, header='infer')  # Load data into DataFrame

    # Create list name of all fields
    fields_name = frame.columns.values.tolist()

    # Clean up the header built by CloudCompare ('//X')
    for index, field in enumerate(fields_name):
        if field == '//X':
            fields_name[index] = 'X'

    # Search X coordinate
    if 'X' in fields_name:
        index_x = fields_name.index('X')
    elif 'x' in fields_name:
        index_x = fields_name.index('x')
    else:
        print("There is no X field !")

    # Search Y coordinate
    if 'Y' in fields_name:
        index_y = fields_name.index('Y')
    elif 'y' in fields_name:
        index_y = fields_name.index('y')
    else:
        print("There is no Y field !")

    # Search Z coordinate
    if 'Z' in fields_name:
        index_z = fields_name.index('Z')
    elif 'z' in fields_name:
        index_z = fields_name.index('z')
    else:
        print("There is no Z field !")

    # Create DataFrame with X and Y coordinates and DataFrame with Z
    coord = frame.iloc[:, [index_x, index_y]]
    hght = frame.iloc[:, [index_z]]  # Create DataFrame with Z coordinate

    # Create dataFrame of targets
    trgt = frame.loc[:, ['target']]

    # Select only features fields by removing X, Y, Z and target fields
    features_name = fields_name
    for field in ['X', 'Y', 'Z', 'target']:
        features_name.remove(field)

    # Remove the raw_classification of some LiDAR point clouds
    if raw_classif is not None:
        if isinstance(raw_classif, str):
            for index, field in enumerate(fields_name):
                if raw_classif in field:
                    features_name.remove(fields_name[index])

    features_data = frame.loc[:, features_name]  # formatted data without extra fields

    # Replace NAN values by median
    features_data.fillna(value=features_data.median(0), inplace=True)  # features_data.median(0) computes median/col

    return features_data, coord, hght, trgt


def split_dataset(data_values, target_values, train_ratio=0.8, test_ratio=0.2, threshold=500000):
    """
    Split the input data and target in data_train, data_test, target_train and target_test.
    Check the length of the dataset. If length > 500 kpts, trainset = 400 kpts and testset = 100 kpts.
    :param data_values: the np.ndarray with the data features.
    :param target_values: the np.ndarray with the target.
    :param rd_state: (optional) random_state.
    :param train_ratio: (optional) Ratio of the size of training dataset.
    :param test_ratio: (optional) Ratio of the size of testing dataset.
    :param threshold: (optional) Number of samples beyond which the dataset is splitted with two integers,
    for train_size and test_size. The threshold is paired with train_ratio and test_ratio.
    :return: data_train, data_test, target_train and target_test as np.ndarray.
    """
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

    return data_train, data_test, target_train, target_test


def scale_dataset(data_to_scale, method='Standard'):
    """
    Scale the dataset according different methods: 'Standard', 'Robust', 'MinMax'.
    :param data_to_scale: dataset to scale.
    :param method: (optional) Set method to scale dataset.
    :return: The training and testing datasets: data_train_scaled, data_test_scaled.
    """
    # Perform the data scaling according the chosen method
    if method is 'Standard':
        scaler = StandardScaler()
    elif method is 'Robust':
        scaler = RobustScaler()
    elif method is 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        print("\nWARNING:"
              "\nScaling method '{}' was not recognized. Replaced by 'StandardScaler' method.\n".format(str(method)))

    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=data_to_scale.columns.values.tolist())

    return data_scaled


def check_parameters(classifier, fit_params):
    """
    Check if the given parameters match with the given classifier.
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
            print("\nValueError: Invalid parameter '{}' for {}, "
                  "it has been ignored!".format(str(key), clf_name))

    return classifier


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
        classifier = check_parameters(classifier, fit_params)

    else:
        classifier = RandomForestClassifier(n_estimators=100,
                                            max_depth=5,
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
                                                n_estimators=10,
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


def training_with_grid(learning_algo, training_data, training_target, param_grid=None):
    """
    Train model with GridSearchCV meta-estimator according the chosen learning algorithm.
    :param learning_algo: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param param_grid: The parameters for the GridSearchCV.
    :return:
    """


def training_with_nogrid(classifier, training_data, training_target):
    """
    Train model with cross-validation according the chosen classifier.
    :param classifier: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :return: model: the training model.
    """
    # Set cross_validation method with train_size 80% and validation_size 20%
    cross_val = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2, random_state=0)

    # Get the training scores
    training_scores = cross_val_score(classifier, training_data, training_target, cv=cross_val, n_jobs=-1)
    print("Training model scores performed with cross-validation:\n{}\n".format(training_scores))

    # Set the classifier with training_data and target
    classifier.fit(training_data, training_target)

    return classifier

# -------------------------
# --------- MAIN ----------
# -------------------------


if __name__ == '__main__':
    # Learning algorithm 'rf', 'gb', 'svm', 'ann'
    algo = 'ann'
    parameters = {"loss": 'deviance',
                  "n_estimators": 50,
                  "max_depth": 3,
                  "min_samples_leaf": 1000,
                  "n_jobs": -1,
                  "random_state": 0}

    # With or without GridSearchCV
    grid = False
    # path to the CSV file
    raw_data = "D:/PostDoc/Python/DataTest/20150603_Classif_plus_Geom_50kpts.csv"
    # raw_data = "D:/PostDoc/Python/DataTest/20150603_targeted_nantest_10kpts.csv"  # Test file with many nan values

    # Format the data as data / XY / Z / target DataFrames
    data, xy_coord, z_height, target = format_dataset(raw_data, raw_classif='lassif')

    # Scale the dataset
    data = scale_dataset(data, method='Standard')

    # Create samples for training and testing
    X_train_val, X_test, y_train_val, y_test = split_dataset(data.values, target.values)

    # Set the chosen learning classifier
    if algo is 'rf':
        clf = set_random_forest(parameters)
    elif algo is 'gb':
        clf = set_gradient_boosting(parameters)
    elif algo is 'svm':
        clf = set_linear_svc(parameters)
    elif algo is 'ann':
        clf = set_mlp_classifier()

    # What type of training
    if grid:
        print('Training with grid')
    else:
        # Train model with cross_val and train_val datasets
        model = training_with_nogrid(clf, X_train_val, y_train_val)
        print(model)

