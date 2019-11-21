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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

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


def split_dataset(data_values, target_values, rd_state=0, train_ratio=0.8, test_ratio=0.2, threshold=500000):
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
    if len(data_values[:, 0]) > threshold:
        train_ratio = train_ratio * threshold
        test_ratio = test_ratio * threshold

    data_train, data_test, target_train, target_test = train_test_split(data_values, target_values,
                                                                        random_state=rd_state,
                                                                        train_size=train_ratio,
                                                                        test_size=test_ratio)
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


def training_with_grid(learning_algo, training_data, training_target, param_grid=None):
    """
    Train model with GridSearchCV meta-estimator according the chosen learning algorithm.
    :param learning_algo: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :param param_grid: The parameters for the GridSearchCV.
    :return:
    """


def training_with_nogrid(learning_algo, training_data, training_target):
    """
    Train model without GridSearchCV meta-estimator according the chosen learning algorithm.
    This function performs cross-validation.
    :param learning_algo: Set the algorithm to train the model.
    :param training_data: The training dataset.
    :param training_target: The targets corresponding to the training dataset.
    :return:
    """


# -------------------------
# --------- MAIN ----------
# -------------------------


if __name__ == '__main__':
    # path to the CSV file
    # raw_data = "D:/PostDoc/Python/DataTest/20150603_Classif_plus_Geom_50kpts.csv"
    raw_data = "D:/PostDoc/Python/DataTest/20150603_targeted_nantest_10kpts.csv"  # Test file with many nan values

    # Format the data as data / XY / Z / target DataFrames
    data, xy_coord, z_height, target = format_dataset(raw_data, raw_classif='lassif')

    # Scale the dataset
    data = scale_dataset(data, method='Standard')

    # Create samples for training and testing
    X_train, X_test, y_train, y_test = split_dataset(data.values, target.values, rd_state=0)

    # Learning algorithm 'rf', 'gb', 'svm', 'ann'
    algo = 'rf'

    # With GridSearch or not
    grid = True
