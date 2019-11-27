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
#        'common.py' from classer library to predict dataset          #
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

import joblib

import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

def format_dataset(path_raw_data, mode='training', raw_classif=None):
    """
    To format the input data as panda DataFrame. Exclude XYZ fields.
    :param path_raw_data: Path of the input data as text file (.CSV).
    :param mode: Set the mode ['training', 'predict'] to check mandatory 'target' field in case of training.
    :param raw_classif: (optional): set the field name of the raw_classification of some LiDAR point clouds.
    :return: features_data, coord, height and target as DataFrames.
    """

    print("1. Formatting data as pandas.Dataframe...", end='')
    # Load data into DataFrame
    frame = pd.read_csv(path_raw_data, header='infer')

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
        index_x = None

    # Search Y coordinate
    if 'Y' in fields_name:
        index_y = fields_name.index('Y')
    elif 'y' in fields_name:
        index_y = fields_name.index('y')
    else:
        index_y = None

    # Search Z coordinate
    if 'Z' in fields_name:
        index_z = fields_name.index('Z')
    elif 'z' in fields_name:
        index_z = fields_name.index('z')
    else:
        index_z = None

    # Create DataFrame with X and Y coordinates
    if index_x is not None and index_y is not None:
        coord = frame.iloc[:, [index_x, index_y]]
    else:
        raise ValueError("There is no X or Y field, or both!")

    # Create DataFrame with Z coordinate
    if index_z is not None:
        hght = frame.iloc[:, [index_z]]
    else:
        raise ValueError("There is no Z field!")

    # Create dataFrame of targets
    if mode == 'training':
        if 'target' in fields_name:
            trgt = frame.loc[:, ['target']]
        else:
            trgt = None
            raise ValueError("A 'target' field is mandatory for training!")
    else:
        if 'target' in fields_name:
            trgt = frame.loc[:, ['target']]
        else:
            trgt = None

    # Select only features fields by removing X, Y, Z and target fields
    features_name = fields_name
    for field in ['X', 'Y', 'Z', 'target']:
        features_name.remove(field)

    # Remove the raw_classification of some LiDAR point clouds
    if raw_classif is not None:
        if isinstance(raw_classif, str):
            for index, field in enumerate(features_name):
                if raw_classif in field:
                    features_name.remove(fields_name[index])

    # formatted data without extra fields
    features_data = frame.loc[:, features_name]

    # Replace NAN values by median
    features_data.fillna(value=features_data.median(0), inplace=True)  # .median(0) computes median/col

    print(" Done.")

    return features_data, coord, hght, trgt


def scale_dataset(data_to_scale, method='Standard'):
    """
    Scale the dataset according different methods: 'Standard', 'Robust', 'MinMax'.
    :param data_to_scale: dataset to scale.
    :param method: (optional) Set method to scale dataset.
    :return: The training and testing datasets: data_train_scaled, data_test_scaled.
    """

    print("2. Scaling the data...", end='')
    # Perform the data scaling according the chosen method
    if method is 'Standard':
        scaler = StandardScaler()  # Scale data with mean and std
    elif method is 'Robust':
        scaler = RobustScaler()  # Scale data with median and interquartile
    elif method is 'MinMax':
        scaler = MinMaxScaler()  # Scale data between 0-1 for each feature and translate (mean=0)
    else:
        scaler = StandardScaler()
        print("\nWARNING:"
              "\nScaling method '{}' was not recognized. Replaced by 'StandardScaler' method.\n".format(str(method)))

    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=data_to_scale.columns.values.tolist())

    print(" Done.")
    return data_scaled
