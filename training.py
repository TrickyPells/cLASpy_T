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

import pandas as pd

from sklearn.model_selection import train_test_split

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def format_dataset(path_raw_data):
    """
    To format the input data as panda DataFrame. Exclude XYZ fields.
    :param path_raw_data: Path of the input data as text file (.CSV)
    :return: features_data, coord, height and target as DataFrames
    """
    frame = pd.read_csv(path_raw_data, header='infer')  # Load data into DataFrame

    # Create list name of all fields
    fields_name = frame.columns.values.tolist()

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

    #  Select only features fields by removing X, Y, Z and target fields
    features_name = frame.columns.values.tolist()
    for field in ['X', 'Y', 'Z', 'target']:
        features_name.remove(field)
    features_data = frame.loc[:, features_name]  # formatted data without extra fields

    return features_data, coord, hght, trgt


def split_dataset(data_values, target_values, rd_state=0):
    """
    Split the input data and target in data_train, data_test, target_train and target_test.
    Check the length of the dataset. If length > 500 kpts, trainset = 400 kpts and testset = 100 kpts
    :param data_values: the np.ndarray with the data features
    :param target_values: the np.ndarray with the target
    :param rd_state: (optional) random_state
    :return: data_train, data_test, target_train and target_test as np.ndarray
    """
    # Define ratio for splitting
    train_ratio = 0.8
    test_ratio = 0.2

    # Check if the length of dataset is longer than 500 kpts
    if len(data_values[:, 0]) > 500000:
        train_ratio = 400000
        test_ratio = 100000

    data_train, data_test, target_train, target_test = train_test_split(data_values, target_values,
                                                                        random_state=rd_state,
                                                                        train_size=train_ratio,
                                                                        test_size=test_ratio)
    return data_train, data_test, target_train, target_test


# -------------------------
# --------- MAIN ----------
# -------------------------


if __name__ == '__main__':
    # path to the CSV file
    raw_data = "D:/PostDoc/Python/DataTest/20150603_Classif_plus_Geom_50kpts.csv"

    # Format the data as DataFrame
    data, xy_coord, z_height, target = format_dataset(raw_data)

    # Create samples for training and testing
    X_train, X_test, y_train, y_test = split_dataset(data.values, target.values, rd_state=0)

    # print("data_train: {} {}\ndata_test: {} {}\ntarget_train: {} {}\ntarget_test: {} {}".format(
    #     type(X_train), X_train.shape, type(X_test), X_test.shape,
    #     type(y_train), y_train.shape, type(y_test), y_test.shape))

