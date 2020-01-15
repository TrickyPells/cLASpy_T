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

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
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

    print("\n1. Formatting data as pandas.Dataframe...", end='')
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

    print("\n2. Scaling data...", end='')
    # Perform the data scaling according the chosen method
    if method == 'Standard':
        method = StandardScaler()  # Scale data with mean and std
    elif method == 'Robust':
        method = RobustScaler()  # Scale data with median and interquartile
    elif method == 'MinMax':
        method = MinMaxScaler()  # Scale data between 0-1 for each feature and translate (mean=0)
    else:
        method = StandardScaler()
        print("\nWARNING:"
              "\nScaling method '{}' was not recognized. Replaced by 'StandardScaler' method.\n".format(str(method)))

    method.fit(data_to_scale)
    data_scaled = method.transform(data_to_scale)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=data_to_scale.columns.values.tolist())
    print(" Done.")

    return data_scaled


def precision_recall(conf_mat):
    """
    Compute precision, recal and global accuracy from confusion matrix.
    :param conf_mat: The confusion matrix as a numpy.array.
   :return conf_mat_up: Confusion matrix wth precision, recall and global accuracy
    """
    # Change type array(int) as array(float)
    conf_mat = np.array(conf_mat).astype(float)

    n_rows_cols = conf_mat.shape[0]
    rows_sums = np.sum(conf_mat, axis=1)
    cols_sums = np.sum(conf_mat, axis=0)

    # Compute the recalls
    recalls = list()
    for row in range(0, n_rows_cols):
        try:
            recalls.append((float(conf_mat[row, row]) / float(rows_sums[row])))
        except ZeroDivisionError:
            recalls.append(np.nan)
    conf_mat_up = np.insert(conf_mat, n_rows_cols, recalls, axis=1)

    # Compute the precision
    precisions = list()
    for col in range(0, n_rows_cols):
        try:
            precisions.append(float(conf_mat[col, col]) / float(cols_sums[col]))
        except ZeroDivisionError:
            precisions.append(np.nan)

    # Compute overall accuracy
    try:
        precisions.append(float(np.trace(conf_mat) / float(np.sum(conf_mat))))
    except ZeroDivisionError:
        precisions.append(np.nan)

    conf_mat_up = np.insert(conf_mat_up, n_rows_cols, precisions, axis=0)

    # Save the new confusion matrix
    conf_mat_up = pd.DataFrame(conf_mat_up).round(decimals=3)

    return conf_mat_up


def save_predictions(target_pred, file_name, xy_fields=None,
                     z_field=None, data_fields=None, target_field=None):
    """
    Save the report of the classsification algorithms with test dataset.
    :param target_pred: The point cloud classified.
    :param file_name: The path and name of the file.
    :param xy_fields: The X and Y fields from the raw_data.
    :param z_field: The Z field from the raw_data
    :param data_fields: The data fields from the raw_data.
    :param target_field: The target field from the raw_data.
    :return:
    """
    # Set the np.array of target_pred pd.Dataframe
    if target_pred.shape[0] > 1:
        target_pred = pd.DataFrame(target_pred, columns=['Predictions'])
    elif target_pred.shape[1] > 1:
        target_pred = pd.DataFrame(target_pred)
    else:
        raise ValueError("The predicted target field is empty!")

    # Set the list of DataFrames
    final_classif_list = list()

    # Fill the DataFrame
    if xy_fields is not None:
        if isinstance(xy_fields, pd.DataFrame):
            xy_fields = xy_fields.round(decimals=4)
        final_classif_list.append(xy_fields)

    if z_field is not None:
        if isinstance(z_field, pd.DataFrame):
            z_field = z_field.round(decimals=4)
        final_classif_list.append(z_field)

    if data_fields is not None:
        if isinstance(data_fields, pd.DataFrame):
            data_fields = data_fields.round(decimals=4)
        final_classif_list.append(data_fields)

    if target_field is not None:
        final_classif_list.append(target_field)

    final_classif_list.append(target_pred)
    final_classif = pd.concat(final_classif_list, axis=1)

    final_classif.to_csv(file_name, sep=',', header=True, index=False)


def save_feature_importance(model, feature_names, feature_filename):
    """
    Save the feature importances of RandomForest or GradientBoosting models into file.
    :param model: The RandomForest or GradientBoostingClassifier
    :param feature_names: The names of the features.
    :param feature_filename: The filename and path of the feature importance figure file.
    :return:
    """
    # Get the feature importances
    importances = model.feature_importances_

    # Create feature_imp_dict
    feature_imp_dict = dict()
    for key, imp in zip(feature_names, importances):
        feature_imp_dict[key] = imp

    # Alphabetic order for feature_names
    feature_names = sorted(feature_names)
    importances_sorted = list()
    for key in feature_names:
        importances_sorted.append(feature_imp_dict[key])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature Importances")
    plt.barh(feature_names, importances_sorted, color='b')
    plt.xlabel("Feature importances")
    # plt.yticks(indices, feature_names)
    plt.ylim([-1, len(feature_names)])
    plt.ylabel("Features")
    plt.savefig(feature_filename, bbox_inches="tight")
