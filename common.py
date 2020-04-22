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
#        'common.py' from classer library to format, scale, compute   #
#        the precision and importance of params                       #
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
    Format the input data as panda DataFrame. Exclude XYZ fields.
    :param path_raw_data: Path of the input data as text file (.CSV).
    :param mode: Set the mode ['train', 'pred'] to check mandatory 'target' field in case of training.
    :param raw_classif: (optional): set the field name of the raw_classification of some LiDAR point clouds.
    :return: features_data, coord, height and target as DataFrames.
    """

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
    feat_name = fields_name
    for field in ['X', 'Y', 'Z', 'target']:
        feat_name.remove(field)

    # Remove the raw_classification of some LiDAR point clouds
    if raw_classif is not None:
        if isinstance(raw_classif, str):
            for index, field in enumerate(feat_name):
                if raw_classif in field:
                    feat_name.remove(fields_name[index])

    # formatted data without extra fields
    feat_data = frame.loc[:, feat_name]

    # Replace NAN values by median
    feat_data.fillna(value=feat_data.median(0), inplace=True)  # .median(0) computes median/col

    print(" Done.")

    return feat_data, coord, hght, trgt


def scale_dataset(data_to_scale, method='Standard'):
    """
    Scale the dataset according different methods: 'Standard', 'Robust', 'MinMax'.
    :param data_to_scale: dataset to scale.
    :param method: (optional) Set method to scale dataset.
    :return: The training and testing datasets: data_train_scaled, data_test_scaled.
    """

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


def nbr_pts(data_length, samples_size=None):
    """
    Set the number of point for according the magnitude.
    :param samples_size: Float of the number of point for training.
    :param data_length: Total number of points in data file.
    :return: nbr_pts: Integer of the number of points.
    """
    # Initiation
    magnitude = 1000000

    # Tests
    try:
        data_length = float(data_length)
    except ValueError as ve:
        raise ValueError("'data_length' parameter must be a number")

    if samples_size is not None:
        try:
            samples_size = float(samples_size)
        except ValueError as ve:
            samples_size = 0.05
            print("ValueError: 'samples_size' must be a number\n"
                  "'samples_size' set to default 0.05 Mpts.")

    # Sample size > Data size
    if samples_size is None or samples_size * magnitude >= data_length:
        int_nbr_pts = int(data_length)  # Number of points of entire point cloud
    else:
        int_nbr_pts = int(samples_size * magnitude)  # Number of points < sample size

    return int_nbr_pts


def format_nbr_pts(number_pts):
    """
    Format the nbr_pts as string for the filename.
    :param number_pts: Integer of the number of points used to format in string.
    :return: String of the point number write according the magnitude suffix.
    """
    # Initiation
    magnitude = 1000000

    # Format as Mpts or kpts according number of points
    if number_pts >= magnitude:  # number of points > 1Mpts
        str_nbr_pts = str(np.round(number_pts / magnitude, 1))
        if str_nbr_pts.split('.')[-1][0] == '0':  # round number if there is zero after point ('xxx.0x')
            str_nbr_pts = str_nbr_pts.split('.')[0]
        else:
            str_nbr_pts = '_'.join(str_nbr_pts.split('.'))  # replace '.' by '_' if not rounded
        str_nbr_pts = str_nbr_pts + 'Mpts_'

    else:  # number of points < 1M
        number_pts = int(number_pts / 1000.)
        str_nbr_pts = str(number_pts) + 'kpts_'

    return str_nbr_pts


def write_report(filename, mode, algo, data_file, start_time, elapsed_time, applied_param,
                 feat_names, scale_method, data_len, train_len=None, test_len=None,
                 model=None, grid_results=None, cv_results=None,
                 conf_mat=None, score_report=None):
    """
    Write the report of training or predictions in .TXT file.
    :param filename: Entire path and filename without extension.
    :param mode: 'train' or 'pred' mode.
    :param algo: Algorithm used for training or predictions.
    :param data_file: Data file used to make training or predictions.
    :param start_time: Time when the script began.
    :param elapsed_time: Time spent between begin and end.
    :param feat_names: List of the all feature names.
    :param scale_method: Method used to scale data.
    :param data_len: Length of the used data.
    :param train_len: Length of the train data set.
    :param test_len: Length of the test data set.
    :param applied_param: Parameters applied to create model or make predictions.
    :param model: Model used to make predictions.
    :param grid_results: Results of the GridSearchCV.
    :param cv_results: Results of the Cross Validation.
    :param conf_mat: Confusion matrix if target field exists.
    :param score_report: Report of all score if target field exists.
    :return:
    """
    # Write the header of the report file
    with open(filename + '.txt', 'w', encoding='utf-8') as report:
        report.write('Report of ' + algo + ' ' + mode)
        report.write('\n\nDatetime: ' + start_time.strftime("%Y-%m-%d %H:%M:%S"))
        report.write('\nFile: ' + data_file)
        report.write('\n\nFeatures:\n' + '\n'.join(feat_names))
        report.write('\n\nScaling method: ' + scale_method)

        # Write the train and test size
        if mode == 'training':
            report.write('\n\nNumber of points for training: ' + str(data_len) + ' pts')
            report.write('\nTrain size: ' + str(train_len) + ' pts')
            report.write('\nTest size: ' + str(test_len) + ' pts')

        # Write the number of point to predict
        if mode == 'prediction':
            report.write('\n\nNumber of points to predict: ' + str(data_len) + ' pts')
            report.write('\nModel used: ' + model)

        # Write the GridSearchCV results
        if grid_results is not None:
            report.write('\n\n\nResults of the GridSearchCV:\n')
            report.write(grid_results.to_string(index=False))

        # Write the Cross validation results
        if cv_results is not None:
            report.write('\n\n\nResults of the Cross-Validation:\n')
            report.write(pd.DataFrame(cv_results).to_string(index=False, header=False))

        # Write applied parameters
        report.write('\n\n\nParameters:\n' + '\n'.join(applied_param))

        # Write the confusion matrix results
        if conf_mat is not None:
            report.write('\n\n\nConfusion Matrix:\n')
            report.write(conf_mat.to_string())

        # Write the score report of the classification
        if score_report is not None:
            report.write('\n\n\nClassification Report:\n')
            report.write(score_report)

        # Write elapsed time
        if mode == 'training':
            report.write('\n\nModel trained in {}'.format(elapsed_time))
        else:
            report.write('\n\nPredictions done in {}'.format(elapsed_time))


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
