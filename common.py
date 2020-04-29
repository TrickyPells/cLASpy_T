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
#                'common.py' from classer library                     #
#                    By Xavier PELLERIN LE BAS                        #
#                         November 2019                               #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description: functions shared by training and prediction modes     #
#  to format, scale, compute the precision and plot some figures      #
#                                                                     #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------

import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

def introduction(algo, csv_file):
    """
    Write the introduction, create folder to store results
    and return the start_time
    :param algo: Algorithm used
    :param csv_file: CSV file of used data
    :return: raw_data, folder_path and start time
    """
    print("\n####### POINT CLOUD CLASSIFICATION #######\n"
          "Algorithm used: {}\n"
          "Path to CSV file: {}\n".format(algo, csv_file))

    # Create a folder to store models, reports and predictions
    print("Create a new folder to store the result files...", end='')
    raw_data = csv_file
    raw_data = '/'.join(raw_data.split('\\'))  # Change '\' in '/'
    folder_path = '.'.join(raw_data.split('.')[:-1])  # remove extension so give folder path
    try:
        os.mkdir(folder_path)  # Using file path to make new folder
        print(" Done.")
    except (TypeError, FileExistsError):
        print(" Folder already exists.")

    # Timestamp for created files
    start_time = datetime.now()

    return raw_data, folder_path, start_time


def format_dataset(path_raw_data, mode='training', raw_classif=None):
    """
    Format the input data as panda DataFrame. Exclude XYZ fields.
    :param path_raw_data: Path of the input data as text file (.CSV).
    :param mode: Set the mode ['training', 'prediction'] to check mandatory 'target' field in case of training.
    :param raw_classif: (optional): set the field name of the raw_classification of some LiDAR point clouds.
    :return: features_data, coord, height and target as DataFrames.
    """
    # Load data into DataFrame
    frame = pd.read_csv(path_raw_data, sep=',', header='infer')

    # Create list name of all fields
    fields_name = frame.columns.values.tolist()

    # Clean up the header built by CloudCompare ('//X')
    for field in fields_name:
        if field == '//X':
            frame = frame.rename(columns={"//X": "X"}, errors='raise')

    # Update list name of all fields
    fields_name = frame.columns.values.tolist()

    # Search X, Y, Z, target and raw_classif fields
    field_x = None
    field_y = None
    field_z = None
    field_t = None
    for field in fields_name:
        if field.casefold() == 'x':  # casefold() to be non case-sensitive
            field_x = field
        if field.casefold() == 'y':
            field_y = field
        if field.casefold() == 'z':
            field_z = field
        if field.casefold() == 'target':
            field_t = field

    # Create XY field -> 'coord'
    if field_x is None or field_y is None:
        raise ValueError("There is no X or Y field, or both!")
    coord = frame[[field_x, field_y]]
    # coord = frame.loc[:, [field_x, field_y]]

    # Create Z field -> 'height'
    if field_z is None:
        raise ValueError("There is no Z field!")
    height = frame.loc[:, field_z]

    # Create 'target' field
    if mode == 'training' and field_t is None:
        raise ValueError("A 'target' field is mandatory for training!")
    if field_t:
        target = frame.loc[:, field_t]
    else:
        target = None

    # Select only features fields by removing X, Y, Z and target fields
    feat_name = fields_name
    for field in [field_x, field_y, field_z, field_t]:
        if field:
            feat_name.remove(field)

    # Remove the raw_classification of some LiDAR point clouds
    if isinstance(raw_classif, str):
        for field in feat_name:
            if raw_classif.casefold() in field.casefold():
                feat_name.remove(field)

    # data without extra fields
    feat_data = frame.loc[:, feat_name]

    # Replace NAN values by median
    feat_data.fillna(value=feat_data.median(0), inplace=True)  # .median(0) computes median/col

    return feat_data, coord, height, target


def plot_pca(filename, pca, ft_names):
    """
    Create and save figure of PCA principal components.
    :param filename: Filename for the matshow figure of principal components.
    :param pca: PCA already fitted with data.
    :param ft_names: List of the feature names.
    :return:
    """
    # Export principal components as picture
    plt.matshow(pca.components_, cmap='seismic')
    # plt.title("PCA Principal Components")
    plt.yticks(list(range(0, len(pca.components_)), list(range(1, len(pca.components) + 1))))
    plt.colorbar()
    plt.xticks(range(len(ft_names)), ft_names, rotation=60, ha='left')
    plt.xlabel("Features")
    plt.ylabel("Principal Components")
    plt.savefig(filename)


def set_pca(n_components):
    """
    Set the PCA according to the number of principal components
    :param n_components: Number of principal components.
    :return: PCA object
    """
    pca = PCA(n_components=n_components)

    return pca


def apply_pca(pca, data):
    """
    Apply PCA transformation to the data.
    :param pca: PCA to apply.
    :param data: Data to tranform.
    :return: data_pca as pandas.core.frame.DataFrame
    """
    #  Create list of component names
    pca_compo_list = list()
    for idx, compo in enumerate(pca.components_):
        pca_compo_list.append('Principal_Compo_' + str(idx + 1))

    # Apply PCA transformation to the data
    data_pca = pca.transform(data)  # Becomes np.array
    data_pca = pd.DataFrame.from_records(data_pca,
                                         columns=pca_compo_list)  # Becomes pd.DataFrame

    return data_pca


def set_pipeline(scaler, classifier, pca=None):
    """
    Set the pipeline for GridSearchCV and Cross-Validation
    :param scaler: Scaler used.
    :param classifier: Classifier used for training
    :param pca: (optional) Principal Component Analysis.
    :return: Pipeline
    """
    # Two configurations depending of PCA existence
    if pca:
        pipe = Pipeline([("scaler", scaler),
                         ("pca", pca),
                         ("classifier", classifier)])

    else:
        pipe = Pipeline([("scaler", scaler),
                         ("classifier", classifier)])

    return pipe


def set_scaler(method='Standard'):
    """
    Set the scaler according to different methods: 'Standard', 'Robust', 'MinMax'.
    :param method: Set method to scale dataset.
    :return: Scaler.
    """
    # Set the data scaling according the chosen method
    if method == 'Standard':
        scaler = StandardScaler()  # Scale data with mean and std
    elif method == 'Robust':
        scaler = RobustScaler()  # Scale data with median and interquartile
    elif method == 'MinMax':
        scaler = MinMaxScaler()  # Scale data between 0-1 for each feature and translate (mean=0)
    else:
        scaler = StandardScaler()
        print("\nWARNING:"
              "\nScaling method '{}' was not recognized. Replaced by 'StandardScaler' method.\n".format(str(method)))

    return scaler


def precision_recall(conf_mat):
    """
    Compute precision, recall and global accuracy from confusion matrix.
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
                 feat_names, scaler, data_len, train_len=None, test_len=None,
                 pca_compo=None, model=None, grid_results=None, cv_results=None,
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
    :param scaler: Method used to scale data.
    :param data_len: Length of the used data.
    :param train_len: Length of the train data set.
    :param test_len: Length of the test data set.
    :param applied_param: Parameters applied to create model or make predictions.
    :param pca_compo: Principal components of the PCA
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
        report.write('\n\nScaling method:\n{}'.format(scaler))

        # Write the train and test size
        if mode == 'training':
            report.write('\n\nNumber of points for training: ' + str(data_len) + ' pts')
            report.write('\nTrain size: ' + str(train_len) + ' pts')
            report.write('\nTest size: ' + str(test_len) + ' pts')

        # Write the number of point to predict
        if mode == 'prediction':
            report.write('\n\nNumber of points to predict: ' + str(data_len) + ' pts')
            report.write('\nModel used: ' + model)

        if pca_compo:
            report.write('\n\nPCA Components:\n')
            report.write(pca_compo)

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
        if score_report:
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
    :param model: Pipeline with RandomForest or GradientBoosting Classifier
    :param feature_names: Names of the features.
    :param feature_filename: Filename and path of the feature importance figure file.
    :return:
    """
    # Get the feature importances
    importances = model.named_steps['classifier'].feature_importances_

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
