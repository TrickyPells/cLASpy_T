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
#                'common.py' from clASpy_T library                    #
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

import pylas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# -------------------------
# ------ VARIABLES --------
# -------------------------

# Define point_format dict for LAS files
point_format = dict()

gps_time = ['gps_time']
nir = ['nir']
rgb = ['red', 'green', 'blue']
wavepacket = ['wavepacket_index', 'wavepacket_offset', 'wavepacket_size',
              'return_point_wave_location', 'x_t', 'y_t', 'z_t']

# Point formats for LAS 1.2 to 1.4
point_format[0] = ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                   'scan_direction_flag', 'edge_of_flight_line', 'classification',
                   'synthetic', 'key_point', 'withheld', 'scan_angle_rank',
                   'user_data', 'point_source_id']
point_format[1] = point_format[0] + gps_time
point_format[2] = point_format[0] + rgb
point_format[3] = point_format[0] + gps_time + rgb

# Point formats for LAS 1.3 to 1.4
point_format[4] = point_format[0] + gps_time + wavepacket
point_format[5] = point_format[0] + gps_time + rgb + wavepacket

# Point formats for LAS 1.4
point_format[6] = ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                   'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
                   'scan_direction_flag', 'edge_of_flight_line', 'classification',
                   'user_data', 'scan_angle_rank', 'point_source_id', 'gps_time']
point_format[7] = point_format[6] + rgb
point_format[8] = point_format[6] + rgb + nir
point_format[9] = point_format[6] + wavepacket
point_format[10] = point_format[6] + rgb + nir + wavepacket

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def introduction(algo, file_path):
    """
    Prompt the introduction, create folder to store results
    and return the start_time
    :param algo: Algorithm used
    :param file_path: CSV or LAS file of used data
    :return: raw_data, folder_path and start time
    """
    # Prompt information about algorithm and file
    data_path = os.path.normpath(file_path)
    print("\n####### POINT CLOUD CLASSIFICATION #######\n"
          "Algorithm used: {}".format(algo))

    # Check CSV or LAS file
    root_ext = os.path.splitext(data_path)  # split file_path into root and extension
    if root_ext[1] == '.csv':
        print("Path to CSV file: {}\n".format(data_path))
    elif root_ext[1] == '.las':
        print("Path to LAS file: {}\n".format(data_path))
    else:
        print("Unknown Extension file !")

    # Create a folder to store models, reports and predictions
    print("Create a new folder to store the result files...", end='')
    folder_path = root_ext[0]  # remove extension so give folder path
    try:
        os.mkdir(folder_path)  # Using file path to create new folder
        print(" Done.")
    except (TypeError, FileExistsError):
        print(" Folder already exists.")

    # Start time for timestamp for files and time computation
    start_time = datetime.now()

    return data_path, folder_path, start_time


def file_to_pandasframe(data_path):
    """
    Convert CSV or LAS data in a Pandas DataFrame
    :param data_path: Path to the CSV or LAS data
    :return: frame as DataFrame
    """
    # Load data into DataFrame
    root_ext = os.path.splitext(data_path)  # split file_path into root and extension
    if root_ext[1] == '.csv':
        frame = pd.read_csv(data_path, sep=',', header='infer')
        # Replace 'space' by '_' and clean up the header built by CloudCompare ('//X')
        for field in frame.columns.values.tolist():
            field_ = field.replace(' ', '_')
            frame = frame.rename(columns={field: field_})
            if field == '//X':
                frame = frame.rename(columns={"//X": "X"})

    elif root_ext[1] == '.las':
        las = pylas.read(data_path)
        print("LAS Version: {}".format(las.header.version))
        print("LAS point format: {}".format(las.point_format.id))
        print("Number of points: {}".format(las.header.point_count))
        extra_dims = list(las.point_format.extra_dimension_names)  # Only get the extra dimensions
        frame = pd.DataFrame()
        for dim in extra_dims:
            frame[dim] = las[dim]
    else:
        raise ValueError("Unknown Extension file !")

    return frame


def format_dataset(data_path, mode='training'):
    """
    Format the input data as panda DataFrame. Exclude XYZ fields.
    :param data_path: Path of the input data as text file (.CSV).
    :param mode: Set the mode ['training', 'prediction'] to check mandatory 'target' field in case of training.
    :return: features_data, coord, height and target as DataFrames.
    """
    # Load data into Pandas DataFrame
    frame = file_to_pandasframe(data_path)

    # Create list name of all fields
    all_field_names = frame.columns.values.tolist()

    # Search X, Y, Z, target and raw_classif fields
    field_x = None
    field_y = None
    field_z = None
    field_t = None

    for field in all_field_names:
        if field.casefold() == 'x':  # casefold() to be non case-sensitive
            field_x = field
        if field.casefold() == 'y':
            field_y = field
        if field.casefold() == 'z':
            field_z = field
        if field.casefold() == 'target':
            field_t = field

    # Target is mandatory for training
    if mode == 'training' and field_t is None:
        raise ValueError("A 'target' field is mandatory for training!")

    # Create target field if exist
    if field_t:
        target = frame.loc[:, field_t]
    else:
        target = None

    # Select only features fields by removing X, Y, Z and target fields
    sel_feat_names = all_field_names
    for field in [field_x, field_y, field_z, field_t]:
        if field:
            sel_feat_names.remove(field)

    # Sort data by field names
    sel_feat_names.sort()  # Sort to be compatible between formats

    # data without X, Y, Z and target fields
    data = frame.filter(sel_feat_names, axis=1)

    # Replace NAN values by median
    data.fillna(value=data.median(0), inplace=True)  # .median(0) computes median/col

    return data, target


def plot_pca(filename, pca, ft_names):
    """
    Create and save figure of principal components.
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
    :param data: Data to transform.
    :return: data_pca as pandas.core.frame.DataFrame
    """
    #  Create list of component names
    pca_compo_list = list()
    for idx, compo in enumerate(pca.components_):
        pca_compo_list.append('Principal_Compo_' + str(idx + 1))

    # Apply PCA transformation to the data
    data_pca = pca.transform(data)  # Becomes np.array
    data_pca = pd.DataFrame.from_records(data_pca, columns=pca_compo_list)  # Becomes pd.DataFrame

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


def nbr_pts(data_length, sample_size=None, mode='training'):
    """
    Set the number of point according the magnitude.
    :param sample_size: Float of the number of point for training.
    :param data_length: Total number of points in data file.
    :param mode: 'training', 'unsupervised' and 'prediction' modes.
    :return: nbr_pts: Integer of the number of points.
    """
    # Initiation
    magnitude = 1000000

    # Tests data_length and sample_size exist as number
    try:
        data_length = float(data_length)
    except ValueError as ve:
        raise ValueError("'data_length' parameter must be a number")

    if sample_size is not None:  # Check if the sample_size and is a number
        try:
            samples_size = float(sample_size)
        except ValueError as ve:
            sample_size = 0.05
            print("ValueError: sample size must be a number\n"
                  "sample size set to default: 0.05 Mpts.")

    # Crop data only on training mode
    if mode == 'training':
        # Sample size > Data size
        if sample_size is None or sample_size * magnitude >= data_length:
            int_nbr_pts = int(data_length)  # Number of points of entire point cloud
        else:
            int_nbr_pts = int(sample_size * magnitude)  # Number of points < sample size
    else:
        int_nbr_pts = int(data_length)

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
    :param pca_compo: Principal components of the PCA.
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
    Save the feature importance of RandomForest or GradientBoosting models into file.
    :param model: Pipeline with RandomForest or GradientBoosting Classifier
    :param feature_names: Names of the features.
    :param feature_filename: Filename and path of the feature importance figure file.
    :return:
    """
    # Get the feature importance
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

    # Plot the feature importances
    plt.figure()
    plt.title("Feature Importances")
    plt.barh(feature_names, importances_sorted, color='b')
    plt.xlabel("Feature importances")
    # plt.yticks(indices, feature_names)
    plt.ylim([-1, len(feature_names)])
    plt.ylabel("Features")
    plt.savefig(feature_filename, bbox_inches="tight")
