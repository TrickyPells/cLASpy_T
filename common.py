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
import yaml
import json
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
                   'raw_classification', 'flag_byte', 'synthetic', 'key_point',
                   'withheld', 'scan_angle_rank', 'user_data', 'point_source_id', 'pt_src_id']
point_format[1] = point_format[0] + gps_time
point_format[2] = point_format[0] + rgb
point_format[3] = point_format[0] + gps_time + rgb

# Point formats for LAS 1.3 to 1.4
point_format[4] = point_format[0] + gps_time + wavepacket
point_format[5] = point_format[0] + gps_time + rgb + wavepacket

# Point formats for LAS 1.4
point_format[6] = ['X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                   'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
                   'raw_classification', 'flag_byte', 'scan_direction_flag',
                   'edge_of_flight_line', 'classification', 'user_data',
                   'scan_angle_rank', 'point_source_id', 'pt_src_id', 'gps_time']
point_format[7] = point_format[6] + rgb
point_format[8] = point_format[6] + rgb + nir
point_format[9] = point_format[6] + wavepacket
point_format[10] = point_format[6] + rgb + nir + wavepacket

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def fullname_algo(algo):
    """
    Give the fullname of selected algorithm.
    :param algo: the selected algo.
    :return: the fullname of the algorithm
    """
    # Get fullname of algorithm according algo
    if algo == 'rf':
        algorithm = 'RandomForestClassifier'
    elif algo == 'gb':
        algorithm = 'GradientBoostingClassifier'
    elif algo == 'ann':
        algorithm = 'MLPClassifier'
    elif algo == 'kmeans':
        algorithm = 'KMeans'
    else:
        raise ValueError("Choose a machine learning algorithm ('--algo')!")

    return algorithm


def shortname_algo(algorithm):
    """
    Give the short name of selected algorithm
    :param algorithm: the selected algo.
    :return: the fullname of the algorithm
    """
    # Get short name of algorithm
    if algorithm == 'RandomForestClassifier':
        algo = 'rf'
    elif algorithm == 'GradientBoostingClassifier':
        algo = 'gb'
    elif algorithm == 'MLPClassifier':
        algo = 'ann'
    elif algorithm == 'KMeans':
        algo = 'kmeans'
    else:
        raise ValueError("Choose a machine learning algorithm ('--algo')!")

    return algo


def update_arguments(args):
    """
    Update the arguments from the config file given in args.config.
    :param args: the argument parser
    """
    # Open the config file
    args.config = os.path.normpath(args.config)
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    args.input_data = os.path.normpath(config['input_file'])
    args.output = os.path.normpath(config['output_folder'])
    args.samples = config['samples']
    args.algo = shortname_algo(config['algorithm'])
    args.parameters = config['parameters']
    args.features = config['feature_names']


def introduction(algorithm, file_path, folder_path=None):
    """
    Prompt the introduction, create folder to store results
    and return the start_time.
    :param algorithm: algorithm used.
    :param file_path: CSV or LAS file of used data.
    :param folder_path: the folder where to save result files.
    :return: raw_data, folder_path and start time.
    """
    # Prompt information about algorithm and file
    data_path = os.path.normpath(file_path)
    print("\n####### POINT CLOUD CLASSIFICATION #######\n"
          "Algorithm used: {}".format(algorithm))

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
    if folder_path is None:
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
        raise ValueError("Unknown Extension file!")

    return frame


def get_selected_features(features, temp_features):
    """
    :param features: the wanted features (selected by user).
    :param temp_features: all features in input_data except ('x', 'y', 'z' and 'target').
    :return: selected_features, the final selected features.
    """
    # Initialization
    selected_features = list()

    # Check if features is a list()
    if isinstance(features, list):
        features = [feature.casefold() for feature in features]
        for feature in temp_features:
            print("\nGet selected features:")
            if feature.casefold() in features:
                selected_features.append(feature)
                print(" - {} feature added".format(feature))
        # print comparison between given feature list and final selected features
        features.sort()
        selected_features.sort()
        print("\nLength of given feature list: {}".format(len(features)))
        print(features)
        print("\nLength of final selected features: {}".format(len(selected_features)))
        print("{}\n".format(selected_features))
        if len(features) == len(selected_features):
            print(" --> All required features are present!\n")
        else:
            raise ValueError("One or several features are missing in 'input_data'!")
    else:
        raise TypeError("Selected features must be a list of string!")

    return selected_features


def format_dataset(data_path, mode='training', features=None):
    """
    Format the input data as panda DataFrame. Exclude XYZ fields.
    :param data_path: path of the input data.
    :param mode: set the mode ['training', 'predict', 'segment'] to check mandatory 'target' field in case of training.
    :param features: the features selected to train the model or make predictions.
    :return: features_data and target as DataFrames.
    """
    # Load data into Pandas DataFrame
    frame = file_to_pandasframe(data_path)

    # Create list name of all fields
    all_features = frame.columns.values.tolist()

    # Search X, Y, Z, target and raw_classif fields
    field_x = None
    field_y = None
    field_z = None
    field_t = None

    for field in all_features:
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
    temp_features = all_features
    for field in [field_x, field_y, field_z, field_t]:
        if field:
            temp_features.remove(field)

    # Get only the selected features among temp_features
    if features is None:
        print("All features in input_data will be used!")
        selected_features = temp_features

    elif isinstance(features, str):
        features = yaml.safe_load(features)
        selected_features = get_selected_features(features, temp_features)

    elif isinstance(features, list):
        selected_features = get_selected_features(features, temp_features)

    else:
        raise TypeError("Selected features must be a list of string!")

    # Sort data by field names
    selected_features.sort()  # Sort to be compatible between formats

    # data without X, Y, Z and target fields
    data = frame.filter(selected_features, axis=1)

    # Replace NAN values by median
    data.fillna(value=data.median(0), inplace=True)  # .median(0) computes median by column

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


def number_of_points(data_length, sample_size=None, magnitude=1000000):
    """
    Set the number of point according the magnitude.
    :param sample_size: float of the number of point for training.
    :param data_length: total number of points in data file.
    :param magnitude: the order of magnitude.
    :return: int_nbr_pts: integer of the number of points.
    """
    # Tests data_length and sample_size exist as number
    try:
        data_length = float(data_length)
    except ValueError:
        raise ValueError("ValueError: 'data_length' parameter must be a number!")

    if sample_size is not None:  # Check if the sample_size is a number
        try:
            sample_size = float(sample_size)
        except ValueError:
            sample_size = 0.05
            print("ValueError: 'sample_size' must be a number!\n"
                  "Sample size set to default value: 0.05 Mpts.")

    # Sample size > Data size or not ?
    if sample_size is None or sample_size * magnitude >= data_length:
        int_nbr_pts = int(data_length)  # Number of points of entire point cloud
    else:
        int_nbr_pts = int(sample_size * magnitude)  # Sample size

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
    :param mode: 'training', 'predict' or 'segment' modes.
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
            report.write('\n\nNumber of points for training: {:,} pts'.format(data_len).replace(',', ' '))
            report.write('\nTrain size: {:,} pts'.format(train_len).replace(',', ' '))
            report.write('\nTest size: {:,} pts'.format(test_len).replace(',', ' '))

        # Write the number of point to predict
        if mode == 'predict':
            report.write('\n\nNumber of points to predict: {:,} pts'.format(data_len).replace(',', ' '))
            report.write('\nModel used: ' + model)

        # Write the number of point to segment
        if mode == 'segment':
            report.write('\n\nNumber of points to segment: {:,} pts'.format(data_len).replace(',', ' '))

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
