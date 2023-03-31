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
#             'cLASpy_Classes.py' from cLASpy_T library               #
#                    By Xavier PELLERIN LE BAS                        #
#                          August 2021                                #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description:                                                       #
#                                                                     #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------

import os
import joblib
import yaml
import laspy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# ------ VARIABLES --------
# -------------------------
# Version of cLASpy_Core
cLASpy_Core_version = '0.2.0'  # 0.2.0 version with classes

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
# -------- CLASSES --------
# -------------------------


class ClaspyTrainer:
    """
    ClaspyTrainer is basic class of cLASpy_T.
    Used to create object to train model according the selected algorithm.
    """

    def __init__(self, input_data, output_data=None, algo=None, algorithm=None,
                 parameters=None, features=None, grid_search=False, grid_param=None,
                 pca=None, n_jobs=-1, random_state=0, samples=None, scaler='Standard',
                 scoring='accuracy', train_ratio=0.5, png_features=False):
        """Initialize cLASpy_Trainer object"""

        # Set variables
        self.mode = 'training'
        self.has_target = False  # Boolean True if target field exists in data file
        self.start_time = datetime.now()  # Set the start time
        self.elapsed_time = None
        self.timestamp = self.start_time.strftime("%m%d_%H%M")  # Timestamp for file creation

        # Data
        self.data_path = os.path.normpath(input_data)  # Path to the datafile
        self.root_ext = os.path.splitext(self.data_path)  # Split path into root and extension
        self.data_type = self.root_ext[1]  # Get the type of data file ('.csv', or '.las')
        self.folder_path = output_data

        # Algorithm
        self.algo = algo
        self.algorithm = algorithm
        self.parameters = parameters
        self.features = features
        self.pca = pca
        self.samples = samples
        self.scaler_method = scaler
        self.scoring = scoring
        self.train_ratio = train_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Grid Search CV
        self.grid_search = grid_search
        self.grid_parameters = grid_param
        self.png_features = png_features

        # Set some variable members to None
        self.magnitude = 1000000  # Magnitude for number_of_points() and format_nbr_pts()
        self.classifier = None  # Classifier object
        self.conf_matrix = None  # Confusion matrix
        self.data = None  # All data except target values
        self.data_features = None  # Feature names from data file
        self.data_test = None  # Data to test model
        self.data_train = None  # Data to train model
        self.feat_importance = None  # Boolean to plot feature importance or not
        self.frame = None  # Loaded data as pandas.DataFrame
        self.model = None  # Model object
        self.model_to_load = None  # Model to load (path file)
        self.nbr_points = None  # Integer of number of points
        self.pipeline = None  # Pipeline object
        self.pipe_params = None  # Pipeline parameters
        self.report_filename = None  # Filename for the classification report
        self.results = None
        self.scaler = None  # Scaler object
        self.target = None  # Target values
        self.target_name = None  # name of target field in data file
        self.target_test = None  # Target to test model
        self.target_test_pred = None  # Prediction results after testing
        self.target_train = None  # Target to train model
        self.test_ratio = None  # Ratio test_size / (test_size + train_size)
        self.test_report = None
        self.test_size = None  # Size of the test dataset
        self.train_size = None  # Size of the train dataset

    def fullname_algo(self):
        """
        Return the fullname of the algorithm
        """
        # Get fullname of algorithm according algo
        if self.algo == 'rf':
            return 'RandomForestClassifier'
        elif self.algo == 'gb':
            return 'GradientBoostingClassifier'
        elif self.algo == 'ann':
            return 'MLPClassifier'
        else:
            raise ValueError("Choose a valid machine learning algorithm ('--algo')!")

    def shortname_algo(self):
        """
        Return the fullname of the algorithm
        """
        # Get short name of algorithm
        if self.algorithm == 'RandomForestClassifier':
            return 'rf'
        elif self.algorithm == 'GradientBoostingClassifier':
            return 'gb'
        elif self.algorithm == 'MLPClassifier':
            return 'ann'
        else:
            raise ValueError("Choose a valid machine learning algorithm ('--algo')!")

    def number_of_points(self):
        """
        Set the number of point according the magnitude.
        self.samples: float of the number of point for training.
        len(self.data): total number of points in data file.
        self.magnitude: the order of magnitude.
        Set self.samples as an integer of the number samples used.
        """
        # Check if self.samples is float or integer
        if isinstance(self.samples, float):
            if self.samples * self.magnitude >= len(self.data):
                self.samples = len(self.data)
            else:
                self.samples = self.samples * self.magnitude
        elif isinstance(self.samples, int):
            if self.samples >= len(self.data):
                self.samples = len(self.data)
        else:  # if self.samples is not float or int
            self.samples = len(self.data)

    def format_nbr_pts(self):
        """
        Format the nbr_pts as string for the filename.
        self.samples: Integer of the number of points used to format in string.
        :return: String of the point number write according the magnitude suffix.
        """
        # Format as Mpts or kpts according number of points
        if self.samples >= self.magnitude:  # number of points > 1Mpts
            str_nbr_pts = str(np.round(self.samples / self.magnitude, 1))
            if str_nbr_pts.split('.')[-1][0] == '0':  # round number if there is zero after point ('xxx.0x')
                str_nbr_pts = str_nbr_pts.split('.')[0]
            else:
                str_nbr_pts = '_'.join(str_nbr_pts.split('.'))  # replace '.' by '_' if not rounded
            str_nbr_pts = str_nbr_pts + 'Mpts_'

        else:  # number of points < 1M
            str_nbr_pts = int(self.samples / 1000)
            str_nbr_pts = str(str_nbr_pts) + 'kpts_'

        return str_nbr_pts

    def set_classifier(self):
        """
        Set the classifier according to the selected algorithm.
        """
        # Algorithm
        if self.algo is None and self.algorithm is not None:
            self.algo = self.shortname_algo()
        elif self.algo is not None and self.algorithm is None:
            self.algorithm = self.fullname_algo()
        else:
            raise ValueError("Choose a valid machine learning algorithm!")

        # PCA
        if isinstance(self.pca, int) or self.pca is None:
            pass
        else:
            raise TypeError("PCA must be a integer or None!")

        # Samples
        if isinstance(self.samples, int) or isinstance(self.samples, float):
            pass
        elif self.samples is None:
            pass
        else:
            raise TypeError("Samples must be a number (integer, float) or None!")

        # feature importance
        if self.grid_search or self.algorithm == 'MLPClassifier':
            self.png_features = False
        else:
            self.png_features = self.png_features

        # Check parameters exists
        if self.parameters is not None:
            if isinstance(self.parameters, str):
                self.parameters = yaml.safe_load(self.parameters)
            elif isinstance(self.parameters, dict):
                self.parameters = self.parameters
            else:
                raise TypeError("Algorithm parameters must be dict or string type!")

        # Set the chosen learning classifier
        if self.algorithm == 'RandomForestClassifier':
            self.set_random_forest()

        elif self.algorithm == 'GradientBoostingClassifier':
            self.set_gradient_boosting()

        elif self.algorithm == 'MLPClassifier':
            self.set_mlp_classifier()

        else:
            raise ValueError("No valid classifier!")

    def check_parameters(self):
        """
        Check if the given parameters match with the given classifier
        and set the classifier with the well defined parameters.
        self.parameters: Parameters to check in dict.
        """
        # Get the type of classifier
        if self.algorithm == 'RandomForestClassifier':
            double_float = ['min_weight_fraction_leaf']

        elif self.algorithm == 'GradientBoostingClassifier':
            double_float = ['min_weight_fraction_leaf',
                            'learning_rate', 'subsample']

        elif self.algorithm == 'MLPClassifier':
            double_float = ['alpha', 'learning_rate_init', 'power_t',
                            'beta_1', 'beta_2', 'epsilon']
        else:
            double_float = list()

        # Check if the parameters are valid for the given classifier
        for key in self.parameters.keys():  # To change str in float
            if key in double_float:
                self.parameters[key] = float(self.parameters[key])
            try:
                temp_dict = {key: self.parameters[key]}
                self.classifier.set_params(**temp_dict)
            except ValueError:
                print("ValueError: Invalid parameter '{}' for {}, "
                      "it was skipped!".format(str(key), self.algorithm))

    def check_grid_parameters(self):
        """
        Check if the given grid_params match with the given classifier.
        self.pipeline: Pipeline set with 'scaler' and 'classifier' (at least).
        self.grid_parameters: Grid parameters to check in dict.
        :return: Grid parameters with only the well defined parameters for the pipeline.
        """
        # String report of if bad parameter
        check_grid_param_str = "\n"

        # Get the list of valid parameters from the classifier
        param_names = self.pipeline.named_steps['classifier']._get_param_names()

        # Keep only the valid parameters
        well_params = dict()  # Dictionary with only good parameters
        if self.grid_parameters is not None:
            for key in self.grid_parameters.keys():
                if key in param_names:
                    well_params[key] = self.grid_parameters[key]
                else:
                    check_grid_param_str += "GridSearchCV: Invalid parameter '{}' for {}, it was skipped!" \
                        .format(str(key), self.algorithm)

        # Check if well_params is empty dict and set predefined parameters
        if not well_params:
            if self.algorithm == 'RandomForestClassifier' or self.algorithm == 'GradientBoostingClassifier':
                well_params = {'n_estimators': [50, 100, 500],
                               'max_depth': [8, 11, 14],
                               'min_samples_leaf': [100, 500, 1000]}
            elif self.algorithm == 'MLPClassifier':
                well_params = {'hidden_layer_sizes': [(25, 25), (25, 25, 25), (25, 50, 25)],
                               'activation': ('tanh', 'relu'),
                               'alpha': [0.0001, 0.01, 1.0]}

        # Rename parameter dictionary keys for Pipeline
        self.pipe_params = dict()
        for key in well_params.keys():
            new_key = 'classifier__' + key
            self.pipe_params[new_key] = well_params[key]

        return check_grid_param_str

    def set_random_forest(self):
        """
        Set the classifier as RandomForestClassier.
        self.parameters: A dict with the parameters to set up.
        """
        # Set the classifier
        if isinstance(self.parameters, dict):
            self.classifier = RandomForestClassifier()
            self.check_parameters()  # Check and set parameters

        else:
            self.classifier = RandomForestClassifier(n_estimators=100,
                                                     max_depth=8,
                                                     min_samples_leaf=500,
                                                     n_jobs=-1,
                                                     random_state=0)

    def set_gradient_boosting(self):
        """
        Set the classifier as GradientBoostingClassifier.
        self.parameters: A dict with the parameters to set up.
        """
        # Set the classifier
        if isinstance(self.parameters, dict):
            self.classifier = GradientBoostingClassifier()
            self.check_parameters()  # Check and set parameters
        else:
            self.classifier = GradientBoostingClassifier(loss='deviance',
                                                         n_estimators=100,
                                                         max_depth=3,
                                                         min_samples_leaf=1000,
                                                         random_state=0)

    def set_mlp_classifier(self):
        """
        Set the classifier as MLPClassifier.
        self.parameters: a dict with the parameters to set up.
        """
        # Set the classifier
        if isinstance(self.parameters, dict):
            self.classifier = MLPClassifier()
            self.check_parameters()

        else:
            self.classifier = MLPClassifier(hidden_layer_sizes=(50, 50),
                                            activation='relu',
                                            solver='adam',
                                            alpha=0.0001,
                                            max_iter=10000,
                                            random_state=0)

    def training_gridsearch(self, verbose=True):
        """
        Train model with GridSearchCV meta-estimator according the chosen learning algorithm.
        self.pipeline: set the algorithm to train the model.
        self.data_train: the training dataset.
        self.target_train: the targets corresponding to the training dataset.
        self.random_state: Set the random_state for the StratifiedShuffleSplit.
        self.grid_parameters: the parameters for the GridSearchCV.
        self.n_jobs: the number of CPU used.
        self.scoring: set the scorer according scikit-learn documentation.
        :return: Report of the training.
        """
        # Set verbose
        if verbose:
            verb = 1
        else:
            verb = 0

        # String of gridsearch
        gridsearch_str = "\n"

        # Set cross_validation method with train_size 80% and validation_size 20%
        cross_val = StratifiedShuffleSplit(n_splits=5,
                                           train_size=0.8,
                                           test_size=0.2,
                                           random_state=self.random_state)

        # Set the GridSearchCV
        gridsearch_str += "\tSearching best parameters...\n"
        self.model = GridSearchCV(self.pipeline,
                                  param_grid=self.pipe_params,
                                  n_jobs=self.n_jobs,
                                  cv=cross_val,
                                  scoring=self.scoring,
                                  verbose=verb,
                                  error_score=np.nan)

        # Training the model to find the best parameters
        self.model.fit(self.data_train, self.target_train)
        self.results = pd.DataFrame(self.model.cv_results_)
        gridsearch_str += "\tBest score: {0:.4f}\n".format(self.model.best_score_)
        gridsearch_str += "\tBest parameters: {}\n".format(self.model.best_params_)
        gridsearch_str += "\tModel trained!\n"

        return gridsearch_str

    def training_nogridsearch(self, verbose=True):
        """
        Train model with cross-validation according the chosen classifier.
        self.pipeline: set the algorithm to train the model.
        self.data_train: the training dataset.
        self.target_train: the targets corresponding to the training dataset.
        self.random_state: Set the random_state for the StratifiedShuffleSplit.
        self.n_jobs: the number of used CPU.
        self.scoring: set the scorer according scikit-learn documentation.
        :return: Report of the training.
        """
        # Set verbose
        if verbose:
            verb = 2
        else:
            verb = 0

        # String of CrossValidation
        crossval_str = "\n"

        # Set cross_validation method with train_size 80% and validation_size 20%
        cross_val = StratifiedShuffleSplit(n_splits=5,
                                           train_size=0.8,
                                           test_size=0.2,
                                           random_state=self.random_state)

        # Get the training scores
        self.results = cross_validate(self.pipeline,
                                      self.data_train,
                                      self.target_train,
                                      cv=cross_val,
                                      n_jobs=self.n_jobs,
                                      scoring=self.scoring,
                                      verbose=verb,
                                      return_estimator=True)

        crossval_str += "\n\tTraining model scores with cross-validation:\n\t{}\n".format(self.results["test_score"])

        # Set the classifier with training_data and target
        self.model = self.results["estimator"][1]  # Get the second estimator from the cross_validation
        self.results = self.results["test_score"]
        crossval_str += "\nModel trained!\n"

        return crossval_str

    def add_precision_recall(self):
        """
        Compute precision, recall and global accuracy from confusion matrix.
        :self.conf_matrix: The confusion matrix as a numpy.array.
        :return conf_mat_up: Confusion matrix wth precision, recall and global accuracy
        """
        # Change type array(int) as array(float)
        self.conf_matrix = np.array(self.conf_matrix).astype(float)

        n_rows_cols = self.conf_matrix.shape[0]
        rows_sums = np.sum(self.conf_matrix, axis=1)
        cols_sums = np.sum(self.conf_matrix, axis=0)

        # Compute the recalls
        recalls = list()
        for row in range(0, n_rows_cols):
            try:
                recalls.append((float(self.conf_matrix[row, row]) / float(rows_sums[row])))
            except ZeroDivisionError:
                recalls.append(np.nan)
        self.conf_matrix = np.insert(self.conf_matrix, n_rows_cols, recalls, axis=1)

        # Compute the precision
        precisions = list()
        for col in range(0, n_rows_cols):
            try:
                precisions.append(float(self.conf_matrix[col, col]) / float(cols_sums[col]))
            except ZeroDivisionError:
                precisions.append(np.nan)

        # Compute overall accuracy
        try:
            precisions.append(float(np.trace(self.conf_matrix) / float(np.sum(self.conf_matrix))))
        except ZeroDivisionError:
            precisions.append(np.nan)

        self.conf_matrix = np.insert(self.conf_matrix, n_rows_cols, precisions, axis=0)

        # Return the new confusion matrix
        self.conf_matrix = pd.DataFrame(self.conf_matrix).round(decimals=3)
        return self.conf_matrix

    def save_feature_importance(self):
        """
        Save the feature importance of RandomForest or GradientBoosting models into file.
        self.model: Pipeline with RandomForest or GradientBoosting Classifier
        :return: The dictionary of the feature importances.
        """
        # Get features names and feature importances
        feature_names = self.data.columns.values.tolist()
        importances = self.model.named_steps['classifier'].feature_importances_

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

        feature_filename = self.report_filename + '_feat_imp.png'
        plt.savefig(feature_filename, bbox_inches="tight")

        return feature_imp_dict

    def introduction(self, verbose=True):
        """
        Return the introduction in a string
        """
        introduction = str("\nAlgorithm used: {}\n".format(self.algorithm))
        if self.data_type == '.csv':
            introduction += "Path to CSV file: {}\n".format(self.data_path)
        elif self.data_type == '.las':
            introduction += "Path to LAS file: {}\n".format(self.data_path)
        else:
            self.introduction += "Unknown Extension file !\n"

        # Create a folder to store models, reports and predictions
        introduction += "\nCreate a new folder to store the result files..."
        if self.folder_path is None:
            self.folder_path = self.root_ext[0]  # remove extension to give the folder path

        try:
            os.makedirs(self.folder_path)  # Using file path to create new folders recursively
            introduction += " Done.\n"
        except (TypeError, FileExistsError):
            introduction += " Folder already exists.\n"

        # Add info of point cloud from file
        introduction += self.point_cloud_info(verbose=True)

        if verbose:
            return introduction

    def point_cloud_info(self, verbose=True):
        """
        Return the point cloud info
        """
        # Point cloud info string
        point_cloud_info = "\n"

        # Load data into DataFrame
        if self.data_type == '.csv':
            with open(self.data_path, 'r') as file:
                self.nbr_points = sum(1 for line in file) - 1
            point_cloud_info += "Number of points: {:,}\n".format(self.nbr_points)

        elif self.data_type == '.las':
            las = laspy.open(self.data_path, mode='r')
            point_cloud_info += "LAS Version: {}\n".format(las.header.version)
            point_cloud_info += "LAS point format: {}\n".format(las.header.point_format.id)
            self.nbr_points = las.header.point_count
            point_cloud_info += "Number of points: {:,}\n".format(self.nbr_points)
            las.close()

        else:
            point_cloud_info += "Unknown Extension file!\n"

        if verbose:
            return point_cloud_info

    def get_data_features(self):
        """
        Get all features (or field names) from data file.
        """
        # Get all features from header
        if self.data_type == '.csv':
            csv_frame = pd.read_csv(self.data_path, sep=',', header='infer', nrows=10)
            # Clean up the header built by CloudCompare ('//X')
            for field in csv_frame.columns.values.tolist():
                if field == '//X':
                    csv_frame = csv_frame.rename(columns={"//X": "X"})
            data_features = csv_frame.columns.values.tolist()

        elif self.data_type == '.las':
            las = laspy.open(self.data_path, mode='r')
            data_features = list(las.header.point_format.dimension_names)
            las.close()

        else:
            raise TypeError("Unrecognized file extension!")

        return data_features

    def get_selected_features(self, data_features, verbose=True):
        """
        self.features: the wanted features (selected by user).
        :param data_features: all features in input_data.
        :param verbose: Set verbose as True or False.
        :return: the final selected features and string report
        """
        # Initialization
        selected_feat_str = "\n"  # String to report info
        selected_features = list()  # The final list of all features found in input data

        # Check if features is a list()
        if isinstance(self.features, list):
            selected_feat_str += "\nGet selected features:"
            for feature in self.features:
                for dt_feature in data_features:
                    # Compare data_feature with 'space' replaced by '_'
                    if dt_feature.replace(' ', '_').casefold() == feature.casefold():
                        selected_features.append(dt_feature)  # Put feature with 'space'
                        selected_feat_str += " - {} asked --> {} found\n".format(feature, dt_feature)

            selected_feat_str += "\nNumber of wanted features: {}\n".format(len(self.features))
            selected_feat_str += "Number of final selected features: {}\n".format(len(selected_features))

            if len(self.features) == len(selected_features):
                self.features = selected_features  # Selected feature with 'space', not '_'
                selected_feat_str += " --> All required features are present!\n"
            else:
                differences = list(set(self.features) - set(selected_features))
                raise ValueError("{} features are missing in 'input_data'!".format(differences))
        else:
            raise TypeError("Selected features must be a list of string!")

        if verbose:
            return selected_feat_str

    def load_data_csv(self):
        """
        Load data from CSV file according asked features (self.features)
        or all features as default features, except X, Y and Z.
        Extracts target field if exists.
        :return: data and target as pandas.DataFrame
        """
        # If asked features is empty, load default features (all except X, Y, Z)
        if self.features is None:
            self.features = list()
            # Remove X, Y and Z fields from self.data_features
            for field in self.data_features:
                if field.casefold() not in ['x', 'y', 'z']:  # casefold() -> non case-sensitive
                    self.features.append(field)  # if 'target' exists, will be in self.features

        # Load data with pandas.read_csv()
        data = pd.read_csv(self.data_path,
                           sep=',',
                           header='infer',
                           usecols=self.features)  # dtype=feature_dtype dict like LAS

        # Extract 'target' from data frame
        target = None
        if self.has_target:
            target = pd.DataFrame.loc[:, self.target_name]  # use dtype uint8
            data.drop(columns=self.target_name, inplace=True)

        return data, target

    def load_data_las(self):
        """
        Load data from LAS file according asked features (self.features)
        or extra_dimensions as default features.
        Extracts target field if exists.
        :return: data and target pandas.DataFrames
        """
        # Read LAS file
        las = laspy.read(self.data_path)

        # If asked features is empty, load LAS Extra Dimensions
        if self.features is None:
            points_selected_dimensions = las.points[list(las.header.point_format.extra_dimension_names)]
        else:
            # Get LAS points from selected features
            points_selected_dimensions = las.points[self.features]

        # Create DataFrame with np.float32 for features
        data = pd.DataFrame(points_selected_dimensions.array).astype(np.float32)

        # Extract 'target' as dataframe
        target = None
        if self.has_target:
            data.drop(columns=self.target_name, inplace=True)  # Drop target field from data
            target = pd.DataFrame(las.points[[self.target_name]].array)

        return data, target

    def format_dataset(self, verbose=True):
        """
        Format dataset from CSV and LAS file as pandas Dataframe
        """
        # Report string
        format_data_str = "\n"

        # Get all feature names from data file
        self.data_features = self.get_data_features()

        # Get default features when asked features are empty (self.features)
        if self.features is None:  # Use all features except LAS standard fields and X, Y, Z
            format_data_str += "All features in input_data will be used!\n" \
                               "Except X, Y, Z and LAS standard dimensions!\n"

        # If asked features exists (self.features), check all are presents in self.data_features
        elif isinstance(self.features, str):
            self.features = yaml.safe_load(self.features)  # asked features from str to list
            format_data_str += self.get_selected_features(self.data_features)

        elif isinstance(self.features, list):
            format_data_str += self.get_selected_features(self.data_features)

        else:
            raise TypeError("Selected features must be a list of string!")

        # Check if 'target' field exists in self.data_features
        for field in self.data_features:
            if field.casefold() == 'target':
                self.has_target = True
                self.target_name = field

        # Target is mandatory for training
        if self.has_target is False and not isinstance(self, ClaspyPredicter) and not isinstance(self, ClaspySegmenter):
            raise ValueError("A 'target' field is mandatory for training!")

        # Load data according asked features from CSV or LAS
        if self.data_type == '.csv':
            self.data, self.target = self.load_data_csv()

        if self.data_type == '.las':
            self.data, self.target = self.load_data_las()

        # Replace NAN values by median
        self.data.fillna(value=self.data.median(0), inplace=True)  # .median(0) computes median by column

        # Set filename for output result files
        self.number_of_points()
        str_nbr_pts = self.format_nbr_pts()
        self.report_filename = str(
            self.folder_path + '/' + self.mode[0:5] + '_' + self.algo + str_nbr_pts + str(self.timestamp))

        if verbose:
            return format_data_str

    def split_dataset(self, verbose=True):
        """
        Split the data and target in data_train, data_test, target_train and target_test.
        self.data_values: the np.ndarray with the data features.
        self.target_values: the np.ndarray with the target.
        self.random_state: set the random_state for the split dataset.
        self.train_ratio: Ratio of the size of training dataset.
        self.samples: Number of samples beyond which the dataset
        is split with two integers, for train_size and test_size.
        The samples is paired with train_ratio and test_ratio.
        :return: data_train, data_test, target_train and target_test as np.ndarray,
        the string of the process.
        """
        # String Split of dataset
        split_dataset_str = "\n"

        # Random state
        split_dataset_str += "Random state to split data: {}\n".format(self.random_state)

        # Set the train_size and test_size according sample size
        self.test_ratio = 1. - self.train_ratio
        self.train_size = int(self.samples * self.train_ratio)
        self.test_size = int(self.samples * self.test_ratio)

        self.data_train, self.data_test, self.target_train, self.target_test = \
            train_test_split(self.data, self.target,
                             random_state=self.random_state,
                             train_size=self.train_size,
                             test_size=self.test_size,
                             stratify=self.target)

        # Convert target_train and target_test column-vectors as 1d array
        # self.target_train = self.target_train.reshape(self.train_size)
        # self.target_test = self.target_test.reshape(self.test_size)

        split_dataset_str += "\tNumber of used points: {:,} pts\n". \
            format(self.train_size + self.test_size).replace(',', ' ')
        split_dataset_str += "\tSize of train|test datasets: {:,} pts | {:,} pts\n". \
            format(self.train_size, self.test_size).replace(',', ' ')

        if verbose:
            return split_dataset_str

    def set_scaler_pca(self, verbose=True):
        """
        Set the scaler according to different methods: 'Standard', 'Robust', 'MinMax'.
        self.scaler: Set method to scale dataset.
        """
        # String Scaler dataset
        scale_dataset_str = "\n"

        # Set the data scaling according the chosen method
        if self.scaler_method == 'Standard':
            self.scaler = StandardScaler()  # Scale data with mean and std
        elif self.scaler_method == 'Robust':
            self.scaler = RobustScaler()  # Scale data with median and interquartile
        elif self.scaler_method == 'MinMax':
            self.scaler = MinMaxScaler()  # Scale data between 0-1 for each feature and translate (mean=0)
        else:
            self.scaler = StandardScaler()
            scale_dataset_str += "\nWARNING:\nScaling method '{}' was not recognized. " \
                                 "Replaced by 'StandardScaler' method.\n".format(str(self.scaler_method))

        # Create Pipeline for GridSearchCV or simple CrossValidation
        if self.pca == 0:
            self.pca = None

        if self.pca is not None:
            if isinstance(self.pca, int):
                self.pca = PCA(n_components=self.pca)
                self.pipeline = Pipeline([("scaler", self.scaler),
                                          ("pca", self.pca),
                                          ("classifier", self.classifier)])
            else:
                self.pipeline = Pipeline([("scaler", self.scaler), ("classifier", self.classifier)])
                scale_dataset_str += 'PCA must None or int type. PCA will not be set!'
        else:
            self.pipeline = Pipeline([("scaler", self.scaler), ("classifier", self.classifier)])

        if verbose:
            return scale_dataset_str

    def train_model(self, verbose=True):
        """
        Perform training with GridSearch CV or simple CrossValidation.
        :param verbose: Set it True to return report as string
        """
        # String Train model
        train_model_str = "\n"

        # Check for GridSearchCV or CrossValidation
        if self.grid_search:
            train_model_str += "Random state for the StratifiedShuffleSplit: {}\n".format(self.random_state)

            # Check self.grid_parameters exists
            if self.grid_parameters is not None:
                if isinstance(self.grid_parameters, str):
                    self.grid_parameters = yaml.safe_load(self.grid_parameters)
                elif isinstance(self.grid_parameters, dict):
                    pass
                else:
                    self.grid_parameters = None

            # Check self.grid_parameters are valid
            train_model_str += self.check_grid_parameters()  # return the string report of checking

            # Get model
            train_model_str += self.training_gridsearch(verbose=verbose)

        else:
            train_model_str += "Random state for the StratifiedShuffleSplit: {}".format(self.random_state)

            # Get model
            train_model_str += self.training_nogridsearch(verbose=False)

        # Importance of each feature in RF and GB
        if self.png_features:
            self.feat_importance = self.save_feature_importance()
            train_model_str += "\nFeature importances save as PNG file.\n"
        else:
            self.feat_importance = self.data.columns.values.tolist()

        if verbose:
            return train_model_str

    def confusion_matrix(self, verbose=True):
        """
        Compute precision matrix and other statistics.
        :return: Report as str
        """
        # String report of the confusion matrix
        confusion_str = "\n"

        if isinstance(self, ClaspyPredicter):
            self.conf_matrix = confusion_matrix(self.target.values, self.y_proba.transpose()[0])
            self.conf_matrix = self.add_precision_recall()  # add statistics to confusion matrix report
            self.test_report = classification_report(self.target.values, self.y_proba.transpose()[0], zero_division=0)
        else:
            # Make prediction over test first
            self.target_test_pred = self.model.predict(self.data_test)
            self.conf_matrix = confusion_matrix(self.target_test, self.target_test_pred)
            self.conf_matrix = self.add_precision_recall()  # add statistics to confusion matrix report
            self.test_report = classification_report(self.target_test, self.target_test_pred, zero_division=0)

        confusion_str += self.test_report

        if verbose:
            return confusion_str

    def save_model(self, verbose=True):
        """
        Save algorithm name, used features and model
        """
        # String report for saving model
        save_model_str = "\n"

        # Model_dilename and model_dict
        model_filename = self.report_filename + '.model'
        model_dict = dict()
        model_dict['algorithm'] = self.algorithm
        model_dict['feature_names'] = self.data.columns.values.tolist()
        model_dict['model'] = self.model

        # Use joblib to save in model file
        joblib.dump(model_dict, model_filename)

        save_model_str += "Model path: {}/\n".format('/'.join(model_filename.split('/')[:-1]))
        save_model_str += "Model file: {}".format(model_filename.split('/')[-1])

        if verbose:
            return save_model_str

    def classification_report(self, verbose=True):
        """
        Write the report of training or predictions in .TXT file.
        :param verbose: Return report string if True
        self.report_filename: Entire path and filename without extension.
        self.mode: 'training', 'predict' or 'segment' modes.
        self.algorithm: Algorithm used for training or predictions.
        :self.data_path: Data file used to make training or predictions.
        self.start_time: Time when the script began.
        elapsed_time: Time spent between begin and end.
        self.feat_importance: List or Dict of the all feature names.
        self.scaler: Method used to scale data.
        self.samples: Number of samples used.
        self.pca: Principal components of the PCA.
        self.model: Model used to make predictions.
        self.results: Results of the GridSearchCV or Cross Validation.
        self.conf_matrix: Confusion matrix if target field exists.
        self.test_report: Report of all score if target field exists.
        :return: elapsed_time
        """
        # Get the model parameters to print in the report
        if self.mode == 'segment':
            applied_parameters = ["{}: {}".format(param, self.classifier.get_params()[param])
                                  for param in self.classifier.get_params()]
        else:
            applied_parameters = ["{}: {}".format(param, self.model.get_params()[param])
                                  for param in self.model.get_params()]

        # Compute elapsed time
        self.elapsed_time = datetime.now() - self.start_time

        # Write the header of the report file
        with open(self.report_filename + '.txt', 'w', encoding='utf-8') as report:
            report.write('Report of ' + self.algorithm + ' ' + self.mode)
            report.write('\n\nDatetime: ' + self.start_time.strftime("%Y-%m-%d %H:%M:%S"))
            report.write('\nFile: ' + self.data_path)
            report.write('\n\nScaling method:\n{}'.format(self.scaler))

            # Write features depending if it's list or dict
            if self.feat_importance:
                if isinstance(self.feat_importance, list):
                    report.write("\n\nFeatures:\n" + "\n".join(self.feat_importance))
                if isinstance(self.feat_importance, dict):
                    report.write("\n\nFeatures,\tImportances:\n")
                    for key in self.feat_importance:
                        report.write("{},\t{:.5f}\n".format(str(key), self.feat_importance[key]))

            # Write the train and test size
            if self.mode == 'training':
                report.write('\n\nNumber of points for training: {:,} pts'.format(self.samples).replace(',', ' '))
                report.write('\nTrain size: {:,} pts'.format(len(self.data_train)).replace(',', ' '))
                report.write('\nTest size: {:,} pts'.format(len(self.data_test)).replace(',', ' '))

            # Write the number of point to predict
            if self.mode == 'predict':
                report.write('\n\nNumber of points to predict: {:,} pts'.format(len(self.data)).replace(',', ' '))
                report.write('\nModel used: ' + self.model_to_load)

            # Write the number of point to segment
            if self.mode == 'segment':
                report.write('\n\nNumber of points to segment: {:,} pts'.format(len(self.data)).replace(',', ' '))

            # if self.pca is not None:
            #     report.write('\n\nPCA Components:\n')
            #     report.write(pca_compo)

            # Write the GridSearchCV results
            if self.results is not None:
                if self.grid_search:
                    report.write('\n\n\nResults of the GridSearchCV:\n')
                    report.write(self.results.to_string(index=False))

                # Write the Cross validation results
                else:
                    report.write('\n\n\nResults of the Cross-Validation:\n')
                    report.write(pd.DataFrame(self.results).to_string(index=False, header=False))

            # Write applied parameters
            report.write('\n\n\nParameters:\n' + '\n'.join(applied_parameters))

            # Write the confusion matrix results
            if self.conf_matrix is not None:
                report.write('\n\n\nConfusion Matrix:\n')
                report.write(self.conf_matrix.to_string())

            # Write the score report of the classification
            if self.test_report:
                report.write('\n\n\nClassification Report:\n')
                report.write(self.test_report)

            # Write elapsed time
            if self.mode == 'training':
                report.write('\n\nModel trained in {}'.format(self.elapsed_time))
            else:
                report.write('\n\nPredictions done in {}'.format(self.elapsed_time))

        if verbose:
            if self.mode == 'training':
                return "\nTraining done in {}\n".format(self.elapsed_time)
            elif self.mode == 'predict':
                return "\nPredictions done in {}\n".format(self.elapsed_time)
            elif self.mode == 'segment':
                return "\nSegmentation done in {}\n".format(self.elapsed_time)


class ClaspyPredicter(ClaspyTrainer):
    """
    ClaspyPredicter create object to predict classes according the selected model.
    """

    def __init__(self, input_data, model, output_data=None, n_jobs=-1, samples=None):
        """Initialize the ClaspyPredicter"""
        ClaspyTrainer.__init__(self, input_data=input_data, output_data=output_data,
                               n_jobs=n_jobs, samples=samples)

        # Set specific variables for ClaspyPredicter
        self.pca_compo = None
        self.mode = 'predict'
        self.model_to_load = model
        self.feat_importance = None
        self.results = None

    def load_model(self, verbose=True):
        """
        Load model from file.model
        :param verbose: Set to True to get return string
        :return load_model_str: verbose return
        """
        # Set vrebose return
        load_model_str = "\n"

        # Load model
        if isinstance(self.model_to_load, str):
            self.model = joblib.load(self.model_to_load)
        else:
            raise TypeError("'model' to import must be a string of the path!")

        # Retrieve algorithm name, features names and model
        self.algorithm = self.model['algorithm']
        self.algo = self.shortname_algo()
        self.features = self.model['feature_names']
        self.model = self.model['model']

        # Check if model created by GridSearchCV or Pipeline
        if isinstance(self.model, GridSearchCV):
            self.model = self.model.best_estimator_
        elif isinstance(self.model, Pipeline):
            pass
        else:
            raise IOError('Loading model failed!\n'
                          'Model to load must be GridSearchCV or Pipeline!')

        # Fill scaler and pca (if any)
        self.scaler = self.model.named_steps['scaler']
        try:
            self.pca = self.model.named_steps['pca']
        except KeyError:
            self.pca = None
            load_model_str += "\tAny PCA data to load from model.\n"

        if verbose:
            return load_model_str

    def scale_dataset(self, verbose=True):
        """Scale dataset according the scaler and PCA retrieved"""
        # Set string for scale_dataset$
        scale_str = "\n"

        # Transform data according scaler
        self.data = self.scaler.transform(self.data)
        self.data = pd.DataFrame.from_records(self.data, columns=self.features)

        # Transform data if PCA exist
        if self.pca is not None:
            self.data = self.pca.transform(self.data)
            self.data = pd.DataFrame.from_records(self.data, columns=self.pca.components_)
            self.pca_compo = np.array2string(self.pca.components_)
            scale_str += "Scale dataset with Scaler and PCA transforms\n"
        else:
            scale_str += "Scale dataset with Scaler transform\n"

        # return verbose
        if verbose:
            return scale_str

    def predict(self, verbose=True):
        """Make predictions on the scaled data"""
        # Set string for predictions
        predict_str = "\n"

        # Make prediction
        self.y_proba = self.model.named_steps['classifier'].predict_proba(self.data)

        # Get the best probability and the corresponding class
        y_best_proba = np.amax(self.y_proba, axis=1)
        y_best_class = np.argmax(self.y_proba, axis=1)

        # Add best proba and bet class to probability per class
        self.y_proba = np.insert(self.y_proba, 0, y_best_proba, axis=1)
        self.y_proba = np.insert(self.y_proba, 0, y_best_class, axis=1)

        # If target is present, get the confusion matrix
        if self.target is not None:
            predict_str += "Target field find -> Create confusion matrix\n"
            predict_str += self.confusion_matrix(verbose=True)
        else:
            self.conf_matrix = None
            self.test_report = None

        if verbose:
            return predict_str

    def save_predictions(self, verbose=True):
        """Save the classified point cloud"""
        # String for saving point cloud
        save_pt_cloud_str = "\n"
        save_pt_cloud_str += self.report_filename

        # Set header for the predictions
        if len(self.y_proba.shape) > 1 and self.y_proba.shape[1] > 2:
            # Get number of class in prediction array (number of column - 2)
            numb_class = self.y_proba.shape[1] - 2
            pred_header = ['Prediction', 'BestProba'] + ['ProbaClass_' + str(cla) for cla in range(0, numb_class)]
        else:
            pred_header = ['Prediction']

        predictions = pd.DataFrame(self.y_proba, columns=pred_header, dtype='float32').round(decimals=4)

        if self.data_type == '.csv':
            # Join predictions to self.frame (input_data)
            self.frame = self.frame.join(predictions)
            self.frame.to_csv(self.report_filename + '.csv', sep=',', header=True, index=False)
        elif self.data_type == '.las':
            # Reopen original las file
            output_las = laspy.read(self.data_path)

            # Create ExtraBytesParams for additonnal dimensions
            extrabytes_list = list()
            extrabytes_list.append(laspy.ExtraBytesParams(name='Prediction',
                                                          type=np.uint16,
                                                          description="Prediction done by the model"))

            dimensions = predictions.columns.values.tolist()  # Get liste of dimensions
            if predictions.shape[1] > 1:
                dimensions.remove('Prediction')  # Remove prediction dimension already added
                for dim in dimensions:
                    extrabytes_list.append(laspy.ExtraBytesParams(name=dim,
                                                                  type=np.float32,
                                                                  description='Probability for this class'))

            # Add Extra dimensions at once
            output_las.add_extra_dims(extrabytes_list)

            # Fill output_las with new data
            for dim in predictions.columns.values.tolist():
                output_las[dim] = predictions[dim].values

            # Write output_las file
            output_las.write(self.report_filename + '.las')

        if verbose:
            return save_pt_cloud_str


class ClaspySegmenter(ClaspyTrainer):
    """
    ClaspySegmenter create object to segment data into clusters according the selected algorithm.
    """

    def __init__(self, input_data, output_data=None, parameters=None, features=None,
                 pca=None, n_jobs=-1, random_state=0, samples=None, scaler='Standard'):
        """Initialize the ClaspySegmenter"""
        ClaspyTrainer.__init__(self, input_data=input_data, output_data=output_data,
                               parameters=parameters, features=features, pca=pca, n_jobs=n_jobs,
                               random_state=random_state, samples=samples, scaler=scaler)

        # Set specific varaibles for ClaspySegmenter
        self.mode = 'segment'

        # Algorithm
        self.algo = 'kmeans'
        self.algorithm = 'KMeans'
        self.conf_matrix = None
        self.test_report = None
        self.feat_importance = None
        self.results = None

    def set_classifier(self):
        """
        Set the classifier according to the selected algorithm.
        """
        # PCA
        if isinstance(self.pca, int) or self.pca is None:
            pass
        else:
            raise TypeError("PCA must be a integer or None!")

        # Samples
        if isinstance(self.samples, int) or isinstance(self.samples, float):
            pass
        elif self.samples is None:
            pass
        else:
            raise TypeError("Samples must be a number (integer, float) or None!")

        # Check parameters exists
        if self.parameters is not None:
            if isinstance(self.parameters, str):
                self.parameters = yaml.safe_load(self.parameters)
            else:
                self.parameters = self.parameters

        # Set the chosen learning classifier
        if self.algorithm == 'KMeans':
            self.set_kmeans()
        else:
            raise ValueError("No valid classifier!")

    def check_parameters(self):
        """
        Check if the given parameters match with the given classifier
        and set the classifier with the well defined parameters.
        self.parameters: Parameters to check in dict.
        """
        # Get the type of classifier
        if self.algorithm == 'KMeans':
            double_float = ['tol']
        else:
            double_float = list()

        # Check if the parameters are valid for the given classifier
        for key in self.parameters.keys():  # To change str in float
            if key in double_float:
                self.parameters[key] = float(self.parameters[key])
            try:
                temp_dict = {key: self.parameters[key]}
                self.classifier.set_params(**temp_dict)
            except ValueError:
                print("ValueError: Invalid parameter '{}' for {}, "
                      "it was skipped!".format(str(key), self.algorithm))

    def set_kmeans(self):
        """
        Set the clustering algorithm as KMeans.
        :return: classifier: the desired classifier with the required parameters
        """
        # Set the classifier
        if isinstance(self.parameters, dict):
            self.classifier = KMeans()
            self.check_parameters()  # Check and set parameters
        else:
            self.classifier = KMeans()

    def segment(self, verbose=True):
        """Segment scaled data"""
        # Set string for segmentation
        segment_str = "\n"

        # Do segmentation
        self.y_cluster = self.classifier.fit_predict(self.data)

        if verbose:
            return segment_str

    def save_clusters(self, verbose=True):
        """Save the segmented point cloud"""
        # String for saving point cloud
        save_pt_cloud_str = "\n"
        save_pt_cloud_str += self.report_filename

        # Set header for the predictions
        cluster_header = ['Cluster']
        clusters = pd.DataFrame(self.y_cluster, columns=cluster_header, dtype='float32').round(decimals=4)

        if self.data_type == '.csv':
            # Join predictions to self.frame (input_data)
            self.frame = self.frame.join(clusters)
            self.frame.to_csv(self.report_filename + '.csv', sep=',', header=True, index=False)
        elif self.data_type == '.las':
            # Reopen original las file
            output_las = laspy.read(self.data_path)

            # Create ExtraBytesParams for additonnal dimensions
            output_las.add_extra_dim(laspy.ExtraBytesParams(name='Cluster',
                                                            type=np.uint16,
                                                            description="K-Means clustering segmentation"))

            # Fill output_las with new data
            output_las['Cluster'] = clusters['Cluster'].values

            # Write output_las file
            output_las.write(self.report_filename + '.las')

        if verbose:
            return save_pt_cloud_str
