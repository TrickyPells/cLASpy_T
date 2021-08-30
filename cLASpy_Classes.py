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
#       'training.py' from cLASpy_T library to train model            #
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

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
# -------- CLASSES --------
# -------------------------


class ClaspyTrainer:
    def __init__(self, input_data, output_data=None, algo=None, algorithm=None,
                 parameters=None, features=None, grid_search=False, grid_param=None,
                 pca=None, n_jobs=-1, random_state=0, samples=None, scaler='Standard',
                 scoring='accuracy', train_ratio=0.5, png_features=True):
        """Initialize cLASpy_Trainer object"""

        # Set variables
        self.mode = 'training'
        self.start_time = datetime.now()  # Set the start time
        self.timestamp = self.start_time.strftime("%m%d_%H%M")  # Timestamp for file creation

        # Data
        self.data_path = os.path.normpath(input_data)  # Path to the datafile
        self.root_ext = os.path.splitext(self.data_path)  # Split path into root and extension
        self.data_type = self.root_ext[1]  # Get the type of data file ('.csv', or '.las')
        self.folder_path = output_data

        # Algorithm
        if algo is None and algorithm is not None:
            self.algorithm = algorithm
            self.algo = self.shortname_algo()
        elif algo is not None and algorithm is None:
            self.algo = algo
            self.algorithm = self.fullname_algo()
        else:
            raise ValueError("Choose a valid machine learning algorithm!")

        self.parameters = parameters
        self.features = features
        if isinstance(pca, int):
            self.pca = pca
        elif pca is None:
            self.pca = None
        else:
            raise TypeError("PCA must be a integer or None!")

        if isinstance(samples, int) or isinstance(samples, float):
            self.samples = samples
        elif samples is None:
            self.samples = None
        else:
            raise TypeError("Samples must be a number (integer, float) or None!")

        self.scaler_method = scaler
        self.scoring = scoring
        self.train_ratio = train_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Grid Search CV
        self.grid_search = grid_search
        self.grid_parameters = grid_param

        # feature importance
        if self.grid_search or self.algorithm == 'MLPClassifier':
            self.png_features = False
        else:
            self.png_features = png_features

        # Set the classifier (self.classifier)
        self.set_classifier()

        # Set some varaible members to None
        self.model_to_load = None

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

    def number_of_points(self, magnitude=1000000):
        """
        Set the number of point according the magnitude.
        self.samples: float of the number of point for training.
        len(self.data): total number of points in data file.
        :param magnitude: the order of magnitude.
        Set self.samples as an integer of the number samples used.
        """
        # Check if self.samples is float or integer
        if isinstance(self.samples, float):
            if self.samples * magnitude >= len(self.data):
                self.samples = len(self.data)
            else:
                self.samples = self.samples * magnitude
        elif isinstance(self.samples, int):
            if self.samples >= len(self.data):
                self.samples = len(self.data)
        else:  # if self.samples is not float or int
            self.samples = len(self.data)

    def format_nbr_pts(self, magnitude=1000000):
        """
        Format the nbr_pts as string for the filename.
        self.samples: Integer of the number of points used to format in string.
        :return: String of the point number write according the magnitude suffix.
        """
        # Format as Mpts or kpts according number of points
        if self.samples >= magnitude:  # number of points > 1Mpts
            str_nbr_pts = str(np.round(self.samples / magnitude, 1))
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
        # Check parameters exists
        if self.parameters is not None:
            if isinstance(self.parameters, str):
                self.parameters = yaml.safe_load(self.parameters)
            else:
                self.parameters = self.parameters

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
                    check_grid_param_str += "GridSearchCV: Invalid parameter '{}' for {}, it was skipped!"\
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

    def get_selected_features(self, data_features):
        """
        self.features: the wanted features (selected by user).
        :param data_features: all features in input_data except ('x', 'y', 'z' and 'target').
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
                    if dt_feature.casefold() == feature.casefold():
                        selected_features.append(dt_feature)
                        selected_feat_str += " - {} asked --> {} found\n".format(feature, dt_feature)

            selected_feat_str += "\nNumber of wanted features: {}\n".format(len(self.features))
            selected_feat_str += "Number of final selected features: {}\n".format(len(selected_features))

            if len(self.features) == len(selected_features):
                self.features = selected_features
                selected_feat_str += " --> All required features are present!\n"
            else:
                differences = list(set(self.features) - set(selected_features))
                raise ValueError("{} features are missing in 'input_data'!".format(differences))
        else:
            raise TypeError("Selected features must be a list of string!")

        return selected_feat_str

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

    def training_gridsearch(self):
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
                                  param_grid=self.grid_parameters,
                                  n_jobs=self.n_jobs,
                                  cv=cross_val,
                                  scoring=self.scoring,
                                  verbose=1,
                                  error_score=np.nan)

        # Training the model to find the best parameters
        self.model.fit(self.data_train, self.target_train)
        self.results = pd.DataFrame(self.model.cv_results_)
        gridsearch_str += "\tBest score: {0:.4f}\n".format(self.model.best_score_)
        gridsearch_str += "\tBest parameters: {}\n".format(self.model.best_params_)
        gridsearch_str += "\tModel trained!\n"

        return gridsearch_str

    def training_nogridsearch(self):
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
                                      verbose=2,
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

        # Save the new confusion matrix
        self.conf_matrix = pd.DataFrame(self.conf_matrix).round(decimals=3)

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

    def introduction(self, prompt=True):
        """
        Return the introduction in a string
        """
        self.introduction = str("\n######## POINT CLOUD CLASSIFICATION #########\n"
                                "Algorithm used: {}\n".format(self.algorithm))
        if self.data_type == '.csv':
            self.introduction += "Path to CSV file: {}\n".format(self.data_path)
        elif self.data_type == '.las':
            self.introduction += "Path to LAS file: {}\n".format(self.data_path)
        else:
            self.introduction += "Unknown Extension file !\n"

        # Create a folder to store models, reports and predictions
        self.introduction += "\nCreate a new folder to store the result files..."
        if self.folder_path is None:
            self.folder_path = self.root_ext[0]  # remove extension to give the folder path

        try:
            os.makedirs(self.folder_path)  # Using file path to create new folders recursively
            self.introduction += " Done.\n"
        except (TypeError, FileExistsError):
            self.introduction += " Folder already exists.\n"

        if prompt:
            return self.introduction

    def point_cloud_info(self, prompt=True):
        """
        Load data into dataframe and return the point cloud info
        """
        # Point cloud info string
        point_cloud_info = "\n"

        # Load data into DataFrame
        if self.data_type == '.csv':
            self.frame = pd.read_csv(self.data_path, sep=',', header='infer')
            self.nbr_points = len(self.frame)
            point_cloud_info += "Number of points: {:,}\n".format(self.nbr_points)

            # Replace 'space' by '_' and clean up the header built by CloudCompare ('//X')
            for field in self.frame.columns.values.tolist():
                field_ = field.replace(' ', '_')
                self.frame = self.frame.rename(columns={field: field_})
                if field == '//X':
                    self.frame = self.frame.rename(columns={"//X": "X"})
        elif self.data_type == '.las':
            las = laspy.file.File(self.data_path, mode='r')
            point_cloud_info += "LAS Version: {}\n".format(las.reader.version)
            point_cloud_info += "LAS point format: {}\n".format(las.header.data_format_id)
            self.nbr_points = las.header.records_count
            point_cloud_info += "Number of points: {:,}\n".format(self.nbr_points)

            # Get the data by dimensions
            frame = pd.DataFrame()
            for dim in las.point_format.specs:
                frame[dim.name] = las.get_reader().get_dimension(dim.name)
        else:
            point_cloud_info += "Unknown Extension file!\n"

        if prompt:
            return point_cloud_info

    def format_dataset(self, prompt=True):
        """
        Format dataset as XY & Z & target Dataframe, remove raw_classification from file
        and return point cloud informations and 'Done' when finished
        """
        # Report string
        format_data_str = "\n"

        # Search X, Y, Z, target and raw_classif fields
        field_x = None
        field_y = None
        field_z = None
        field_t = None

        for field in self.frame.columns.values.tolist():
            if field.casefold() == 'x':  # casefold() -> non case-sensitive
                field_x = field
            elif field.casefold() == 'y':
                field_y = field
            elif field.casefold() == 'z':
                field_z = field
            elif field.casefold() == 'target':
                field_t = field

        # Create target varaible if exist
        if field_t:
            self.target = self.frame.loc[:, field_t]
        else:
            self.target = None

        # Target is mandatory for training
        if isinstance(self, ClaspyTrainer) and self.target is None:
            raise ValueError("A 'target' field is mandatory for training!")

        # Create temp list of features
        temp_features = self.frame.columns.values.tolist()

        # Remove target field from self.frame if exist
        if self.target is not None:
            temp_features.remove(field_t)

        # Get only the selected features among temp_features
        if self.features is None:  # Use all features except standard LAS fields and X, Y, Z
            format_data_str += "All features in input_data will be used!\n"
            selected_features = temp_features
            for field in [field_x, field_y, field_z]:  # remove X, Y, Z
                selected_features.remove(field)
            if self.data_type == '.las':
                las = laspy.file.File(self.data_path, mode='r')
                standard_dimensions = point_format[las.header.data_format_id]
                for field in standard_dimensions:  # remove standard LAS dimensions
                    selected_features.remove(field)

        elif isinstance(self.features, str):
            self.features = yaml.safe_load(self.features)
            format_data_str += self.get_selected_features(temp_features)

        elif isinstance(self.features, list):
            format_data_str += self.get_selected_features(temp_features)

        else:
            raise TypeError("Selected features must be a list of string!")

        # Sort data by field names
        self.features.sort()  # Sort to be compatible between formats

        # data without target field
        self.data = self.frame.filter(self.features, axis=1)

        # Replace NAN values by median
        self.data.fillna(value=self.data.median(0), inplace=True)  # .median(0) computes median by column

        # Set filename for output result files
        self.number_of_points()
        str_nbr_pts = self.format_nbr_pts()
        self.report_filename = str(self.folder_path + '/train_' + self.algo + str_nbr_pts + str(self.timestamp))

        if prompt:
            return format_data_str

    def split_dataset(self, prompt=True):
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
        self.target_train = self.target_train.reshape(self.train_size)
        self.target_test = self.target_test.reshape(self.test_size)

        split_dataset_str += "\tNumber of used points: {:,} pts\n".\
            format(self.train_size + self.test_size).replace(',', ' ')
        split_dataset_str += "\tSize of train|test datasets: {:,} pts | {:,} pts\n".\
            format(self.train_size, self.test_size).replace(',', ' ')

        if prompt:
            return split_dataset_str

    def scale_dataset(self, prompt=True):
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

        if prompt:
            return scale_dataset_str

    def train_model(self, prompt=True):
        """
        Perform training with GridSearch CV or simple CrossValidation.
        :param prompt: Set it True to return report as string
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
            train_model_str += self.training_gridsearch()

        else:
            train_model_str += "Random state for the StratifiedShuffleSplit: {}".format(self.random_state)

            # Get model
            train_model_str += self.training_nogridsearch()

        # Importance of each feature in RF and GB
        if self.png_features:
            self.feat_importance = self.save_feature_importance()
            train_model_str += "\nFeature importances save as PNG file.\n"
        else:
            self.feat_importance = self.data.columns.values.tolist()

        if prompt:
            return train_model_str

    def confusion_matrix(self, prompt=True):
        """
        Compute precision matrix and other statistics.
        :return: Report as str
        """
        # String report of the confusion matrix
        confusion_str = "\n"

        # Make prediction over test set with the model
        self.target_test_pred = self.model.predict(self.data_test)
        self.conf_matrix = confusion_matrix(self.target_test, self.target_test_pred)
        self.add_precision_recall()  # add statistics to confusion matrix report
        self.test_report = classification_report(self.target_test, self.target_test_pred, zero_division=0)

        confusion_str += self.test_report

        if prompt:
            return confusion_str

    def save_model(self, prompt=True):
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

        if prompt:
            return save_model_str

    def classification_report(self, prompt=True):
        """
        Write the report of training or predictions in .TXT file.
        :param prompt: Return report string if True
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
        applied_parameters = ["{}: {}".format(param, self.model.get_params()[param])
                              for param in self.model.get_params()]

        # Compute elapsed time
        elapsed_time = datetime.now() - self.start_time

        # Write the header of the report file
        with open(self.report_filename + '.txt', 'w', encoding='utf-8') as report:
            report.write('Report of ' + self.algorithm + ' ' + self.mode)
            report.write('\n\nDatetime: ' + self.start_time.strftime("%Y-%m-%d %H:%M:%S"))
            report.write('\nFile: ' + self.data_path)
            report.write('\n\nScaling method:\n{}'.format(self.scaler))

            # Write features depending if it's list or dict
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
                report.write('\n\nModel trained in {}'.format(elapsed_time))
            else:
                report.write('\n\nPredictions done in {}'.format(elapsed_time))

        if prompt:
            return "\nTraining done in {}\n".format(elapsed_time)

