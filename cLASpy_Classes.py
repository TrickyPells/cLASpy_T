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
import laspy
import pandas as pd
from datetime import datetime

from common import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# -------------------------
# -------- CLASSES --------
# -------------------------

class ClaspyTrainer:
    def __init__(self, input_data, output_data=None, algorithm=None,
                 parameters=None, features=None, grid_search=False, param_grid=None,
                 pca=False, n_jobs=-1, random_state=0, samples=None, scaler='Standard',
                 scoring='accuracy', train_ratio=0.5, png_features=True):
        """Initialize cLASpy_Trainer object"""

        # Set variables
        self.start_time = datetime.now()  # Set the start time
        self.timestamp = self.start_time.strftime("%m%d_%H%M")  # Timestamp for file creation

        # Data
        self.data_path = os.path.normpath(input_data)  # Path to the datafile
        self.root_ext = os.path.splitext(self.data_path)  # Split path into root and extension
        self.data_type = self.root_ext[1]  # Get the type of data file ('.csv', or '.las')
        self.folder_path = output_data

        # Algorithm
        self.algorithm = algorithm
        self.parameters = parameters
        self.features = features
        self.pca = pca
        self.samples = samples
        self.scaler = scaler
        self.scoring = scoring
        self.train_ratio = train_ratio
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.png_features = png_features

        # Grid Search CV
        self.grid_search = grid_search
        self.grid_parameters = param_grid

        # Set the classifier (self.classifier)
        self.set_classifier()

        #

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

    def prompt_introduction(self):
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

        return self.introduction

    def prompt_point_cloud_info(self):
        """
        Load data into dataframe and return the point cloud info
        """
        # Point cloud info string
        point_cloud_info = "\n"

        # Load data into DataFrame
        if self.data_type == '.csv':
            self.frame = pd.read_csv(self.data_path, sep=',', header='infer')
            point_cloud_info += "Number of points: {:,}\n".format(len(self.frame))

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
            point_cloud_info += "Number of points: {:,}\n".format(las.header.records_count)

            # Get the data by dimensions
            frame = pd.DataFrame()
            for dim in las.point_format.specs:
                frame[dim.name] = las.get_reader().get_dimension(dim.name)
        else:
            point_cloud_info += "Unknown Extension file!\n"

        return point_cloud_info

    def prompt_format_dataset(self):
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
            selected_features = get_selected_features(self.features, temp_features)

        elif isinstance(self.features, list):
            selected_features = get_selected_features(self.features, temp_features)

        else:
            raise TypeError("Selected features must be a list of string!")

        # Sort data by field names
        selected_features.sort()  # Sort to be compatible between formats

        # data without target field
        self.data = self.frame.filter(selected_features, axis=1)

        # Replace NAN values by median
        self.data.fillna(value=self.data.median(0), inplace=True)  # .median(0) computes median by column
        
        return format_data_str


