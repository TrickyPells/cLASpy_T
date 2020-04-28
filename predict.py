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
#       'predict.py' from classer library to predict dataset          #
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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def load_model(path_to_model):
    """
    Load the given model using joblib.
    :param path_to_model: The path to the model to load
    :return: loaded_model
    """

    # Check if path_to_model variable is str and load model
    if isinstance(path_to_model, str):
        loaded_model = joblib.load(path_to_model)
    else:
        raise TypeError("Argument 'model_to_import' must be a string!")

    # Check if the model is GridSearchCV or Pipeline
    if isinstance(loaded_model, GridSearchCV):
        loaded_model = loaded_model.best_estimator_
    elif isinstance(loaded_model, Pipeline):
        pass
    else:
        raise IOError('Loading model failed !\n'
                      'Model to load must be GridSearchCV or Pipeline type !')

    # Fill classifier, scaler and pca
    model = loaded_model.named_steps['classifier']  # Load classifier
    scaler = loaded_model.named_steps['scaler']  # Load scaler
    try:
        pca = loaded_model.named_steps['pca']  # Load PCA if exists
    except KeyError as ke:
        print('\tAny PCA data to load from model.')
        pca = None
    else:
        print('\tPCA data load from model.')

    return model, scaler, pca


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
