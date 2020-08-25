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

from training import *

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

def set_kmeans_cluster(fit_params):
    """
    Set the clustering algorithm as KMeans.
    :param fit_params: A dict with the parameters to set up.
    :return: classifier: the desired classifier with the required parameters
    """
    # Set the classifier
    if isinstance(fit_params, dict):
        fit_params['random_state'] = 0
        classifier = KMeans()
        classifier = check_parameters(classifier, fit_params)  # Check and set parameters

    else:
        classifier = KMeans()

    return classifier


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

    return model, scaler, pca


def predict_with_proba(model, data_to_predic):
    """
    Make predictions with probability for each class.
    :param model: The model to use to make predictions.
    :param data_to_predic: The data to predict.
    :return: The prediction, the best probability and the probability for each class.
    """
    # Get the probability for each class
    y_proba = model.predict_proba(data_to_predic)

    # Get the best probability and the corresponding class
    y_best_proba = np.amax(y_proba, axis=1)
    y_best_class = np.argmax(y_proba, axis=1)

    # Add best proba and bet class to probability per class
    y_proba = np.insert(y_proba, 0, y_best_proba, axis=1)
    y_proba = np.insert(y_proba, 0, y_best_class, axis=1)

    return y_proba


def save_predictions(predictions, file_name, xy_fields=None,
                     z_field=None, data_fields=None, target_field=None):
    """
    Save the report of the classification algorithms with test dataset.
    :param predictions: Array with first column as class predicted,
    second column as probability of predicted class,
    following columns as probabilities for each class.
    :param file_name: The path and name of the file.
    :param xy_fields: The X and Y fields from the raw_data.
    :param z_field: The Z field from the raw_data
    :param data_fields: The data fields from the raw_data.
    :param target_field: The target field from the raw_data.
    :return:
    """
    # Set header for the predictions
    if len(predictions.shape) > 1:
        # Get number of class in prediction array (number of column - 2)
        numb_class = predictions.shape[1] - 2
        if numb_class <= -1:
            pred_header = ['Prediction']
        else:
            pred_header = ['Prediction', 'BestProba'] + ['Proba' + str(cla) for cla in range(0, numb_class)]

    else:
        pred_header = ['Prediction']

    # Set the np.array of target_pred pd.Dataframe
    predictions = pd.DataFrame(predictions, columns=pred_header).round(decimals=3)

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

    final_classif_list.append(predictions)
    final_classif = pd.concat(final_classif_list, axis=1)

    final_classif.to_csv(file_name, sep=',', header=True, index=False)
