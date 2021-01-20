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
#       'predict.py' from cLASpy_T library to predict dataset         #
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

import os
import pylas
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


def save_pred_csv(predictions, csv_name, source_file):
    """
    Save the predictions in copy of CSV source_file.
    :param predictions: DataFrame of the predictions.
    :param csv_name: Output file name with path.
    :param source_file: CSV source file name with path.
    """
    # Get CSV data in copy_data
    copy_data = pd.read_csv(source_file, sep=',', header='infer')

    # Join copy of the data with the predictions
    final_data = copy_data.join(predictions)

    # Free memory
    copy_data = None
    predictions = None

    # Write all data in the CSV output file
    csv_name = str(csv_name + '.csv')
    final_data.to_csv(csv_name, sep=',', header=True, index=False)


def save_pred_las(predictions, las_name, source_file):
    """
    Save the predictions in copy of LAS source_file.
    :param predictions: DataFrame of the predictions.
    :param las_name: Output file name with path.
    :param source_file: LAS source file name with path.
    """
    # Get LAS data in copy_data
    copy_data = pylas.read(source_file)

    # Add dimensions according the predictions dataframe
    copy_data.add_extra_dim(name='Prediction', type='u1',  # First dimension is 'u1' type
                            description='Prediction done by the model')
    copy_data['Prediction'] = predictions['Prediction']

    if predictions.shape[1] > 1:
        dimension_list = predictions.columns.values.tolist()
        dimension_list.remove('Prediction')
        for dim in dimension_list:
            copy_data.add_extra_dim(name=dim, type='f4', description='Probability for this class')
            copy_data[dim] = predictions[dim]

    # Write all data in the LAS output file
    las_name = str(las_name + '.las')
    copy_data.write(las_name)


def save_predictions(predictions, file_name, source_file):
    """
    Save the predictions in copy of source_file.
    :param predictions: Array with first column as class predicted,
    second column as probability of predicted class,
    following columns as probabilities for each class.
    :param file_name: The path and name of the file.
    :param source_file: The path to the input file.
    """
    # Set header for the predictions
    if predictions.shape[1] > 2:
        # Get number of class in prediction array (number of column - 2)
        numb_class = predictions.shape[1] - 2
        pred_header = ['Prediction', 'BestProba'] + ['Proba' + str(cla) for cla in range(0, numb_class)]
    else:
        pred_header = ['Prediction']

    # Set the np.array of target_pred pd.Dataframe
    predictions = pd.DataFrame(predictions, columns=pred_header, dtype='float32').round(decimals=3)

    # Reload data into DataFrame
    root_ext = os.path.splitext(source_file)  # split file path into root and extension
    if root_ext[1] == '.csv':  # Copy the CSV source file
        print("Write the CSV output file...", end='')
        save_pred_csv(predictions, file_name, source_file)
        print(" Done!")

    elif root_ext[1] == '.las':  # Copy the LAS source file
        print("Write the LAS output file...", end='')
        save_pred_las(predictions, file_name, source_file)
        print(" Done!")

    else:
        print("Unknown extension!")
