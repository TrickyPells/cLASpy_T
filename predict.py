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

import laspy

from training import *
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

    # Retrieve algorithm, model and field_names
    algorithm = loaded_model['algorithm']
    feature_names = loaded_model['feature_names']
    loaded_model = loaded_model['model']

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
    except KeyError:
        print('\tAny PCA data to load from model.')
        pca = None

    return algorithm, model, scaler, pca, feature_names


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
    copy_data = laspy.file.File(source_file, mode='r')

    # Create new LAS file
    las_name = str(las_name + '.las')
    output_las = laspy.file.File(las_name, mode='w', header=copy_data.header)

    # Define new dimensions according the predictions dataframe
    output_las.define_new_dimension(name='Prediction', data_type=5,
                                    description='Prediction done by the model')

    dimension_list = predictions.columns.values.tolist()
    if predictions.shape[1] > 1:
        dimension_list.remove('Prediction')
        for dim in dimension_list:
            output_las.define_new_dimension(name=dim, data_type=9,
                                            description='Probability fir this class')

    # Fill output_las with original dimensions
    for dim in copy_data.point_format:
        data = copy_data.reader.get_dimension(dim.name)
        output_las.writer.set_dimension(dim.name, data)
    copy_data.close()

    # Now fill output_las with new data
    output_las.Prediction = predictions['Prediction']
    if predictions.shape[1] > 1:
        for dim in dimension_list:
            output_las.writer.set_dimension(dim, predictions[dim].values)
    output_las.close()


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
    if len(predictions.shape) > 1 and predictions.shape[1] > 2:
        # Get number of class in prediction array (number of column - 2)
        numb_class = predictions.shape[1] - 2
        pred_header = ['Prediction', 'BestProba'] + ['ProbaClass_' + str(cla) for cla in range(0, numb_class)]
    else:
        pred_header = ['Prediction']

    # Set the np.array of target_pred pd.Dataframe
    predictions = pd.DataFrame(predictions, columns=pred_header, dtype='float32').round(decimals=4)

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


# -------------------------
# --------- MAIN ----------
# -------------------------


def predict(args):
    """
    Make predictions according the passed arguments.
    :param args: the passed arguments.
    """
    # Set mode for common functions
    mode = 'predict'

    # Config file exists ?
    if args.config:
        arguments_from_config(args)  # Get the arguments from the config file

    # Get model, scaler and pca
    print("\nStep 1/6: Loading model...", end='')
    model_to_load = args.model  # Set variable for the report
    algorithm, model, scaler, pca, feature_names = load_model(model_to_load)
    print("Done\n")

    # INTRODUCTION
    data_path, folder_path, start_time = introduction(algorithm, args.input_data, folder_path=args.output)
    timestamp = start_time.strftime("%m%d_%H%M")  # Timestamp for file creation MonthDay_HourMinute

    # FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
    print("\nStep 2/6: Formatting data as pandas.Dataframe...")
    data, target = format_dataset(data_path, mode=mode, features=feature_names)

    # Get the number of points
    nbr_pts = number_of_points(data.shape[0])
    str_nbr_pts = format_nbr_pts(nbr_pts)  # Format nbr_pts as string for filename

    # Set the report filename
    algo = shortname_algo(algorithm)
    report_filename = str(folder_path + '/' + mode + '_' + algo + str_nbr_pts + str(timestamp))

    # Get the field names
    # ADD TEST TO CHECK ALL MANDATORY FIELDS ARE PRESENT
    feature_names = data.columns.values.tolist()

    # Apply scaler to data
    print("\nStep 3/6: Scaling data...")
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame.from_records(data_scaled, columns=feature_names)

    # Apply pca to data if exists
    if pca:
        data_scaled = apply_pca(pca, data_scaled)
        pca_compo = np.array2string(pca.components_)
    else:
        pca_compo = None

    # Predic target of input data
    print("\nStep 4/6: Making predictions for entire dataset...")
    # y_pred = model.predict(data_scaled.values)
    y_pred = predict_with_proba(model, data_scaled.values)

    if target is not None:
        # Save confusion matrix
        print("\nCreating confusion matrix...")
        conf_mat = confusion_matrix(target.values, y_pred.transpose()[0])
        conf_mat = precision_recall(conf_mat)  # return Dataframe
        test_report = classification_report(target.values, y_pred.transpose()[0])  # Get classification report
        print("\n{}\n".format(test_report))
    else:
        conf_mat = None
        test_report = None

    # Save classifaction result as point cloud file with all data
    print("\nStep 5/6: Saving classified point cloud:")
    predic_filename = report_filename
    print(predic_filename)
    save_predictions(y_pred, predic_filename, data_path)

    # Create and save prediction report
    print("\nStep 6/6: Creating classification report:")
    print(report_filename + '.txt')

    # Get the model parameters to print in the report
    applied_parameters = ["{}: {}".format(param, model.get_params()[param])
                          for param in model.get_params()]

    # Compute elapsed time
    spent_time = datetime.now() - start_time

    # Write the entire report
    write_report(report_filename,
                 mode=mode,
                 algo=algorithm,
                 data_file=args.input_data,
                 start_time=start_time,
                 elapsed_time=spent_time,
                 feat_names=feature_names,
                 scaler=scaler,
                 data_len=nbr_pts,
                 applied_param=applied_parameters,
                 pca_compo=pca_compo,
                 model=model_to_load,
                 conf_mat=conf_mat,
                 score_report=test_report)

    print("\nPredictions done in {}".format(spent_time))


def segment(args):
    """
    Segment input_data point cloud according the passed arguments.
    :param args: the passed arguments.
    """
    # Set mode for common functions
    mode = 'segment'

    # Config file exists ?
    if args.config:
        arguments_from_config(args)  # Get the arguments from the config file

    # Get the classifier and update the selected algorithm
    algorithm, classifier = get_classifier(args, mode=mode)

    # INTRODUCTION
    data_path, folder_path, start_time = introduction('KMeans', args.input_data, folder_path=args.output)
    timestamp = start_time.strftime("%m%d_%H%M")  # Timestamp for file creation MonthDay_HourMinute

    # FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
    print("\nStep 1/4: Formatting data as pandas.Dataframe...")
    data, target = format_dataset(data_path, mode=mode, features=args.features)

    # Get the number of points
    nbr_pts = number_of_points(data.shape[0])
    str_nbr_pts = format_nbr_pts(nbr_pts)  # Format nbr_pts as string for filename

    # Set the report filename
    report_filename = str(folder_path + '/' + mode + '_' + args.algo + str_nbr_pts + str(timestamp))

    # Get the field names
    feature_names = data.columns.values.tolist()

    # Clustering the input data
    print("\nStep 2/4: Clustering the dataset...")
    y_pred = classifier.fit_predict(data)

    # Save clustering result as point cloud file with all data
    print("\nStep 3/4: Saving segmented point cloud as CSV file:")
    predic_filename = report_filename
    print(predic_filename)
    save_predictions(y_pred, predic_filename, data_path)
    scaler = None

    # Create and save prediction report
    print("\nStep 4/4: Creating segmentation report:")
    print(report_filename + '.txt')

    # Get the model parameters to print in the report
    applied_parameters = ["{}: {}".format(param, classifier.get_params()[param])
                          for param in classifier.get_params()]

    # Compute elapsed time
    spent_time = datetime.now() - start_time

    # Write the entire report
    write_report(report_filename,
                 mode=mode,
                 algo=algorithm,
                 data_file=args.input_data,
                 start_time=start_time,
                 elapsed_time=spent_time,
                 feat_names=feature_names,
                 scaler=scaler,
                 data_len=nbr_pts,
                 applied_param=applied_parameters)

    print("\nSegmentation done in {}".format(spent_time))
