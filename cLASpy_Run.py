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
#           The run part of the GUI of cLASpy_T library               #
#                    By Xavier PELLERIN LE BAS                        #
#                         February 2022                               #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description:                                                       #
#                                                                     #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------

import sys
import io
import re
import json
import threading
import time
import psutil
import argparse
import textwrap
import subprocess
import traceback

from contextlib import redirect_stdout
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cLASpy_Classes import *
from cLASpy_GUI import warning_box, error_box

# -------------------------
# ---- ARGUMENT_PARSER ----
# -------------------------

# Create global parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description=textwrap.dedent('''\
                                 ------------------------------------------------------------------------
                                                            cLASpy_Run
                                                     ------------------------'''))

# Add subparsers
subparsers = parser.add_subparsers(help="cLASpy_T modes:\n\n")

# Create sub-command for training
parser_train = subparsers.add_parser('train', help="training mode",
                                     formatter_class=argparse.RawTextHelpFormatter)

parser_train.add_argument("settings",
                          help="give the temporary configuration file with\n"
                               "all parameters and selected scalar fields",
                          type=str)

parser_train.set_defaults(func='train')  # Use training function

# Create sub-command for predictions
parser_predict = subparsers.add_parser('predict', help="prediction mode",
                                       formatter_class=argparse.RawTextHelpFormatter)

parser_predict.add_argument("settings",
                            help="give the temporary configuration file with\n"
                                 "all parameters and selected scalar fields",
                            type=str)


parser_predict.set_defaults(func='predict')  # Use predict function

# Create sub-command for segmentation
parser_segment = subparsers.add_parser('segment', help="segmentation mode",
                                       formatter_class=argparse.RawTextHelpFormatter)

parser_segment.add_argument("settings",
                            help="give the temporary configuration file with\n"
                                 "all parameters and selected scalar fields",
                            type=str)


parser_segment.set_defaults(func='segment')  # Use segment function

# parse the args and call whatever function was selected
args = parser.parse_args()


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

def percent_parser(output):
    """
    Search regular expression to know the progress of a process.
    :param output: The output string from process.
    :return: integer converted in percent.
    """
    progress_regex = re.compile("Step (\d)/(\d):")
    finish_regex = re.compile(" done in ")

    match_progress = progress_regex.search(output)
    if match_progress:
        num_progress = int(match_progress.group(1))
        denom_progress = int(match_progress.group(2))

        progress = int(((num_progress - 1) / denom_progress) * 100)
        return progress

    match_finish = finish_regex.search(output)
    if match_finish:
        progress = 100
        return progress


def run_train(config_file=None):
    """
    Run the Claspy_Run GUI and perform training according the passed config file.
    :param config_file: parser arguments
    """
    # Set the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Execute the Main window
    ex = ClaspyRun()
    ex.show()
    sys.exit(app.exec_())


def run_predict(config_file):
    """
    Run the Claspy_Run GUI and perform predictions according the passed config file.
    :param config_file: parser arguments
    """


def run_segment(config_file):
    """
    Run the Claspy_Run GUI and perform segmentation according the passed config file.
    :param config_file: parser arguments
    """

# -------------------------
# ------- CLASSES ---------
# -------------------------


class Worker(QObject):
    """
    Worker thread
    :param fn: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.progress

    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.finished.emit()  # Done


class ClaspyRun(QMainWindow):
    def __init__(self, settings, parent=None):
        super(ClaspyRun, self).__init__(parent)

        # Store settings arguments
        self.settings = settings

        # -------------- Window --------------
        self.setWindowTitle("Run of cLASpy_T")
        self.setWindowIcon(QIcon('Ressources/pythie_alpha_64px.png'))
        self.mainWidget = QWidget()

        # ----------- Command part -----------
        self.plainTextCommand = QPlainTextEdit()
        self.plainTextCommand.setReadOnly(True)
        self.plainTextCommand.setStyleSheet(
            """QPlainTextEdit {background-color: #333;
                               color: #EEEEEE;}""")

        # Save button
        self.buttonSaveCommand = QPushButton("Save Command Output")
        self.buttonSaveCommand.clicked.connect(self.save_output_command)

        # Clear button
        self.buttonClear = QPushButton("Clear")
        self.buttonClear.clicked.connect(self.plainTextCommand.clear)

        self.hLayoutSaveClear = QHBoxLayout()
        self.hLayoutSaveClear.addWidget(self.buttonSaveCommand)
        self.hLayoutSaveClear.addWidget(self.buttonClear)

        # Progress bar
        self.progressBar = QProgressBar()
        self.progressBar.setMaximum(100)

        # Fill layout of right part
        self.vLayoutRight = QVBoxLayout()
        self.vLayoutRight.addWidget(self.plainTextCommand)
        self.vLayoutRight.addLayout(self.hLayoutSaveClear)
        self.vLayoutRight.addWidget(self.progressBar)

        self.groupCommand = QGroupBox("Command Output")
        self.groupCommand.setLayout(self.vLayoutRight)

        # ------------ Process ---------------
        self.threadpool = QThreadPool()
        self.thread = QThread()
        self.process = None

        # ----------- Button box -------------
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttonBox.rejected.connect(self.reject)

        self.buttonRunTrain = QPushButton("Run")
        self.buttonRunTrain.clicked.connect(self.run_train)
        self.buttonBox.addButton(self.buttonRunTrain, QDialogButtonBox.ActionRole)
        self.buttonStop = QPushButton("Stop")
        self.buttonStop.clicked.connect(self.reject)
        self.buttonStop.setEnabled(False)
        self.buttonBox.addButton(self.buttonStop, QDialogButtonBox.ActionRole)

        # ------------- Main layout ------------------
        self.vMainLayout = QVBoxLayout(self.mainWidget)
        self.vMainLayout.addWidget(self.groupCommand)
        self.vMainLayout.addWidget(self.buttonBox)

        # setCentralWidget
        self.setCentralWidget(self.mainWidget)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Geometry
        self.setGeometry(0, 0, 1280, 960)

    def save_output_command(self):
        """
        Save the Output Command in a text file.
        """
        cmd_output = self.plainTextCommand.toPlainText()
        cmd_output_file = QFileDialog.getSaveFileName(None, 'Save Command Output as TXT file',
                                                      '', "TXT files (*.txt);;")
        if cmd_output_file[0] != '':
            with open(cmd_output_file[0], 'w') as text_file:
                text_file.write(cmd_output)

    def message(self, s):
        """
        Print message 's' in the plainTextCommand (QPlainTextEdit())
        :param s: the message to print into plainTextCommand
        """
        self.plainTextCommand.appendPlainText(s)

    def update_progress(self, n):
        """Update progressBar according given percent as integer"""
        self.progressBar.setValue(n)

    def run_train(self):
        # Update training configuration
        self.update_config()

        # Check if some features are selected
        if self.sel_feat_count <= 0:
            warning_box("No feature field selected!\nPlease select the features you need.",
                        title="Warning: No features selected!")
        else:
            # Pass the function to execute
            # Step 3
            self.worker = Worker(self.train)

            # Step 4
            self.worker.moveToThread(self.thread)

            # Step 5
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.finished.connect(self.thread_complete)
            self.worker.progress.connect(self.update_progress)

            # Execute
            self.thread.start()

            # Update buttons
            self.buttonRunTrain.setEnabled(False)
            self.buttonRunPredict.setEnabled(False)
            self.buttonRunSegment.setEnabled(False)
            self.buttonStop.setEnabled(True)

    def train(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Training Mode   - - - - - - - - - - - - - - -")

        # Train with GridSearchCV or not
        if self.train_config['grid_search']:
            algo_parameters = None
            grid_parameters = self.train_config['param_grid']
        else:
            algo_parameters = self.train_config['parameters']
            grid_parameters = None

        # Set the classifier
        trainer = ClaspyTrainer(input_data=self.lineLocalFile.text(),
                                output_data=self.lineLocalFolder.text(),
                                algo=self.algo,
                                algorithm=None,
                                parameters=algo_parameters,
                                features=self.train_config['feature_names'],
                                grid_search=self.train_config['grid_search'],
                                grid_param=grid_parameters,
                                pca=self.spinPCA.value(),
                                n_jobs=self.spinNJobCV.value(),
                                random_state=self.train_config['random_state'],
                                samples=self.train_config['samples'],
                                scaler=self.comboScaler.currentText(),
                                scoring=self.comboScorer.currentText(),
                                train_ratio=self.train_config['training_ratio'],
                                png_features=self.train_config['png_features'])

        # Set the classifier according parameters
        trainer.set_classifier()

        # Introduction
        intro = trainer.introduction(verbose=True)
        self.message(intro)
        progress_callback.emit(int(0 * 100 / 7))

        # Part 1/7 - Format dataset
        self.message("\nStep 1/7: Formatting data as pandas.DataFrame...")
        step1 = trainer.format_dataset(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 7))

        # Part 2/7 - Split data into training and testing sets
        self.message("\nStep 2/7: Splitting data in train and test sets...")
        step2 = trainer.split_dataset(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 7))

        # Part 3/7 - Scale dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 3/7: Scaling data...")
        step3 = trainer.set_scaler_pca(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 7))

        # Part 4/7 - Train model
        if trainer.grid_search:  # Training with GridSearchCV
            self.message('\nStep 4/7: Training model with GridSearchCV...\n')
        else:  # Training with Cross Validation
            self.message("\nStep 4/7: Training model with cross validation...\n")

        step4 = trainer.train_model(verbose=True)  # Perform both training
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 7))

        # Part 5/7 - Create confusion matrix
        self.message("\nStep 5/7: Creating confusion matrix...")
        step5 = trainer.confusion_matrix(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 7))

        # Part 6/7 - Save algorithm, model, scaler, pca and feature_names
        self.message("\nStep 6/7: Saving model and scaler in file:")
        step6 = trainer.save_model(verbose=True)
        self.message(step6)
        progress_callback.emit(int(6 * 100 / 7))

        # Part 7/7 - Create and save prediction report
        self.message("\nStep 7/7: Creating classification report:")
        self.message(trainer.report_filename + '.txt')
        step7 = trainer.classification_report(verbose=True)
        self.message(step7)
        progress_callback.emit(int(7 * 100 / 7))

        # Kill the remaining python interpreters (1+18)
        return "Training done!"

    def run_predict(self):
        # Update buttons
        self.buttonRunTrain.setEnabled(False)
        self.buttonRunPredict.setEnabled(False)
        self.buttonRunSegment.setEnabled(False)
        self.buttonStop.setEnabled(True)

        # Check if model and input file features match
        self.check_model_features()
        if self.predict_features:
            # Update predict configuration
            self.update_config()

            # Pass the function to execute
            self.worker = Worker(self.predict)
            self.worker.signals.result.connect(self.message)
            self.worker.signals.finished.connect(self.thread_complete)
            self.worker.signals.progress.connect(self.update_progress)

            # Execute
            self.threadpool.start(self.worker)

    def predict(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Prediction Mode   - - - - - - - - - - - - - - -")

        # Set the classifier for prediction
        predicter = ClaspyPredicter(model=self.lineModelFile.text(),
                                    input_data=self.lineLocalFile.text(),
                                    output_data=self.lineLocalFolder.text())

        # Load model
        self.message("\nStep 1/6: Loading model...")
        step1 = predicter.load_model(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 6))

        # Introduction
        intro = predicter.introduction(verbose=True)
        self.message(intro)

        # Format dataset
        self.message("\nStep 2/6: Formatting data as pandas.Dataframe...")
        step2 = predicter.format_dataset(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 6))

        # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 3/6: Scaling data...")
        step3 = predicter.scale_dataset(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 6))

        # Predict target of input data
        self.message("\nStep 4/6: Making predictions for entire dataset...")
        step4 = predicter.predict(verbose=True)
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 6))

        # Save classification results as point cloud file with all data
        self.message("\nStep 5/6: Saving classified point cloud:")
        step5 = predicter.save_predictions(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 6))

        # Create and save prediction report
        self.message("\nStep 6/6: Creating classification report:")
        self.message(predicter.report_filename + '.txt')
        step6 = predicter.classification_report(verbose=True)
        self.message(step6)
        progress_callback.emit(int(6 * 100 / 6))

        # End of the function
        return "Predictions done!"

    def run_segment(self):
        # Update buttons
        self.buttonRunTrain.setEnabled(False)
        self.buttonRunPredict.setEnabled(False)
        self.buttonRunSegment.setEnabled(False)
        self.buttonStop.setEnabled(True)

        # Update segmentation configuration
        self.update_config()

        # Check if some features are selected
        if self.sel_feat_count <= 0:
            warning_box("No feature field selected!\nPlease select the features you need!",
                        "No features selected")
        else:
            # Pass the function to execute
            self.worker = Worker(self.segment)
            self.worker.signals.result.connect(self.message)
            self.worker.signals.finished.connect(self.thread_complete)
            self.worker.signals.progress.connect(self.update_progress)

            # Execute
            self.threadpool.start(self.worker)

    def segment(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Segmentation Mode   - - - - - - - - - - - - - - -")

        # Set the classifier
        segmenter = ClaspySegmenter(input_data=self.lineLocalFile.text(),
                                    output_data=self.lineLocalFolder.text(),
                                    parameters=self.segment_config['parameters'],
                                    features=self.segment_config['feature_names'])

        # Set the classifier according parameters
        segmenter.set_classifier()

        # Introduction
        intro = segmenter.introduction(verbose=True)
        self.message(intro)
        progress_callback.emit(int(0 * 100 / 5))

        # Format dataset
        self.message("\nStep 1/5: Formatting data as pandas.DataFrame...")
        step1 = segmenter.format_dataset(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 5))

        # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 2/5: Scaling data...")
        step2 = segmenter.set_scaler_pca(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 5))

        # Split data into training and testing sets
        self.message("\nStep 3/5: Clustering the dataset...")
        step3 = segmenter.segment(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 5))

        # Save algorithm, model, scaler, pca and feature_names
        self.message("\nStep 4/5: Saving segmented point cloud in file...")
        step4 = segmenter.save_clusters(verbose=True)
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 5))

        # Create and save prediction report
        self.message("\nStep 5/5: Creating classification report:")
        step5 = segmenter.classification_report(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 5))

        # End of the function
        return "Segmentation done!"

    def handle_state(self, state):
        states = {QProcess.NotRunning: "Not Running",
                  QProcess.Starting: "Starting",
                  QProcess.Running: "Running"}

        state_name = states[state]
        self.statusBar.showMessage("cLASpy_T is {}".format(state_name), 3000)

    def thread_complete(self):
        self.statusBar.showMessage("cLASpy_T finished !", 5000)
        self.threadpool.releaseThread()
        self.progressBar.reset()
        self.enable_open_results()
        self.buttonStop.setEnabled(False)
        self.buttonRunTrain.setEnabled(True)
        self.buttonRunPredict.setEnabled(True)
        self.buttonRunSegment.setEnabled(True)

    def stop_thread(self):
        try:
            self.thread.exit()
            self.thread.wait()
            # self.thread.quit()
            # self.worker.deleteLater()
            # self.thread.deleteLater()
        except:  # too generic
            self.statusBar.showMessage("Exception raised: Thread not stopped!", 3000)

        else:
            self.buttonStop.setEnabled(False)
            self.buttonRunTrain.setEnabled(True)
            self.buttonRunPredict.setEnabled(True)
            self.buttonRunSegment.setEnabled(True)
            self.plainTextCommand.appendPlainText("\n********************"
                                                  "\nProcess stopped by user!"
                                                  "\n********************")

    def reject(self):
        """
        Close the GUI
        """
        if self.WelcomeAgain:
            self.welcome_window.close()
        self.close()

    def closeEvent(self, event):
        if self.WelcomeAgain:
            self.welcome_window.close()
        self.close()
        event.accept()


if args.func == 'train':
    run_train(args)
elif args.func == 'predict':
    run_predict(args)
elif args.func == 'segment':
    run_segment(args)
else:
    raise KeyError("No valid function selected!")
