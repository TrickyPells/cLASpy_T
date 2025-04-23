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
#          The run part of the GUI for cLASpy_T library               #
#                    By Xavier PELLERIN LE BAS                        #
#                         February 2022                               #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description:                                                       #
#     - 0.3.2 : update scikit-learn 0.24 > 1.5.0                      #
#     - 0.3.0 : with laspy2 support                                   #
#                                                                     #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------
import os
import signal
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
                                 ----------------------------------------------------------
                                                    cLASpy_Run
                                             ------------------------'''))

parser.add_argument("config",
                    help="give the temporary configuration file with\n"
                         "all parameters and selected scalar fields",
                    type=str)

# parse the args and call whatever function was selected
args = parser.parse_args()


# -------------------------
# ------ FUNCTIONS --------
# -------------------------

def run(arguments):
    """
    Run the Claspy_Run GUI and perform training, predictions or segmentation
    according the passed config file.
    :param arguments: parser arguments
    """
    # Set the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Execute the Main window
    ex = ClaspyRun(arguments=arguments)
    ex.show()
    # ex.run_worker()
    sys.exit(app.exec_())

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
    def __init__(self, arguments, parent=None):
        super(ClaspyRun, self).__init__(parent)

        # Store settings arguments
        self.arguments = arguments

        # -------- Arguments from config file ----------
        arguments.config = os.path.normpath(args.config)
        with open(args.config, 'r') as config_file:
            self.config = json.load(config_file)

        # Common arguments
        self.mode = self.config['version'].split('_')[-1]
        self.input_data = os.path.normpath(self.config['input_file'])
        self.output = os.path.normpath(self.config['output_folder'])

        # Create ClaspyObjects
        if self.mode == 'train':
            self.trainer = self.create_trainer()
        elif self.mode == 'predi':
            self.predicter = self.create_predicter()
        elif self.mode == 'segme':
            self.segmenter = self.create_segmenter()
        else:
            raise ValueError("Invalid mode for ClaspyRun()")

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
        self.hLayoutSaveClear = QHBoxLayout()
        self.hLayoutSaveClear.addWidget(self.buttonSaveCommand)

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

        # ----------- Button box -------------
        self.buttonBox = QDialogButtonBox()

        self.buttonStop = QPushButton("Stop")
        self.buttonStop.clicked.connect(self.stop_thread)
        self.buttonStop.setEnabled(True)
        self.buttonBox.addButton(self.buttonStop, QDialogButtonBox.ActionRole)

        self.buttonClose = QPushButton("Close")
        self.buttonClose.clicked.connect(self.close)
        self.buttonClose.setEnabled(False)
        self.buttonBox.addButton(self.buttonClose, QDialogButtonBox.ActionRole)

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
        self.setGeometry(0, 0, 640, 800)

        # Process ID
        self.pid = os.getpid()
        self.thread = QThread()
        self.run_worker()
        # self.worker = Worker(self.train)

    def create_trainer(self):
        """
        Create the ClaspyTrainer object according config file
        """
        # Check the mode of ClaspyRun
        if self.mode != 'train':
            raise ValueError('Function create_trainer() only in training mode !')

        # Special train arguments
        if self.config['grid_search']:
            param_grid = self.config['param_grid']
            parameters = None
        else:
            param_grid = None
            parameters = self.config['parameters']

        try:
            pca = self.config['pca']
        except KeyError:
            pca = None

        # Create ClaspyTrainer object
        trainer = ClaspyTrainer(input_data=self.config['input_file'],
                                output_data=self.config['output_folder'],
                                algo=None,
                                algorithm=self.config['algorithm'],
                                parameters=parameters,
                                features=self.config['feature_names'],
                                grid_search=self.config['grid_search'],
                                grid_param=param_grid,
                                pca=pca,
                                n_jobs=self.config['n_jobs_cv'],
                                random_state=self.config['random_state'],
                                samples=self.config['samples'],
                                scaler=self.config['scaler'],
                                scoring=self.config['scorer'],
                                train_ratio=self.config['training_ratio'],
                                png_features=self.config['png_features'])

        return trainer

    def create_predicter(self):
        """
        Create the ClaspyPredicter object according config file
        """
        # Check the mode of ClaspyRun
        if self.mode != 'predi':
            raise ValueError('Function create_predicter() only in predict mode !')

        # Create ClaspyPredicter object
        predicter = ClaspyPredicter(input_data=self.config['input_file'],
                                    output_data=self.config['output_folder'],
                                    model=self.config['model'])

        return predicter

    def create_segmenter(self):
        """
        Create the ClaspySegmenter object according config file
        """
        # Check the mode of ClaspyRun
        if self.mode != 'segme':
            raise ValueError('Function create_segmenter() only in segmentation mode !')

        # Create ClaspySegmenter object
        segmenter = ClaspySegmenter(input_data=self.config['input_file'],
                                    output_data=self.config['output_folder'],
                                    parameters=self.config['parameters'],
                                    features=self.config['feature_names'])

        return segmenter

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
        self.plainTextCommand.ensureCursorVisible()

    def update_progress(self, n):
        """Update progressBar according given percent as integer"""
        self.progressBar.setValue(n)

    def run_worker(self):
        # Update button
        self.buttonClose.setEnabled(False)
        self.buttonStop.setEnabled(True)

        # Create ClaspyObjects
        if self.mode == 'train':
            #self.trainer = self.create_trainer()
            self.worker = Worker(self.train)
        elif self.mode == 'predi':
            #self.predicter = self.create_predicter()
            self.worker = Worker(self.predict)
        elif self.mode == 'segme':
            #self.segmenter = self.create_segmenter()
            self.worker = Worker(self.segment)

        # Set Worker as thread
        self.thread = QThread()
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

    def train(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Training Mode   - - - - - - - - - - - - - - -")

        # Set the classifier according parameters
        self.trainer.set_classifier()

        # Introduction
        intro = self.trainer.introduction(verbose=True)
        self.message(intro)
        progress_callback.emit(int(0 * 100 / 7))

        # Part 1/7 - Format dataset
        self.message("\nStep 1/7: Formatting data as pandas.DataFrame...")
        step1 = self.trainer.format_dataset(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 7))

        # Part 2/7 - Split data into training and testing sets
        self.message("\nStep 2/7: Splitting data in train and test sets...")
        step2 = self.trainer.split_dataset(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 7))

        # Part 3/7 - Scale dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 3/7: Scaling data...")
        step3 = self.trainer.set_scaler_pca(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 7))

        # Part 4/7 - Train model
        if self.trainer.grid_search:  # Training with GridSearchCV
            self.message('\nStep 4/7: Training model with GridSearchCV...\n')
        else:  # Training with Cross Validation
            self.message("\nStep 4/7: Training model with cross validation...\n")

        step4 = self.trainer.train_model(verbose=True)  # Perform both training
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 7))

        # Part 5/7 - Create confusion matrix
        self.message("\nStep 5/7: Creating confusion matrix...")
        step5 = self.trainer.confusion_matrix(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 7))

        # Part 6/7 - Save algorithm, model, scaler, pca and feature_names
        self.message("\nStep 6/7: Saving model and scaler in file:")
        step6 = self.trainer.save_model(verbose=True)
        self.message(step6)
        progress_callback.emit(int(6 * 100 / 7))

        # Part 7/7 - Create and save prediction report
        self.message("\nStep 7/7: Creating training report:")
        self.message("\n" + self.trainer.report_filename + '.txt')
        step7 = self.trainer.classification_report(verbose=True)
        self.message(step7)
        progress_callback.emit(int(7 * 100 / 7))

        return "Training done!"

    def predict(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Prediction Mode   - - - - - - - - - - - - - - -")

        # Load model
        self.message("\nStep 1/6: Loading model...")
        step1 = self.predicter.load_model(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 6))

        # Introduction
        intro = self.predicter.introduction(verbose=True)
        self.message(intro)

        # Format dataset
        self.message("\nStep 2/6: Formatting data as pandas.Dataframe...")
        step2 = self.predicter.format_dataset(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 6))

        # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 3/6: Scaling data...")
        step3 = self.predicter.scale_dataset(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 6))

        # Predict target of input data
        self.message("\nStep 4/6: Making predictions for entire dataset...")
        step4 = self.predicter.predict(verbose=True)
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 6))

        # Save classification results as point cloud file with all data
        self.message("\nStep 5/6: Saving classified point cloud:")
        step5 = self.predicter.save_predictions(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 6))

        # Create and save prediction report
        self.message("\nStep 6/6: Creating prediction report:")
        self.message("\n" + self.predicter.report_filename + '.txt')
        step6 = self.predicter.classification_report(verbose=True)
        self.message(step6)
        progress_callback.emit(int(6 * 100 / 6))

        return "Predictions done!"

    def segment(self, progress_callback):
        self.message("\n# # # # # # # # # #  CLASPY_T  # # # # # # # # # # # #"
                     "\n"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n"
                     "\n - - - - - - - - - - - - - -   Segmentation Mode   - - - - - - - - - - - - - - -")

        # Set the classifier according parameters
        self.segmenter.set_classifier()

        # Introduction
        intro = self.segmenter.introduction(verbose=True)
        self.message(intro)
        progress_callback.emit(int(0 * 100 / 5))

        # Format dataset
        self.message("\nStep 1/5: Formatting data as pandas.DataFrame...")
        step1 = self.segmenter.format_dataset(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1 * 100 / 5))

        # Scale the dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 2/5: Scaling data...")
        step2 = self.segmenter.set_scaler_pca(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2 * 100 / 5))

        # Split data into training and testing sets
        self.message("\nStep 3/5: Clustering the dataset...")
        step3 = self.segmenter.segment(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3 * 100 / 5))

        # Save algorithm, model, scaler, pca and feature_names
        self.message("\nStep 4/5: Saving segmented point cloud in file...")
        step4 = self.segmenter.save_clusters(verbose=True)
        self.message(step4)
        progress_callback.emit(int(4 * 100 / 5))

        # Create and save prediction report
        self.message("\nStep 5/5: Creating segmentation report:")
        self.message("\n" + self.segmenter.report_filename + '.txt')
        step5 = self.segmenter.classification_report(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5 * 100 / 5))

        return "Segmentation done!"

    def handle_state(self, state):
        states = {QProcess.NotRunning: "Not Running",
                  QProcess.Starting: "Starting",
                  QProcess.Running: "Running"}

        state_name = states[state]
        self.statusBar.showMessage("cLASpy_T run is {}".format(state_name), 5000)

    def thread_complete(self):
        self.statusBar.showMessage("cLASpy_T run finished!", 5000)

        self.progressBar.reset()
        self.buttonStop.setEnabled(False)
        self.buttonClose.setEnabled(True)

    def stop_thread(self):
        self.buttonStop.setEnabled(False)
        self.buttonStop.setText("Stopping...")

        time.sleep(1)
        parent = psutil.Process(self.pid)
        for child in parent.children(recursive=True):
            child.kill()

        self.buttonStop.setText("Stop")
        self.buttonClose.setEnabled(True)

        self.plainTextCommand.appendPlainText("\n********************"
                                              "\nProcess stopped by user!"
                                              "\n********************")

    def closeEvent(self, event):
        self.stop_thread()
        time.sleep(1)
        self.close()


# execute run() function
run(args)
