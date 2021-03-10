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
#          The GUI 'cLASpy_GUI.py' for the cLASpy_T library           #
#                    By Xavier PELLERIN LE BAS                        #
#                         February 2021                               #
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
import time
import laspy

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from common import *

# -------------------------
# -------- CLASS ----------
# -------------------------


class ClaspyGui(QMainWindow):
    def __init__(self, parent=None):
        super(ClaspyGui, self).__init__(parent)
        self.setWindowTitle("cLASpy_GUI")
        self.setGeometry(300, 200, 900, 700)
        self.mainWidget = QWidget()

        # Left part of GUI

        # GroupBox for point cloud file
        self.groupPtCld = QGroupBox("Point Cloud")

        self.labelLocalServer = QLabel("Compute on:")
        self.layoutLocalServer = QHBoxLayout()
        self.radioLocal = QRadioButton("local")
        self.layoutLocalServer.addWidget(self.radioLocal)
        self.radioServer = QRadioButton("server")
        self.layoutLocalServer.addWidget(self.radioServer)
        self.radioLocal.setChecked(True)

        # Stacks for input point cloud
        self.stack_Local = QWidget()
        self.stack_Server = QWidget()
        self.stackui_local()
        self.stackui_server()
        self.stackInput = QStackedWidget(self)
        self.stackInput.addWidget(self.stack_Local)
        self.stackInput.addWidget(self.stack_Server)

        # Info about the input point cloud
        self.labelPtCldFormat = QLabel()
        self.labelPtCount = QLabel()

        # Set the sample size
        self.spinSampleSize = QDoubleSpinBox()
        self.spinSampleSize.setMaximumWidth(80)
        self.spinSampleSize.setMinimum(0)
        self.spinSampleSize.setDecimals(6)
        self.spinSampleSize.setWrapping(True)
        self.hLayoutSize = QHBoxLayout()
        self.hLayoutSize.addWidget(self.spinSampleSize)
        self.hLayoutSize.addWidget(QLabel("Million points"))

        # Set the training/testing ratio
        self.spinTrainRatio = QDoubleSpinBox()
        self.spinTrainRatio.setMaximumWidth(60)
        self.spinTrainRatio.setMinimum(0)
        self.spinTrainRatio.setMaximum(1)
        self.spinTrainRatio.setSingleStep(0.1)
        self.spinTrainRatio.setDecimals(3)
        self.spinTrainRatio.setWrapping(True)
        self.spinTrainRatio.setValue(0.5)

        # Label: Target field exist ?
        self.labelTarget = QLabel()

        # Fill the point cloud groupBox with QFormLayout
        self.formLayoutPtCld = QFormLayout()
        self.formLayoutPtCld.addRow(self.labelLocalServer, self.layoutLocalServer)
        self.formLayoutPtCld.addRow(self.stackInput)
        self.formLayoutPtCld.addRow("Format:", self.labelPtCldFormat)
        self.formLayoutPtCld.addRow("Number of points:", self.labelPtCount)
        self.formLayoutPtCld.addRow("Number of samples:", self.hLayoutSize)
        self.formLayoutPtCld.addRow("Training ratio:", self.spinTrainRatio)
        self.formLayoutPtCld.addRow("Target field:", self.labelTarget)
        self.groupPtCld.setLayout(self.formLayoutPtCld)

        # Selection of the algorithm
        self.groupAlgorithm = QGroupBox("Algorithm")

        self.listAlgorithms = QListWidget()
        self.listAlgorithms.setMaximumSize(120, 80)
        self.listAlgorithms.insertItem(0, "Random Forest")
        self.listAlgorithms.insertItem(1, "Gradient Boosting")
        self.listAlgorithms.insertItem(2, "Neural Network")
        self.listAlgorithms.insertItem(3, "K-Means Clustering")
        self.listAlgorithms.setCurrentItem(self.listAlgorithms.item(0))

        # Stacks for the parameters of the algo
        self.stack_RF = QWidget()
        self.stack_GB = QWidget()
        self.stack_NN = QWidget()
        self.stack_KM = QWidget()
        self.stackui_rf()
        self.stackui_gb()
        self.stackui_nn()
        self.stackui_km()
        self.stackAlgo = QStackedWidget(self)
        self.stackAlgo.addWidget(self.stack_RF)
        self.stackAlgo.addWidget(self.stack_GB)
        self.stackAlgo.addWidget(self.stack_NN)
        self.stackAlgo.addWidget(self.stack_KM)

        # Fill the algorithm groupBox with QFormLayout
        self.formLayoutAlgo = QFormLayout()
        self.formLayoutAlgo.addRow("Select algorithm:", self.listAlgorithms)
        self.formLayoutAlgo.addRow("Algorithm parameters:", self.stackAlgo)
        self.groupAlgorithm.setLayout(self.formLayoutAlgo)

        # Fill the left layout
        self.vLayoutLeft = QVBoxLayout()
        self.vLayoutLeft.addWidget(self.groupPtCld)
        self.vLayoutLeft.addWidget(self.groupAlgorithm)

        # Right part of GUI
        self.groupFeatures = QGroupBox("Feature Fields")

        # List of features
        self.labelFeatures = QLabel("Select features:\n\n\n"
                                    "(press Ctrl+Shift\n"
                                    "for multiple selection)")
        self.labelFeatures.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.listFeatures = QListWidget()
        self.listFeatures.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listFeatures.setSortingEnabled(True)

        # Number of selected features
        self.labelNbrSelFeatures = QLabel()
        self.number_selected_features()

        # Fill the feature groupBox with layout
        self.formLayoutRight = QFormLayout()
        self.formLayoutRight.addRow(self.labelFeatures, self.listFeatures)
        self.formLayoutRight.addRow("Number of selected features:", self.labelNbrSelFeatures)
        self.groupFeatures.setLayout(self.formLayoutRight)

        # Button box for
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonRun = QPushButton("Run cLASpy_T")
        self.buttonRun.clicked.connect(self.run_claspy_t)
        self.buttonBox.addButton(self.buttonRun, QDialogButtonBox.ActionRole)

        # Fill the main layout
        self.hMainLayout = QHBoxLayout()
        self.hMainLayout.addLayout(self.vLayoutLeft)
        self.hMainLayout.addWidget(self.groupFeatures)

        self.vMainLayout = QVBoxLayout(self.mainWidget)
        self.vMainLayout.addLayout(self.hMainLayout)
        self.vMainLayout.addWidget(self.buttonBox)

        # setCentralWidget
        self.setCentralWidget(self.mainWidget)

        # MenuBar and Menu
        bar = self.menuBar()
        menu_file = bar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        menu_file.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        menu_file.addAction(save_action)

        quit_action = QAction("Quit", self)
        menu_file.addAction(quit_action)

        menu_file.triggered[QAction].connect(self.menu_trigger)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Function call
        self.listAlgorithms.currentRowChanged.connect(self.display_stack_algo)
        self.radioLocal.toggled.connect(self.display_stack_input)
        self.listFeatures.itemSelectionChanged.connect(self.number_selected_features)

    def menu_trigger(self, action):
        if action.text() == "Open":
            self.open_config()

        elif action.text() == "Save":
            self.save_config()

        elif action.text() == "Quit":
            self.reject()

    def stackui_local(self):
        form_layout = QFormLayout()

        self.lineLocalFile = QLineEdit()
        self.lineLocalFile.setPlaceholderText("Select LAS or CSV file as input")
        self.toolButtonFile = QToolButton()
        self.toolButtonFile.setText("Browse")
        self.hLayoutFile = QHBoxLayout()
        self.hLayoutFile.addWidget(self.lineLocalFile)
        self.hLayoutFile.addWidget(self.toolButtonFile)
        form_layout.addRow("Input file:", self.hLayoutFile)

        self.lineLocalFolder = QLineEdit()
        self.lineLocalFolder.setPlaceholderText("Folder where save result files")
        self.toolButtonFolder = QToolButton()
        self.toolButtonFolder.setText("Browse")
        self.hLayoutFolder = QHBoxLayout()
        self.hLayoutFolder.addWidget(self.lineLocalFolder)
        self.hLayoutFolder.addWidget(self.toolButtonFolder)
        form_layout.addRow("Output folder:", self.hLayoutFolder)

        self.stack_Local.setLayout(form_layout)

        self.toolButtonFile.clicked.connect(self.get_file)
        self.lineLocalFile.editingFinished.connect(self.open_file)
        self.toolButtonFolder.clicked.connect(self.get_folder)

    def stackui_server(self):
        form_layout = QFormLayout()

        self.lineFile = QLineEdit()
        self.lineFile.setPlaceholderText("Local file to get features")
        self.toolButtonFile = QToolButton()
        self.toolButtonFile.setText("Browse")
        self.hLayoutFile = QHBoxLayout()
        self.hLayoutFile.addWidget(self.lineFile)
        self.hLayoutFile.addWidget(self.toolButtonFile)
        form_layout.addRow("Local input file:", self.hLayoutFile)

        self.lineServerFile = QLineEdit()
        self.lineServerFile.setPlaceholderText("File on server to compute on")
        form_layout.addRow("Server input file:", self.lineServerFile)

        self.lineServerFolder = QLineEdit()
        self.lineServerFolder.setPlaceholderText("Folder on server where save result files")
        form_layout.addRow("Server output folder:", self.lineServerFolder)

        self.stack_Server.setLayout(form_layout)

        self.toolButtonFile.clicked.connect(self.get_file)
        self.lineFile.editingFinished.connect(self.open_file)

    def display_stack_input(self):
        if self.radioLocal.isChecked():
            self.stackInput.setCurrentIndex(0)
        else:
            self.stackInput.setCurrentIndex(1)

    def get_file(self):
        self.statusBar.showMessage("Select file...", 3000)
        filename = QFileDialog.getOpenFileName(self, 'Select CSV or LAS file',
                                               '', "LAS files (*.las);;CSV files (*.csv)")

        if filename[0] != '':
            if self.radioLocal.isChecked():
                self.lineLocalFile.setText(os.path.normpath(filename[0]))
            else:
                self.lineFile.setText(os.path.normpath(filename[0]))

            self.open_file()

    def open_file(self):
        if self.radioLocal.isChecked():
            file_path = os.path.normpath(self.lineLocalFile.text())
        else:
            file_path = os.path.normpath(self.lineFile.text())

        root_ext = os.path.splitext(file_path)
        if self.radioLocal.isChecked():
            self.lineLocalFolder.setText(os.path.splitext(root_ext[0])[0])

        if root_ext[1] == '.csv':
            feature_names = ["Encore", "en", "Test"]
            self.target = False
        elif root_ext[1] == '.las':
            feature_names = self.open_las(file_path)
        else:
            feature_names = ["File error:", "Unknown extension file!"]
            self.statusBar.showMessage("File error: Unknown extension file!", 3000)

        # Check if the target field exist
        try:
            self.target
        except AttributeError:
            self.target = False

        if self.target:
            self.labelTarget.setText("Target field is available")
        else:
            self.labelTarget.setText("Target field not found!!")

        # Rewrite listFeature
        self.listFeatures.clear()
        for item in feature_names:
            self.listFeatures.addItem(str(item))

    def open_las(self, file_path):
        """
        Open LAS file and only return the extra dimension.
        :param file_path: The path to the LAS file.
        :return: The list of the extra dimensions (with 'Target' field).
        """
        # Initialize target bool
        self.target = False

        # Infos about LAS file
        las = laspy.file.File(file_path, mode='r')
        version = las.header.version
        data_format = las.header.data_format_id
        point_count = las.header.records_count

        # Set value of the train size
        number_mpts = float(point_count / 1000000.)  # Number of million points
        self.spinSampleSize.setMaximum(number_mpts)
        if number_mpts > 2:
            self.spinSampleSize.setValue(1)
        else:
            self.spinSampleSize.setValue(number_mpts/2.)

        # Show LAS version and number of points in status bar
        point_count = '{:,}'.format(point_count).replace(',', ' ')  # Format with thousand separator
        self.labelPtCldFormat.setText("LAS version {}  |  Data format: {}".format(version, data_format))
        self.labelPtCount.setText("{}".format(point_count))
        self.statusBar.showMessage("{} points | LAS Version: {}".format(point_count, version),
                                   5000)

        # List of the standard dimensions according the data_format of LAS
        standard_dimensions = point_format[data_format]

        # List of all dimensions
        extra_dimensions = list()
        for dim in las.point_format.specs:
            extra_dimensions.append(str(dim.name))

        # Remove the standard dimensions
        for dim in standard_dimensions:
            if dim in extra_dimensions:
                extra_dimensions.remove(dim)

        # Find 'target' dimension if exist
        for trgt in ['target', 'Target', 'TARGET']:
            if trgt in extra_dimensions:
                self.target = True
                self.targetName = trgt
                extra_dimensions.remove(trgt)

        return extra_dimensions

    def get_folder(self):
        """
        Create folder to save model and report files
        """
        file_path = os.path.splitext(self.lineLocalFile.text())[0]
        # Open QFile Dialog with empty lineFile
        if file_path == '':
            foldername = QFileDialog.getExistingDirectory(self, 'Select output folder')

        else:
            foldername = QFileDialog.getExistingDirectory(self, 'Select output folder',
                                                          file_path + '/..')

        if foldername != '':
            if self.radioLocal.isChecked():
                self.lineLocalFolder.setText(foldername)

    def number_selected_features(self):
        self.max_count = 0
        self.selected_count = 0
        if self.listFeatures is None or self.listFeatures.count() == 0:
            self.max_count = 0
            self.selected_count = 0
        else:
            self.max_count = self.listFeatures.count()
            self.selected_count = len(self.listFeatures.selectedItems())

        self.labelNbrSelFeatures.setText("{} / {}".format(self.selected_count,
                                                          self.max_count))

    def stackui_rf(self):
        form_layout = QFormLayout()

        self.RFspinRandomState = QSpinBox()
        self.RFspinRandomState.setMaximumWidth(80)
        self.RFspinRandomState.setMinimum(-1)
        self.RFspinRandomState.setMaximum(999999)
        self.RFspinRandomState.setValue(-1)
        self.RFspinRandomState.setToolTip("Controls the randomness to split the data into\n"
                                          "train/test sets and to build the trees.")
        form_layout.addRow("random_state:", self.RFspinRandomState)

        self.RFspinEstimators = QSpinBox()
        self.RFspinEstimators.setMaximumWidth(80)
        self.RFspinEstimators.setMinimum(2)
        self.RFspinEstimators.setMaximum(999999)
        self.RFspinEstimators.setValue(100)
        self.RFspinEstimators.setToolTip("The number of trees in the forest.")
        form_layout.addRow("n_estimators:", self.RFspinEstimators)

        self.RFcriterion = ["gini", "entropy"]
        self.RFcomboCriterion = QComboBox()
        self.RFcomboCriterion.setMaximumWidth(80)
        self.RFcomboCriterion.addItems(self.RFcriterion)
        self.RFcomboCriterion.setCurrentIndex(self.RFcriterion.index("gini"))
        self.RFcomboCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.RFcomboCriterion)

        self.RFspinMaxDepth = QSpinBox()
        self.RFspinMaxDepth.setMaximumWidth(80)
        self.RFspinMaxDepth.setMinimum(0)
        self.RFspinMaxDepth.setMaximum(9999)
        self.RFspinMaxDepth.setValue(0)
        self.RFspinMaxDepth.setToolTip("The maximum depth of the tree.")
        form_layout.addRow("max_depth:", self.RFspinMaxDepth)

        self.RFspinSamplesSplit = QSpinBox()
        self.RFspinSamplesSplit.setMaximumWidth(80)
        self.RFspinSamplesSplit.setMinimum(2)
        self.RFspinSamplesSplit.setMaximum(999999)
        self.RFspinSamplesSplit.setValue(2)
        self.RFspinSamplesSplit.setToolTip("The minimum number of samples required\n"
                                           "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.RFspinSamplesSplit)

        self.RFspinSamplesLeaf = QSpinBox()
        self.RFspinSamplesLeaf.setMaximumWidth(80)
        self.RFspinSamplesLeaf.setMaximum(999999)
        self.RFspinSamplesLeaf.setValue(1)
        self.RFspinSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                          "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.RFspinSamplesLeaf)

        self.RFspinWeightLeaf = QDoubleSpinBox()
        self.RFspinWeightLeaf.setMaximumWidth(80)
        self.RFspinWeightLeaf.setDecimals(4)
        self.RFspinWeightLeaf.setMaximum(1)
        self.RFspinWeightLeaf.setValue(0)
        self.RFspinWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                         "of weights required to be at a leaf node. Samples\n"
                                         "have equal weight when sample_weight=0.")
        form_layout.addRow("min_weight_fraction_leaf:", self.RFspinWeightLeaf)

        self.maxFeatures = ["auto", "sqrt", "log2"]
        self.RFcomboMaxFeatures = QComboBox()
        self.RFcomboMaxFeatures.setMaximumWidth(80)
        self.RFcomboMaxFeatures.addItems(self.maxFeatures)
        self.RFcomboMaxFeatures.setCurrentIndex(self.maxFeatures.index("auto"))
        self.RFcomboMaxFeatures.setToolTip("The number of features to consider\n"
                                           "when looking for the best split.")
        form_layout.addRow("max_features:", self.RFcomboMaxFeatures)

        self.RFspinNJob = QSpinBox()
        self.RFspinNJob.setMaximumWidth(80)
        self.RFspinNJob.setMinimum(-1)
        self.RFspinNJob.setValue(0)
        self.RFspinNJob.setToolTip("The number of jobs to run in parallel.\n"
                                   "'0' means one job at the same time.\n"
                                   "'-1' means using all processors.")
        form_layout.addRow("n_jobs:", self.RFspinNJob)

        label_h_line = QLabel()
        label_h_line.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        form_layout.addRow(label_h_line)

        self.RFcheckImportance = QCheckBox()
        self.RFcheckImportance.setChecked(False)
        self.RFcheckImportance.setToolTip("Export the impurity-based feature importances\n"
                                          "as PNG image. The higher, the more important\n"
                                          "feature. It is also known as the Gini importance.")
        form_layout.addRow("Export feature importances:", self.RFcheckImportance)

        self.stack_RF.setLayout(form_layout)

    def stackui_gb(self):
        form_layout = QFormLayout()

        self.GBspinRandomState = QSpinBox()
        self.GBspinRandomState.setMaximumWidth(80)
        self.GBspinRandomState.setMinimum(-1)
        self.GBspinRandomState.setMaximum(999999)
        self.GBspinRandomState.setValue(-1)
        self.GBspinRandomState.setToolTip("Controls the randomness to split the data into\n"
                                          "train/test sets and to build the trees.")
        form_layout.addRow("random_state:", self.GBspinRandomState)

        self.GBspinEstimators = QSpinBox()
        self.GBspinEstimators.setMaximumWidth(80)
        self.GBspinEstimators.setMinimum(2)
        self.GBspinEstimators.setMaximum(9999999)
        self.GBspinEstimators.setValue(100)
        self.GBspinEstimators.setToolTip("The number of boosting stages to perform.\n"
                                         "Gradient boosting is fairly robust to over-\n"
                                         "fitting so a large number usually results in\n"
                                         "better performance.")
        form_layout.addRow("n_estimators:", self.GBspinEstimators)

        self.GBcriterion = ["friedman_mse", "mse"]
        self.GBcomboCriterion = QComboBox()
        self.GBcomboCriterion.setMaximumWidth(80)
        self.GBcomboCriterion.addItems(self.GBcriterion)
        self.GBcomboCriterion.setCurrentIndex(self.GBcriterion.index("friedman_mse"))
        self.GBcomboCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.GBcomboCriterion)

        self.GBspinMaxDepth = QSpinBox()
        self.GBspinMaxDepth.setMaximumWidth(80)
        self.GBspinMaxDepth.setMinimum(0)
        self.GBspinMaxDepth.setMaximum(9999)
        self.GBspinMaxDepth.setValue(3)
        self.GBspinMaxDepth.setToolTip("The maximum depth of the individual regression estimators.\n"
                                       "The maximum depth limits the number of nodes in the tree. ")
        form_layout.addRow("max_depth:", self.GBspinMaxDepth)

        self.GBspinSamplesSplit = QSpinBox()
        self.GBspinSamplesSplit.setMaximumWidth(80)
        self.GBspinSamplesSplit.setMinimum(2)
        self.GBspinSamplesSplit.setMaximum(999999)
        self.GBspinSamplesSplit.setValue(2)
        self.GBspinSamplesSplit.setToolTip("The minimum number of samples required\n"
                                           "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.GBspinSamplesSplit)

        self.GBspinSamplesLeaf = QSpinBox()
        self.GBspinSamplesLeaf.setMaximumWidth(80)
        self.GBspinSamplesLeaf.setMaximum(999999)
        self.GBspinSamplesLeaf.setValue(1)
        self.GBspinSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                          "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.GBspinSamplesLeaf)

        self.GBspinWeightLeaf = QDoubleSpinBox()
        self.GBspinWeightLeaf.setMaximumWidth(80)
        self.GBspinWeightLeaf.setDecimals(4)
        self.GBspinWeightLeaf.setMaximum(1)
        self.GBspinWeightLeaf.setValue(0)
        self.GBspinWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                         "of weights required to be at a leaf node. Samples\n"
                                         "have equal weight when sample_weight=0.")
        form_layout.addRow("min_weight_fraction_leaf:", self.GBspinWeightLeaf)

        self.maxFeatures = ["None", "auto", "sqrt", "log2"]
        self.GBcomboMaxFeatures = QComboBox()
        self.GBcomboMaxFeatures.setMaximumWidth(80)
        self.GBcomboMaxFeatures.addItems(self.maxFeatures)
        self.GBcomboMaxFeatures.setCurrentIndex(self.maxFeatures.index("None"))
        self.GBcomboMaxFeatures.setToolTip("The number of features to consider\n"
                                           "when looking for the best split.")
        form_layout.addRow("max_features:", self.GBcomboMaxFeatures)

        self.loss = ["deviance", "exponential"]
        self.GBcomboLoss = QComboBox()
        self.GBcomboLoss.setMaximumWidth(80)
        self.GBcomboLoss.addItems(self.loss)
        self.GBcomboLoss.setCurrentIndex(self.loss.index("deviance"))
        self.GBcomboLoss.setToolTip("The loss function to be optimized. ‘deviance’ refers to logistic\n"
                                    "regression for classification with probabilistic outputs. For loss \n"
                                    "‘exponential’ gradient boosting recovers the AdaBoost algorithm.")
        form_layout.addRow("loss:", self.GBcomboLoss)

        self.GBspinLearningRate = QDoubleSpinBox()
        self.GBspinLearningRate.setMaximumWidth(80)
        self.GBspinLearningRate.setDecimals(6)
        self.GBspinLearningRate.setMinimum(0)
        self.GBspinLearningRate.setValue(0.1)
        self.GBspinLearningRate.setToolTip("Learning rate shrinks the contribution of each tree\n"
                                           "by learning_rate. There is a trade-off between\n"
                                           "learning_rate and n_estimators.")
        form_layout.addRow("learning_rate:", self.GBspinLearningRate)

        self.GBspinSubsample = QDoubleSpinBox()
        self.GBspinSubsample.setMaximumWidth(80)
        self.GBspinSubsample.setDecimals(4)
        self.GBspinSubsample.setMinimum(0)
        self.GBspinSubsample.setValue(1)
        self.GBspinSubsample.setToolTip("The fraction of samples to be used for fitting\n"
                                        "the individual base learners. If smaller than\n"
                                        "1.0 this results in Stochastic Gradient Boosting.")
        form_layout.addRow("subsample:", self.GBspinSubsample)

        label_h_line = QLabel()
        label_h_line.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        form_layout.addRow(label_h_line)

        self.GBcheckImportance = QCheckBox()
        self.GBcheckImportance.setChecked(False)
        self.GBcheckImportance.setToolTip("Export the impurity-based feature importances\n"
                                          "as PNG image. The higher, the more important\n"
                                          "feature. It is also known as the Gini importance.")
        form_layout.addRow("Export feature importances:", self.GBcheckImportance)

        self.stack_GB.setLayout(form_layout)

    def stackui_nn(self):
        form_layout = QFormLayout()

        self.NNspinRandomState = QSpinBox()
        self.NNspinRandomState.setMaximumWidth(80)
        self.NNspinRandomState.setMinimum(-1)
        self.NNspinRandomState.setMaximum(999999)
        self.NNspinRandomState.setValue(-1)
        self.NNspinRandomState.setToolTip("Determines random number generation for weights and\n"
                                          "bias initialization, train-test split if early stopping is used,\n"
                                          "and batch sampling when solver=’sgd’or ‘adam’.")
        form_layout.addRow("random_state:", self.NNspinRandomState)

        self.NNlineHiddenLayers = QLineEdit()
        self.NNlineHiddenLayers.setPlaceholderText("Example: 50,100,50")
        self.NNlineHiddenLayers.setToolTip("The ith element represents the number of neurons\n"
                                           "in the ith hidden layer.")
        form_layout.addRow("hidden_layer_sizes:", self.NNlineHiddenLayers)

        self.NNactivation = ["identity", "logistic", "tanh", "relu"]
        self.NNcomboActivation = QComboBox()
        self.NNcomboActivation.setMaximumWidth(80)
        self.NNcomboActivation.addItems(self.NNactivation)
        self.NNcomboActivation.setCurrentIndex(self.NNactivation.index("relu"))
        self.NNcomboActivation.setToolTip("Activation function for the hidden layer.")
        form_layout.addRow("activation:", self.NNcomboActivation)

        self.NNsolver = ["lbfgs", "sgd", "adam"]
        self.NNcomboSolver = QComboBox()
        self.NNcomboSolver.setMaximumWidth(80)
        self.NNcomboSolver.addItems(self.NNsolver)
        self.NNcomboSolver.setCurrentIndex(self.NNsolver.index("adam"))
        self.NNcomboSolver.setToolTip("The solver for weight optimization.\n"
                                      "-'lbfgs' optimizer from quasi-Newton method family.\n"
                                      "-'sgd' refers to stochastic gradient descent.\n"
                                      "-'adam' refers to a stochastic gradient-based optimizer,\n"
                                      "  proposed by Kingma, Diederik, and Jimmy Ba.")
        form_layout.addRow("solver:", self.NNcomboSolver)

        self.NNspinAlpha = QDoubleSpinBox()
        self.NNspinAlpha.setMaximumWidth(80)
        self.NNspinAlpha.setDecimals(8)
        self.NNspinAlpha.setMinimum(0)
        self.NNspinAlpha.setMaximum(999999)
        self.NNspinAlpha.setValue(0.0001)
        self.NNspinAlpha.setToolTip("L2 penalty (regularization term) parameter.")
        form_layout.addRow("alpha:", self.NNspinAlpha)

        self.NNspinBatchSize = QSpinBox()
        self.NNspinBatchSize.setMaximumWidth(80)
        self.NNspinBatchSize.setMinimum(-1)
        self.NNspinBatchSize.setMaximum(999999)
        self.NNspinBatchSize.setValue(-1)
        self.NNspinBatchSize.setToolTip("Size of minibatches for stochastic optimizers.\n"
                                        "If the solver is ‘lbfgs’, the classifier will not use minibatch.")
        form_layout.addRow("batch_size:", self.NNspinBatchSize)

        self.NNlearningRate = ["constant", "invscaling", "adaptive"]
        self.NNcomboLearningRate = QComboBox()
        self.NNcomboLearningRate.setMaximumWidth(80)
        self.NNcomboLearningRate.addItems(self.NNlearningRate)
        self.NNcomboLearningRate.setCurrentIndex(self.NNlearningRate.index("constant"))
        self.NNcomboLearningRate.setEnabled(False)
        self.NNcomboLearningRate.setToolTip("Learning rate schedule for weight updates.")
        form_layout.addRow("learning_rate:", self.NNcomboLearningRate)

        self.NNspinLearningRateInit = QDoubleSpinBox()
        self.NNspinLearningRateInit.setMaximumWidth(80)
        self.NNspinLearningRateInit.setDecimals(6)
        self.NNspinLearningRateInit.setMinimum(0)
        self.NNspinLearningRateInit.setMaximum(9999)
        self.NNspinLearningRateInit.setValue(0.001)
        self.NNspinLearningRateInit.setToolTip("The initial learning rate used. It controls\n"
                                               "the step-size in updating the weights.\n"
                                               "Only used when solver=’sgd’ or ‘adam’.")
        form_layout.addRow("learning_rate_init:", self.NNspinLearningRateInit)

        self.NNspinPowerT = QDoubleSpinBox()
        self.NNspinPowerT.setMaximumWidth(80)
        self.NNspinPowerT.setDecimals(6)
        self.NNspinPowerT.setMinimum(0)
        self.NNspinPowerT.setMaximum(9999)
        self.NNspinPowerT.setValue(0.5)
        self.NNspinPowerT.setEnabled(False)
        self.NNspinPowerT.setToolTip("The exponent for inverse scaling learning rate.\n"
                                     "It is used in updating effective learning rate\n"
                                     "when the learning_rate is set to ‘invscaling’.\n"
                                     "Only used when solver=’sgd’.")
        form_layout.addRow("power_t:", self.NNspinPowerT)

        self.NNspinMaxIter = QSpinBox()
        self.NNspinMaxIter.setMaximumWidth(80)
        self.NNspinMaxIter.setMinimum(1)
        self.NNspinMaxIter.setMaximum(99999)
        self.NNspinMaxIter.setValue(200)
        self.NNspinMaxIter.setToolTip("The solver iterates until convergence or this number of iterations.\n"
                                      "For stochastic solvers ('sgd' or 'adam'), note that this determines\n"
                                      "the number of epochs (how many times each data point will be\n"
                                      "used), not the number of gradient steps.")
        form_layout.addRow("max_iter:", self.NNspinMaxIter)

        self.NNcheckShuffle = QCheckBox()
        self.NNcheckShuffle.setChecked(True)
        self.NNcheckShuffle.setToolTip("Whether to shuffle samples in each iteration.\n"
                                       "Only used when solver=’sgd’ or ‘adam’.")
        form_layout.addRow("shuffle:", self.NNcheckShuffle)

        self.NNspinBeta_1 = QDoubleSpinBox()
        self.NNspinBeta_1.setMaximumWidth(80)
        self.NNspinBeta_1.setDecimals(6)
        self.NNspinBeta_1.setMinimum(0)
        self.NNspinBeta_1.setMaximum(1)
        self.NNspinBeta_1.setValue(0.9)
        self.NNspinBeta_1.setToolTip("Exponential decay rate for estimates of first\n"
                                     "moment vector in adam, should be in [0, 1).\n"
                                     "Only used when solver=’adam’")
        form_layout.addRow("beta_1:", self.NNspinBeta_1)

        self.NNspinBeta_2 = QDoubleSpinBox()
        self.NNspinBeta_2.setMaximumWidth(80)
        self.NNspinBeta_2.setDecimals(6)
        self.NNspinBeta_2.setMinimum(0)
        self.NNspinBeta_2.setMaximum(1)
        self.NNspinBeta_2.setValue(0.999)
        self.NNspinBeta_2.setToolTip("Exponential decay rate for estimates of second\n"
                                     "moment vector in adam, should be in [0, 1).\n"
                                     "Only used when solver=’adam’")
        form_layout.addRow("beta_2:", self.NNspinBeta_2)

        self.NNspinEpsilon = QDoubleSpinBox()
        self.NNspinEpsilon.setMaximumWidth(80)
        self.NNspinEpsilon.setDecimals(8)
        self.NNspinEpsilon.setMinimum(0)
        self.NNspinEpsilon.setValue(0.00000001)
        self.NNspinEpsilon.setToolTip("Value for numerical stability in adam.\n"
                                      "Only used when solver=’adam’")
        form_layout.addRow("epsilon:", self.NNspinEpsilon)

        self.stack_NN.setLayout(form_layout)

        self.NNcomboSolver.currentTextChanged.connect(self.nn_solver_options)
        self.NNcomboLearningRate.currentTextChanged.connect(self.nn_solver_options)

    def nn_solver_options(self):
        solver = self.NNcomboSolver.currentText()
        learning_rate = self.NNcomboLearningRate.currentText()

        if solver == 'lbfgs':
            self.NNspinBatchSize.setEnabled(False)
            self.NNcomboLearningRate.setEnabled(False)
            self.NNspinLearningRateInit.setEnabled(False)
            self.NNcheckShuffle.setEnabled(False)
            self.NNspinBeta_1.setEnabled(False)
            self.NNspinBeta_2.setEnabled(False)
            self.NNspinEpsilon.setEnabled(False)
        elif solver == 'sgd':
            self.NNspinBatchSize.setEnabled(True)
            self.NNcomboLearningRate.setEnabled(True)
            self.NNspinLearningRateInit.setEnabled(True)
            self.NNcheckShuffle.setEnabled(True)
            self.NNspinBeta_1.setEnabled(False)
            self.NNspinBeta_2.setEnabled(False)
            self.NNspinEpsilon.setEnabled(False)
        elif solver == 'adam':
            self.NNspinBatchSize.setEnabled(True)
            self.NNcomboLearningRate.setEnabled(False)
            self.NNspinLearningRateInit.setEnabled(True)
            self.NNcheckShuffle.setEnabled(True)
            self.NNspinBeta_1.setEnabled(True)
            self.NNspinBeta_2.setEnabled(True)
            self.NNspinEpsilon.setEnabled(True)
        else:
            self.NNspinBatchSize.setEnabled(False)
            self.NNcomboLearningRate.setEnabled(False)
            self.NNspinLearningRateInit.setEnabled(False)
            self.NNspinPowerT.setEnabled(False)
            self.NNcheckShuffle.setEnabled(False)
            self.NNspinBeta_1.setEnabled(False)
            self.NNspinBeta_2.setEnabled(False)
            self.NNspinEpsilon.setEnabled(False)

        if solver == 'sgd' and learning_rate == 'invscaling':
            self.NNspinPowerT.setEnabled(True)
        else:
            self.NNspinPowerT.setEnabled(False)

    def stackui_km(self):
        form_layout = QFormLayout()

        self.KMspinRandomState = QSpinBox()
        self.KMspinRandomState.setMaximumWidth(80)
        self.KMspinRandomState.setMinimum(-1)
        self.KMspinRandomState.setMaximum(999999)
        self.KMspinRandomState.setValue(-1)
        self.KMspinRandomState.setToolTip("Controls the randomness to split the data into train/test\n"
                                          "sets and to determine the generation for centroid.")
        form_layout.addRow("random_state:", self.KMspinRandomState)

        self.KMspinNClusters = QSpinBox()
        self.KMspinNClusters.setMaximumWidth(80)
        self.KMspinNClusters.setMinimum(2)
        self.KMspinNClusters.setMaximum(9999)
        self.KMspinNClusters.setValue(8)
        self.KMspinNClusters.setToolTip("The number of clusters to form as well\n"
                                        "as the number of centroids to generate.")
        form_layout.addRow("n_clusters:", self.KMspinNClusters)

        self.KMinit = ["k-means++", "random"]
        self.KMcomboInit = QComboBox()
        self.KMcomboInit.setMaximumWidth(80)
        self.KMcomboInit.addItems(self.KMinit)
        self.KMcomboInit.setCurrentIndex(self.KMinit.index("k-means++"))
        self.KMcomboInit.setToolTip("Method for initialization.")
        form_layout.addRow("init:", self.KMcomboInit)

        self.KMspinNInit = QSpinBox()
        self.KMspinNInit.setMaximumWidth(80)
        self.KMspinNInit.setMinimum(1)
        self.KMspinNInit.setMaximum(9999)
        self.KMspinNInit.setValue(10)
        self.KMspinNInit.setToolTip("Number of time the k-means algorithm will be\n"
                                    "run with different centroid seeds. The final\n"
                                    "results will be the best output of n_init\n"
                                    "consecutive runs in terms of inertia.")
        form_layout.addRow("n_init:", self.KMspinNInit)

        self.KMspinMaxIter = QSpinBox()
        self.KMspinMaxIter.setMaximumWidth(80)
        self.KMspinMaxIter.setMinimum(1)
        self.KMspinMaxIter.setMaximum(99999)
        self.KMspinMaxIter.setValue(300)
        self.KMspinMaxIter.setToolTip("Maximum number of iterations of the k-means\n"
                                      "algorithm for a single run.")
        form_layout.addRow("max_iter:", self.KMspinMaxIter)

        self.KMspinTol = QDoubleSpinBox()
        self.KMspinTol.setMaximumWidth(80)
        self.KMspinTol.setDecimals(8)
        self.KMspinTol.setMinimum(0)
        self.KMspinTol.setMaximum(9999)
        self.KMspinTol.setValue(0.0001)
        self.KMspinTol.setToolTip("Relative tolerance with regards to Frobenius norm\n"
                                  "of the difference in the cluster centers of two\n"
                                  "consecutive iterations to declare convergence.")
        form_layout.addRow("tol:", self.KMspinTol)

        self.KMalgorithm = ["auto", "full", "elkan"]
        self.KMcomboAlgorithm = QComboBox()
        self.KMcomboAlgorithm.setMaximumWidth(80)
        self.KMcomboAlgorithm.addItems(self.KMalgorithm)
        self.KMcomboAlgorithm.setCurrentIndex(self.KMalgorithm.index("auto"))
        self.KMcomboAlgorithm.setToolTip("K-means algorithm to use.")
        form_layout.addRow("algorithm:", self.KMcomboAlgorithm)

        self.stack_KM.setLayout(form_layout)

    def display_stack_algo(self, i):
        self.stackAlgo.setCurrentIndex(i)
        algo_list = ["Random Forest", "Gradient Boosting", "Neural Network", "K-Means clustering"]
        self.statusBar.showMessage(algo_list[i] + " parameters", 2000)

    def open_config(self):
        """
        Open configuration JSON file and set all saved parameters.
        """
        # Get config filename
        self.statusBar.showMessage("Select config file...", 3000)
        filename = QFileDialog.getOpenFileName(self, 'Select config file',
                                               '', "JSON files (*.json)")

        # Open config file and get data
        with open(filename[0], 'r') as config_file:
            config = json.load(config_file)

        # Update point cloud parameters
        if config['local_compute']:
            self.radioLocal.setChecked(True)
            self.lineLocalFile.setText(config['input_file'])
            self.open_file()
            self.lineLocalFolder.setText(config['output_folder'])
        else:
            self.radioServer.setChecked(True)
            self.lineFile.setText(config['local_input'])
            self.open_file()
            self.lineServerFile.setText(config['input_file'])
            self.lineServerFolder.setText(config['output_folder'])

        self.spinSampleSize.setValue(config['samples'])
        self.spinTrainRatio.setValue(config['training_ratio'])

        # Update algorithm parameters
        algorithm = config['algorithm']
        if algorithm == 'RandomForestClassifier':
            self.listAlgorithms.setCurrentItem(self.listAlgorithms.item(0))
            self.RFcheckImportance.setChecked(config['png_features'])

            # Get algo parameter dict
            parameters = config['parameters']

            # n_estimators
            self.RFspinEstimators.setValue(parameters['n_estimators'])
            # criterion
            self.RFcomboCriterion.setCurrentText(parameters['criterion'])
            # max_depth
            if parameters['max_depth'] is None:
                self.RFspinMaxDepth.setValue(0)
            else:
                self.RFspinMaxDepth.setValue(parameters['max_depth'])
            # min_samples_split
            self.RFspinSamplesSplit.setValue(parameters['min_samples_split'])
            # min_samples_leaf
            self.RFspinSamplesLeaf.setValue(parameters['min_samples_leaf'])
            # min_weight_fraction_leaf
            self.RFspinWeightLeaf.setValue(parameters['min_weight_fraction_leaf'])
            # max_features
            self.RFcomboMaxFeatures.setCurrentText(parameters['max_features'])
            # n_jobs
            if parameters['n_jobs'] is None:
                self.RFspinNJob.setValue(0)
            else:
                self.RFspinNJob.setValue(parameters['n_jobs'])
            # random_state
            if parameters['random_state'] is None:
                self.RFspinRandomState.setValue(-1)
            else:
                self.RFspinRandomState.setValue(parameters['random_state'])

        elif algorithm == 'GradientBoostingClassifier':
            self.listAlgorithms.setCurrentItem(self.listAlgorithms.item(1))
            self.RFcheckImportance.setChecked(config['png_features'])

            # Get algo parameter dict
            parameters = config['parameters']

            # loss
            self.GBcomboLoss.setCurrentText(parameters['loss'])
            # learning_rate
            self.GBspinLearningRate.setValue(parameters['learning_rate'])
            # n_estimators
            self.GBspinEstimators.setValue(parameters['n_estimators'])
            # subsample
            self.GBspinSubsample.setValue(parameters['subsample'])
            # criterion
            self.GBcomboCriterion.setCurrentText(parameters['criterion'])
            # min_samples_split
            self.GBspinSamplesSplit.setValue(parameters['min_samples_split'])
            # min_samples_leaf
            self.GBspinSamplesLeaf.setValue(parameters['min_samples_leaf'])
            # min_weight_fraction_leaf
            self.GBspinWeightLeaf.setValue(parameters['min_weight_fraction_leaf'])
            # max_depth
            if parameters['max_depth'] is None:
                self.GBspinMaxDepth.setValue(0)
            else:
                self.GBspinMaxDepth.setValue(parameters['max_depth'])
            # random_state
            if parameters['random_state'] is None:
                self.GBspinRandomState.setValue(-1)
            else:
                self.GBspinRandomState.setValue(parameters['random_state'])
            # max_features
            if parameters['max_features'] is None:
                self.GBcomboMaxFeatures.setCurrentText('None')
            else:
                self.GBcomboMaxFeatures.setCurrentText(parameters['max_features'])

        elif algorithm == 'MLPClassifier':
            self.listAlgorithms.setCurrentItem(self.listAlgorithms.item(2))

            # Get algo parameter dict
            parameters = config['parameters']

            # hidden_layer_sizes
            hidden_layers = [str(layer) for layer in parameters['hidden_layer_sizes']]
            self.NNlineHiddenLayers.setText(','.join(hidden_layers))
            # activation
            self.NNcomboActivation.setCurrentText(parameters['activation'])
            # solver
            self.NNcomboSolver.setCurrentText(parameters['solver'])
            self.nn_solver_options()
            # alpha
            self.NNspinAlpha.setValue(parameters['alpha'])
            # batch_size
            if self.NNspinBatchSize.isEnabled():
                if parameters['batch_size'] == 'auto':
                    self.NNspinBatchSize.setValue(-1)
                else:
                    self.NNspinBatchSize.setValue(parameters['batch_size'])
            # learning_rate
            if self.NNcomboLearningRate.isEnabled():
                self.NNcomboLearningRate.setCurrentText(parameters['learning_rate'])
            # learning_rate_init
            if self.NNspinLearningRateInit.isEnabled():
                self.NNspinLearningRateInit.setValue(parameters['learning_rate_init'])
            # power_t
            if self.NNspinPowerT.isEnabled():
                self.NNspinPowerT.setValue(parameters['power_t'])
            # max_iter
            self.NNspinMaxIter.setValue(parameters['max_iter'])
            # shuffle
            if self.NNcheckShuffle.isEnabled():
                self.NNcheckShuffle.setChecked(parameters['shuffle'])
            # random_state
            if parameters['random_state'] is None:
                self.NNspinRandomState.setValue(-1)
            else:
                self.NNspinRandomState.setValue(parameters['random_state'])
            # beta_1, beta_2 and epsilon
            if self.NNspinBeta_1.isEnabled():
                self.NNspinBeta_1.setValue(parameters['beta_1'])
            if self.NNspinBeta_2.isEnabled():
                self.NNspinBeta_2.setValue(parameters['beta_2'])
            if self.NNspinEpsilon.isEnabled():
                self.NNspinEpsilon.setValue(parameters['epsilon'])

        elif algorithm == 'KMeans':
            self.listAlgorithms.setCurrentItem(self.listAlgorithms.item(3))

            # Get algo parameter dict
            parameters = config['parameters']

            # n_clusters
            self.KMspinNClusters.setValue(parameters['n_clusters'])
            # init
            self.KMcomboInit.setCurrentText(parameters['init'])
            # n_init
            self.KMspinNInit.setValue(parameters['n_init'])
            # max_iter
            self.KMspinMaxIter.setValue(parameters['max_iter'])
            # tol
            self.KMspinTol.setValue(parameters['tol'])
            # random_state
            if parameters['random_state'] is None:
                self.KMspinRandomState.setValue(-1)
            else:
                self.KMspinRandomState.setValue(parameters['random_state'])
            # algorithm
            self.KMcomboAlgorithm.setCurrentText(parameters['algorithm'])

        else:
            self.statusBar.showMessage('Error: Unknown algorithm from config file!',
                                       3000)

        # Get the selected features
        feature_names = config['feature_names']
        for feature in feature_names:
            item = self.listFeatures.findItems(feature, Qt.MatchExactly)
            if len(item) > 0:
                row = self.listFeatures.row(item[0])
                self.listFeatures.item(row).setSelected(True)

    def save_config(self):
        """
        Save configuration as JSON file.
        """
        # Create the json directory
        json_dict = dict()

        # Save input file, output folder and sample size
        if self.radioLocal.isChecked():
            json_dict['local_compute'] = True
            json_dict['input_file'] = self.lineLocalFile.text()
            json_dict['output_folder'] = self.lineLocalFolder.text()
        else:
            json_dict['local_compute'] = False
            json_dict['local_input'] = self.lineFile.text()
            json_dict['input_file'] = self.lineServerFile.text()
            json_dict['output_folder'] = self.lineServerFolder.text()

        json_dict['samples'] = self.spinSampleSize.value()
        json_dict['training_ratio'] = self.spinTrainRatio.value()

        # Get the selected algorithm and the parameters
        param_dict = dict()
        selected_algo = self.listAlgorithms.selectedItems()[0].text()
        # if Random Forest
        if selected_algo == "Random Forest":
            json_dict['algorithm'] = 'RandomForestClassifier'
            json_dict['png_features'] = self.RFcheckImportance.isChecked()

            # n_estimators
            param_dict['n_estimators'] = self.RFspinEstimators.value()
            # criterion
            param_dict['criterion'] = self.RFcomboCriterion.currentText()
            # max_depth
            if self.RFspinMaxDepth.value() == 0:
                param_dict['max_depth'] = None
            else:
                param_dict['max_depth'] = self.RFspinMaxDepth.value()
            # min_samples_split
            param_dict['min_samples_split'] = self.RFspinSamplesSplit.value()
            # min_samples_leaf
            param_dict['min_samples_leaf'] = self.RFspinSamplesLeaf.value()
            # min_weight_fraction_leaf
            param_dict['min_weight_fraction_leaf'] = self.RFspinWeightLeaf.value()
            # max_features
            param_dict['max_features'] = self.RFcomboMaxFeatures.currentText()
            # n_jobs
            if self.RFspinNJob.value() == 0:
                param_dict['n_jobs'] = None
            else:
                param_dict['n_jobs'] = self.RFspinNJob.value()
            # random_state
            if self.RFspinRandomState.value() == -1:
                param_dict['random_state'] = None
            else:
                param_dict['random_state'] = self.RFspinRandomState.value()

        # if Gradient Boosting
        elif selected_algo == "Gradient Boosting":
            json_dict['algorithm'] = 'GradientBoostingClassifier'
            json_dict['png_features'] = self.GBcheckImportance.isChecked()

            # loss
            param_dict['loss'] = self.GBcomboLoss.currentText()
            # learning_rate
            param_dict['learning_rate'] = self.GBspinLearningRate.value()
            # n_estimators
            param_dict['n_estimators'] = self.GBspinEstimators.value()
            # subsample
            param_dict['subsample'] = self.GBspinSubsample.value()
            # criterion
            param_dict['criterion'] = self.GBcomboCriterion.currentText()
            # min_samples_split
            param_dict['min_samples_split'] = self.GBspinSamplesSplit.value()
            # min_samples_leaf
            param_dict['min_samples_leaf'] = self.GBspinSamplesLeaf.value()
            # min_weight_fraction_leaf
            param_dict['min_weight_fraction_leaf'] = self.GBspinWeightLeaf.value()
            # max_depth
            if self.GBspinMaxDepth.value() == 0:
                param_dict['max_depth'] = None
            else:
                param_dict['max_depth'] = self.GBspinMaxDepth.value()
            # random_state
            if self.GBspinRandomState.value() == -1:
                param_dict['random_state'] = None
            else:
                param_dict['random_state'] = self.GBspinRandomState.value()
            # max_features
            if self.GBcomboMaxFeatures.currentText() == 'None':
                param_dict['max_features'] = None
            else:
                param_dict['max_features'] = self.GBcomboMaxFeatures.currentText()

        # if Neural Network
        elif selected_algo == "Neural Network":
            json_dict['algorithm'] = 'MLPClassifier'

            # hidden_layer_sizes
            hidden_layers = self.NNlineHiddenLayers.text().replace(' ', '')
            if hidden_layers == '':
                param_dict['hidden_layer_sizes'] = (100,)
            else:
                param_dict['hidden_layer_sizes'] = [int(layer) for layer in hidden_layers.split(',')]
            # activation
            param_dict['activation'] = self.NNcomboActivation.currentText()
            # solver
            param_dict['solver'] = self.NNcomboSolver.currentText()
            # alpha
            param_dict['alpha'] = self.NNspinAlpha.value()
            # batch_size
            if self.NNspinBatchSize.isEnabled():
                if self.NNspinBatchSize.value() == -1:
                    param_dict['batch_size'] = 'auto'
                else:
                    param_dict['batch_size'] = self.NNspinBatchSize.value()
            # learning_rate
            if self.NNcomboLearningRate.isEnabled():
                param_dict['learning_rate'] = self.NNcomboLearningRate.currentText()
            # learning_rate_init
            if self.NNspinLearningRateInit.isEnabled():
                param_dict['learning_rate_init'] = self.NNspinLearningRateInit.value()
            # power_t
            if self.NNspinPowerT.isEnabled():
                param_dict['power_t'] = self.NNspinPowerT.value()
            # max_iter
            param_dict['max_iter'] = self.NNspinMaxIter.value()
            # shuffle
            if self.NNcheckShuffle.isEnabled():
                param_dict['shuffle'] = self.NNcheckShuffle.isChecked()
            # random_state
            if self.NNspinRandomState.value() == -1:
                param_dict['random_state'] = None
            else:
                param_dict['random_state'] = self.NNspinRandomState.value()
            # beta_1, beta_2 and epsilon
            if self.NNspinBeta_1.isEnabled():
                param_dict['beta_1'] = self.NNspinBeta_1.value()
            if self.NNspinBeta_2.isEnabled():
                param_dict['beta_2'] = self.NNspinBeta_2.value()
            if self.NNspinEpsilon.isEnabled():
                param_dict['epsilon'] = self.NNspinEpsilon.value()

        # if K-Means Clustering
        elif selected_algo == "K-Means Clustering":
            json_dict['algorithm'] = 'KMeans'

            # n_clusters
            param_dict['n_clusters'] = self.KMspinNClusters.value()
            # init
            param_dict['init'] = self.KMcomboInit.currentText()
            # n_init
            param_dict['n_init'] = self.KMspinNInit.value()
            # max_iter
            param_dict['max_iter'] = self.KMspinMaxIter.value()
            # tol
            param_dict['tol'] = self.KMspinTol.value()
            # random_state
            if self.KMspinRandomState.value() == -1:
                param_dict['random_state'] = None
            else:
                param_dict['random_state'] = self.KMspinRandomState.value()
            # algorithm
            param_dict['algorithm'] = self.KMcomboAlgorithm.currentText()

        # if anything else
        else:
            self.statusBar.showMessage('Error: Unknown selected algorithm!',
                                       3000)

        json_dict['parameters'] = param_dict

        # Save the random_state for data split, GridSearchCV or cross_val
        json_dict['random_state'] = param_dict['random_state']

        # Get the current selected features
        self.selectedFeatures = [item.text() for item in self.listFeatures.selectedItems()]
        self.selectedFeatures.sort()
        json_dict['feature_names'] = self.selectedFeatures

        # Save the JSON file
        self.statusBar.showMessage("Saving JSON config file...", 3000)
        json_file = QFileDialog.getSaveFileName(None, 'Save JSON config file',
                                                '', "JSON files (*.json)")

        if json_file[0] != '':
            with open(json_file[0], 'w') as config_file:
                json.dump(json_dict, config_file)
                self.statusBar.showMessage("Config file saved: {}".format(json_file[0]),
                                           5000)

    def run_claspy_t(self):
        print("Run cLASpy_T")

    def reject(self):
        """
        Close the GUI
        """
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClaspyGui()
    ex.show()
    sys.exit(app.exec_())
