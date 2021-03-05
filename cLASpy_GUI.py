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

        # Create the left part of GUI
        self.labelLocalServer = QLabel("Compute on:")
        self.layoutLocalServer = QHBoxLayout(self)
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

        self.labelHLine_1 = QLabel()
        self.labelHLine_1.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        # Info about the input point cloud
        self.labelPtCldInfo = QLabel("Point cloud info:")
        self.labelPtCldFormat = QLabel("Format: ")
        self.labelPtCount = QLabel("Number of points: ")
        self.vLayoutInfo = QVBoxLayout()
        self.vLayoutInfo.addWidget(self.labelPtCldFormat)
        self.vLayoutInfo.addWidget(self.labelPtCount)

        self.labelSampleSize = QLabel("Number of samples:")
        self.spinSampleSize = QDoubleSpinBox()
        self.spinSampleSize.setMaximumWidth(80)
        self.spinSampleSize.setMinimum(0)
        self.spinSampleSize.setDecimals(6)
        self.spinSampleSize.setWrapping(True)
        self.labelMillionPoints = QLabel("Million points")
        self.hLayoutSize = QHBoxLayout()
        self.hLayoutSize.addWidget(self.spinSampleSize)
        self.hLayoutSize.addWidget(self.labelMillionPoints)

        self.labelHLine_2 = QLabel()
        self.labelHLine_2.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        # Selection of the algorithm
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

        # Fill the layout of the left part
        self.formLayoutLeft = QFormLayout()
        self.formLayoutLeft.addRow(self.labelLocalServer, self.layoutLocalServer)
        self.formLayoutLeft.addRow(self.stackInput)
        self.formLayoutLeft.addRow(self.labelHLine_1)
        self.formLayoutLeft.addRow(self.labelPtCldInfo, self.vLayoutInfo)
        self.formLayoutLeft.addRow(self.labelSampleSize, self.hLayoutSize)
        self.formLayoutLeft.addRow(self.labelHLine_2)
        self.formLayoutLeft.addRow("Select algorithm:", self.listAlgorithms)
        self.formLayoutLeft.addRow("Algorithm parameters:", self.stackAlgo)

        # Create the right part of GUI
        self.labelTarget = QLabel("Target field")
        self.labelFeatures = QLabel("Select features:\n\n\n"
                                    "(press Ctrl+Shift\n"
                                    "for multiple selection)")
        self.labelFeatures.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.listFeatures = QListWidget()
        self.listFeatures.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listFeatures.setSortingEnabled(True)

        self.vLayoutFeatures = QVBoxLayout()
        self.vLayoutFeatures.addWidget(self.labelTarget)
        self.vLayoutFeatures.addWidget(self.listFeatures)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        self.buttonBox.accepted.connect(self.save_config)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonRun = QPushButton("Run cLASpy_T")
        self.buttonRun.clicked.connect(self.run_claspy_t)
        self.buttonBox.addButton(self.buttonRun, QDialogButtonBox.ActionRole)

        # Fill the layout of the right part
        self.formLayoutRight = QFormLayout()
        self.formLayoutRight.addRow(self.labelFeatures, self.vLayoutFeatures)
        self.formLayoutRight.addWidget(self.buttonBox)

        # Fill the main layout
        self.hMainLayout = QHBoxLayout(self.mainWidget)
        self.hMainLayout.addLayout(self.formLayoutLeft)
        self.labelVLine = QLabel()
        self.labelVLine.setFrameStyle(QFrame.VLine | QFrame.Sunken)
        self.hMainLayout.addWidget(self.labelVLine)
        self.hMainLayout.addLayout(self.formLayoutRight)

        # Save, Run and Close buttons
        self.setCentralWidget(self.mainWidget)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Function call
        self.listAlgorithms.currentRowChanged.connect(self.display_stack_algo)
        self.radioLocal.toggled.connect(self.display_stack_input)

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
                                               'D:/PostDoc_Temp/Article_Classif_Orne',
                                               "LAS files (*.las);;CSV files (*.csv)")

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
            self.statusBar.showMessage("File error: Unknown extension file!")

        # Check if the target field exist
        if self.target:
            self.labelTarget.setText("Target field is available")
        else:
            self.labelTarget.setText("Mandatory target field not found!!")

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
        self.labelPtCldFormat.setText("Format: LAS version {} | Data format: {}".format(version, data_format))
        self.labelPtCount.setText("Number of points: {}".format(point_count))
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

    def stackui_rf(self):
        form_layout = QFormLayout()

        self.RFspinRandomState = QSpinBox()
        self.RFspinRandomState.setMaximumWidth(80)
        self.RFspinRandomState.setMinimum(-1)
        self.RFspinRandomState.setMaximum(999999)
        self.RFspinRandomState.setValue(-1)
        form_layout.addRow("random_state:", self.RFspinRandomState)

        self.RFspinEstimators = QSpinBox()
        self.RFspinEstimators.setMaximumWidth(80)
        self.RFspinEstimators.setMinimum(2)
        self.RFspinEstimators.setMaximum(999999)
        self.RFspinEstimators.setValue(100)
        form_layout.addRow("n_estimators:", self.RFspinEstimators)

        self.RFcriterion = ["gini", "entropy"]
        self.RFcomboCriterion = QComboBox()
        self.RFcomboCriterion.setMaximumWidth(80)
        self.RFcomboCriterion.addItems(self.RFcriterion)
        self.RFcomboCriterion.setCurrentIndex(self.RFcriterion.index("gini"))
        form_layout.addRow("criterion:", self.RFcomboCriterion)

        self.RFspinMaxDepth = QSpinBox()
        self.RFspinMaxDepth.setMaximumWidth(80)
        self.RFspinMaxDepth.setMinimum(0)
        self.RFspinMaxDepth.setMaximum(9999)
        self.RFspinMaxDepth.setValue(0)
        form_layout.addRow("max_depth:", self.RFspinMaxDepth)

        self.RFspinSamplesSplit = QSpinBox()
        self.RFspinSamplesSplit.setMaximumWidth(80)
        self.RFspinSamplesSplit.setMinimum(2)
        self.RFspinSamplesSplit.setMaximum(999999)
        self.RFspinSamplesSplit.setValue(2)
        form_layout.addRow("min_samples_split:", self.RFspinSamplesSplit)

        self.RFspinSamplesLeaf = QSpinBox()
        self.RFspinSamplesLeaf.setMaximumWidth(80)
        self.RFspinSamplesLeaf.setMaximum(999999)
        self.RFspinSamplesLeaf.setValue(1)
        form_layout.addRow("min_samples_leaf:", self.RFspinSamplesLeaf)

        self.RFspinWeightLeaf = QDoubleSpinBox()
        self.RFspinWeightLeaf.setMaximumWidth(80)
        self.RFspinWeightLeaf.setDecimals(4)
        self.RFspinWeightLeaf.setMaximum(1)
        self.RFspinWeightLeaf.setValue(0)
        form_layout.addRow("min_weight_fraction_leaf:", self.RFspinWeightLeaf)

        self.maxFeatures = ["auto", "sqrt", "log2"]
        self.RFcomboMaxFeatures = QComboBox()
        self.RFcomboMaxFeatures.setMaximumWidth(80)
        self.RFcomboMaxFeatures.addItems(self.maxFeatures)
        self.RFcomboMaxFeatures.setCurrentIndex(self.maxFeatures.index("auto"))
        form_layout.addRow("max_features:", self.RFcomboMaxFeatures)

        self.RFspinNJob = QSpinBox()
        self.RFspinNJob.setMaximumWidth(80)
        self.RFspinNJob.setMinimum(-1)
        self.RFspinNJob.setValue(0)
        form_layout.addRow("n_jobs:", self.RFspinNJob)

        label_h_line = QLabel()
        label_h_line.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        form_layout.addRow(label_h_line)

        self.RFcheckImportance = QCheckBox()
        self.RFcheckImportance.setChecked(False)
        form_layout.addRow("Export feature importances:", self.RFcheckImportance)

        self.stack_RF.setLayout(form_layout)

    def stackui_gb(self):
        form_layout = QFormLayout()

        self.GBspinRandomState = QSpinBox()
        self.GBspinRandomState.setMaximumWidth(80)
        self.GBspinRandomState.setMinimum(-1)
        self.GBspinRandomState.setMaximum(999999)
        self.GBspinRandomState.setValue(-1)
        form_layout.addRow("random_state:", self.GBspinRandomState)

        self.GBspinEstimators = QSpinBox()
        self.GBspinEstimators.setMaximumWidth(80)
        self.GBspinEstimators.setMinimum(2)
        self.GBspinEstimators.setMaximum(9999999)
        self.GBspinEstimators.setValue(100)
        form_layout.addRow("n_estimators:", self.GBspinEstimators)

        self.GBcriterion = ["friedman_mse", "mse"]
        self.GBcomboCriterion = QComboBox()
        self.GBcomboCriterion.setMaximumWidth(80)
        self.GBcomboCriterion.addItems(self.GBcriterion)
        self.GBcomboCriterion.setCurrentIndex(self.GBcriterion.index("friedman_mse"))
        form_layout.addRow("criterion:", self.GBcomboCriterion)

        self.GBspinMaxDepth = QSpinBox()
        self.GBspinMaxDepth.setMaximumWidth(80)
        self.GBspinMaxDepth.setMinimum(0)
        self.GBspinMaxDepth.setMaximum(9999)
        self.GBspinMaxDepth.setValue(3)
        form_layout.addRow("max_depth:", self.GBspinMaxDepth)

        self.GBspinSamplesSplit = QSpinBox()
        self.GBspinSamplesSplit.setMaximumWidth(80)
        self.GBspinSamplesSplit.setMinimum(2)
        self.GBspinSamplesSplit.setMaximum(999999)
        self.GBspinSamplesSplit.setValue(2)
        form_layout.addRow("min_samples_split:", self.GBspinSamplesSplit)

        self.GBspinSamplesLeaf = QSpinBox()
        self.GBspinSamplesLeaf.setMaximumWidth(80)
        self.GBspinSamplesLeaf.setMaximum(999999)
        self.GBspinSamplesLeaf.setValue(1)
        form_layout.addRow("min_samples_leaf:", self.GBspinSamplesLeaf)

        self.GBspinWeightLeaf = QDoubleSpinBox()
        self.GBspinWeightLeaf.setMaximumWidth(80)
        self.GBspinWeightLeaf.setDecimals(4)
        self.GBspinWeightLeaf.setMaximum(1)
        self.GBspinWeightLeaf.setValue(0)
        form_layout.addRow("min_weight_fraction_leaf:", self.GBspinWeightLeaf)

        self.maxFeatures = ["None", "auto", "sqrt", "log2"]
        self.GBcomboMaxFeatures = QComboBox()
        self.GBcomboMaxFeatures.setMaximumWidth(80)
        self.GBcomboMaxFeatures.addItems(self.maxFeatures)
        self.GBcomboMaxFeatures.setCurrentIndex(self.maxFeatures.index("None"))
        form_layout.addRow("max_features:", self.GBcomboMaxFeatures)

        self.loss = ["deviance", "exponential"]
        self.GBcomboLoss = QComboBox()
        self.GBcomboLoss.setMaximumWidth(80)
        self.GBcomboLoss.addItems(self.loss)
        self.GBcomboLoss.setCurrentIndex(self.loss.index("deviance"))
        form_layout.addRow("loss:", self.GBcomboLoss)

        self.GBspinLearningRate = QDoubleSpinBox()
        self.GBspinLearningRate.setMaximumWidth(80)
        self.GBspinLearningRate.setDecimals(6)
        self.GBspinLearningRate.setMinimum(0)
        self.GBspinLearningRate.setValue(0.1)
        form_layout.addRow("learning_rate:", self.GBspinLearningRate)

        self.GBspinSubsample = QDoubleSpinBox()
        self.GBspinSubsample.setMaximumWidth(80)
        self.GBspinSubsample.setDecimals(4)
        self.GBspinSubsample.setMinimum(0)
        self.GBspinSubsample.setValue(1)
        form_layout.addRow("subsample:", self.GBspinSubsample)

        label_h_line = QLabel()
        label_h_line.setFrameStyle(QFrame.HLine | QFrame.Sunken)
        form_layout.addRow(label_h_line)

        self.GBcheckImportance = QCheckBox()
        self.GBcheckImportance.setChecked(False)
        form_layout.addRow("Export feature importances:", self.GBcheckImportance)

        self.stack_GB.setLayout(form_layout)

    def stackui_nn(self):
        form_layout = QFormLayout()

        self.NNspinRandomState = QSpinBox()
        self.NNspinRandomState.setMaximumWidth(80)
        self.NNspinRandomState.setMinimum(-1)
        self.NNspinRandomState.setMaximum(999999)
        self.NNspinRandomState.setValue(-1)
        form_layout.addRow("random_state:", self.NNspinRandomState)

        self.NNlineHiddenLayers = QLineEdit()
        self.NNlineHiddenLayers.setPlaceholderText("Example: 50,100,50")
        form_layout.addRow("hidden_layer_sizes:", self.NNlineHiddenLayers)

        self.NNactivation = ["identity", "logistic", "tanh", "relu"]
        self.NNcomboActivation = QComboBox()
        self.NNcomboActivation.setMaximumWidth(80)
        self.NNcomboActivation.addItems(self.NNactivation)
        self.NNcomboActivation.setCurrentIndex(self.NNactivation.index("relu"))
        form_layout.addRow("activation:", self.NNcomboActivation)

        self.NNsolver = ["lbfgs", "sgd", "adam"]
        self.NNcomboSolver = QComboBox()
        self.NNcomboSolver.setMaximumWidth(80)
        self.NNcomboSolver.addItems(self.NNsolver)
        self.NNcomboSolver.setCurrentIndex(self.NNsolver.index("adam"))
        form_layout.addRow("solver:", self.NNcomboSolver)

        self.NNspinAlpha = QDoubleSpinBox()
        self.NNspinAlpha.setMaximumWidth(80)
        self.NNspinAlpha.setDecimals(8)
        self.NNspinAlpha.setMinimum(0)
        self.NNspinAlpha.setMaximum(999999)
        self.NNspinAlpha.setValue(0.0001)
        form_layout.addRow("alpha:", self.NNspinAlpha)

        self.NNspinBatchSize = QSpinBox()
        self.NNspinBatchSize.setMaximumWidth(80)
        self.NNspinBatchSize.setMinimum(-1)
        self.NNspinBatchSize.setMaximum(999999)
        self.NNspinBatchSize.setValue(-1)
        form_layout.addRow("batch_size:", self.NNspinBatchSize)

        self.NNlearningRate = ["constant", "invscaling", "adaptive"]
        self.NNcomboLearningRate = QComboBox()
        self.NNcomboLearningRate.setMaximumWidth(80)
        self.NNcomboLearningRate.addItems(self.NNlearningRate)
        self.NNcomboLearningRate.setCurrentIndex(self.NNlearningRate.index("constant"))
        self.NNcomboLearningRate.setEnabled(False)
        form_layout.addRow("learning_rate:", self.NNcomboLearningRate)

        self.NNspinLearningRateInit = QDoubleSpinBox()
        self.NNspinLearningRateInit.setMaximumWidth(80)
        self.NNspinLearningRateInit.setDecimals(6)
        self.NNspinLearningRateInit.setMinimum(0)
        self.NNspinLearningRateInit.setMaximum(9999)
        self.NNspinLearningRateInit.setValue(0.001)
        form_layout.addRow("learning_rate_init:", self.NNspinLearningRateInit)

        self.NNspinPowerT = QDoubleSpinBox()
        self.NNspinPowerT.setMaximumWidth(80)
        self.NNspinPowerT.setDecimals(6)
        self.NNspinPowerT.setMinimum(0)
        self.NNspinPowerT.setMaximum(9999)
        self.NNspinPowerT.setValue(0.5)
        self.NNspinPowerT.setEnabled(False)
        form_layout.addRow("power_t:", self.NNspinPowerT)

        self.NNspinMaxIter = QSpinBox()
        self.NNspinMaxIter.setMaximumWidth(80)
        self.NNspinMaxIter.setMinimum(1)
        self.NNspinMaxIter.setMaximum(99999)
        self.NNspinMaxIter.setValue(200)
        form_layout.addRow("max_iter:", self.NNspinMaxIter)

        self.NNcheckShuffle = QCheckBox()
        self.NNcheckShuffle.setChecked(True)
        form_layout.addRow("shuffle:", self.NNcheckShuffle)

        self.NNspinBeta_1 = QDoubleSpinBox()
        self.NNspinBeta_1.setMaximumWidth(80)
        self.NNspinBeta_1.setDecimals(6)
        self.NNspinBeta_1.setMinimum(0)
        self.NNspinBeta_1.setMaximum(1)
        self.NNspinBeta_1.setValue(0.9)
        form_layout.addRow("beta_1:", self.NNspinBeta_1)

        self.NNspinBeta_2 = QDoubleSpinBox()
        self.NNspinBeta_2.setMaximumWidth(80)
        self.NNspinBeta_2.setDecimals(6)
        self.NNspinBeta_2.setMinimum(0)
        self.NNspinBeta_2.setMaximum(1)
        self.NNspinBeta_2.setValue(0.999)
        form_layout.addRow("beta_2:", self.NNspinBeta_2)

        self.NNspinEpsilon = QDoubleSpinBox()
        self.NNspinEpsilon.setMaximumWidth(80)
        self.NNspinEpsilon.setDecimals(8)
        self.NNspinEpsilon.setMinimum(0)
        self.NNspinEpsilon.setValue(0.00000001)
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
        form_layout.addRow("random_state:", self.KMspinRandomState)

        self.KMspinNClusters = QSpinBox()
        self.KMspinNClusters.setMaximumWidth(80)
        self.KMspinNClusters.setMinimum(2)
        self.KMspinNClusters.setMaximum(9999)
        self.KMspinNClusters.setValue(8)
        form_layout.addRow("n_clusters:", self.KMspinNClusters)

        self.KMinit = ["k-means++", "random"]
        self.KMcomboInit = QComboBox()
        self.KMcomboInit.setMaximumWidth(80)
        self.KMcomboInit.addItems(self.KMinit)
        self.KMcomboInit.setCurrentIndex(self.KMinit.index("k-means++"))
        form_layout.addRow("init:", self.KMcomboInit)

        self.KMspinNInit = QSpinBox()
        self.KMspinNInit.setMaximumWidth(80)
        self.KMspinNInit.setMinimum(1)
        self.KMspinNInit.setMaximum(9999)
        self.KMspinNInit.setValue(10)
        form_layout.addRow("n_init:", self.KMspinNInit)

        self.KMspinMaxIter = QSpinBox()
        self.KMspinMaxIter.setMaximumWidth(80)
        self.KMspinMaxIter.setMinimum(1)
        self.KMspinMaxIter.setMaximum(99999)
        self.KMspinMaxIter.setValue(300)
        form_layout.addRow("max_iter:", self.KMspinMaxIter)

        self.KMspinTol = QDoubleSpinBox()
        self.KMspinTol.setMaximumWidth(80)
        self.KMspinTol.setDecimals(8)
        self.KMspinTol.setMinimum(0)
        self.KMspinTol.setMaximum(9999)
        self.KMspinTol.setValue(0.0001)
        form_layout.addRow("tol:", self.KMspinTol)

        self.KMalgorithm = ["auto", "full", "elkan"]
        self.KMcomboAlgorithm = QComboBox()
        self.KMcomboAlgorithm.setMaximumWidth(80)
        self.KMcomboAlgorithm.addItems(self.KMalgorithm)
        self.KMcomboAlgorithm.setCurrentIndex(self.KMalgorithm.index("auto"))
        form_layout.addRow("algorithm:", self.KMcomboAlgorithm)

        self.stack_KM.setLayout(form_layout)

    def display_stack_algo(self, i):
        self.stackAlgo.setCurrentIndex(i)
        algo_list = ["Random Forest", "Gradient Boosting", "Neural Network", "K-Means clustering"]
        self.statusBar.showMessage(algo_list[i] + " parameters", 2000)

    def save_config(self):
        """
        Save configuration as JSON file.
        """
        # Create the json directory
        json_dict = dict()

        # Save input file, output folder and sample size
        if self.radioLocal.isChecked():
            json_dict['input_file'] = self.lineLocalFile.text()
            json_dict['output_folder'] = self.lineLocalFolder.text()
        else:
            json_dict['input_file'] = self.lineServerFile.text()
            json_dict['output_folder'] = self.lineServerFolder.text()

        json_dict['samples'] = self.spinSampleSize.value()

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
        elif selected_algo == 'Gradient Boosting':
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
