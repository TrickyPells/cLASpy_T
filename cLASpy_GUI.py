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
import io
import re
import json
import time
import psutil
import subprocess
import traceback

from contextlib import redirect_stdout
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cLASpy_Classes import *

# -------------------------
# ------ FUNCTIONS --------
# -------------------------


def get_platform():
    """
    :return: the operating system.
    """
    platform_dict = {'linux': 'Linux',
                     'linux1': 'Linux',
                     'linux2': 'Linux',
                     'darwin': 'OS X',
                     'win32': 'Windows'}

    if sys.platform not in platform_dict:
        return sys.platform

    return platform_dict[sys.platform]


def list2str(given_list, join_c=','):
    """
    Format the given list as string.
    :param given_list: The list to convert to string.
    :param join_c: The joining characters.
    :return: The string.
    """
    # Check given_list is a list
    if isinstance(given_list, list) is False:
        raise TypeError("list2str() expected list instance, {} found".format(type(given_list)))

    # Create new list with only str
    list_of_string = list()
    if len(given_list) > 0:
        for item in given_list:
            list_of_string.append(str(item))

    # Join this new list according the joining characters
    return join_c.join(list_of_string)


def format_numberlist(number_str, as_type='str'):
    """
    Format a string of integers separated by comma or whitespace (\\s).
    :param number_str: the string of one or several numbers.
    :param as_type: the type of the return (list or str).
    :return: formatted list or string of the integer.
    """
    number_list_regex = re.compile("[0-9*,\\s]*")

    if isinstance(number_str, str) is False:
        raise TypeError("format_numberlist() only accept string as input")

    # Remove '[' and ']'
    number_str = number_str.replace(']', '')
    number_str = number_str.replace('[', '')

    match = number_list_regex.search(number_str)
    if match is not None:
        numberlist = re.sub('[\\s]', ',', match.group(0))  # Replace all whitespace characters [ \t\n\r\f\v] by comma
        numberlist = re.sub('[,]+', ',', numberlist)  # Replace 1 or more repetitions of comma by one comma

        # Remove the last comma
        if len(numberlist) > 0 and numberlist[-1] == ',':
            numberlist = numberlist[0:-1]

        # Format as selected type
        if as_type == 'str':
            return numberlist
        elif as_type == 'list':
            if len(numberlist) == 0:
                return list()
            else:
                numberlist = numberlist.split(',')
                return [int(number) for number in numberlist]
        else:
            raise TypeError("Parameter 'as_type' has invalid type (give 'list' or 'str')")
    else:
        print("No match found for integer number!")
        return None


def format_floatlist(float_str, as_type='str'):
    """
    Format a string of floats separated by comma or whitespace (\\s).
    :param float_str: the string of one or several floats.
    :param as_type: the type of the return (list or str).
    :return: formatted list or string of the floats.
    """
    float_list_regex = re.compile("[0-9*.?0-9*,\\s]*")

    if isinstance(float_str, str) is False:
        raise TypeError("format_floatlist() only accept string as input")

    # Remove '[' and ']'
    float_str = float_str.replace(']', '')
    float_str = float_str.replace('[', '')

    match = float_list_regex.search(float_str)
    if match is not None:
        floatlist = re.sub('[\\s]', ',', match.group(0))
        floatlist = re.sub('[,]+', ',', floatlist)
        floatlist = re.sub('[.]+', '.', floatlist)

        if len(floatlist) > 0 and floatlist[-1] == ',':
            floatlist = floatlist[0:-1]

        if as_type == 'str':
            return floatlist
        elif as_type == 'list':
            if len(floatlist) == 0:
                return list()
            else:
                floatlist = floatlist.split(',')
                return [float(number) for number in floatlist]
        else:
            raise TypeError("Parameter 'as_type' has invalid type (give 'list' or 'str')")
    else:
        print('No match found for floating number!')
        return None


def format_layerlist(layer_str, as_type='str'):
    """
    Format a string of list of hidden layers separated by '[]' and comma or whitespace (\\s).
    :param layer_str: the string of one or several lists of hidden layers.
    :param as_type: the type of the return (list or str).
    :return: formatted list or string of the hidden layers.
    """
    layer_list_regex = re.compile("[\]]|[\[]")

    if isinstance(layer_str, str) is False:
        raise TypeError("format_layerlist() only accept string as input")

    # Split at each '],[' or ']['
    layer_list = layer_list_regex.split(layer_str)

    # Remove empty item ('' or ',')
    for item in layer_list:
        if item == '' or item == ',':
            layer_list.remove(item)

    # Convert to lists of integer
    for index, item in enumerate(layer_list):
        item_list = item.split(',')
        item_list = [int(value) for value in item_list]
        layer_list[index] = item_list

    if len(layer_list) > 0 and layer_list[-1] == ',':
        layer_list = layer_list[0:-1]

    if as_type == 'str':
        return str(layer_list)[1:-1]  # String of list without '[' and ']'
    elif as_type == 'list':
        if len(layer_list) == 0:
            return list()
        else:
            return layer_list
    else:
        raise TypeError("Parameter 'as_type' has invalid type (give 'list' or 'str')")


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


def warning_box(message, title="Warning"):
    nofeatures = QMessageBox()
    nofeatures.setIcon(QMessageBox.Warning)
    nofeatures.setText(message)
    nofeatures.setWindowTitle(title)
    nofeatures.setStandardButtons(QMessageBox.Ok)
    nofeatures.buttonClicked.connect(nofeatures.close)
    nofeatures.exec_()


def error_box(message, title="Error"):
    """Show message about algorithm parameter error in a message box"""
    parameter_box = QMessageBox()
    parameter_box.setIcon(QMessageBox.Critical)
    parameter_box.setText(message)
    parameter_box.setWindowTitle(title)
    parameter_box.setStandardButtons(QMessageBox.Ok)
    parameter_box.buttonClicked.connect(parameter_box.close)
    parameter_box.exec_()


def new_seed(random_state_spin):
    """
    Give a new seed to random state
    """
    # Upadte the randomState
    high_value = 2 ** 31 - 1  # 2,147,483,647
    seed = np.random.randint(0, high_value)
    random_state_spin.setValue(seed)


# -------------------------
# ------- CLASSES ---------
# -------------------------


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    progress
        int indicating % progress
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
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
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class ClaspyGui(QMainWindow):
    def __init__(self, parent=None):
        super(ClaspyGui, self).__init__(parent)
        self.setWindowTitle("cLASpy_T")
        self.setWindowIcon(QIcon('Ressources/pythie_alpha_64px.png'))
        self.mainWidget = QWidget()

        # Variable Initialization
        self.platform = get_platform()
        self.cLASpy_Core_version = cLASpy_Core_version
        self.cLASpy_GUI_version = '0.1.1'
        self.cLASpy_train_version = self.cLASpy_GUI_version + '_train'
        self.cLASpy_predi_version = self.cLASpy_GUI_version + '_predi'
        self.cLASpy_segme_version = self.cLASpy_GUI_version + '_segme'

        self.options_dict = dict()
        self.train_config = dict()
        self.predi_config = dict()
        self.segme_config = dict()
        self.file_type = 'NONE'
        self.process = None

        # Regular expression for list of integer and float
        self.intlist_validator = QRegExpValidator(QRegExp("[0-9*,\\s]*"), self)
        self.floatlist_validator = QRegExpValidator(QRegExp("[0-9*.?0-9*,\\s]*"), self)

        # Initialize options according claspy_t option file
        try:
            with open("claspy_options.json", 'r') as options_file:
                self.options_dict = json.load(options_file)
                # self.pythonPath = self.options_dict['python_path']
                self.WelcomeAgain = self.options_dict['welcome_window']
        except FileNotFoundError:
            # self.pythonPath = ''
            self.WelcomeAgain = True

        # Call the Welcome window
        if self.WelcomeAgain:
            self.display_welcome_window()

        # Left part of GUI
        self.parameter_part()

        # Central part of GUI
        self.feature_part()
        self.groupCoordinates.setEnabled(False)
        self.groupCoordinates.setVisible(False)
        self.groupStandardLAS.setEnabled(False)
        self.groupStandardLAS.setVisible(False)

        # Right part of GUI -----
        self.command_part()
        self.threadpool = QThreadPool()

        # self.plainTextCommand = QPlainTextEdit()
        # self.plainTextCommand.setReadOnly(True)
        # self.plainTextCommand.setStyleSheet(
        #     """QPlainTextEdit {background-color: #333;
        #                        color: #EEEEEE;}""")
        #
        # # Save button
        # self.buttonSaveCommand = QPushButton("Save Command Output")
        # self.buttonSaveCommand.clicked.connect(self.save_output_command)
        #
        # # Clear button
        # self.buttonClear = QPushButton("Clear")
        # self.buttonClear.clicked.connect(self.plainTextCommand.clear)
        #
        # self.hLayoutSaveClear = QHBoxLayout()
        # self.hLayoutSaveClear.addWidget(self.buttonSaveCommand)
        # self.hLayoutSaveClear.addWidget(self.buttonClear)
        #
        # # Progress bar
        # self.progressBar = QProgressBar()
        # self.progressBar.setMaximum(100)
        #
        # # Fill layout of right part
        # self.vLayoutRight = QVBoxLayout()
        # self.vLayoutRight.addWidget(self.plainTextCommand)
        # self.vLayoutRight.addLayout(self.hLayoutSaveClear)
        # self.vLayoutRight.addWidget(self.progressBar)
        #
        # self.groupCommand = QGroupBox("Command Output")
        # self.groupCommand.setLayout(self.vLayoutRight)
        # # ------ End of command part

        # Button box
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Close)
        self.buttonBox.rejected.connect(self.reject)

        self.buttonRunTrain = QPushButton("Train")
        self.buttonRunTrain.clicked.connect(self.run_train)
        self.buttonBox.addButton(self.buttonRunTrain, QDialogButtonBox.ActionRole)
        self.buttonRunPredict = QPushButton("Predict")
        self.buttonRunPredict.clicked.connect(self.run_predict)
        self.buttonBox.addButton(self.buttonRunPredict, QDialogButtonBox.ActionRole)
        self.buttonRunSegment = QPushButton("Segment")
        self.buttonRunSegment.clicked.connect(self.run_segment)
        self.buttonBox.addButton(self.buttonRunSegment, QDialogButtonBox.ActionRole)
        self.buttonStop = QPushButton("Stop")
        self.buttonStop.clicked.connect(self.stop_process)
        self.buttonStop.setEnabled(False)
        self.buttonBox.addButton(self.buttonStop, QDialogButtonBox.ActionRole)
        self.buttonRunPredict.setVisible(False)
        self.buttonRunSegment.setVisible(False)

        # Fill the main layout
        self.splitterMain = QSplitter(Qt.Horizontal)
        self.splitterMain.addWidget(self.groupParameters)
        self.splitterMain.addWidget(self.groupFeatures)
        self.splitterMain.addWidget(self.groupCommand)
        self.splitterMain.setSizes(([200, 100, 468]))

        self.vMainLayout = QVBoxLayout(self.mainWidget)
        self.vMainLayout.addWidget(self.splitterMain)
        self.vMainLayout.addWidget(self.buttonBox)

        # setCentralWidget
        self.setCentralWidget(self.mainWidget)

        # MenuBar and Menu
        bar = self.menuBar()

        # File Menu
        menu_file = bar.addMenu("File")
        # Open action
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        menu_file.addAction(open_action)
        # Save action
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        menu_file.addAction(save_action)
        # Quit action
        quit_action = QAction("Quit", self)
        menu_file.addAction(quit_action)
        menu_file.triggered[QAction].connect(self.menu_file_trigger)

        # Edit Menu
        menu_edit = bar.addMenu("Edit")
        # Option action
        options_action = QAction("Options", self)
        menu_edit.addAction(options_action)
        menu_edit.triggered[QAction].connect(self.menu_edit_trigger)

        # Help Menu
        menu_help = bar.addMenu("Help")
        # About action
        about_action = QAction("About", self)
        menu_help.addAction(about_action)
        menu_help.triggered[QAction].connect(self.menu_help_trigger)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Geometry
        self.setGeometry(0, 0, 1024, 768)

    # Welcome window
    def display_welcome_window(self, welcome=True):
        """
        Display the Welcome Window with cLASpy_T logo, institution and licence.
        """
        # Create window with QWidget() object
        self.welcome_window = QWidget()

        # Logo of cLASpy_T Part
        label_pythia = QLabel()
        pixmap_pythia = QPixmap('Ressources/WelcomeWindowLow.png')
        pixmap_pythia.scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
        label_pythia.setPixmap(pixmap_pythia)

        # cLASpy_T versions
        if welcome:
            label_welcometo = QLabel('Welcome to')
        else:
            label_welcometo = QLabel('About')
        label_welcometo.setFont(QFont('Arial', 16))
        label_welcometo.setAlignment(Qt.AlignCenter)
        label_softname = QLabel('cLASpy_T')
        label_softname.setFont(QFont('Arial', 20, QFont.Bold))
        label_softname.setAlignment(Qt.AlignCenter)
        label_webpage = QLabel('github.com/TrickyPells/cLASpy_T')
        label_webpage.setAlignment(Qt.AlignCenter)
        label_version = QLabel('Core version: ' + str(self.cLASpy_Core_version) +
                               ' | GUI version: ' + str(self.cLASpy_GUI_version))
        label_version.setAlignment(Qt.AlignCenter)

        # Institution Logos
        pixmap_cnrs = QPixmap('Ressources/cnrs.png')
        pixmap_cnrs.scaled(64, 64, Qt.KeepAspectRatio, Qt.FastTransformation)
        label_cnrs = QLabel()
        label_cnrs.setPixmap(pixmap_cnrs)
        pixmap_m2c = QPixmap('Ressources/m2c.png')
        pixmap_m2c.scaledToHeight(64)
        label_m2c = QLabel()
        label_m2c.setPixmap(pixmap_m2c)
        # pixmap_rsg = QPixmap('Ressources/rsg.png')
        label_rsg = QLabel('Remote\nSensing\nGroup')
        label_rsg.setFont(QFont('Arial', 10, QFont.Bold))
        label_rsg.setAlignment(Qt.AlignCenter)

        hlayout_institutions = QHBoxLayout()
        hlayout_institutions.addWidget(label_cnrs)
        hlayout_institutions.addWidget(label_rsg)
        hlayout_institutions.addWidget(label_m2c)

        # Authors
        label_createdby = QLabel('Created by:')
        label_createdby.setAlignment(Qt.AlignCenter)
        label_author1 = QLabel('Xavier PELLERIN (xavier.peller1@gmail.com)\n'
                               'and\n'
                               'Laurent FROIDEVAL (email laurent.froideval)\n'
                               'Christophe CONESSA (email christophe.conessa)')
        label_author1.setAlignment(Qt.AlignCenter)

        # CheckBox Welcome Window
        self.checkWelcomeWin = QCheckBox("Do not show this window again !")
        if self.WelcomeAgain:
            self.checkWelcomeWin.setChecked(False)
        else:
            self.checkWelcomeWin.setChecked(True)
        self.checkWelcomeWin.toggled.connect(lambda: self.welcome_again(self.checkWelcomeWin))

        # Layout of the Authors, Insitutions and Licence part
        vlayout_softpart = QVBoxLayout()
        vlayout_softpart.addWidget(label_welcometo)
        vlayout_softpart.addWidget(label_softname)
        vlayout_softpart.addWidget(label_webpage)
        vlayout_softpart.addWidget(label_version)
        vlayout_softpart.addLayout(hlayout_institutions)
        vlayout_softpart.addWidget(label_createdby)
        vlayout_softpart.addWidget(label_author1)
        if welcome:
            vlayout_softpart.addWidget(self.checkWelcomeWin)

        vlayout_softpart.setAlignment(Qt.AlignVCenter)

        # Create layout of the window
        hbox = QHBoxLayout()
        hbox.addWidget(label_pythia)
        hbox.addLayout(vlayout_softpart)
        self.welcome_window.setLayout(hbox)

        # Set the apparence of window
        if welcome:
            self.welcome_window.setWindowTitle("Welcome to cLASpy_T")
        else:
            self.welcome_window.setWindowTitle("About cLASpy_T")
        self.welcome_window.setWindowIcon(QIcon('Ressources/pythie_alpha_512px.png'))
        self.welcome_window.setGeometry(128, 128, 512, 384)
        self.welcome_window.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.welcome_window.show()

    def welcome_again(self, checkbox):
        if checkbox.isChecked():
            self.WelcomeAgain = False
        else:
            self.WelcomeAgain = True

        self.options_dict['welcome_window'] = self.WelcomeAgain
        with open("claspy_options.json", 'w') as options_file:
            json.dump(self.options_dict, options_file, indent=4)

    # Menu and Options
    def menu_file_trigger(self, action):
        if action.text() == "Open":
            self.open_config()

        elif action.text() == "Save":
            self.update_config()
            self.save_config()

        elif action.text() == "Quit":
            self.reject()

    def menu_edit_trigger(self, action):
        if action.text() == "Options":
            self.options()

    def menu_help_trigger(self, action):
        if action.text() == "About":
            self.display_welcome_window(welcome=False)

    def options(self):
        self.dialogOptions = QDialog()

        # Add lineEdit to set the python interpreter
        # self.labelPython = QLabel("Python path:")
        # self.linePython = QLineEdit()
        # self.linePython.setMinimumWidth(200)
        # if self.platform == 'Windows':
        #     self.linePython.setPlaceholderText("Give 'python.exe' path")
        # else:
        #     self.linePython.setPlaceholderText("Give python path")
        # if self.pythonPath != '':
        #     self.linePython.setText(self.pythonPath)

        # self.toolButtonPython = QToolButton()
        # self.toolButtonPython.setText("Browse")
        # self.toolButtonPython.clicked.connect(self.find_python)

        # self.hLayoutPython = QHBoxLayout()
        # self.hLayoutPython.addWidget(self.linePython)
        # self.hLayoutPython.addWidget(self.toolButtonPython)

        # CheckBox Welcome Window
        check_welcome_window = QCheckBox('Do not show the welcome window again !')
        if self.WelcomeAgain:
            check_welcome_window.setChecked(False)
        else:
            check_welcome_window.setChecked(True)
        check_welcome_window.toggled.connect(lambda:self.welcome_again(check_welcome_window))

        # Button box for Options
        self.buttonOptionsBox = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonOptionsBox.button(QDialogButtonBox.Apply).clicked.connect(self.save_options)
        self.buttonOptionsBox.accepted.connect(self.save_close_options)
        self.buttonOptionsBox.rejected.connect(self.dialogOptions.reject)

        self.formLayoutOptions = QFormLayout(self.dialogOptions)
        # self.formLayoutOptions.addRow(self.labelPython, self.hLayoutPython)
        self.formLayoutOptions.addRow(QLabel('Welcome window: '), check_welcome_window)
        self.formLayoutOptions.addRow(self.buttonOptionsBox)

        self.dialogOptions.setWindowTitle("Options")
        self.setWindowModality(Qt.ApplicationModal)
        self.dialogOptions.exec_()

    # def find_python(self):
    #     if self.platform == 'Windows':
    #         python_exe = QFileDialog.getOpenFileName(self, 'Select \'python.exe\'',
    #                                                  '', "Executables (*.exe);;")
    #     else:
    #         python_exe = QFileDialog.getOpenFileName(self, 'Select python interpreter')
    #
    #     if python_exe[0] != '':
    #         self.linePython.setText(os.path.normpath(python_exe[0]))

    def save_options(self):
        # self.pythonPath = self.linePython.text()
        # self.options_dict['python_path'] = self.pythonPath
        # self.statusBar.showMessage("Python path: {}".format(self.pythonPath), 3000)

        self.options_dict['welcome_window'] = self.WelcomeAgain
        with open("claspy_options.json", 'w') as options_file:
            json.dump(self.options_dict, options_file, indent=4)

    def save_close_options(self):
        # call save_otpion()
        self.save_options()
        self.dialogOptions.close()

        # # and close dialog
        # if self.linePython.text() != '':
        #     self.dialogOptions.close()

    # Parameter part
    def parameter_part(self):
        """
        Part to set all parameters (Input data, algorithms, algo parameters...)
        """
        # Global parameter group
        self.groupParameters = QGroupBox()
        self.groupParameters.setObjectName("WithoutBorder")
        self.groupParameters.setStyleSheet("QGroupBox#WithoutBorder { border: 0px }")

        self.labelLocalServer = QLabel("Run cLASpy_T:")
        self.labelLocalServer.setAlignment(Qt.AlignLeft)
        self.pushLocal = QPushButton("on Local")
        self.pushLocal.setToolTip("To run cLASpy_T on this computer")
        self.pushLocal.setCheckable(True)
        self.pushLocal.setChecked(True)
        self.pushLocal.clicked.connect(self.display_stack_local)

        self.pushServer = QPushButton("on Server")
        self.pushServer.setToolTip("To run cLASpy_T on a remote server")
        self.pushServer.setCheckable(True)
        self.pushServer.setChecked(False)
        self.pushServer.clicked.connect(self.display_stack_server)

        self.layoutLocalServer = QVBoxLayout()
        self.layoutLocalServer.addWidget(self.pushLocal)
        self.layoutLocalServer.addWidget(self.pushServer)

        # GroupBox for point cloud file
        self.groupPtCld = QGroupBox("Point Cloud")

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

        # Fill the point cloud groupBox with QFormLayout
        self.formLayoutPtCld = QFormLayout()
        self.formLayoutPtCld.addRow(self.stackInput)
        self.formLayoutPtCld.addRow("Format:", self.labelPtCldFormat)
        self.formLayoutPtCld.addRow("Number of points:", self.labelPtCount)
        self.groupPtCld.setLayout(self.formLayoutPtCld)

        # Create tabs
        self.tabTrain = QWidget()  # Widget for tab train
        self.tabPredict = QWidget()  # Widget for tab predict
        self.tabSegment = QWidget()  # Widget for tab segment

        # Make tabs scrollable
        self.scrollTabTrain = QScrollArea()
        self.scrollTabTrain.setFrameShape(QFrame.NoFrame)
        self.scrollTabTrain.setWidgetResizable(True)
        self.scrollTabTrain.setWidget(self.tabTrain)

        self.scrollTabPredict = QScrollArea()
        self.scrollTabPredict.setFrameShape(QFrame.NoFrame)
        self.scrollTabPredict.setWidgetResizable(True)
        self.scrollTabPredict.setWidget(self.tabPredict)

        self.scrollTabSegment = QScrollArea()
        self.scrollTabSegment.setFrameShape(QFrame.NoFrame)
        self.scrollTabSegment.setWidgetResizable(True)
        self.scrollTabSegment.setWidget(self.tabSegment)

        self.tabui_train()
        self.tabui_predict()
        self.tabui_segment()

        self.tabModes = QTabWidget()  # TabWidget to group all tabs
        self.tabModes.addTab(self.scrollTabTrain, "Training")
        self.tabModes.addTab(self.scrollTabPredict, "Prediction")
        self.tabModes.addTab(self.scrollTabSegment, "Segmentation")

        self.tabModes.currentChanged.connect(self.tab_modes_action)

        # Fill layout of parameter part
        self.vLayoutParameters = QVBoxLayout()
        self.vLayoutParameters.addWidget(self.labelLocalServer)
        self.vLayoutParameters.addLayout(self.layoutLocalServer)
        self.vLayoutParameters.addWidget(self.groupPtCld)
        self.vLayoutParameters.setStretchFactor(self.groupPtCld, 1)
        self.vLayoutParameters.addWidget(self.tabModes)
        self.vLayoutParameters.setStretchFactor(self.tabModes, 4)

        self.groupParameters.setLayout(self.vLayoutParameters)

    # Run on local or server
    def stackui_local(self):
        form_layout = QFormLayout()

        # Line for local file input
        self.lineLocalFile = QLineEdit()
        self.lineLocalFile.setPlaceholderText("Select LAS or CSV file as input")
        self.lineLocalFile.editingFinished.connect(self.open_file)
        self.toolButtonFile = QToolButton()
        self.toolButtonFile.setText("Browse")
        self.toolButtonFile.clicked.connect(self.get_file)
        self.hLayoutFile = QHBoxLayout()
        self.hLayoutFile.addWidget(self.lineLocalFile)
        self.hLayoutFile.addWidget(self.toolButtonFile)
        form_layout.addRow("Input file:", self.hLayoutFile)

        # Line for local folder output
        self.lineLocalFolder = QLineEdit()
        self.lineLocalFolder.setPlaceholderText("Folder where save result files")
        self.lineLocalFolder.textChanged.connect(self.enable_open_results)
        self.toolButtonFolder = QToolButton()
        self.toolButtonFolder.setText("Browse")
        self.toolButtonFolder.clicked.connect(self.get_folder)
        self.hLayoutFolder = QHBoxLayout()
        self.hLayoutFolder.addWidget(self.lineLocalFolder)
        self.hLayoutFolder.addWidget(self.toolButtonFolder)
        form_layout.addRow("Output folder:", self.hLayoutFolder)

        # Button to open result folder
        self.buttonResultFolder = QPushButton('Open Result Directory')
        self.buttonResultFolder.setEnabled(False)
        self.buttonResultFolder.clicked.connect(self.open_results)
        form_layout.addWidget(self.buttonResultFolder)

        # Set the layout
        self.stack_Local.setLayout(form_layout)

    def stackui_server(self):
        form_layout = QFormLayout()

        self.lineFile = QLineEdit()
        self.lineFile.setPlaceholderText("Local file to get features")
        self.lineFile.editingFinished.connect(self.open_file)
        self.toolButtonFile = QToolButton()
        self.toolButtonFile.setText("Browse")
        self.toolButtonFile.clicked.connect(self.get_file)
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

    def display_stack_local(self):
        if self.pushLocal.isChecked():
            self.stackInput.setCurrentIndex(0)
            self.pushServer.setChecked(False)  # Disable push button to compute on server
            self.buttonRunTrain.setEnabled(True)  # Enable training on local
            self.buttonRunPredict.setEnabled(True)  # Enable prediction on local
            self.buttonRunSegment.setEnabled(True)  # Enable segmentation on local
            self.lineServerModel.setEnabled(False)  # Disable the line for model on server

    def display_stack_server(self):
        if self.pushServer.isChecked():
            self.stackInput.setCurrentIndex(1)
            self.pushLocal.setChecked(False)  # Disable push button to compute on server
            self.buttonRunTrain.setEnabled(False)  # Enable training on local
            self.buttonRunPredict.setEnabled(False)  # Enable prediction on local
            self.buttonRunSegment.setEnabled(False)  # Enable segmentation on local
            self.lineServerModel.setEnabled(True)  # Enable the line for model on server

    def get_file(self):
        self.statusBar.showMessage("Select file...", 3000)
        filename = QFileDialog.getOpenFileName(self, 'Select CSV or LAS file',
                                               '', "LAS files (*.las);;CSV files (*.csv)")

        if filename[0] != '':
            if self.pushLocal.isChecked():
                self.lineLocalFile.setText(os.path.normpath(filename[0]))
            else:
                self.lineFile.setText(os.path.normpath(filename[0]))

            self.open_file()

    def open_file(self):
        if self.pushLocal.isChecked():
            file_path = os.path.normpath(self.lineLocalFile.text())
        else:
            file_path = os.path.normpath(self.lineFile.text())

        root_ext = os.path.splitext(file_path)
        if self.pushLocal.isChecked():
            self.lineLocalFolder.setText(os.path.splitext(root_ext[0])[0])

        if root_ext[1] == '.csv':
            self.file_type = 'CSV'
            feature_names = self.open_csv(file_path)
        elif root_ext[1] == '.las':
            self.file_type = 'LAS'
            feature_names, standard_fields = self.open_las(file_path)
            self.listStandardLAS.setEnabled(True)  # Enable List of standard LAS fields
            self.listStandardLAS.clear()
            for item in standard_fields:
                self.listStandardLAS.addItem(str(item))
        else:
            feature_names = ["File error:", "Unknown extension file!"]
            self.statusBar.showMessage("File error: Unknown extension file!", 3000)

        # Check if the target field exist
        try:
            self.target
        except AttributeError:
            self.target = False

        if self.target:
            self.labelTarget.setText("Target field is available.")
            self.SeglabelTarget.setText("Will be discarded for segmentation.")
        else:
            self.labelTarget.setText("Not found!!")
            self.SeglabelTarget.setText("Not mandatory.")

        # Rewrite listExtraFeature
        self.listExtraFeatures.clear()
        for item in feature_names:
            self.listExtraFeatures.addItem(str(item))
        self.number_selected_features()

        # Update the feature part
        self.enable_advanced_features()

    def open_csv(self, file_path):
        """
        Open the CSV file with pandas and return the list of the feature names.
        :param file_path: THe CSV file.
        :return: List of the feature names.
        """
        # Initialize target bool
        self.target = False

        # Infos about CSV file
        frame = pd.read_csv(file_path, sep=',', header='infer')
        version = 'CSV'
        point_count = len(frame)

        # Set value of the train size
        number_mpts = float(point_count / 1000000.)  # Number of million points
        self.spinSampleSize.setMaximum(number_mpts)
        if number_mpts > 2:
            self.spinSampleSize.setValue(1)
        else:
            self.spinSampleSize.setValue(number_mpts / 2.)

        # Show CSV and number of points in status bar
        point_count = '{:,}'.format(point_count).replace(',', ' ')  # Format with thousand separator
        self.labelPtCldFormat.setText("{}".format(version))
        self.labelPtCount.setText("{}".format(point_count))
        self.statusBar.showMessage("{} points on {} file".format(point_count, version), 5000)

        # Get the extra dimensions and coordinates (column names)
        extra_dimensions = list()
        self.checkX.setEnabled(False)
        self.checkY.setEnabled(False)
        self.checkZ.setEnabled(False)
        for dim in frame.columns.values.tolist():
            if dim == '//X':
                dim = 'X'
            if dim in ['X', 'x']:
                self.checkX.setEnabled(True)
            elif dim in ['Y', 'y']:
                self.checkY.setEnabled(True)
            elif dim in ['Z', 'z']:
                self.checkZ.setEnabled(True)
            else:
                extra_dimensions.append(dim.replace(' ', '_'))  # Replace 'space' with '_'

        # Find 'target' dimension if exist
        for trgt in ['target', 'Target', 'TARGET']:
            if trgt in extra_dimensions:
                self.target = True
                self.targetName = trgt
                extra_dimensions.remove(trgt)

        return extra_dimensions

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
            self.spinSampleSize.setValue(number_mpts / 2.)

        # Show LAS version and number of points in status bar
        point_count = '{:,}'.format(point_count).replace(',', ' ')  # Format with thousand separator
        self.labelPtCldFormat.setText("LAS version {}  |  Data format: {}".format(version, data_format))
        self.labelPtCount.setText("{}".format(point_count))
        self.statusBar.showMessage("{} points | LAS Version: {}".format(point_count, version),
                                   5000)

        # List of the standard dimensions according the data_format of LAS (prefilled_dimensions)
        prefilled_dimensions = point_format[data_format]

        # Get the extra_dimensions
        extra_dimensions = list()
        for dim in las.point_format.specs:
            extra_dimensions.append(str(dim.name))  # List of all dimension
        for dim in prefilled_dimensions:  # Remove the pre-filled dimensions
            if dim in extra_dimensions:
                extra_dimensions.remove(dim)

        # Get the standard dimensions in LAS file (for real, not prefilled)
        standard_dimensions = list()
        for dim in las.point_format.specs:
            standard_dimensions.append(str(dim.name))
        for dim in extra_dimensions:  # Remove the extra_dimensions, so get only the standard_dimension
            if dim in standard_dimensions:
                standard_dimensions.remove(dim)
        for coord in ['X', 'Y', 'Z']:  # Remove X, Y, Z (already checkable in Advanced Features)
            if coord in standard_dimensions:
                standard_dimensions.remove(coord)

        # Find 'target' dimension if exist
        for trgt in ['target', 'Target', 'TARGET']:
            if trgt in extra_dimensions:
                self.target = True
                self.targetName = trgt
                extra_dimensions.remove(trgt)

        return extra_dimensions, standard_dimensions

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
            if self.pushLocal.isChecked():
                self.lineLocalFolder.setText(foldername)

    def enable_open_results(self):
        folder_path = os.path.normpath(self.lineLocalFolder.text())
        if QDir(folder_path).exists():
            self.buttonResultFolder.setEnabled(True)
        else:
            self.buttonResultFolder.setEnabled(False)

    def open_results(self):
        self.enable_open_results()
        if self.buttonResultFolder.isEnabled():
            folder_path = os.path.normpath(self.lineLocalFolder.text())
            if self.platform == 'Windows':
                self.dialog_results = QProcess()
                self.dialog_results.setProgram("explorer")
                self.dialog_results.setArguments([folder_path])
                self.dialog_results.startDetached()
            elif self.platform == 'Linux':
                self.dialog_results = QProcess()
                self.dialog_results.setProgram("xdg-open")
                self.dialog_results.setArguments([folder_path])
                self.dialog_results.startDetached()
            else:
                print("ErrorOS: Operating System not supported !")

    # Tab of train
    def tabui_train(self):
        # Group Training
        self.groupTrain = QGroupBox("Training Parameters")

        # Label: Target field exist ?
        self.labelTarget = QLabel()

        # ComboBox of Algorithms and GridSearchCV
        self.comboAlgorithms = QComboBox()
        self.comboAlgorithms.insertItem(0, "Random Forest")
        self.comboAlgorithms.insertItem(1, "Gradient Boosting")
        self.comboAlgorithms.insertItem(2, "Neural Network")
        self.comboAlgorithms.setCurrentText("Random Forest")
        self.comboAlgorithms.currentIndexChanged.connect(self.display_stack_algo)

        # CheckBox for GridSearchCV
        self.checkGridSearchCV = QCheckBox()
        self.checkGridSearchCV.setChecked(False)
        self.checkGridSearchCV.setToolTip("Perform training with GridSearchCV.\n"
                                          "(See the scikit-learn documentation)")
        self.checkGridSearchCV.stateChanged.connect(self.display_stack_algo)

        # Set the sample size
        self.spinSampleSize = QDoubleSpinBox()
        self.spinSampleSize.setMinimum(0)
        self.spinSampleSize.setDecimals(6)
        self.spinSampleSize.setWrapping(True)
        self.hLayoutSize = QHBoxLayout()
        self.hLayoutSize.addWidget(self.spinSampleSize)
        self.hLayoutSize.addWidget(QLabel("Millions points"))

        # Set the training/testing ratio
        self.spinTrainRatio = QDoubleSpinBox()
        self.spinTrainRatio.setMinimum(0)
        self.spinTrainRatio.setMaximum(1)
        self.spinTrainRatio.setSingleStep(0.1)
        self.spinTrainRatio.setDecimals(3)
        self.spinTrainRatio.setWrapping(True)
        self.spinTrainRatio.setValue(0.5)

        # Set the scaler
        self.scalerNames = ['Standard', 'Robust', 'MinMax']
        self.comboScaler = QComboBox()
        self.comboScaler.addItems(self.scalerNames)
        self.comboScaler.setCurrentText("Standard")
        self.comboScaler.setToolTip("Set the method to scale the data.")

        # Set the Principal Component Analysis
        self.spinPCA = QSpinBox()
        self.spinPCA.setMinimum(0)
        self.spinPCA.setMaximum(9999)
        self.spinPCA.setValue(0)
        self.spinPCA.setToolTip("Set the Principal Component Analysis\n"
                                "and the number of principal components.")

        # Set the scorer
        self.scorerNames = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score',
                            'f1_micro', 'f1_macro', 'f1_weighted',
                            'precision_micro', 'precision_macro', 'precision_weighted',
                            'recall_micro', 'recall_macro', 'recall_weighted',
                            'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_ovr_weighted', 'roc_ovo_weighted']
        self.comboScorer = QComboBox()
        self.comboScorer.addItems(self.scorerNames)
        self.comboScorer.setCurrentText("accuracy")
        self.comboScorer.setToolTip("Set the scorer for GridSearchCV or Cross_validation\n"
                                    "(see the scikit-learn documentation).")

        # Set the random state to split data, GridSearchCV, CrossVal...
        self.spinRandomState = QSpinBox()
        self.spinRandomState.setMinimum(0)
        self.spinRandomState.setMaximum(2147483647)  # 2^31 - 1
        self.spinRandomState.setValue(0)
        self.spinRandomState.setToolTip("Controls the randomness to split the data into train/test and\n"
                                        "the shuffle split for GridSearchCV or Cross-Validation.")
        self.pushRandomSeed = QPushButton("New Seed")
        self.pushRandomSeed.clicked.connect(lambda: new_seed(self.spinRandomState))
        h_layout_random = QHBoxLayout()
        h_layout_random.addWidget(self.spinRandomState)
        h_layout_random.addWidget(self.pushRandomSeed)

        # Set the number of jobs only for GridSearchCV or Cross-Validation
        self.spinNJobCV = QSpinBox()
        self.spinNJobCV.setMinimum(-1)
        self.spinNJobCV.setValue(-1)
        self.spinNJobCV.setToolTip("Set the number of jobs for GridSearchCV and Cross-Validation.\n"
                                   "In case of RandomForest, Total CPU used = N_jobs CV * n_jobs.")

        # Fill layout of training parameters
        self.formTrainParam = QFormLayout()
        self.formTrainParam.addRow("Target field:", self.labelTarget)
        self.formTrainParam.addRow("Select Algorithm:", self.comboAlgorithms)
        self.formTrainParam.addRow("GridSearchCV:", self.checkGridSearchCV)
        self.formTrainParam.addRow("Number of samples:", self.hLayoutSize)
        self.formTrainParam.addRow("Training ratio:", self.spinTrainRatio)
        self.formTrainParam.addRow("Scaler:", self.comboScaler)
        self.formTrainParam.addRow("PCA:", self.spinPCA)
        self.formTrainParam.addRow("Scorer:", self.comboScorer)

        self.formTrainParam.addRow("Random State:", h_layout_random)
        self.formTrainParam.addRow("N_jobs CV:", self.spinNJobCV)
        self.groupTrain.setLayout(self.formTrainParam)

        # Algorithm group
        self.groupAlgorithm = QGroupBox("Algorithm parameters")

        # Stacks for the parameters of the algo
        self.stack_RF = QWidget()
        self.stack_RF_grid = QWidget()
        self.stack_GB = QWidget()
        self.stack_GB_grid = QWidget()
        self.stack_NN = QWidget()
        self.stack_NN_grid = QWidget()
        self.stackui_rf()
        self.stackui_gb()
        self.stackui_nn()
        self.stackui_rf_grid()
        self.stackui_gb_grid()
        self.stackui_nn_grid()
        self.stackAlgo = QStackedWidget(self)
        # self.stackAlgo.setMaximumHeight(310)
        self.stackAlgo.addWidget(self.stack_RF)
        self.stackAlgo.addWidget(self.stack_GB)
        self.stackAlgo.addWidget(self.stack_NN)
        self.stackAlgo.addWidget(self.stack_RF_grid)
        self.stackAlgo.addWidget(self.stack_GB_grid)
        self.stackAlgo.addWidget(self.stack_NN_grid)

        # Fill layout of algorithm parameters
        self.vLayoutAlgoParam = QVBoxLayout()
        self.vLayoutAlgoParam.addWidget(self.stackAlgo)
        self.groupAlgorithm.setLayout(self.vLayoutAlgoParam)

        # Fill the left layout
        self.vLayoutTabTrain = QVBoxLayout()
        self.vLayoutTabTrain.addWidget(self.groupTrain)
        self.vLayoutTabTrain.addWidget(self.groupAlgorithm)

        self.tabTrain.setLayout(self.vLayoutTabTrain)

    def stackui_rf(self):
        form_layout = QFormLayout()

        self.RFspinRandomState = QSpinBox()
        self.RFspinRandomState.setMinimum(-1)
        self.RFspinRandomState.setMaximum(2147483647)
        self.RFspinRandomState.setValue(-1)
        self.RFspinRandomState.setToolTip("Controls the randomness to build the trees.")
        self.RFpushRandomState = QPushButton('New Seed')
        self.RFpushRandomState.clicked.connect(lambda: new_seed(self.RFspinRandomState))
        h_layout_random = QHBoxLayout()
        h_layout_random.addWidget(self.RFspinRandomState)
        h_layout_random.addWidget(self.RFpushRandomState)
        form_layout.addRow("random_state:", h_layout_random)

        self.RFspinEstimators = QSpinBox()
        self.RFspinEstimators.setMinimum(2)
        self.RFspinEstimators.setMaximum(999999)
        self.RFspinEstimators.setValue(100)
        self.RFspinEstimators.setToolTip("The number of trees in the forest.")
        form_layout.addRow("n_estimators:", self.RFspinEstimators)

        self.RFcriterion = ["gini", "entropy"]
        self.RFcomboCriterion = QComboBox()
        self.RFcomboCriterion.addItems(self.RFcriterion)
        self.RFcomboCriterion.setCurrentText("gini")
        self.RFcomboCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.RFcomboCriterion)

        self.RFspinMaxDepth = QSpinBox()
        self.RFspinMaxDepth.setMinimum(0)
        self.RFspinMaxDepth.setMaximum(9999)
        self.RFspinMaxDepth.setValue(0)
        self.RFspinMaxDepth.setToolTip("The maximum depth of the tree.")
        form_layout.addRow("max_depth:", self.RFspinMaxDepth)

        self.RFspinSamplesSplit = QSpinBox()
        self.RFspinSamplesSplit.setMinimum(2)
        self.RFspinSamplesSplit.setMaximum(999999)
        self.RFspinSamplesSplit.setValue(2)
        self.RFspinSamplesSplit.setToolTip("The minimum number of samples required\n"
                                           "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.RFspinSamplesSplit)

        self.RFspinSamplesLeaf = QSpinBox()
        self.RFspinSamplesLeaf.setMaximum(999999)
        self.RFspinSamplesLeaf.setValue(1)
        self.RFspinSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                          "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.RFspinSamplesLeaf)

        self.RFspinWeightLeaf = QDoubleSpinBox()
        self.RFspinWeightLeaf.setDecimals(4)
        self.RFspinWeightLeaf.setMaximum(1)
        self.RFspinWeightLeaf.setValue(0)
        self.RFspinWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                         "of weights required to be at a leaf node. Samples\n"
                                         "have equal weight when sample_weight=0.")
        form_layout.addRow("min_weight_fraction_leaf:", self.RFspinWeightLeaf)

        self.RFmaxFeatures = ["auto", "sqrt", "log2"]
        self.RFcomboMaxFeatures = QComboBox()
        self.RFcomboMaxFeatures.addItems(self.RFmaxFeatures)
        self.RFcomboMaxFeatures.setCurrentText("auto")
        self.RFcomboMaxFeatures.setToolTip("The number of features to consider\n"
                                           "when looking for the best split.")
        form_layout.addRow("max_features:", self.RFcomboMaxFeatures)

        self.RFspinNJob = QSpinBox()
        self.RFspinNJob.setMinimum(-1)
        self.RFspinNJob.setValue(-1)
        self.RFspinNJob.setToolTip("The number of jobs to run in parallel.\n"
                                   "'0' means one job at the same time.\n"
                                   "'-1' means using all processors.")
        form_layout.addRow("n_jobs:", self.RFspinNJob)
        form_layout.addRow(QHLine())

        self.RFcheckImportance = QCheckBox()
        self.RFcheckImportance.setChecked(False)
        self.RFcheckImportance.setToolTip("Export the impurity-based feature importances\n"
                                          "as PNG image. The higher, the more important\n"
                                          "feature. It is also known as the Gini importance.")
        form_layout.addRow("Export feature importances:", self.RFcheckImportance)

        self.stack_RF.setLayout(form_layout)

    def stackui_rf_grid(self):
        form_layout = QFormLayout()

        # List of Random State
        self.RFgridlineRandomState = QLineEdit()
        self.RFgridlineRandomState.setValidator(self.intlist_validator)
        self.RFgridlineRandomState.setPlaceholderText("0,42,562,25685...")
        self.RFgridlineRandomState.setToolTip("Controls the randomness to build the trees.")
        form_layout.addRow("random_state:", self.RFgridlineRandomState)

        # List of Number of estimators (trees)
        self.RFgridlineEstimators = QLineEdit()
        self.RFgridlineEstimators.setValidator(self.intlist_validator)
        self.RFgridlineEstimators.setPlaceholderText("100,500,1000...")
        self.RFgridlineEstimators.setToolTip("List of the number of trees in the forest.")
        form_layout.addRow("n_estimators:", self.RFgridlineEstimators)

        # List of Criterion to mesure the quality of a split (use self.RFcriterion)
        self.RFgridlistCriterion = QListWidget()
        for idx, criterion in zip(range(0, len(self.RFcriterion)), self.RFcriterion):
            self.RFgridlistCriterion.insertItem(idx, criterion)
        self.RFgridlistCriterion.setCurrentItem(self.RFgridlistCriterion.item(0))
        # self.RFgridlistCriterion.setFlow(QListView.LeftToRight)
        self.RFgridlistCriterion.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.RFgridlistCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.RFgridlistCriterion)

        # List of Max depth for each tree
        self.RFgridlineMaxDepth = QLineEdit()
        self.RFgridlineMaxDepth.setValidator(self.intlist_validator)
        self.RFgridlineMaxDepth.setPlaceholderText("5,8,12...")
        self.RFgridlineMaxDepth.setToolTip("List of maximum depth of trees.")
        form_layout.addRow("max_depth:", self.RFgridlineMaxDepth)

        # List of Samples at Split
        self.RFgridlineSamplesSplit = QLineEdit()
        self.RFgridlineSamplesSplit.setValidator(self.intlist_validator)
        self.RFgridlineSamplesSplit.setPlaceholderText("2,50,100,500...")
        self.RFgridlineSamplesSplit.setToolTip("The minimum number of samples required\n"
                                               "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.RFgridlineSamplesSplit)

        # List of Samples at Leaf
        self.RFgridlineSamplesLeaf = QLineEdit()
        self.RFgridlineSamplesLeaf.setValidator(self.intlist_validator)
        self.RFgridlineSamplesLeaf.setPlaceholderText("1,50,100,500...")
        self.RFgridlineSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                              "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.RFgridlineSamplesLeaf)

        # List of Weight Leaf
        self.RFgridlineWeightLeaf = QLineEdit()
        self.RFgridlineWeightLeaf.setValidator(self.floatlist_validator)
        self.RFgridlineWeightLeaf.setPlaceholderText("0.00,0.01,0.05,0.2...")
        self.RFgridlineWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                             "of weights required to be at a leaf node. Samples\n"
                                             "have equal weight when sample_weight=0.00")
        form_layout.addRow("min_weight_fraction_leaf:", self.RFgridlineWeightLeaf)

        # List of Maximum Features (use self.RFmaxFeatures)
        self.RFgridlistMaxFeatures = QListWidget()
        for idx, method in zip(range(0, len(self.RFmaxFeatures)), self.RFmaxFeatures):
            self.RFgridlistMaxFeatures.insertItem(idx, method)
        self.RFgridlistMaxFeatures.setCurrentItem(self.RFgridlistMaxFeatures.item(0))
        # self.RFgridlistMaxFeatures.setFlow(QListView.LeftToRight)
        self.RFgridlistMaxFeatures.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.RFgridlistMaxFeatures.setToolTip("The number of features to consider\n"
                                              "when looking for the best split.")
        form_layout.addRow("max_features:", self.RFgridlistMaxFeatures)

        # Number of Jobs
        self.RFgridlineNJob = QLineEdit()
        self.RFgridlineNJob.setValidator(self.intlist_validator)
        self.RFgridlineNJob.setPlaceholderText("0,1,3,16...")
        self.RFgridlineNJob.setToolTip("The number of jobs to run in parallel.\n"
                                       "Differs from RandomForest without GridSearchCV:\n"
                                       "'0' means all processors. Leave empty = 1x'0'")
        form_layout.addRow("n_jobs:", self.RFgridlineNJob)

        self.stack_RF_grid.setLayout(form_layout)
        self.stack_RF_grid.setMaximumHeight(400)

    def stackui_gb(self):
        form_layout = QFormLayout()

        self.GBspinRandomState = QSpinBox()
        self.GBspinRandomState.setMinimum(-1)
        self.GBspinRandomState.setMaximum(2147483647)
        self.GBspinRandomState.setValue(-1)
        self.GBspinRandomState.setToolTip("Controls the randomness to build the trees.")
        self.GBpushRandomState = QPushButton('New Seed')
        self.GBpushRandomState.clicked.connect(lambda: new_seed(self.GBspinRandomState))
        h_layout_random = QHBoxLayout()
        h_layout_random.addWidget(self.GBspinRandomState)
        h_layout_random.addWidget(self.GBpushRandomState)
        form_layout.addRow("random_state:", h_layout_random)

        self.GBspinEstimators = QSpinBox()
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
        self.GBcomboCriterion.addItems(self.GBcriterion)
        self.GBcomboCriterion.setCurrentText("friedman_mse")
        self.GBcomboCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.GBcomboCriterion)

        self.GBspinMaxDepth = QSpinBox()
        self.GBspinMaxDepth.setMinimum(0)
        self.GBspinMaxDepth.setMaximum(9999)
        self.GBspinMaxDepth.setValue(3)
        self.GBspinMaxDepth.setToolTip("The maximum depth of the individual regression estimators.\n"
                                       "The maximum depth limits the number of nodes in the tree. ")
        form_layout.addRow("max_depth:", self.GBspinMaxDepth)

        self.GBspinSamplesSplit = QSpinBox()
        self.GBspinSamplesSplit.setMinimum(2)
        self.GBspinSamplesSplit.setMaximum(999999)
        self.GBspinSamplesSplit.setValue(2)
        self.GBspinSamplesSplit.setToolTip("The minimum number of samples required\n"
                                           "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.GBspinSamplesSplit)

        self.GBspinSamplesLeaf = QSpinBox()
        self.GBspinSamplesLeaf.setMaximum(999999)
        self.GBspinSamplesLeaf.setValue(1)
        self.GBspinSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                          "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.GBspinSamplesLeaf)

        self.GBspinWeightLeaf = QDoubleSpinBox()
        self.GBspinWeightLeaf.setDecimals(4)
        self.GBspinWeightLeaf.setMaximum(1)
        self.GBspinWeightLeaf.setValue(0)
        self.GBspinWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                         "of weights required to be at a leaf node. Samples\n"
                                         "have equal weight when sample_weight=0.")
        form_layout.addRow("min_weight_fraction_leaf:", self.GBspinWeightLeaf)

        self.GBmaxFeatures = ["None", "auto", "sqrt", "log2"]
        self.GBcomboMaxFeatures = QComboBox()
        self.GBcomboMaxFeatures.addItems(self.GBmaxFeatures)
        self.GBcomboMaxFeatures.setCurrentText("None")
        self.GBcomboMaxFeatures.setToolTip("The number of features to consider\n"
                                           "when looking for the best split.")
        form_layout.addRow("max_features:", self.GBcomboMaxFeatures)

        self.loss = ["deviance", "exponential"]
        self.GBcomboLoss = QComboBox()
        self.GBcomboLoss.addItems(self.loss)
        self.GBcomboLoss.setCurrentText("deviance")
        self.GBcomboLoss.setToolTip("The loss function to be optimized. ‘deviance’ refers to logistic\n"
                                    "regression for classification with probabilistic outputs. For loss \n"
                                    "‘exponential’ gradient boosting recovers the AdaBoost algorithm.")
        form_layout.addRow("loss:", self.GBcomboLoss)

        self.GBspinLearningRate = QDoubleSpinBox()
        self.GBspinLearningRate.setDecimals(6)
        self.GBspinLearningRate.setMaximum(999999)
        self.GBspinLearningRate.setMinimum(0)
        self.GBspinLearningRate.setValue(0.1)
        self.GBspinLearningRate.setToolTip("Learning rate shrinks the contribution of each tree\n"
                                           "by learning_rate. There is a trade-off between\n"
                                           "learning_rate and n_estimators.")
        form_layout.addRow("learning_rate:", self.GBspinLearningRate)

        self.GBspinSubsample = QDoubleSpinBox()
        self.GBspinSubsample.setDecimals(4)
        self.GBspinSubsample.setMaximum(999999)
        self.GBspinSubsample.setMinimum(0)
        self.GBspinSubsample.setValue(1)
        self.GBspinSubsample.setToolTip("The fraction of samples to be used for fitting\n"
                                        "the individual base learners. If smaller than\n"
                                        "1.0 this results in Stochastic Gradient Boosting.")
        form_layout.addRow("subsample:", self.GBspinSubsample)

        form_layout.addRow(QHLine())

        self.GBcheckImportance = QCheckBox()
        self.GBcheckImportance.setChecked(False)
        self.GBcheckImportance.setToolTip("Export the impurity-based feature importances\n"
                                          "as PNG image. The higher, the more important\n"
                                          "feature. It is also known as the Gini importance.")
        form_layout.addRow("Export feature importances:", self.GBcheckImportance)

        self.stack_GB.setLayout(form_layout)

    def stackui_gb_grid(self):
        form_layout = QFormLayout()

        # List of Random State
        self.GBgridlineRandomState = QLineEdit()
        self.GBgridlineRandomState.setValidator(self.intlist_validator)
        self.GBgridlineRandomState.setPlaceholderText("0,42,562,25685...")
        self.GBgridlineRandomState.setToolTip("Controls the randomness to build the trees.")
        form_layout.addRow("random_state:", self.GBgridlineRandomState)

        # List of Number of Estimators
        self.GBgridlineEstimators = QLineEdit()
        self.GBgridlineEstimators.setValidator(self.intlist_validator)
        self.GBgridlineEstimators.setPlaceholderText("100,500,1000...")
        self.GBgridlineEstimators.setToolTip("The number of boosting stages to perform.\n"
                                             "Gradient boosting is fairly robust to over-\n"
                                             "fitting so a large number usually results in\n"
                                             "better performance.")
        form_layout.addRow("n_estimators:", self.GBgridlineEstimators)

        # List of Criterion
        self.GBcriterion = ["friedman_mse", "mse"]
        self.GBgridlistCriterion = QListWidget()
        for idx, criterion in zip(range(0, len(self.GBcriterion)), self.GBcriterion):
            self.GBgridlistCriterion.insertItem(idx, criterion)
        self.GBgridlistCriterion.setCurrentItem(self.GBgridlistCriterion.item(0))
        self.GBgridlistCriterion.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.GBgridlistCriterion.setToolTip("The function to measure the quality of a split.")
        form_layout.addRow("criterion:", self.GBgridlistCriterion)

        # List of Max Depth
        self.GBgridlineMaxDepth = QLineEdit()
        self.GBgridlineMaxDepth.setValidator(self.intlist_validator)
        self.GBgridlineMaxDepth.setPlaceholderText("5,8,12...")
        self.GBgridlineMaxDepth.setToolTip("The maximum depth of the individual regression estimators.\n"
                                           "The maximum depth limits the number of nodes in the tree. ")
        form_layout.addRow("max_depth:", self.GBgridlineMaxDepth)

        # List of Samples at Split
        self.GBgridlineSamplesSplit = QLineEdit()
        self.GBgridlineSamplesSplit.setValidator(self.intlist_validator)
        self.GBgridlineSamplesSplit.setPlaceholderText("2,50,100,500...")
        self.GBgridlineSamplesSplit.setToolTip("The minimum number of samples required\n"
                                               "to split an internal node.")
        form_layout.addRow("min_samples_split:", self.GBgridlineSamplesSplit)

        # List of Samples at Leaf
        self.GBgridlineSamplesLeaf = QLineEdit()
        self.GBgridlineSamplesLeaf.setValidator(self.intlist_validator)
        self.GBgridlineSamplesLeaf.setPlaceholderText("1,50,100,500...")
        self.GBgridlineSamplesLeaf.setToolTip("The minimum number of samples required\n"
                                              "to be at a leaf node.")
        form_layout.addRow("min_samples_leaf:", self.GBgridlineSamplesLeaf)

        # List of Weight Leaf
        self.GBgridlineWeightLeaf = QLineEdit()
        self.GBgridlineWeightLeaf.setValidator(self.floatlist_validator)
        self.GBgridlineWeightLeaf.setPlaceholderText("0.00,0.01,0.05,0.2...")
        self.GBgridlineWeightLeaf.setToolTip("The minimum weighted fraction of the sum total\n"
                                             "of weights required to be at a leaf node. Samples\n"
                                             "have equal weight when sample_weight=0.")
        form_layout.addRow("min_weight_fraction_leaf:", self.GBgridlineWeightLeaf)

        # List of Maximum Features (use self.RFmaxFeatures)
        self.GBgridlistMaxFeatures = QListWidget()
        for idx, method in zip(range(0, len(self.RFmaxFeatures)), self.RFmaxFeatures):
            self.GBgridlistMaxFeatures.insertItem(idx, method)
        self.GBgridlistMaxFeatures.setCurrentItem(self.GBgridlistMaxFeatures.item(0))
        self.GBgridlistMaxFeatures.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.GBgridlistMaxFeatures.setToolTip("The number of features to consider\n"
                                              "when looking for the best split.")
        form_layout.addRow("max_features:", self.GBgridlistMaxFeatures)

        # List of Loss methods (use self.loss)
        self.GBgridlistLoss = QListWidget()
        for idx, method in zip(range(0, len(self.loss)), self.loss):
            self.GBgridlistLoss.insertItem(idx, method)
        self.GBgridlistLoss.setCurrentItem(self.GBgridlistLoss.item(0))
        self.GBgridlistLoss.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.GBgridlistLoss.setFlow(QListView.LeftToRight)
        self.GBgridlistLoss.setToolTip("The loss function to be optimized. ‘deviance’ refers to logistic\n"
                                       "regression for classification with probabilistic outputs. For loss \n"
                                       "‘exponential’ gradient boosting recovers the AdaBoost algorithm.")
        form_layout.addRow("loss:", self.GBgridlistLoss)

        # List of Learning Rate float
        self.GBgridlineLearningRate = QLineEdit()
        self.GBgridlineLearningRate.setValidator(self.floatlist_validator)
        self.GBgridlineLearningRate.setPlaceholderText("0.00001,0.001,0.1...")
        self.GBgridlineLearningRate.setToolTip("Learning rate shrinks the contribution of each tree\n"
                                               "by learning_rate. There is a trade-off between\n"
                                               "learning_rate and n_estimators.")
        form_layout.addRow("learning_rate:", self.GBgridlineLearningRate)

        # List of Subsample
        self.GBgridlineSubsample = QLineEdit()
        self.GBgridlineSubsample.setValidator(self.floatlist_validator)
        self.GBgridlineSubsample.setPlaceholderText("0.01,0.1,1.0...")
        self.GBgridlineSubsample.setToolTip("The fraction of samples to be used for fitting\n"
                                            "the individual base learners. If smaller than\n"
                                            "1.0 this results in Stochastic Gradient Boosting.")
        form_layout.addRow("subsample:", self.GBgridlineSubsample)

        self.stack_GB_grid.setLayout(form_layout)
        self.stack_GB_grid.setMaximumHeight(500)

    def stackui_nn(self):
        form_layout = QFormLayout()

        self.NNspinRandomState = QSpinBox()
        self.NNspinRandomState.setMinimum(-1)
        self.NNspinRandomState.setMaximum(2147483647)
        self.NNspinRandomState.setValue(-1)
        self.NNspinRandomState.setToolTip("Determines random number generation for weights and\n"
                                          "bias initialization, train-test split if early stopping is used,\n"
                                          "and batch sampling when solver=’sgd’or ‘adam’.")
        self.NNpushRandomState = QPushButton('New Seed')
        self.NNpushRandomState.clicked.connect(lambda: new_seed(self.NNspinRandomState))
        h_layout_random = QHBoxLayout()
        h_layout_random.addWidget(self.NNspinRandomState)
        h_layout_random.addWidget(self.NNpushRandomState)
        form_layout.addRow("random_state:", h_layout_random)

        self.NNlineHiddenLayers = QLineEdit()
        self.NNlineHiddenLayers.setPlaceholderText("Example: 50,100,50")
        self.NNlineHiddenLayers.setToolTip("The ith element represents the number of neurons\n"
                                           "in the ith hidden layer.")
        form_layout.addRow("hidden_layer_sizes:", self.NNlineHiddenLayers)

        self.NNactivation = ["identity", "logistic", "tanh", "relu"]
        self.NNcomboActivation = QComboBox()
        self.NNcomboActivation.addItems(self.NNactivation)
        self.NNcomboActivation.setCurrentText("relu")
        self.NNcomboActivation.setToolTip("Activation function for the hidden layer.")
        form_layout.addRow("activation:", self.NNcomboActivation)

        self.NNsolver = ["lbfgs", "sgd", "adam"]
        self.NNcomboSolver = QComboBox()
        self.NNcomboSolver.addItems(self.NNsolver)
        self.NNcomboSolver.setCurrentText("adam")
        self.NNcomboSolver.setToolTip("The solver for weight optimization.\n"
                                      "-'lbfgs' optimizer from quasi-Newton method family.\n"
                                      "-'sgd' refers to stochastic gradient descent.\n"
                                      "-'adam' refers to a stochastic gradient-based optimizer,\n"
                                      "  proposed by Kingma, Diederik, and Jimmy Ba.")
        form_layout.addRow("solver:", self.NNcomboSolver)

        self.NNspinAlpha = QDoubleSpinBox()
        self.NNspinAlpha.setDecimals(8)
        self.NNspinAlpha.setMinimum(0)
        self.NNspinAlpha.setMaximum(999999)
        self.NNspinAlpha.setValue(0.0001)
        self.NNspinAlpha.setToolTip("L2 penalty (regularization term) parameter.")
        form_layout.addRow("alpha:", self.NNspinAlpha)

        self.NNspinBatchSize = QSpinBox()
        self.NNspinBatchSize.setMinimum(-1)
        self.NNspinBatchSize.setMaximum(999999)
        self.NNspinBatchSize.setValue(-1)
        self.NNspinBatchSize.setToolTip("Size of minibatches for stochastic optimizers.\n"
                                        "If the solver is ‘lbfgs’, the classifier will not use minibatch.")
        form_layout.addRow("batch_size:", self.NNspinBatchSize)

        self.NNlearningRate = ["constant", "invscaling", "adaptive"]
        self.NNcomboLearningRate = QComboBox()
        self.NNcomboLearningRate.addItems(self.NNlearningRate)
        self.NNcomboLearningRate.setCurrentText("constant")
        self.NNcomboLearningRate.setEnabled(False)
        self.NNcomboLearningRate.setToolTip("Learning rate schedule for weight updates.")
        form_layout.addRow("learning_rate:", self.NNcomboLearningRate)

        self.NNspinLearningRateInit = QDoubleSpinBox()
        self.NNspinLearningRateInit.setDecimals(6)
        self.NNspinLearningRateInit.setMinimum(0)
        self.NNspinLearningRateInit.setMaximum(9999)
        self.NNspinLearningRateInit.setValue(0.001)
        self.NNspinLearningRateInit.setToolTip("The initial learning rate used. It controls\n"
                                               "the step-size in updating the weights.\n"
                                               "Only used when solver=’sgd’ or ‘adam’.")
        form_layout.addRow("learning_rate_init:", self.NNspinLearningRateInit)

        self.NNspinPowerT = QDoubleSpinBox()
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
        self.NNspinBeta_1.setDecimals(6)
        self.NNspinBeta_1.setMinimum(0)
        self.NNspinBeta_1.setMaximum(1)
        self.NNspinBeta_1.setValue(0.9)
        self.NNspinBeta_1.setToolTip("Exponential decay rate for estimates of first\n"
                                     "moment vector in adam, should be in [0, 1).\n"
                                     "Only used when solver=’adam’")
        form_layout.addRow("beta_1:", self.NNspinBeta_1)

        self.NNspinBeta_2 = QDoubleSpinBox()
        self.NNspinBeta_2.setDecimals(6)
        self.NNspinBeta_2.setMinimum(0)
        self.NNspinBeta_2.setMaximum(1)
        self.NNspinBeta_2.setValue(0.999)
        self.NNspinBeta_2.setToolTip("Exponential decay rate for estimates of second\n"
                                     "moment vector in adam, should be in [0, 1).\n"
                                     "Only used when solver=’adam’")
        form_layout.addRow("beta_2:", self.NNspinBeta_2)

        self.NNspinEpsilon = QDoubleSpinBox()
        self.NNspinEpsilon.setDecimals(8)
        self.NNspinEpsilon.setMinimum(0)
        self.NNspinEpsilon.setValue(0.00000001)
        self.NNspinEpsilon.setToolTip("Value for numerical stability in adam.\n"
                                      "Only used when solver=’adam’")
        form_layout.addRow("epsilon:", self.NNspinEpsilon)

        self.stack_NN.setLayout(form_layout)

        self.NNcomboSolver.currentTextChanged.connect(self.nn_solver_options)
        self.NNcomboLearningRate.currentTextChanged.connect(self.nn_solver_options)

    def stackui_nn_grid(self):
        form_layout = QFormLayout()

        # List of Random State
        self.NNgridlineRandomState = QLineEdit()
        self.NNgridlineRandomState.setValidator(self.intlist_validator)
        self.NNgridlineRandomState.setPlaceholderText("0,42,562,25685...")
        self.NNgridlineRandomState.setToolTip("Determines random number generation for weights and\n"
                                              "bias initialization, train-test split if early stopping is used,\n"
                                              "and batch sampling when solver=’sgd’or ‘adam’.")
        form_layout.addRow("random_state:", self.NNgridlineRandomState)

        # List of Hidden Layers
        self.NNgridlineHiddenLayers = QLineEdit()
        self.NNgridlineHiddenLayers.setPlaceholderText("[50][25,50,25][25,50,100,100]")
        self.NNgridlineHiddenLayers.setToolTip("List of lists of hidden layers.\n"
                                               "Use brackets !!")
        form_layout.addRow("hidden_layer_sizes:", self.NNgridlineHiddenLayers)

        # List of Activation Function (use self.NNactivation)
        self.NNgridlistActivation = QListWidget()
        for idx, function in zip(range(0, len(self.NNactivation)), self.NNactivation):
            self.NNgridlistActivation.insertItem(idx, function)
        self.NNgridlistActivation.setCurrentItem(self.NNgridlistActivation.item(3))
        self.NNgridlistActivation.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.NNgridlistActivation.setFlow(QListView.LeftToRight)
        self.NNgridlistActivation.setToolTip("Activation function for the hidden layer.")
        form_layout.addRow("activation:", self.NNgridlistActivation)

        # List of Solvers (use self.NNsolver)
        self.NNgridlistSolver = QListWidget()
        for idx, solver in zip(range(0, len(self.NNsolver)), self.NNsolver):
            self.NNgridlistSolver.insertItem(idx, solver)
        self.NNgridlistSolver.setCurrentItem(self.NNgridlistSolver.item(2))
        self.NNgridlistSolver.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.NNgridlistSolver.setFlow(QListView.LeftToRight)
        self.NNgridlistSolver.setToolTip("The solver for weight optimization.\n"
                                         "-'lbfgs' optimizer from quasi-Newton method family.\n"
                                         "-'sgd' refers to stochastic gradient descent.\n"
                                         "-'adam' refers to a stochastic gradient-based optimizer,\n"
                                         "  proposed by Kingma, Diederik, and Jimmy Ba.")
        form_layout.addRow("solver:", self.NNgridlistSolver)

        # List of Alpha
        self.NNgridlineAlpha = QLineEdit()
        self.NNgridlineAlpha.setValidator(self.floatlist_validator)
        self.NNgridlineAlpha.setPlaceholderText("0.00001,0.0001,0.01...")
        self.NNgridlineAlpha.setToolTip("L2 penalty (regularization term) parameter.")
        form_layout.addRow("alpha:", self.NNgridlineAlpha)

        # List of Batch Size
        self.NNgridlineBatchSize = QLineEdit()
        self.NNgridlineBatchSize.setValidator(self.intlist_validator)
        self.NNgridlineBatchSize.setPlaceholderText("50,200,500...")
        self.NNgridlineBatchSize.setToolTip("Size of minibatches for stochastic optimizers.\n"
                                            "If the solver is ‘lbfgs’, the classifier will not use minibatch.")
        form_layout.addRow("batch_size:", self.NNgridlineBatchSize)

        # List of Learning Rate methods (use self.NNlearningRate)
        self.NNgridlistLearningRate = QListWidget()
        for idx, method in zip(range(0, len(self.NNlearningRate)), self.NNlearningRate):
            self.NNgridlistLearningRate.insertItem(idx, method)
        self.NNgridlistLearningRate.setCurrentItem(self.NNgridlistLearningRate.item(0))
        self.NNgridlistLearningRate.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.NNgridlistLearningRate.setFlow(QListView.LeftToRight)
        self.NNgridlistLearningRate.setEnabled(False)
        self.NNgridlistLearningRate.setToolTip("Learning rate schedule for weight updates.")
        form_layout.addRow("learning_rate:", self.NNgridlistLearningRate)

        # List of Learning Rate Initialisation
        self.NNgridlineLearningRateInit = QLineEdit()
        self.NNgridlineLearningRateInit.setValidator(self.floatlist_validator)
        self.NNgridlineLearningRateInit.setPlaceholderText("0.0001,0.001,0.01...")
        self.NNgridlineLearningRateInit.setToolTip("The initial learning rate used. It controls\n"
                                                   "the step-size in updating the weights.\n"
                                                   "Only used when solver=’sgd’ or ‘adam’.")
        form_layout.addRow("learning_rate_init:", self.NNgridlineLearningRateInit)

        # List of PowerT
        self.NNgridlinePowerT = QLineEdit()
        self.NNgridlinePowerT.setValidator(self.floatlist_validator)
        self.NNgridlinePowerT.setEnabled(False)
        self.NNgridlinePowerT.setPlaceholderText("0.1,0.25,0.5...")
        self.NNgridlinePowerT.setToolTip("The exponent for inverse scaling learning rate.\n"
                                         "It is used in updating effective learning rate\n"
                                         "when the learning_rate is set to ‘invscaling’.\n"
                                         "Only used when solver=’sgd’.")
        form_layout.addRow("power_t:", self.NNgridlinePowerT)

        # List of Maximum Iteration
        self.NNgridlineMaxIter = QLineEdit()
        self.NNgridlineMaxIter.setValidator(self.intlist_validator)
        self.NNgridlineMaxIter.setPlaceholderText("200,1000,5000...")
        self.NNgridlineMaxIter.setToolTip("The solver iterates until convergence or this number of iterations.\n"
                                          "For stochastic solvers ('sgd' or 'adam'), note that this determines\n"
                                          "the number of epochs (how many times each data point will be\n"
                                          "used), not the number of gradient steps.")
        form_layout.addRow("max_iter:", self.NNgridlineMaxIter)

        # List Shuffle and not Shuffle
        self.NNgridlistShuffle = QListWidget()
        self.NNgridlistShuffle.insertItem(0, "shuffle")
        self.NNgridlistShuffle.insertItem(1, "no shuffle")
        self.NNgridlistShuffle.setCurrentItem(self.NNgridlistShuffle.item(0))
        self.NNgridlistShuffle.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.NNgridlistShuffle.setFlow(QListView.LeftToRight)
        self.NNgridlistShuffle.setToolTip("Whether to shuffle samples in each iteration.\n"
                                          "Only used when solver=’sgd’ or ‘adam’.")
        form_layout.addRow("shuffle:", self.NNgridlistShuffle)

        # List of Beta_1
        self.NNgridlineBeta_1 = QLineEdit()
        self.NNgridlineBeta_1.setValidator(self.floatlist_validator)
        self.NNgridlineBeta_1.setPlaceholderText("0.5,0.75,0.9")
        self.NNgridlineBeta_1.setToolTip("Exponential decay rate for estimates of first\n"
                                         "moment vector in adam, should be in [0, 1).\n"
                                         "Only used when solver=’adam’")
        form_layout.addRow("beta_1:", self.NNgridlineBeta_1)

        # List of Beta_2
        self.NNgridlineBeta_2 = QLineEdit()
        self.NNgridlineBeta_2.setValidator(self.floatlist_validator)
        self.NNgridlineBeta_2.setPlaceholderText("0.9,0.99,0.999...")
        self.NNgridlineBeta_2.setToolTip("Exponential decay rate for estimates of second\n"
                                         "moment vector in adam, should be in [0, 1).\n"
                                         "Only used when solver=’adam’")
        form_layout.addRow("beta_2:", self.NNgridlineBeta_2)

        # List of Epsilon
        self.NNgridlineEpsilon = QLineEdit()
        self.NNgridlineEpsilon.setValidator(self.floatlist_validator)
        self.NNgridlineEpsilon.setPlaceholderText("0.00000001,0.000001,0.0001...")
        self.NNgridlineEpsilon.setToolTip("Value for numerical stability in adam.\n"
                                          "Only used when solver=’adam’")
        form_layout.addRow("epsilon:", self.NNgridlineEpsilon)

        self.stack_NN_grid.setLayout(form_layout)

        self.NNgridlistSolver.itemSelectionChanged.connect(self.nn_grid_solver_options)
        self.NNgridlistLearningRate.itemSelectionChanged.connect(self.nn_grid_solver_options)

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

    def nn_grid_solver_options(self):
        solver_list = [item.text() for item in self.NNgridlistSolver.selectedItems()]
        learning_rate_list = [item.text() for item in self.NNgridlistLearningRate.selectedItems()]

        self.NNgridlineBatchSize.setEnabled(False)
        self.NNgridlistLearningRate.setEnabled(False)
        self.NNgridlineLearningRateInit.setEnabled(False)
        self.NNgridlinePowerT.setEnabled(False)
        self.NNgridlistShuffle.setEnabled(False)
        self.NNgridlineBeta_1.setEnabled(False)
        self.NNgridlineBeta_2.setEnabled(False)
        self.NNgridlineEpsilon.setEnabled(False)

        if 'sgd' in solver_list:
            self.NNgridlineBatchSize.setEnabled(True)
            self.NNgridlistLearningRate.setEnabled(True)
            self.NNgridlineLearningRateInit.setEnabled(True)
            self.NNgridlistShuffle.setEnabled(True)

        if 'adam' in solver_list:
            self.NNgridlineBatchSize.setEnabled(True)
            self.NNgridlineLearningRateInit.setEnabled(True)
            self.NNgridlistShuffle.setEnabled(True)
            self.NNgridlineBeta_1.setEnabled(True)
            self.NNgridlineBeta_2.setEnabled(True)
            self.NNgridlineEpsilon.setEnabled(True)

        if 'sgd' in solver_list and 'invscaling' in learning_rate_list:
            self.NNgridlinePowerT.setEnabled(True)

    def display_stack_algo(self):
        index = self.comboAlgorithms.currentIndex()

        # stack without GridSearchCV
        if self.checkGridSearchCV.isChecked() is False:
            self.stackAlgo.setCurrentIndex(index)
            if index == 0:  # Random Forest
                self.stackAlgo.setMaximumHeight(self.stack_RF.maximumHeight())
            if index == 1:  # Gradient Boosting
                self.stackAlgo.setMaximumHeight(self.stack_GB.maximumHeight())
            if index == 2:  # Neural Network
                self.stackAlgo.setMaximumHeight(self.stack_NN.maximumHeight())
        else:  # with GridSearchCV
            index += 3
            self.stackAlgo.setCurrentIndex(index)
            if index == 3:  # Random Forest GridSearchCV
                self.stackAlgo.setMaximumHeight(self.stack_RF_grid.maximumHeight())
            if index == 4:  # Gradient Boosting GridSearchCV
                self.stackAlgo.setMaximumHeight(self.stack_GB_grid.maximumHeight())
            if index == 5:  # Neural Network GridSearchCV
                self.stackAlgo.setMaximumHeight(self.stack_NN_grid.maximumHeight())

        algo_list = ["Random Forest", "Gradient Boosting", "Neural Network",
                     "Random Forest (GridSearchCV)",
                     "Gradient Boosting (GridSearchCV)",
                     "Neural Network (GridSearchCV)"]
        self.statusBar.showMessage("{} parameters".format(algo_list[index]), 3000)

    # Tab of predictions
    def tabui_predict(self):
        # Group Prediction
        self.groupModel = QGroupBox("Input Model")

        # Line for local model input
        self.lineModelFile = QLineEdit()
        self.lineModelFile.setPlaceholderText("Select the input model")
        self.lineModelFile.editingFinished.connect(self.open_model)
        self.toolButtonModel = QToolButton()
        self.toolButtonModel.setText("Browse")
        self.toolButtonModel.clicked.connect(self.get_model)
        self.hLocalModel = QHBoxLayout()
        self.hLocalModel.addWidget(self.lineModelFile)
        self.hLocalModel.addWidget(self.toolButtonModel)

        # Line for server model input
        self.lineServerModel = QLineEdit()
        self.lineServerModel.setPlaceholderText("Give the input model on server")
        self.lineServerModel.setEnabled(False)

        # Model layout form
        self.formModel = QFormLayout()
        self.formModel.addRow(QLabel("Local model input:"), self.hLocalModel)
        self.formModel.addRow(QLabel("Server model input:"), self.lineServerModel)
        self.groupModel.setLayout(self.formModel)

        # Group Algo parameters
        self.groupAlgoParam = QGroupBox("Algorithm")

        # Scaler
        self.labelModelScaler = QLabel()

        # PCA
        self.labelModelPCA = QLabel()

        # Name of the algorithm
        self.labelModelName = QLabel()
        self.scrollModelParam = QScrollArea()
        self.scrollModelParam.setFrameShape(QFrame.NoFrame)

        # Form layout for the algo parameters from the loaded model
        self.formAlgoParam = QFormLayout()
        self.formAlgoParam.addRow(QLabel("Scaler:"), self.labelModelScaler)
        self.formAlgoParam.addRow(QHLine())
        self.formAlgoParam.addRow(QLabel("PCA:"), self.labelModelPCA)
        self.formAlgoParam.addRow(QHLine())
        self.formAlgoParam.addRow(QLabel("Algorithm:"), self.labelModelName)
        self.formAlgoParam.addRow(QLabel("Parameters:"), self.scrollModelParam)
        self.groupAlgoParam.setLayout(self.formAlgoParam)

        # Group Features from model
        self.groupModelFeatures = QGroupBox('Features')
        self.listModelFeatures = QListWidget()
        self.listModelFeatures.setSelectionMode(QAbstractItemView.NoSelection)
        self.listModelFeatures.setMinimumHeight(250)
        self.pushModelFeatures = QPushButton('Match Features')
        self.pushModelFeatures.clicked.connect(self.check_model_features)

        v_layout_features = QVBoxLayout()
        v_layout_features.addWidget(self.listModelFeatures)
        v_layout_features.addWidget(self.pushModelFeatures)
        self.groupModelFeatures.setLayout(v_layout_features)

        # Vertical layout for prediction tab
        self.vLayoutTabPredict = QVBoxLayout()
        self.vLayoutTabPredict.addWidget(self.groupModel)
        self.vLayoutTabPredict.addWidget(self.groupAlgoParam)
        self.vLayoutTabPredict.addWidget(self.groupModelFeatures)

        self.tabPredict.setLayout(self.vLayoutTabPredict)

    def get_model(self):
        self.statusBar.showMessage("Select model...", 3000)
        filename = QFileDialog.getOpenFileName(self, 'Select model file',
                                               '', "model files (*.model);;")

        if filename[0] != '':
            if self.pushLocal.isChecked():
                self.lineModelFile.setText(os.path.normpath(filename[0]))

            self.open_model()

    def open_model(self):
        """
        Open the model in self.lineModelFile and write all data in prediction tab.
        """
        # Model reset
        self.model_features = list()
        self.labelModelName.clear()
        self.labelModelScaler.clear()
        self.labelModelPCA.clear()
        self.scrollModelParam.setWidget(QLabel(''))
        self.scrollModelParam.setMinimumHeight(250)
        self.listModelFeatures.clear()

        # Clear feature selection from input file
        self.reset_selection_features()

        # Check the extension of the given file
        model_path = os.path.normpath(self.lineModelFile.text())
        model_root_ext = os.path.splitext(model_path)
        if model_root_ext[1] == '.model':

            # Check if file exist
            try:
                loaded_model = joblib.load(model_path)
            except FileNotFoundError:
                self.statusBar.showMessage("No such model file !", 3000)
            except PermissionError:
                self.statusBar.showMessage("Permission Error !", 3000)
            else:

                # Retrieve algorithm, model
                algorithm = loaded_model['algorithm']
                model = loaded_model['model']

                # Scaler
                scaler = model['scaler']

                # PCA
                try:
                    if model['pca']:
                        pca = model['pca'].get_params()['n_components']
                        pca = str(pca) + ' components'
                    else:
                        pca = 'No PCA applied'
                except KeyError:
                    pca = 'No PCA applied'

                # Parameters
                algo_parameters = str()
                dict_algo_param = model['classifier'].get_params()
                for key in dict_algo_param:
                    algo_parameters += str(key) + ': ' + str(dict_algo_param[key]) + '\n'
                label_model_param = QLabel(algo_parameters)

                # Retrieve features used in model
                self.model_features = loaded_model['feature_names']
                self.listModelFeatures.addItems(self.model_features)

                # Update the data of the loaded model
                self.labelModelName.setText(str(algorithm))
                self.labelModelScaler.setText(str(scaler))
                self.labelModelPCA.setText(str(pca))
                self.scrollModelParam.setWidget(label_model_param)

        else:
            self.statusBar.showMessage("Invalid model file !", 3000)

    def check_model_features(self):
        """
        Check if all features in the model are in the input file.
        """
        # Reset the self.predict_features
        self.predict_features = False  # Bool to lock prediction if a feaure is missing

        # Try if the list of model feature is defined
        try:
            self.model_features
        except AttributeError:
            warning_box("No model feature found!\nPlease, open a valid model.",
                        "No Feature Found")
        else:
            if self.max_feat_count > 0:
                # Clear feature selection
                self.reset_selection_features()

                missing_features = list()
                for m_feature in self.model_features:
                    standard_feature = self.listStandardLAS.findItems(m_feature, Qt.MatchExactly)
                    extra_feature = self.listExtraFeatures.findItems(m_feature, Qt.MatchExactly)

                    # Coordinates X, Y, Z
                    if m_feature == 'X':
                        self.checkAdvancedFeat.setChecked(True)
                        self.checkX.setChecked(True)
                    elif m_feature == 'Y':
                        self.checkAdvancedFeat.setChecked(True)
                        self.checkY.setChecked(True)
                    elif m_feature == 'Z':
                        self.checkAdvancedFeat.setChecked(True)
                        self.checkZ.setChecked(True)

                    # Standard features
                    elif len(standard_feature) > 0:
                        self.checkAdvancedFeat.setChecked(True)
                        row = self.listStandardLAS.row(standard_feature[0])
                        self.listStandardLAS.item(row).setSelected(True)

                    # Extra features
                    elif len(extra_feature) > 0:
                        row = self.listExtraFeatures.row(extra_feature[0])
                        self.listExtraFeatures.item(row).setSelected(True)

                    # Missing features
                    else:
                        missing_features.append(m_feature)

                if missing_features:
                    self.predict_features = False
                    error_box("One or several features are missing in input file:\n"
                              "{}".format(str(missing_features)), "Missing Features")
                else:
                    self.predict_features = True
                    self.statusBar.showMessage("All model features found in input file!",
                                               3000)

            else:
                self.predict_features = False
                warning_box("Features from input file not found!\n"
                            "Make sure a valid input file is opened properly.",
                            "No Feature Found")

    # Tab of segmentation
    def tabui_segment(self):
        # Group Segmentation
        self.groupSegment = QGroupBox("Segmentation Parameters")

        # Label: Target field exist ?
        self.SeglabelTarget = QLabel()
        self.SeglabelTarget.setWordWrap(True)
        self.SeglabelTarget.setToolTip("'Target' field must be discarded for segmentation.\n"
                                       "Otherwise, the segmentation may be based on this field.")

        # Set the scaler
        self.SegcomboScaler = QComboBox()
        self.SegcomboScaler.addItems(self.scalerNames)
        self.SegcomboScaler.setCurrentText("Standard")
        self.SegcomboScaler.setToolTip("Set the method to scale the data.")
        self.SegcomboScaler.setEnabled(False)

        # Set the Principal Component Analysis
        self.SegspinPCA = QSpinBox()
        self.SegspinPCA.setMinimum(0)
        self.SegspinPCA.setMaximum(9999)
        self.SegspinPCA.setValue(0)
        self.SegspinPCA.setToolTip("Set the Principal Component Analysis\n"
                                   "and the number of principal components.")
        self.SegspinPCA.setEnabled(False)

        # Fill the layout for Data parameters of the semgentation
        form_segment = QFormLayout()
        form_segment.addRow("Target field:", self.SeglabelTarget)
        form_segment.addRow("Scaler:", self.SegcomboScaler)
        form_segment.addRow("PCA:", self.SegspinPCA)

        self.groupSegment.setLayout(form_segment)
        self.groupSegment.setEnabled(False)
        self.groupSegment.setToolTip("Not active yet!")

        # Cluster algorithms
        # self.groupCluster = QGroupBox("Cluster Algorithms")
        self.groupCluster = QGroupBox("K-means Algorithm")

        # random_state
        self.KMspinRandomState = QSpinBox()
        self.KMspinRandomState.setMinimum(0)
        self.KMspinRandomState.setMaximum(2147483647)  # 2^31 - 1
        self.KMspinRandomState.setValue(0)
        self.KMspinRandomState.setToolTip("Controls the random generation of centroids.")
        self.KMpushRandomSeed = QPushButton("New Seed")
        self.KMpushRandomSeed.clicked.connect(lambda: new_seed(self.KMspinRandomState))
        h_layout_random = QHBoxLayout()
        h_layout_random.addWidget(self.KMspinRandomState)
        h_layout_random.addWidget(self.KMpushRandomSeed)

        # n_clusters
        self.KMspinNClusters = QSpinBox()
        self.KMspinNClusters.setMinimum(2)
        self.KMspinNClusters.setMaximum(9999)
        self.KMspinNClusters.setValue(8)
        self.KMspinNClusters.setToolTip("The number of clusters to form as well\n"
                                        "as the number of centroids to generate.")

        # init
        self.KMinit = ["k-means++", "random"]
        self.KMcomboInit = QComboBox()
        self.KMcomboInit.addItems(self.KMinit)
        self.KMcomboInit.setCurrentIndex(self.KMinit.index("k-means++"))
        self.KMcomboInit.setToolTip("Method for initialization.")

        # n_init
        self.KMspinNInit = QSpinBox()
        self.KMspinNInit.setMinimum(1)
        self.KMspinNInit.setMaximum(9999)
        self.KMspinNInit.setValue(10)
        self.KMspinNInit.setToolTip("Number of time the k-means algorithm will be\n"
                                    "run with different centroid seeds. The final\n"
                                    "results will be the best output of n_init\n"
                                    "consecutive runs in terms of inertia.")

        # max_iter
        self.KMspinMaxIter = QSpinBox()
        self.KMspinMaxIter.setMinimum(1)
        self.KMspinMaxIter.setMaximum(99999)
        self.KMspinMaxIter.setValue(300)
        self.KMspinMaxIter.setToolTip("Maximum number of iterations of the k-means\n"
                                      "algorithm for a single run.")

        # tol
        self.KMspinTol = QDoubleSpinBox()
        self.KMspinTol.setDecimals(8)
        self.KMspinTol.setMinimum(0)
        self.KMspinTol.setMaximum(9999)
        self.KMspinTol.setValue(0.0001)
        self.KMspinTol.setToolTip("Relative tolerance with regards to Frobenius norm\n"
                                  "of the difference in the cluster centers of two\n"
                                  "consecutive iterations to declare convergence.")

        # algorithm
        self.KMalgorithm = ["auto", "full", "elkan"]
        self.KMcomboAlgorithm = QComboBox()
        self.KMcomboAlgorithm.addItems(self.KMalgorithm)
        self.KMcomboAlgorithm.setCurrentIndex(self.KMalgorithm.index("auto"))
        self.KMcomboAlgorithm.setToolTip("K-means algorithm to use.")

        # form layout for cluster parameters
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.addRow("random_state:", h_layout_random)
        form_layout.addRow("n_clusters:", self.KMspinNClusters)
        form_layout.addRow("init:", self.KMcomboInit)
        form_layout.addRow("n_init:", self.KMspinNInit)
        form_layout.addRow("max_iter:", self.KMspinMaxIter)
        form_layout.addRow("tol:", self.KMspinTol)
        form_layout.addRow("algorithm:", self.KMcomboAlgorithm)

        self.groupCluster.setLayout(form_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.groupSegment)
        layout.addWidget(self.groupCluster)

        self.tabSegment.setLayout(layout)

    def tab_modes_action(self, tab):
        if tab == 0:  # For Training tab
            self.buttonRunTrain.setVisible(True)
            self.buttonRunPredict.setVisible(False)
            self.buttonRunSegment.setVisible(False)
        elif tab == 1:  # For Prediction tab
            self.buttonRunTrain.setVisible(False)
            self.buttonRunPredict.setVisible(True)
            self.buttonRunSegment.setVisible(False)
        elif tab == 2:  # For Segmentation tab
            self.buttonRunTrain.setVisible(False)
            self.buttonRunPredict.setVisible(False)
            self.buttonRunSegment.setVisible(True)
        else:
            raise IndexError("Returning selected tab does not exist!")

    # Feature selection
    def feature_part(self):
        """
        Give the central part of the GUI
        """
        # Global Feature Group
        self.groupFeatures = QGroupBox("Features")

        # Advanced features
        self.checkAdvancedFeat = QCheckBox("Enable advanced features")
        self.checkAdvancedFeat.stateChanged.connect(self.enable_advanced_features)
        self.checkAdvancedFeat.stateChanged.connect(self.number_selected_features)
        self.groupAdvancedFeat = QGroupBox("Advanced Features")

        # Coordinate fields
        self.groupCoordinates = QGroupBox("Coordinates")
        self.checkX = QCheckBox("X")
        self.checkX.setToolTip("Use X field as feature for training.")
        self.checkX.stateChanged.connect(self.number_selected_features)
        self.checkY = QCheckBox("Y")
        self.checkY.setToolTip("Use Y field as feature for training.")
        self.checkY.stateChanged.connect(self.number_selected_features)
        self.checkZ = QCheckBox("Z")
        self.checkZ.setToolTip("Use Z field as feature for training.")
        self.checkZ.stateChanged.connect(self.number_selected_features)
        self.hLayoutCoordinates = QHBoxLayout()
        self.hLayoutCoordinates.addWidget(self.checkX)
        self.hLayoutCoordinates.addWidget(self.checkY)
        self.hLayoutCoordinates.addWidget(self.checkZ)
        self.groupCoordinates.setLayout(self.hLayoutCoordinates)

        # List of standard field of LAS
        self.groupStandardLAS = QGroupBox("Standard LAS fields")
        self.labelStandardLAS = QLabel("\n\n\n\n"
                                       "(press Ctrl for\n"
                                       "multiple selection)")
        self.labelStandardLAS.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.listStandardLAS = QListWidget()
        self.listStandardLAS.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listStandardLAS.setSortingEnabled(False)
        self.listStandardLAS.itemSelectionChanged.connect(self.number_selected_features)
        v_standard_las = QVBoxLayout()
        v_standard_las.addWidget(self.listStandardLAS)
        self.groupStandardLAS.setLayout(v_standard_las)

        # List of features
        self.groupExtraFeatures = QGroupBox("Extra Features")

        self.labelFeatures = QLabel("(press Ctrl for multiple selection)")
        self.labelFeatures.setWordWrap(True)
        self.listExtraFeatures = QListWidget()
        self.listExtraFeatures.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listExtraFeatures.setSortingEnabled(True)
        self.listExtraFeatures.itemSelectionChanged.connect(self.number_selected_features)

        v_extra_features = QVBoxLayout()
        v_extra_features.addWidget(self.listExtraFeatures)
        v_extra_features.addWidget(self.labelFeatures)
        self.groupExtraFeatures.setLayout(v_extra_features)

        # Number of selected features
        self.labelNbrSelFeatures = QLabel()
        self.number_selected_features()
        self.pushResetSelection = QPushButton("Reset Selection")
        self.pushResetSelection.clicked.connect(self.reset_selection_features)

        # Fill the feature groupBox with layout
        self.vLayoutCentral = QVBoxLayout()
        self.vLayoutCentral.addWidget(self.checkAdvancedFeat)
        self.vLayoutCentral.addWidget(self.groupCoordinates)
        self.vLayoutCentral.addWidget(self.groupStandardLAS)
        self.vLayoutCentral.setStretchFactor(self.groupStandardLAS, 1)
        self.vLayoutCentral.addWidget(self.groupExtraFeatures)
        self.vLayoutCentral.setStretchFactor(self.groupExtraFeatures, 4)
        self.vLayoutCentral.addWidget(self.pushResetSelection)
        self.vLayoutCentral.addWidget(self.labelNbrSelFeatures)

        # Fill Feature group
        self.groupFeatures.setLayout(self.vLayoutCentral)

    def enable_advanced_features(self):
        if self.checkAdvancedFeat.isChecked():
            self.groupCoordinates.setVisible(True)
            self.groupCoordinates.setEnabled(True)
            if self.file_type == 'LAS':
                self.groupStandardLAS.setEnabled(True)
                self.groupStandardLAS.setVisible(True)
            else:
                self.groupStandardLAS.setVisible(False)
                self.groupStandardLAS.setEnabled(False)
        else:
            self.groupCoordinates.setVisible(False)
            self.groupCoordinates.setEnabled(False)
            self.groupStandardLAS.setVisible(False)
            self.groupStandardLAS.setEnabled(False)

        self.number_selected_features()

    def number_selected_features(self):
        self.max_feat_count = 0
        self.sel_feat_count = 0

        # Coordinate Features
        if self.checkAdvancedFeat.isChecked():
            if self.checkX.isEnabled():
                self.max_feat_count += 1
                if self.checkX.isChecked():
                    self.sel_feat_count += 1
            if self.checkY.isEnabled():
                self.max_feat_count += 1
                if self.checkY.isChecked():
                    self.sel_feat_count += 1
            if self.checkZ.isEnabled():
                self.max_feat_count += 1
                if self.checkZ.isChecked():
                    self.sel_feat_count += 1

        if self.listStandardLAS is not None and self.groupStandardLAS.isEnabled():
            self.max_feat_count += self.listStandardLAS.count()
            self.sel_feat_count += len(self.listStandardLAS.selectedItems())

        if self.listExtraFeatures is not None:
            self.max_feat_count += self.listExtraFeatures.count()
            self.sel_feat_count += len(self.listExtraFeatures.selectedItems())

        self.labelNbrSelFeatures.setText("Number of selected features: {} / {}".format(
            self.sel_feat_count, self.max_feat_count))

    def reset_selection_features(self):
        """
        Deselect all selected features.
        """
        self.checkX.setChecked(False)
        self.checkY.setChecked(False)
        self.checkZ.setChecked(False)
        self.listStandardLAS.clearSelection()
        self.listExtraFeatures.clearSelection()

    # Configuration for train, prediction and segmentation
    def set_selected_features(self, feature_names):
        """
        Set the selected features in the GUI according the given list.
        :param feature_names: List of the selected features.
        """
        # Loop throught all selected features
        for feature in feature_names:
            # Set the coordinate feature
            if feature == 'X':
                self.checkAdvancedFeat.setChecked(True)
                self.checkX.setChecked(True)
            if feature == 'Y':
                self.checkAdvancedFeat.setChecked(True)
                self.checkY.setChecked(True)
            if feature == 'Z':
                self.checkAdvancedFeat.setChecked(True)
                self.checkZ.setChecked(True)

            # Set the standard LAS features
            if self.file_type == 'LAS':
                standard_item = self.listStandardLAS.findItems(feature, Qt.MatchExactly)
                if len(standard_item) > 0:
                    row = self.listStandardLAS.row(standard_item[0])
                    self.listStandardLAS.item(row).setSelected(True)

            item = self.listExtraFeatures.findItems(feature, Qt.MatchExactly)
            if len(item) > 0:
                row = self.listExtraFeatures.row(item[0])
                self.listExtraFeatures.item(row).setSelected(True)

    def open_config(self):
        """
        Open configuration JSON file and set all saved parameters.
        """
        # Get config filename
        self.statusBar.showMessage("Select config file...", 3000)
        filename = QFileDialog.getOpenFileName(self, 'Select config file',
                                               '', "JSON files (*.json)")

        if filename[0] != '':
            # Open config file and get data
            with open(filename[0], 'r') as config_file:
                config_dict = json.load(config_file)

            # Update point cloud parameters
            if config_dict['local_compute']:
                self.pushLocal.setChecked(True)
                self.display_stack_local()
                self.lineLocalFile.setText(config_dict['input_file'])
                self.open_file()
                self.lineLocalFolder.setText(config_dict['output_folder'])
            else:
                self.pushServer.setChecked(True)
                self.display_stack_server()
                self.lineFile.setText(config_dict['local_input'])
                self.open_file()
                self.lineServerFile.setText(config_dict['input_file'])
                self.lineServerFolder.setText(config_dict['output_folder'])

            # Config version and mode
            try:
                self.config_v_m = config_dict['version']
            except KeyError:
                error_box("Version of config file is not compatible!", "Incompatible config file")
            else:
                self.config_version = self.config_v_m.split('_')[0]
                self.config_mode = self.config_v_m.split('_')[-1]

                if self.config_mode == 'train':
                    self.open_train_config(config_dict)
                elif self.config_mode == 'predi':
                    self.open_predict_config(config_dict)
                elif self.config_mode == 'segme':
                    self.open_segment_config(config_dict)
                else:
                    error_box("No valid mode found on config file!", "No valid mode")

    def open_train_config(self, config_dict):
        """
        Open the configuration file for training.
        :param config_dict: The dictionnary from configuration file.
        """
        # Set sample size and train ratio
        self.spinSampleSize.setValue(config_dict['samples'])
        self.spinTrainRatio.setValue(config_dict['training_ratio'])

        # Set scaler, scorer, random_state and pca
        self.comboScaler.setCurrentIndex(self.scalerNames.index(config_dict['scaler']))
        self.comboScorer.setCurrentIndex(self.scorerNames.index(config_dict['scorer']))
        self.spinNJobCV.setValue(config_dict['n_jobs_cv'])
        self.spinRandomState.setValue(config_dict['random_state'])
        try:
            self.spinPCA.setValue(config_dict['pca'])
        except KeyError:
            self.spinPCA.setValue(0)

        # Update algorithm
        algorithm = config_dict['algorithm']

        # Check GridSearchCV
        if config_dict['grid_search']:
            self.checkGridSearchCV.setChecked(True)
            # Update algorithm parameters
            if algorithm == 'RandomForestClassifier':
                self.algo = 'rf'
                self.comboAlgorithms.setCurrentIndex(0)

                # Get algo parameter dict
                param_dict = config_dict['param_grid']

                # n_estimators
                self.RFgridlineEstimators.setText(list2str(param_dict['n_estimators']))

                # criterion
                criterions = param_dict['criterion']
                for criterion in criterions:
                    item = self.RFgridlistCriterion.findItems(criterion, Qt.MatchExactly)
                    if len(item) > 0:  # If criterion is present
                        row = self.RFgridlistCriterion.row(item[0])
                        self.RFgridlistCriterion.item(row).setSelected(True)

                # max_depth
                self.RFgridlineMaxDepth.setText(list2str(param_dict['max_depth']))

                # min_samples_split
                self.RFgridlineSamplesSplit.setText(list2str(param_dict['min_samples_split']))

                # min_samples_leaf
                self.RFgridlineSamplesLeaf.setText(list2str(param_dict['min_samples_leaf']))

                # min_weight_fraction_leaf
                self.RFgridlineWeightLeaf.setText(list2str(param_dict['min_weight_fraction_leaf']))

                # max_features
                max_features = param_dict['max_features']
                for method in max_features:
                    item = self.RFgridlistMaxFeatures.findItems(method, Qt.MatchExactly)
                    if len(item) > 0:  # If method is present
                        row = self.RFgridlistMaxFeatures.row(item[0])
                        self.RFgridlistMaxFeatures.item(row).setSelected(True)

                # n_jobs
                self.RFgridspinNJob.setValue(param_dict['n_jobs'])

                # random_state
                self.RFgridlineRandomState.setText(list2str(param_dict['random_state']))

            elif algorithm == 'GradientBoostingClassifier':
                self.algo = 'gb'
                self.comboAlgorithms.setCurrentIndex(1)

                # Get algo parameter dict
                param_dict = config_dict['param_grid']

                # loss
                loss = param_dict['loss']
                for method in loss:
                    item = self.GBgridlistLoss.findItems(method, Qt.MatchExactly)
                    if len(item) > 0:  # If method is present
                        row = self.GBgridlistLoss.row(item[0])
                        self.GBgridlistLoss.item(row).setSelected(True)

                # learning_rate
                self.GBgridlineLearningRate.setText(list2str(param_dict['learning_rate']))

                # n_estimators
                self.GBgridlineEstimators.setText(list2str(param_dict['n_estimators']))

                # subsample
                self.GBgridlineSubsample.setText(list2str(param_dict['subsample']))

                # criterion
                criterions = param_dict['criterion']
                for criterion in criterions:
                    item = self.GBgridlistCriterion.findItems(criterion, Qt.MatchExactly)
                    if len(item) > 0:  # If criterion is present
                        row = self.GBgridlistCriterion.row(item[0])
                        self.GBgridlistCriterion.item(row).setSelected(True)

                # min_samples_split
                self.GBgridlineSamplesSplit.setText(list2str(param_dict['min_samples_split']))

                # min_samples_leaf
                self.GBgridlineSamplesLeaf.setText(list2str(param_dict['min_samples_leaf']))

                # min_weight_fraction_leaf
                self.GBgridlineWeightLeaf.setText(list2str(param_dict['min_weight_fraction_leaf']))

                # max_depth
                self.GBgridlineMaxDepth.setText(list2str(param_dict['max_depth']))

                # random_state
                self.GBgridlineRandomState.setText(list2str(param_dict['random_state']))

                # max_features
                max_features = param_dict['max_features']
                for method in max_features:
                    item = self.GBgridlistMaxFeatures.findItems(method, Qt.MatchExactly)
                    if len(item) > 0:  # If method is present
                        row = self.GBgridlistMaxFeatures.row(item[0])
                        self.GBgridlistMaxFeatures.item(row).setSelected(True)

            elif algorithm == 'MLPClassifier':
                self.algo = 'ann'
                self.comboAlgorithms.setCurrentIndex(2)

                # Get algo parameter dict
                param_dict = config_dict['param_grid']

                # hidden_layer_sizes
                self.NNgridlineHiddenLayers.setText(list2str(param_dict['hidden_layer_sizes'], join_c=''))

                # activation
                activations = param_dict['activation']
                for function in activations:
                    item = self.NNgridlistActivation.findItems(function, Qt.MatchExactly)
                    if len(item) > 0:  # If function is present
                        row = self.NNgridlistActivation.row(item[0])
                        self.NNgridlistActivation.item(row).setSelected(True)

                # solver
                solvers = param_dict['solver']
                for solver in solvers:
                    item = self.NNgridlistSolver.findItems(solver, Qt.MatchExactly)
                    if len(item) > 0:  # If solver is present
                        row = self.NNgridlistSolver.row(item[0])
                        self.NNgridlistSolver.item(row).setSelected(True)

                # alpha
                self.NNgridlineAlpha.setText(list2str(param_dict['alpha']))

                # batch_size
                if self.NNgridlineBatchSize.isEnabled():
                    self.NNgridlineBatchSize.setText(list2str(param_dict['batch_size']))

                # learning_rate
                if self.NNgridlistLearningRate.isEnabled():
                    learning_rates = param_dict['learning_rate']
                    for method in learning_rates:
                        item = self.NNgridlistLearningRate.findItems(method, Qt.MatchExactly)
                        if len(item) > 0:  # If method is present
                            row = self.NNgridlistLearningRate.row(item[0])
                            self.NNgridlistLearningRate.item(row).setSelected(True)

                # learning_rate_init
                if self.NNgridlineLearningRateInit.isEnabled():
                    self.NNgridlineLearningRateInit.setText(list2str(param_dict['learning_rate_init']))

                # power_t
                if self.NNgridlinePowerT.isEnabled():
                    self.NNgridlinePowerT.setText(list2str(param_dict['power_t']))

                # max_iter
                self.NNgridlineMaxIter.setText(list2str(param_dict['max_iter']))

                # shuffle
                if self.NNgridlistShuffle.isEnabled():
                    true_shuffle_list = param_dict['shuffle']
                    shuffle_list = list()  # Replace True by 'shuffle' and False by 'no shuffle'
                    for item in true_shuffle_list:
                        if item:
                            shuffle_list.append('shuffle')
                        if item is False:
                            shuffle_list.append('no shuffle')
                    if shuffle_list:
                        for shuffle in shuffle_list:
                            item = self.NNgridlistShuffle.findItems(shuffle, Qt.MatchExactly)
                            if len(item) > 0:  # If shuffle is present
                                row = self.NNgridlistShuffle.row(item[0])
                                self.NNgridlistShuffle.item(row).setSelected(True)

                # random_state
                self.NNgridlineRandomState.setText(list2str(param_dict['random_state']))

                # beta_1, beta_2 and epsilon
                if self.NNgridlineBeta_1.isEnabled():
                    self.NNgridlineBeta_1.setText(list2str(param_dict['beta_1']))

                if self.NNgridlineBeta_2.isEnabled():
                    self.NNgridlineBeta_2.setText(list2str(param_dict['beta_2']))

                if self.NNgridlineEpsilon.isEnabled():
                    self.NNgridlineEpsilon.setText(list2str(param_dict['epsilon']))

            else:
                self.statusBar.showMessage('Error: Unknown selected algorithm!',
                                           3000)

        else:
            self.checkGridSearchCV.setChecked(False)
            # Update algorithm parameters
            if algorithm == 'RandomForestClassifier':
                self.algo = 'rf'
                self.comboAlgorithms.setCurrentIndex(0)
                self.RFcheckImportance.setChecked(config_dict['png_features'])

                # Get algo parameter dict
                param_dict = config_dict['parameters']

                # n_estimators
                self.RFspinEstimators.setValue(param_dict['n_estimators'])
                # criterion
                self.RFcomboCriterion.setCurrentText(param_dict['criterion'])
                # max_depth
                if param_dict['max_depth'] is None:
                    self.RFspinMaxDepth.setValue(0)
                else:
                    self.RFspinMaxDepth.setValue(param_dict['max_depth'])
                # min_samples_split
                self.RFspinSamplesSplit.setValue(param_dict['min_samples_split'])
                # min_samples_leaf
                self.RFspinSamplesLeaf.setValue(param_dict['min_samples_leaf'])
                # min_weight_fraction_leaf
                self.RFspinWeightLeaf.setValue(param_dict['min_weight_fraction_leaf'])
                # max_features
                self.RFcomboMaxFeatures.setCurrentText(param_dict['max_features'])
                # n_jobs
                if param_dict['n_jobs'] is None:
                    self.RFspinNJob.setValue(0)
                else:
                    self.RFspinNJob.setValue(param_dict['n_jobs'])
                # random_state
                if param_dict['random_state'] is None:
                    self.RFspinRandomState.setValue(-1)
                else:
                    self.RFspinRandomState.setValue(param_dict['random_state'])

            elif algorithm == 'GradientBoostingClassifier':
                self.algo = 'gb'
                self.comboAlgorithms.setCurrentIndex(1)
                self.RFcheckImportance.setChecked(config_dict['png_features'])

                # Get algo parameter dict
                param_dict = config_dict['parameters']

                # loss
                self.GBcomboLoss.setCurrentText(param_dict['loss'])
                # learning_rate
                self.GBspinLearningRate.setValue(param_dict['learning_rate'])
                # n_estimators
                self.GBspinEstimators.setValue(param_dict['n_estimators'])
                # subsample
                self.GBspinSubsample.setValue(param_dict['subsample'])
                # criterion
                self.GBcomboCriterion.setCurrentText(param_dict['criterion'])
                # min_samples_split
                self.GBspinSamplesSplit.setValue(param_dict['min_samples_split'])
                # min_samples_leaf
                self.GBspinSamplesLeaf.setValue(param_dict['min_samples_leaf'])
                # min_weight_fraction_leaf
                self.GBspinWeightLeaf.setValue(param_dict['min_weight_fraction_leaf'])
                # max_depth
                if param_dict['max_depth'] is None:
                    self.GBspinMaxDepth.setValue(0)
                else:
                    self.GBspinMaxDepth.setValue(param_dict['max_depth'])
                # random_state
                if param_dict['random_state'] is None:
                    self.GBspinRandomState.setValue(-1)
                else:
                    self.GBspinRandomState.setValue(param_dict['random_state'])
                # max_features
                if param_dict['max_features'] is None:
                    self.GBcomboMaxFeatures.setCurrentText('None')
                else:
                    self.GBcomboMaxFeatures.setCurrentText(param_dict['max_features'])

            elif algorithm == 'MLPClassifier':
                self.algo = 'ann'
                self.comboAlgorithms.setCurrentIndex(2)

                # Get algo parameter dict
                param_dict = config_dict['parameters']

                # hidden_layer_sizes
                hidden_layers = [str(layer) for layer in param_dict['hidden_layer_sizes']]
                self.NNlineHiddenLayers.setText(','.join(hidden_layers))
                # activation
                self.NNcomboActivation.setCurrentText(param_dict['activation'])
                # solver
                self.NNcomboSolver.setCurrentText(param_dict['solver'])
                self.nn_solver_options()
                # alpha
                self.NNspinAlpha.setValue(param_dict['alpha'])
                # batch_size
                if self.NNspinBatchSize.isEnabled():
                    if param_dict['batch_size'] == 'auto':
                        self.NNspinBatchSize.setValue(-1)
                    else:
                        self.NNspinBatchSize.setValue(param_dict['batch_size'])
                # learning_rate
                if self.NNcomboLearningRate.isEnabled():
                    self.NNcomboLearningRate.setCurrentText(param_dict['learning_rate'])
                # learning_rate_init
                if self.NNspinLearningRateInit.isEnabled():
                    self.NNspinLearningRateInit.setValue(param_dict['learning_rate_init'])
                # power_t
                if self.NNspinPowerT.isEnabled():
                    self.NNspinPowerT.setValue(param_dict['power_t'])
                # max_iter
                self.NNspinMaxIter.setValue(param_dict['max_iter'])
                # shuffle
                if self.NNcheckShuffle.isEnabled():
                    self.NNcheckShuffle.setChecked(param_dict['shuffle'])
                # random_state
                if param_dict['random_state'] is None:
                    self.NNspinRandomState.setValue(-1)
                else:
                    self.NNspinRandomState.setValue(param_dict['random_state'])
                # beta_1, beta_2 and epsilon
                if self.NNspinBeta_1.isEnabled():
                    self.NNspinBeta_1.setValue(param_dict['beta_1'])
                if self.NNspinBeta_2.isEnabled():
                    self.NNspinBeta_2.setValue(param_dict['beta_2'])
                if self.NNspinEpsilon.isEnabled():
                    self.NNspinEpsilon.setValue(param_dict['epsilon'])

            else:
                self.statusBar.showMessage('Error: Unknown algorithm from config file!',
                                           3000)

        # Set the selected features
        feature_names = config_dict['feature_names']
        self.set_selected_features(feature_names)

        # Update the train_config dict of the application
        self.update_train_config()

        # Display train_tab
        self.tabModes.setCurrentIndex(0)

    def open_predict_config(self, config_dict):
        """
        Open the configuration file for prediction.
        :param config_dict: The dictionnary from configuration file.
        """
        # Get model
        if config_dict['local_compute']:
            self.lineModelFile.setText(config_dict['model'])
        else:
            self.lineModelFile.setText(config_dict['local_model'])
            self.lineServerModel.setText(config_dict['model'])

        # Update the predict_config of the application
        self.update_predict_config()

        # Display predict_tab
        self.tabModes.setCurrentIndex(1)

    def open_segment_config(self, config_dict):
        """
        Open the configuration file for segmentation.
        :param config_dict: The dictionnary from configuration file.
        """
        # Segementation parameters
        param_dict = config_dict['parameters']

        # random_state
        self.KMspinRandomState.setValue(param_dict['random_state'])

        # n_clusters
        self.KMspinNClusters.setValue(param_dict['n_clusters'])

        # init
        self.KMcomboInit.setCurrentText(param_dict['init'])

        # n_init
        self.KMspinNInit.setValue(param_dict['n_init'])

        # max_iter
        self.KMspinMaxIter.setValue(param_dict['max_iter'])

        # tol
        self.KMspinTol.setValue(param_dict['tol'])

        # algorithm
        self.KMcomboAlgorithm.setCurrentText(param_dict['algorithm'])

        # Set the selected features
        feature_names = config_dict['feature_names']
        self.set_selected_features(feature_names)

        # Update the semgent_config of the application
        self.update_segment_config()

        # Display segment_tab
        self.tabModes.setCurrentIndex(2)

    def get_selected_features(self):
        """
        Get the selected features and return them in list.
        :return: List of all selected features.
        """
        # Initialize the list of selected features
        selected_features = list()

        # Get the coordinates features
        if self.checkAdvancedFeat.isChecked():  # Add X, Y, Z
            if self.checkX.isEnabled() and self.checkX.isChecked():
                selected_features.append('X')
            if self.checkY.isEnabled() and self.checkY.isChecked():
                selected_features.append('Y')
            if self.checkZ.isEnabled() and self.checkZ.isChecked():
                selected_features.append('Z')

            # Get the features from standard LAS features
            for feature in self.listStandardLAS.selectedItems():  # Add Standard LAS features
                selected_features.append(feature.text())

        # Get features from Extra features
        for feature in self.listExtraFeatures.selectedItems():  # Add Extra features
            selected_features.append(feature.text())

        # Sort the selected features
        selected_features.sort()

        return selected_features

    def update_config(self):
        """
        Update the configuration of training, predictions or semgentation according the current tab.
        """
        if self.tabModes.currentIndex() == 0:
            self.update_train_config()
        elif self.tabModes.currentIndex() == 1:
            self.update_predict_config()
        elif self.tabModes.currentIndex() == 2:
            self.update_segment_config()
        else:
            raise IndexError("Returning selected tab does not exist!")

    def update_train_config(self):
        """
        Update the dictionary of the training configuration.
        """
        # Erase previous config
        self.train_config = dict()

        # Version and mode
        self.train_config['version'] = self.cLASpy_train_version

        # Save input file, output folder
        if self.pushLocal.isChecked():
            self.train_config['local_compute'] = True
            self.train_config['input_file'] = self.lineLocalFile.text()
            self.train_config['output_folder'] = self.lineLocalFolder.text()
        else:
            self.train_config['local_compute'] = False
            self.train_config['local_input'] = self.lineFile.text()
            self.train_config['input_file'] = self.lineServerFile.text()
            self.train_config['output_folder'] = self.lineServerFolder.text()

        # Save sample size and train ratio
        self.train_config['samples'] = self.spinSampleSize.value()
        self.train_config['training_ratio'] = self.spinTrainRatio.value()

        # Save scaler, scorer, n_job CV, random_state and pca
        self.train_config['scaler'] = self.comboScaler.currentText()
        self.train_config['scorer'] = self.comboScorer.currentText()
        self.train_config['n_jobs_cv'] = self.spinNJobCV.value()
        self.train_config['random_state'] = self.spinRandomState.value()
        if self.spinPCA.value() != 0:
            self.train_config['pca'] = self.spinPCA.value()

        # Get the selected algorithm and the parameters
        param_dict = dict()
        selected_algo = self.comboAlgorithms.currentText()

        # With GridSearchCV
        if self.checkGridSearchCV.isChecked():
            self.train_config['grid_search'] = True
            # if Random Forest (GridSearchCV)
            if selected_algo == "Random Forest":
                self.algo = 'rf'
                self.train_config['algorithm'] = 'RandomForestClassifier'
                self.train_config['png_features'] = False

                # n_estimators
                n_estimators_list = format_numberlist(self.RFgridlineEstimators.text(), as_type='list')
                if n_estimators_list:
                    param_dict['n_estimators'] = n_estimators_list

                # criterion
                criterion_list = [item.text() for item in self.RFgridlistCriterion.selectedItems()]
                if criterion_list:
                    param_dict['criterion'] = criterion_list

                # max_depth
                max_depth_list = format_numberlist(self.RFgridlineMaxDepth.text(), as_type='list')
                # Get the occurrences of '0' and remove it
                zeros = [value for value in max_depth_list if value == 0]
                [max_depth_list.remove(zero) for zero in zeros]
                if max_depth_list:
                    param_dict['max_depth'] = max_depth_list

                # min_samples_split
                samples_split_list = format_numberlist(self.RFgridlineSamplesSplit.text(), as_type='list')
                # Get values < 2
                inferior_2 = [value for value in samples_split_list if value < 2]
                [samples_split_list.remove(value) for value in inferior_2]
                if samples_split_list:
                    param_dict['min_samples_split'] = samples_split_list

                # min_samples_leaf
                samples_leaf_list = format_numberlist(self.RFgridlineSamplesLeaf.text(), as_type='list')
                # Get values < 1
                inferior_1 = [value for value in samples_leaf_list if value < 1]
                [samples_leaf_list.remove(value) for value in inferior_1]
                if samples_leaf_list:
                    param_dict['min_samples_leaf'] = samples_leaf_list

                # min_weight_fraction_leaf
                weight_leaf_list = format_floatlist(self.RFgridlineWeightLeaf.text(), as_type='list')
                if weight_leaf_list:
                    param_dict['min_weight_fraction_leaf'] = weight_leaf_list

                # max_features
                max_features_list = [item.text() for item in self.RFgridlistMaxFeatures.selectedItems()]
                if max_features_list:
                    param_dict['max_features'] = max_features_list

                # n_jobs
                n_jobs_list = format_numberlist(self.RFgridlineNJob.text(), as_type='list')
                n_jobs_list = [-1 if value == 0 else value for value in n_jobs_list]  # replace '0' by '-1'
                if n_jobs_list:
                    param_dict['n_jobs'] = n_jobs_list

                # random_state
                random_state_list = format_numberlist(self.RFgridlineRandomState.text(), as_type='list')
                # Get the occurrences of number < '0' and remove it
                zeros_random = [value for value in random_state_list if value < 0]
                [random_state_list.remove(value) for value in zeros_random]
                if random_state_list:
                    param_dict['random_state'] = random_state_list

            # if Gradient Boosting with GridSearchCV
            elif selected_algo == "Gradient Boosting":
                self.algo = 'gb'
                self.train_config['algorithm'] = 'GradientBoostingClassifier'
                self.train_config['png_features'] = False

                # loss
                loss_list = [item.text() for item in self.GBgridlistLoss.selectedItems()]
                if loss_list:
                    param_dict['loss'] = loss_list

                # learning_rate
                learning_rate_list = format_floatlist(self.GBgridlineLearningRate.text(), as_type='list')
                if learning_rate_list:
                    param_dict['learning_rate'] = learning_rate_list

                # n_estimators
                n_estimators_list = format_numberlist(self.GBgridlineEstimators.text(), as_type='list')
                if n_estimators_list:
                    param_dict['n_estimators'] = n_estimators_list

                # subsample
                subsample_list = format_floatlist(self.GBgridlineSubsample.text(), as_type='list')
                if subsample_list:
                    param_dict['subsample'] = subsample_list

                # criterion
                criterion_list = [item.text() for item in self.GBgridlistCriterion.selectedItems()]
                if criterion_list:
                    param_dict['criterion'] = criterion_list

                # min_samples_split
                min_samples_split_list = format_numberlist(self.GBgridlineSamplesSplit.text(), as_type='list')
                if min_samples_split_list:
                    param_dict['min_samples_split'] = min_samples_split_list

                # min_samples_leaf
                min_samples_leaf_list = format_numberlist(self.GBgridlineSamplesLeaf.text(), as_type='list')
                if min_samples_leaf_list:
                    param_dict['min_samples_leaf'] = min_samples_leaf_list

                # min_weight_fraction_leaf
                min_weight_leaf_list = format_floatlist(self.GBgridlineWeightLeaf.text(), as_type='list')
                if min_weight_leaf_list:
                    param_dict['min_weight_fraction_leaf'] = min_weight_leaf_list

                # max_depth
                max_depth_list = format_numberlist(self.GBgridlineMaxDepth.text(), as_type='list')
                # Get the occurrences of '0' and remove it
                zeros = [value for value in max_depth_list if value == 0]
                [max_depth_list.remove(zero) for zero in zeros]
                if max_depth_list:
                    param_dict['max_depth'] = max_depth_list

                # random_state
                random_state_list = format_numberlist(self.GBgridlineRandomState.text(), as_type='list')
                # Get the occurrences of number < '0' and remove it
                zeros_random = [value for value in random_state_list if value < 0]
                [random_state_list.remove(value) for value in zeros_random]
                if random_state_list:
                    param_dict['random_state'] = random_state_list

                # max_features
                max_features_list = [item.text() for item in self.GBgridlistMaxFeatures.selectedItems()]
                if max_features_list:
                    param_dict['max_features'] = max_features_list

            # if Neural Network with GridSearchCV
            elif selected_algo == "Neural Network":
                self.algo = 'ann'
                self.train_config['algorithm'] = 'MLPClassifier'
                self.train_config['png_features'] = False

                # hidden_layer_sizes
                hidden_layers_list = format_layerlist(self.NNgridlineHiddenLayers.text(), as_type='list')
                if hidden_layers_list:
                    param_dict['hidden_layer_sizes'] = hidden_layers_list
                else:
                    param_dict['hidden_layer_sizes'] = []

                # activation
                activation_list = [item.text() for item in self.NNgridlistActivation.selectedItems()]
                if activation_list:
                    param_dict['activation'] = activation_list

                # solver
                solver_list = [item.text() for item in self.NNgridlistSolver.selectedItems()]
                if solver_list:
                    param_dict['solver'] = solver_list

                # alpha
                alpha_list = format_floatlist(self.NNgridlineAlpha.text(), as_type='list')
                if alpha_list:
                    param_dict['alpha'] = alpha_list

                # batch_size
                if self.NNgridlineBatchSize.isEnabled():
                    batch_size_list = format_numberlist(self.NNgridlineBatchSize.text(), as_type='list')
                    if batch_size_list:
                        param_dict['batch_size'] = batch_size_list

                # learning_rate
                if self.NNgridlistLearningRate.isEnabled():
                    learning_rate_list = [item.text() for item in self.NNgridlistLearningRate.selectedItems()]
                    if learning_rate_list:
                        param_dict['learning_rate'] = learning_rate_list

                # learning_rate_init
                if self.NNgridlineLearningRateInit.isEnabled():
                    learning_init_list = format_floatlist(self.NNgridlineLearningRateInit.text(), as_type='list')
                    if learning_init_list:
                        param_dict['learning_rate_init'] = learning_init_list

                # power_t
                if self.NNgridlinePowerT.isEnabled():
                    power_t_list = format_floatlist(self.NNgridlinePowerT.text(), as_type='list')
                    if power_t_list:
                        param_dict['power_t'] = power_t_list

                # max_iter
                max_iter_list = format_numberlist(self.NNgridlineMaxIter.text(), as_type='list')
                if max_iter_list:
                    param_dict['max_iter'] = max_iter_list

                # shuffle
                if self.NNgridlistShuffle.isEnabled():
                    shuffle_list = [item.text() for item in self.NNgridlistShuffle.selectedItems()]
                    true_shuffle_list = list()  # Only True or/and False accepted
                    for item in shuffle_list:
                        if item == 'shuffle':
                            true_shuffle_list.append(True)
                        if item == 'no shuffle':
                            true_shuffle_list.append(False)
                    if true_shuffle_list:
                        param_dict['shuffle'] = true_shuffle_list

                # random_state
                random_state_list = format_numberlist(self.NNgridlineRandomState.text(), as_type='list')
                # Get the occurrences of number < '0' and remove it
                zeros_random = [value for value in random_state_list if value < 0]
                [random_state_list.remove(value) for value in zeros_random]
                if random_state_list:
                    param_dict['random_state'] = random_state_list

                # beta_1, beta_2 and epsilon
                if self.NNgridlineBeta_1.isEnabled():
                    beta_1_list = format_floatlist(self.NNgridlineBeta_1.text(), as_type='list')
                    if beta_1_list:
                        param_dict['beta_1'] = beta_1_list

                if self.NNgridlineBeta_2.isEnabled():
                    beta_2_list = format_floatlist(self.NNgridlineBeta_2.text(), as_type='list')
                    if beta_2_list:
                        param_dict['beta_2'] = beta_2_list

                if self.NNgridlineEpsilon.isEnabled():
                    epsilon_list = format_floatlist(self.NNgridlineEpsilon.text(), as_type='list')
                    if epsilon_list:
                        param_dict['epsilon'] = epsilon_list

            # if anything else
            else:
                self.statusBar.showMessage('Error: Unknown selected algorithm!',
                                           3000)

        # Without GridSearchCV
        else:
            self.train_config['grid_search'] = False
            # if Random Forest
            if selected_algo == "Random Forest":
                self.algo = 'rf'
                self.train_config['algorithm'] = 'RandomForestClassifier'
                self.train_config['png_features'] = self.RFcheckImportance.isChecked()

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
                self.algo = 'gb'
                self.train_config['algorithm'] = 'GradientBoostingClassifier'
                self.train_config['png_features'] = self.GBcheckImportance.isChecked()

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
                self.algo = 'ann'
                self.train_config['algorithm'] = 'MLPClassifier'
                self.train_config['png_features'] = False

                # hidden_layer_sizes
                hidden_layers = self.NNlineHiddenLayers.text().replace(' ', '')
                if hidden_layers == '':
                    param_dict['hidden_layer_sizes'] = [25, 50, 25]
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

            # if anything else
            else:
                self.statusBar.showMessage('Error: Unknown selected algorithm!',
                                           3000)

        # Save classifier parameters with GridSearchCV or not
        if self.train_config['grid_search']:
            self.train_config['param_grid'] = param_dict
            try:
                self.train_config.pop('parameters')  # Remove 'parameters'
            except KeyError:
                pass
        else:
            self.train_config['parameters'] = param_dict
            try:
                self.train_config.pop('param_grid')  # Remove 'param_grid'
            except KeyError:
                pass

        # Get the current selected features
        self.selectedFeatures = self.get_selected_features()
        self.train_config['feature_names'] = self.selectedFeatures

    def update_predict_config(self):
        """
        Update the dictionary of the prediction configuration
        """
        # Erase previous config
        self.predict_config = dict()

        # Version and mode
        self.predict_config['version'] = self.cLASpy_predi_version

        # Save input file, output folder
        if self.pushLocal.isChecked():
            self.predict_config['local_compute'] = True
            self.predict_config['input_file'] = self.lineLocalFile.text()
            self.predict_config['output_folder'] = self.lineLocalFolder.text()
            self.predict_config['model'] = self.lineModelFile.text()
        else:
            self.predict_config['local_compute'] = False
            self.predict_config['local_input'] = self.lineFile.text()
            self.predict_config['input_file'] = self.lineServerFile.text()
            self.predict_config['output_folder'] = self.lineServerFolder.text()
            self.predict_config['local_model'] = self.lineModelFile.text()
            self.predict_config['model'] = self.lineServerModel.text()

    def update_segment_config(self):
        """
        Update the dictionary of the segmentation configuration
        """
        # Erase previous config
        self.segment_config = dict()

        # Version and mode
        self.segment_config['version'] = self.cLASpy_segme_version

        # Save input file, output folder
        if self.pushLocal.isChecked():
            self.segment_config['local_compute'] = True
            self.segment_config['input_file'] = self.lineLocalFile.text()
            self.segment_config['output_folder'] = self.lineLocalFolder.text()
        else:
            self.segment_config['local_compute'] = False
            self.segment_config['local_input'] = self.lineFile.text()
            self.segment_config['input_file'] = self.lineServerFile.text()
            self.segment_config['output_folder'] = self.lineServerFolder.text()

        # Save scaler and PCA
        # self.segment_config['scaler'] = self.SegcomboScaler.currentText()
        # if self.SegspinPCA.value() != 0:
        #     self.segment_config['pca'] = self.SegspinPCA.value()

        # Save algorithm parameters
        param_dict = dict()

        # random_state
        param_dict['random_state'] = self.KMspinRandomState.value()

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

        # algorithm
        param_dict['algorithm'] = self.KMcomboAlgorithm.currentText()

        # Save all parameters
        self.segment_config['parameters'] = param_dict

        # Get the selected features
        self.SegSelectedFeatures = self.get_selected_features()
        self.segment_config['feature_names'] = self.SegSelectedFeatures

    def save_config(self):
        """
        Save the configuration of training, predictions or semgentation according the current tab.
        """
        if self.tabModes.currentIndex() == 0:
            self.save_train_config()
        elif self.tabModes.currentIndex() == 1:
            self.save_predict_config()
        elif self.tabModes.currentIndex() == 2:
            self.save_segment_config()
        else:
            raise IndexError("Returning selected tab does not exist!")

    def save_train_config(self):
        """
        Save the training configuration into a JSON file
        """
        self.update_config()

        # Check if features are selected
        if self.sel_feat_count > 0:
            # Save the JSON file
            self.statusBar.showMessage("Saving JSON config file...", 3000)
            json_file = QFileDialog.getSaveFileName(None, 'Save JSON config file',
                                                    '', "JSON files (*.json)")

            if json_file[0] != '':
                with open(json_file[0], 'w') as config_file:
                    json.dump(self.train_config, config_file, indent=4)
                    self.statusBar.showMessage("Config file for training saved: {}".format(json_file[0]),
                                               5000)
        else:
            warning_box("No feature field selected !\nPlease, select the features you need !",
                        "Missing features")

    def save_predict_config(self):
        """
        Save the prediction configuration into a JSON file.
        """
        self.update_config()

        # Save the JSON file
        self.statusBar.showMessage("Saving JSON config file...", 3000)
        json_file = QFileDialog.getSaveFileName(None, 'Save JSON config file',
                                                '', "JSON files (*.json)")

        if json_file[0] != '':
            with open(json_file[0], 'w') as config_file:
                json.dump(self.predict_config, config_file, indent=4)
                self.statusBar.showMessage("Config file for prediction saved: {}".format(json_file[0]),
                                           5000)

    def save_segment_config(self):
        """
        Save the segmentation configuration into a JSON file
        """
        self.update_config()

        # Check if features are selected
        if self.sel_feat_count > 0:
            # Save the JSON file
            self.statusBar.showMessage("Saving JSON config file...", 3000)
            json_file = QFileDialog.getSaveFileName(None, 'Save JSON config file',
                                                    '', "JSON files (*.json)")

            if json_file[0] != '':
                with open(json_file[0], 'w') as config_file:
                    json.dump(self.segment_config, config_file, indent=4)
                    self.statusBar.showMessage("Config file saved: {}".format(json_file[0]),
                                               5000)
        else:
            warning_box("No feature field selected !\nPlease, select the features you need !",
                        "No selected features")

    # Command and Run
    def command_part(self):
        """
        Give the command part of the GUI.
        """
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

    def print_output(self, s):
        self.message(s)

    def update_progress(self, n):
        """Update progressBar according given percent as integer"""
        self.progressBar.setValue(n)

    def run_train(self):
        # Update training configuration
        self.update_config()

        # Check if some features are selected
        if self.sel_feat_count <= 0:
            warning_box("No feature field selected!\nPlease select the features you need!",
                        "No features selected")
        else:
            # Pass the function to execute
            worker = Worker(self.train)
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.update_progress)

            # Execute
            self.threadpool.start(worker)

    def train(self, progress_callback):
        # Remove space in features
        features = str(self.train_config['feature_names']).replace(' ', '')

        self.message("\n# # # # # # # # # #  cLASpy_T  # # # # # # # # # # # #"
                     "\n - - - - - - - - - - - -    TRAIN MODE   - - - - - - - - - - - - - -"
                     "\n * * * * * * * *    Point Cloud Classification   * * * * * * * * *\n")

        # Train with GridSearchCV or not
        if self.train_config['grid_search']:
            # Setting up grid_parameters
            grid_parameters = str(self.train_config['param_grid'])

            # Set the classifier
            trainer = ClaspyTrainer(input_data=self.lineLocalFile.text(),
                                    output_data=self.lineLocalFolder.text(),
                                    algo=self.algo,
                                    algorithm=None,
                                    features=features,
                                    grid_search=True,
                                    grid_param=grid_parameters,
                                    pca=self.spinPCA.value(),
                                    n_jobs=self.spinNJobCV.value(),
                                    random_state=self.train_config['random_state'],
                                    samples=self.train_config['samples'],
                                    scaler=self.comboScaler.currentText(),
                                    scoring=self.comboScorer.currentText(),
                                    train_ratio=self.train_config['training_ratio'])

        else:
            # Setting up parameters, remove parameters set as None
            parameters_dict = self.train_config['parameters']
            keys_to_remove = list()  # Create list of keys to remove, because dictionary must keep the same length
            for key in parameters_dict:
                if parameters_dict[key] is None:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                parameters_dict.pop(key)
            parameters = str(parameters_dict).replace(' ', '')

            # Set the classifier
            trainer = ClaspyTrainer(input_data=self.lineLocalFile.text(),
                                    output_data=self.lineLocalFolder.text(),
                                    algo=self.algo,
                                    algorithm=None,
                                    parameters=parameters,
                                    features=features,
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

        # Kill the remaining python interpreter (1+18)
        return "Training done!"

    def run_predict(self):
        self.check_model_features()  # Check if model and input file features match
        if self.predict_features:
            self.update_config()

            # Create command list to run cLASpy_T
            command = ["cLASpy_T.py", "predict",
                       "-i", self.lineLocalFile.text(),
                       "-o", self.lineLocalFolder.text(),
                       "-m", self.lineModelFile.text()]

            if self.pythonPath != '':
                if self.process is None:
                    self.process = QProcess()
                    self.process.readyReadStandardOutput.connect(self.handle_stdout)
                    self.process.readyReadStandardError.connect(self.handle_stderr)
                    self.process.stateChanged.connect(self.handle_state)
                    self.process.finished.connect(self.thread_complete)
                    self.process.setProgram(self.pythonPath)
                    self.process.setArguments(command)
                    self.process.start()
                    self.processPID = self.process.processId()
                    self.buttonStop.setEnabled(True)
                    self.buttonRunPredict.setEnabled(False)
            else:
                self.plainTextCommand.appendPlainText("Set python path through Edit > Options")

    def run_segment(self):
        self.update_config()

        # Check pythonPath is set
        if self.pythonPath != '':

            # Check if some features are selected
            if self.sel_feat_count > 0:
                features = str(self.segment_config['feature_names']).replace(' ', '')

                # Setting up parameters, remove parameters set as None
                parameters_dict = self.segment_config['parameters']
                # Create list of keys to remove, because dictionary must keep the same length
                keys_to_remove = list()
                for key in parameters_dict:
                    if parameters_dict[key] is None:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    parameters_dict.pop(key)
                parameters = str(parameters_dict).replace(' ', '')

                # Create command list to run cLASpy_T
                command = ["cLASpy_T.py", "segment",
                           "-i", self.lineLocalFile.text(),
                           "-o", self.lineLocalFolder.text(),
                           "-f", features,
                           "-p", parameters]

                # Run semgentation process
                if self.process is None:
                    self.process = QProcess()
                    self.process.readyReadStandardOutput.connect(self.handle_stdout)
                    self.process.readyReadStandardError.connect(self.handle_stderr)
                    self.process.stateChanged.connect(self.handle_state)
                    self.process.finished.connect(self.thread_complete)
                    self.process.setProgram(self.pythonPath)
                    self.process.setArguments(command)
                    self.process.start()
                    self.processPID = self.process.processId()
                    self.buttonStop.setEnabled(True)
                    self.buttonRunSegment.setEnabled(False)

            else:
                warning_box("No feature field selected!\nPlease select the features you need!",
                            "No features selected")
        else:
            warning_box("Set python path through Edit > Options", "Python path not set")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf8')
        self.plainTextCommand.appendPlainText(stdout)
        progress = percent_parser(stdout)
        if progress:
            self.progressBar.setValue(progress)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode('utf8')
        self.plainTextCommand.appendPlainText(stderr)

    def handle_state(self, state):
        states = {QProcess.NotRunning: "Not Running",
                  QProcess.Starting: "Starting",
                  QProcess.Running: "Running"}

        state_name = states[state]
        self.statusBar.showMessage("cLASpy_T is {}".format(state_name), 3000)

    def thread_complete(self):
        self.statusBar.showMessage("cLASpy_T finished !", 5000)
        self.threadpool.releaseThread()
        self.process = None
        self.progressBar.reset()
        self.enable_open_results()
        self.buttonStop.setEnabled(False)
        self.buttonRunTrain.setEnabled(True)
        self.buttonRunPredict.setEnabled(True)
        self.buttonRunSegment.setEnabled(True)

    def stop_process(self):
        parent_process = psutil.Process(self.processPID)
        try:
            for child in parent_process.children(recursive=True):
                child.kill()
            parent_process.kill()
        except:
            self.statusBar.showMessage("ERROR: Process not killed!", 3000)
        else:
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

if __name__ == '__main__':
    # Set the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Execute the Main window
    ex = ClaspyGui()
    # ex.showMaximized()
    ex.show()

    sys.exit(app.exec_())
