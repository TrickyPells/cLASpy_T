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
import os
import json
import pylas

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from laspy import file

from common import *

# -------------------------
# -------- CLASS ----------
# -------------------------


class ClaspyGui(QMainWindow):
    def __init__(self, parent=None):
        super(ClaspyGui, self).__init__(parent)
        self.setWindowTitle("cLASpy_GUI")
        self.setGeometry(400, 300, 400, 500)
        self.mainWidget = QWidget()

        self.labelFile = QLabel("Input file:")
        self.lineFile = QLineEdit()
        self.lineFile.setPlaceholderText("Select LAS or CSV file as input")
        self.toolButtonFile = QToolButton()
        self.toolButtonFile.setText("Browse")

        self.hLayoutFile = QHBoxLayout()
        self.hLayoutFile.addWidget(self.lineFile)
        self.hLayoutFile.addWidget(self.toolButtonFile)

        self.labelHLine_1 = QLabel()
        self.labelHLine_1.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        self.listAlgorithms = QListWidget()
        self.listAlgorithms.setMaximumSize(120, 80)
        self.listAlgorithms.insertItem(0, "Random Forest")
        self.listAlgorithms.insertItem(1, "Gradient Boosting")
        self.listAlgorithms.insertItem(2, "Neural Network")
        self.listAlgorithms.insertItem(3, "K-Means Clustering")

        self.stack_0 = QWidget()
        self.stack_1 = QWidget()
        self.stack_2 = QWidget()
        self.stack_3 = QWidget()
        self.stackui_0()
        self.stackui_1()
        self.stackui_2()
        self.stackui_3()
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.stack_0)
        self.stack.addWidget(self.stack_1)
        self.stack.addWidget(self.stack_2)
        self.stack.addWidget(self.stack_3)

        self.labelHLine_2 = QLabel()
        self.labelHLine_2.setFrameStyle(QFrame.HLine | QFrame.Sunken)

        self.checkTarget = QCheckBox("Target field")
        self.labelFields = QLabel("Select scalar fields:\n\n\n"
                                  "(press Ctrl+Shift\n"
                                  "for multiple selection)")
        self.labelFields.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.listFields = QListWidget()
        self.listFields.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listFields.setSortingEnabled(True)

        self.vLayoutFields = QVBoxLayout()
        self.vLayoutFields.addWidget(self.checkTarget)
        self.vLayoutFields.addWidget(self.listFields)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        self.buttonBox.accepted.connect(self.save_config)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonRun = QPushButton("Run cLASpy_T")
        self.buttonRun.clicked.connect(self.run_claspy_t)
        self.buttonBox.addButton(self.buttonRun, QDialogButtonBox.ActionRole)

        self.formLayout = QFormLayout(self.mainWidget)
        self.formLayout.addRow(self.labelFile, self.hLayoutFile)
        self.formLayout.addRow(self.labelHLine_1)
        self.formLayout.addRow(QLabel("Select algorithm:"), QLabel("Select algorithm parameters:"))
        self.formLayout.addRow(self.listAlgorithms, self.stack)
        self.formLayout.addRow(self.labelHLine_2)
        self.formLayout.addRow(self.labelFields, self.vLayoutFields)
        self.formLayout.addWidget(self.buttonBox)

        self.setCentralWidget(self.mainWidget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.toolButtonFile.clicked.connect(self.get_file)
        self.lineFile.textChanged.connect(self.open_file)
        self.listAlgorithms.currentRowChanged.connect(self.display_stack)

    def get_file(self):
        self.statusBar.showMessage("Select file...", 2000)
        filename = QFileDialog.getOpenFileName(self, 'Select CSV or LAS file',
                                               'D:/PostDoc_Temp/Article_Classif_Orne',
                                               "LAS files (*.las);;CSV files (*.csv)")

        self.lineFile.setText(filename[0])
        # self.statusBar.clearMessage()

    def open_file(self):
        file_path = os.path.normpath(self.lineFile.text())

        root_ext = os.path.splitext(file_path)
        if root_ext[1] == '.csv':
            field_names = ["Encore", "en", "Test"]
            self.target = False
        elif root_ext[1] == '.las':
            field_names = self.open_las(file_path)
        else:
            field_names = ["File error:", "Unknown", "extension file!"]
            self.statusBar.showMessage("File error: Unknown extension file!")

        # Check if the target field exist
        if self.target:
            self.checkTarget.setText("Target field is available")
            self.checkTarget.setEnabled(True)
            self.checkTarget.setChecked(True)
        else:
            self.checkTarget.setText("Target field is not available")
            self.checkTarget.setEnabled(False)

        # Rewrite listField
        self.listFields.clear()
        for item in field_names:
            self.listFields.addItem(str(item))

    def open_las(self, file_path):
        """
        Open LAS file and only return the extra dimension.
        :param file_path: The path to the LAS file.
        :return: The list of the extra dimensions (with 'Target' field).
        """
        las = file.File(file_path, mode='r')
        version = las.header.version
        data_format = las.header.data_format_id
        point_count = las.header.records_count
        point_count = '{:,}'.format(point_count).replace(',', ' ')  # Format with thousand separator

        # Show LAS version and number of points in status bar
        self.statusBar.showMessage("{} points | LAS Version: {}"
                                   .format(point_count, version),
                                   3000)

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

    def stackui_0(self):
        layout = QFormLayout()

        layout.addRow("Name", QLineEdit())
        layout.addRow("Address", QLineEdit())

        self.stack_0.setLayout(layout)

    def stackui_1(self):
        layout = QFormLayout()

        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))
        layout.addRow(QLabel("Sex"), sex)
        layout.addRow("Date of Birth", QLineEdit())

        self.stack_1.setLayout(layout)

    def stackui_2(self):
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Subjects"))
        layout.addWidget(QCheckBox("Physics"))
        layout.addWidget(QCheckBox("Maths"))

        self.stack_2.setLayout(layout)

    def stackui_3(self):
        layout = QFormLayout()

        sex = QHBoxLayout()
        sex.addWidget(QRadioButton("Male"))
        sex.addWidget(QRadioButton("Female"))

        layout_hbox = QHBoxLayout()
        layout_hbox.addWidget(QCheckBox("Physics"))
        layout_hbox.addWidget(QCheckBox("Maths"))

        layout.addRow(QLabel("Sex"), sex)
        layout.addRow("Date of Birth", QLineEdit())
        layout.addRow(QLabel("Subject"), layout_hbox)
        self.stack_3.setLayout(layout)

    def display_stack(self, i):
        self.stack.setCurrentIndex(i)
        algo_list = ["Random Forest", "Gradient Boosting", "Neural Network", "K-Means clustering"]
        self.statusBar.showMessage(algo_list[i] + " parameters", 2000)

    def save_config(self):
        """
        Save configuration as JSON file.
        """
        # Get the current selected fields
        self.selectedFields = [item.text() for item in self.listFields.selectedItems()]

        # If Target is checked, add to listField
        if self.checkTarget.isChecked():
            self.selectedFields.append(self.targetName)

        self.selectedFields.sort()
        print(self.selectedFields)

    def run_claspy_t(self):
        print("Run cLASpy_T")

    def reject(self):
        print("Close pressed !")
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClaspyGui()
    ex.show()
    sys.exit(app.exec_())
