import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Configuration(QWidget):
    def __init__(self, parent=None):
        super(Configuration, self).__init__(parent)
        self.setGeometry(400, 300, 500, 500)
        self.setWindowTitle("cLASpy_T - Configuration")

        self.labelFile = QLabel("Input file:")
        self.lineFile = QLineEdit()
        self.lineFile.setPlaceholderText("Select CSV or LAS file as input")
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

        self.labelFields = QLabel("Please select scalar fields:\n\n"
                                  "(press Ctrl+Shift\n"
                                  "for multiple selection)")
        self.labelFields.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.listFields = QListWidget()
        self.listFields.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listFields.setSortingEnabled(True)

        self.hLayoutFields = QHBoxLayout()
        self.hLayoutFields.addWidget(self.listFields)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.formLayout = QFormLayout(self)
        self.formLayout.addRow(self.labelFile, self.hLayoutFile)
        self.formLayout.addRow(self.labelHLine_1)
        self.formLayout.addRow(QLabel("Select algorithm:"), QLabel("Algorithm parameters:"))
        self.formLayout.addRow(self.listAlgorithms, self.stack)
        self.formLayout.addRow(self.labelHLine_2)
        self.formLayout.addRow(self.labelFields, self.hLayoutFields)
        self.formLayout.addWidget(self.buttonBox)

        self.toolButtonFile.clicked.connect(self.getfile)
        self.lineFile.textChanged.connect(self.openfile)
        self.listAlgorithms.currentRowChanged.connect(self.display_stack)

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

    def getfile(self):
        filename = QFileDialog.getOpenFileName(self, 'Select CSV or LAS file',
                                               '', "CSV files (*.csv);;"
                                                   "LAS files (*.las)")

        self.lineFile.setText(filename[0])

    def openfile(self):
        scalar_list = [*range(0, 100, 1)]
        self.listFields.clear()
        for item in scalar_list:
            self.listFields.addItem(str(item))

    def accept(self):
        print("Ok pressed !")

    def reject(self):
        print("Cancel pressed !")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Configuration()
    ex.show()
    sys.exit(app.exec_())
