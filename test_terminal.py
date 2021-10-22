import sys
from PyQt5 import QtGui, QtCore, QtWidgets

class OutputWrapper(QtCore.QObject):
    outputWritten = QtCore.pyqtSignal(object, object)

    def __init__(self, parent, stdout=True):
        QtCore.QObject.__init__(self, parent)
        if stdout:
            self._stream = sys.stdout
            sys.stdout = self
        else:
            self._stream = sys.stderr
            sys.stderr = self
        self._stdout = stdout

    def write(self, text):
        self._stream.write(text)
        self.outputWritten.emit(text, self._stdout)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __del__(self):
        try:
            if self._stdout:
                sys.stdout = self._stream
            else:
                sys.stderr = self._stream
        except AttributeError:
            pass

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)
        self.terminal = QtWidgets.QTextBrowser(self)
        self._err_color = QtCore.Qt.red
        self.button = QtWidgets.QPushButton('Test', self)
        self.button.clicked.connect(self.handleButton)
        layout.addWidget(self.terminal)
        layout.addWidget(self.button)
        stdout = OutputWrapper(self, True)
        stdout.outputWritten.connect(self.handleOutput)
        stderr = OutputWrapper(self, False)
        stderr.outputWritten.connect(self.handleOutput)

    def handleOutput(self, text, stdout):
        color = self.terminal.textColor()
        self.terminal.setTextColor(color if stdout else self._err_color)
        self.terminal.moveCursor(QtGui.QTextCursor.End)
        self.terminal.insertPlainText(text)
        self.terminal.setTextColor(color)

    def handleButton(self):
        if QtCore.QTime.currentTime().second() % 2:
            print('Printing to stdout...')
        else:
            sys.stderr.write('Printing to stderr...\n')

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 300, 200)
    window.show()
    sys.exit(app.exec_())
