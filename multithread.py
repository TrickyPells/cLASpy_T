from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback, sys

from cLASpy_Classes import ClaspyTrainer, ClaspyPredicter, ClaspySegmenter

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


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.counter = 0

        layout = QVBoxLayout()

        self.l = QLabel("Start")
        b = QPushButton("TRAIN!")
        b.pressed.connect(self.run_train)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)

        layout.addWidget(self.l)
        layout.addWidget(b)
        layout.addWidget(self.text)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.show()

        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def message(self, s):
        self.text.appendPlainText(s)

    def progress_fn(self, n):
        self.message("%d%% done" % n)

    def execute_this_fn(self, progress_callback):
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(int(n*100/4))

        return "Done."

    def train(self, progress_callback):
        self.message("\n# # # # # # # # # #  cLASpy_T  # # # # # # # # # # # #"
                     "\n - - - - - - - -    TRAIN MODE    - - - - - - - - - -"
                     "\n * * * *    Point Cloud Classification    * * * * * *\n")

        # Set the classifier
        trainer = ClaspyTrainer(input_data="Test/Orne_20130525.las",
                                algo="rf",
                                algorithm=None)
        trainer.set_classifier()

        # Intro
        intro = trainer.introduction(verbose=True)
        self.message(intro)
        progress_callback.emit(int(0*100/7))

        # Part 1/7 - Format dataset
        self.message("\nStep 1/7: Formatting data as pandas.DataFrame...")
        step1 = trainer.format_dataset(verbose=True)
        self.message(step1)
        progress_callback.emit(int(1*100/7))

        # Part 2/7 - Split data into training and testing sets
        self.message("\nStep 2/7: Splitting data in train and test sets...")
        step2 = trainer.split_dataset(verbose=True)
        self.message(step2)
        progress_callback.emit(int(2*100/7))

        # Part 3/7 - Scale dataset as 'Standard', 'Robust' or 'MinMaxScaler'
        self.message("\nStep 3/7: Scaling data...")
        step3 = trainer.set_scaler_pca(verbose=True)
        self.message(step3)
        progress_callback.emit(int(3*100/7))

        # Part 4/7 - Train model
        if trainer.grid_search:  # Training with GridSearchCV
            self.message('\nStep 4/7: Training model with GridSearchCV...\n')
        else:  # Training with Cross Validation
            self.message("\nStep 4/7: Training model with cross validation...\n")

        step4 = trainer.train_model(verbose=True)  # Perform both training
        self.message(step4)
        progress_callback.emit(int(4*100/7))

        # Part 5/7 - Create confusion matrix
        self.message("\nStep 5/7: Creating confusion matrix...")
        step5 = trainer.confusion_matrix(verbose=True)
        self.message(step5)
        progress_callback.emit(int(5*100/7))

        # Part 6/7 - Save algorithm, model, scaler, pca and feature_names
        self.message("\nStep 6/7: Saving model and scaler in file:")
        step6 = trainer.save_model(verbose=True)
        self.message(step6)
        progress_callback.emit(int(6*100/7))

        # Part 7/7 - Create and save prediction report
        self.message("\nStep 7/7: Creating classification report:")
        self.message(trainer.report_filename + '.txt')
        step7 = trainer.classification_report(verbose=True)
        self.message(step7)
        progress_callback.emit(int(7*100/7))

        # Kill the remaining python interpreter (1+18)

        return "Training done!"

    def print_output(self, s):
        self.message(s)

    def thread_complete(self):
        self.message("THREAD COMPLETE!")
        self.threadpool.releaseThread()

    def run_train(self):
        # Pass the function to execute
        worker = Worker(self.train)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def recurring_timer(self):
        self.counter += 1
        self.l.setText("Counter: %d" % self.counter)


app = QApplication([])
window = MainWindow()
app.exec_()