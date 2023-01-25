Usages
######

Presentation
============

|claspyt| is devided into 3 main modules: :command:`train`, :command:`predict` and :command:`segment`.

* :command:`train`: performs training model according the selected supervised machine learning algorithm and the provided dataset. The data file must contain fields of features that describe each point and the target field of labels as integers.

* :command:`predict`: performs predictions for new dataset according a previous trained model. The new dataset must contain the same fields of features that the dataset used to train the model. |claspyt| ignore any 'target' field if the new dataset has one.

* :command:`segment`: performs cluster segmentation of a dataset according KMeans algorithm (see `scikit-learn`_ documentation).

.. note::

  In command line mode, use :command:`python cLASpy_T.py train`, :command:`predict` or :command:`segment` with :command:`--help` argument for more details.
  

.. _scikit-learn: https://scikit-learn.org/stable/modules/classes.html

Command line
============

Command line mode is the main and stablest way to use |claspyt|. Do not forget to activate your python virtual environment before calling |claspyt|.

It uses the 3 main modules as subcommands of the main software, *i.e.* :command:`train`, :command:`predict` and :command:`segment`.

For example, if you want to train model:

.. code-block:: console

  python cLASpy_T.py train -a=rf -i=/home/me/data/lidar_training_dataset.las
  
To make predictions:

.. code-block:: console

  python cLASpy_T.py predict -m=/home/me/results/lidar_dataset.model -i=/home/me/data/lidar_survey.las
  
And to segment a dataset:

.. code-block:: console

  python cLASpy_T.py segment -i=/home/me/data/lidar_survey.las
  

:command:`train` module
-----------------------

The 'train' module is used to create a supervised model from the machine learning algorithm that you select. |claspyt| uses `scikit-learn`_ library as main machine learning library, so do not hesitate to look up the documentation.

Available supervised algorithms

  Currently, there 3 available supervised machine learning algorithms from `scikit-learn`_:
  
  * :command:`rf`: `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_ for Random Forest algorithm
  * :command:`gb`: `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`_ for Gradient Boosting algorithm
  * :command:`ann`: `MLPClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier>`_ for Neural Network algorithm


Format of data files

  The input data must be in **LAS** or **CSV** (sep=',') formats.

:command:`predict` module
-------------------------



:command:`segment` module
-------------------------



