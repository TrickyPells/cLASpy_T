Command line
************

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


*'train'* module
================

The 'train' module is used to create a supervised model from the machine learning algorithm that you select. |claspyt| uses `scikit-learn`_ library as main machine learning library, so do not hesitate to look up the documentation.

.. _scikit-learn: https://scikit-learn.org/stable/

Available supervised algorithms
-------------------------------

Currently, there 3 available supervised machine learning algorithms from `scikit-learn`_:

* :command:`rf`: `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_ for Random Forest algorithm
* :command:`gb`: `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`_ for Gradient Boosting algorithm
* :command:`ann`: `MLPClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier>`_ for Neural Network algorithm

Format of data files
--------------------

The input data must be in **LAS** or **CSV** (sep=',') formats.

*Example of CSV file:*

.. code-block:: asc

  X,Y,Z,Target,Intensity,Red,Green,Blue,Roughness (5),Omnivariance (5),Sphericity (5)...
  638.957,916.201,-2.953,1,39.0,104,133,113,0.11013,0.63586,0.00095...

Data file must contain:
^^^^^^^^^^^^^^^^^^^^^^^

* Target field named **'target'** (non case-sensitive): contains the labels for training as integer.
* Fileds of the features that describe each point.

If X, Y and/or Z fields provided, **they are discard for training**, but re-used to write the output file.

To use **'Intensity'** field from **LAS** file, rename it as, for example, **'Original_Intensity'** or **'Amplitude'**.

Arguments
---------

- :command:`-h, --help`
  *Show this help message and exit.*

- :command:`-a, --algo`
  *Set the supervised machine learning algorithm: 'rf', 'gb', 'ann'.*

  * :command:`rf` > **RandomForestClassifier**
  * :command:`gb` > **GradientBoostingClassifier**
  * :command:`ann` > **MLPClassifier**

- :command:`-c, --config`
  *Give the configuration file with all parameters and selected scalar fields.*

  * **On Windows**: C:\path\to\the\config.json
  * **On Linux**: /path/to/the/config.json

- :command:`-i, --input_data`
  *Set the input file of the dataset (LAS or CSV).*

  * **On Windows**: C:\path\to\the\input_data.las
  * **On Linux**: /path/to/the/input_data.las

- :command:`-o, --output`
  *Set the output folder to save all results. Default: Create folder with the path of the input file.*

  * **On Windows**: C:\path\to\the\output_folder
  * **On Linux**: /path/to/the/output_folder

- :command:`-f, --features`
  *Select the features to used to train the model. Give a list of feature names. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -f=['Anisotropy_5m', 'R', 'G', 'B', ...]

- :command:`-g, --grid_search`
  *Perform the training with GridSearchCV (see `scikit-learn`_ documentation).*

- :command:`-k, --param_grid`
  *Set the parameters to pass to the GridSearchCV as lists in a dictionary. If empty, GridSearchCV uses presets.*
  *Wrong parameters will be ignored. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -k="{'n_estimators':[50,100,500],'loss':['deviance', 'exponential'],'hidden_layer_sizes':[[100,100],[50,100,50]]}"

- :command:`-n, --n_jobs`
  *Set the number of threads to use, '-1' means all available threads. Default: -1.*

- :command:`-p, --parameters`
  *Set the parameters to pass to the classifier for training, as a dictionary. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -p="{'n_estimators':50,'max_depth':5,'max_iter':500}"

- :command:`--pca`
  *Set the Principal Component Analysis and the number of principal components.*

- :command:`--png_features`
  *Export the feature importnaces from RandomForest and GradientBoosting algorithms as PNG image.*

- :command:`--random_state`
  *Set the random_state to split dataset in the GridSearchCV and cross-validation.*

- :command:`-s, --samples`
  *Set the number of samples for large dataset (float in million points). samples = train_set + test_set.*

- :command:`--scaler`
  *Set the method to scale the dataset before training. Default: 'Standard'.*

  * :command:`Standard`: `StandardScaler`_ > Standardize features by removing the mean and scaling to unit variance.

.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

  * :command:`MinMax`: `MinMaxScaler`_ > Transform features by scaling each feature to a given range, *e.g.* betwen zero and one.

.. _MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler

  * :command:`Robust`: `RobustScaler`_ > Scale features using statistics that are robust to outliers, *e.g.* between 1st and 3rd quartile.

.. _RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler

- :command:`--scoring`
  *Set scorer for GridSearchCV or cross_val_score. Default: 'accuracy'. See the `scikit-learn_ documentation.*

- :command:`--train_r`
  *Set the train ratio as float [0.0 - 1.0] to split data into train and test datasets. Default: 0.5.*


*'predict'* module
===================



*'segment'* module
===================


