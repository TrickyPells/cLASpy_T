
Command line
#############

|claspyt| could be used through command line or with Graphical User Interface.

|claspyt| is devided into 3 main modules: :command:`train`, :command:`predict` and :command:`segment`.

* :command:`train`: performs training model according the selected supervised machine learning algorithm and the provided dataset. The data file must contain fields of features that describe each point and the target field of labels as integers.

* :command:`predict`: performs predictions for new dataset according a previous trained model. The new dataset must contain the same fields of features that the dataset used to train the model. |claspyt| ignore any 'target' field if the new dataset has one.

* :command:`segment`: performs cluster segmentation of a dataset according KMeans algorithm (see `scikit-learn`_ documentation).

.. note::

 Command line mode is the main and stablest way to use |claspyt|. 
 Do not forget to activate your python virtual environment before calling |claspyt|.
  

Command line mode uses the 3 main modules as subcommands of the main software, *i.e.* :command:`train`, :command:`predict` and :command:`segment`.

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
******************

The 'train' module is used to create a supervised model from the machine learning algorithm that you select. |claspyt| uses `scikit-learn`_ library as main machine learning library, so do not hesitate to look up the documentation.

.. _scikit-learn: https://scikit-learn.org/stable/

Available supervised algorithms
---------------------------------

Currently, there are 3 available supervised machine learning algorithms from `scikit-learn`_:

  * :command:`rf`: for Random Forest algorithm (`RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_)
  * :command:`gb`:  for Gradient Boosting algorithm (`GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`_)
  * :command:`ann`: for Artificial Neural Network algorithm (`MLPClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier>`_)

Format of data files
======================

The input data must be in **LAS** or **CSV** (sep=',') formats.

*Example of CSV file:*

.. code-block:: asc

  X,Y,Z,Target,Intensity,Red,Green,Blue,Roughness (5),Omnivariance (5),Sphericity (5)...
  638.957,916.201,-2.953,1,39.0,104,133,113,0.11013,0.63586,0.00095...

Data file must contain:
------------------------

* Target field named **'target'** (non case-sensitive): contains the labels for training as integer.
* Features that describe each point.

If X, Y and/or Z fields provided, **they are discard for training**, but re-used to write the output file.

To use **'Intensity'** field from **LAS** file, rename it as, for example, **'Original_Intensity'** or **'Amplitude'**.

Arguments
==========

- :command:`-h, --help`
  *Show this help message and exit.*

- :command:`-a, --algo`
  *Set the supervised machine learning algorithm: 'rf', 'gb', 'ann'.*

    * :command:`rf` > **RandomForestClassifier**
    * :command:`gb` > **GradientBoostingClassifier**
    * :command:`ann` > **MLPClassifier**

- :command:`-c, --config`
  *Give the configuration file with all parameters and selected scalar fields.*

    * **on Windows**: C:\\path\\to\\the\\config.json
    * **on Linux**: /path/to/the/config.json

- :command:`-i, --input_data`
  *Set the input file of the dataset (LAS or CSV).*

    * **on Windows**: C:\\path\\to\\the\\input_data.las
    * **on Linux**: /path/to/the/input_data.las

- :command:`-o, --output`
  *Set the output folder to save all results. Default: Create folder with the path of the input file.*

    * **on Windows**: C:\\path\\to\\the\\output_folder
    * **on Linux**: /path/to/the/output_folder

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
    * :command:`MinMax`: `MinMaxScaler`_ > Transform features by scaling each feature to a given range, *e.g.* betwen zero and one.
    * :command:`Robust`: `RobustScaler`_ > Scale features using statistics that are robust to outliers, *e.g.* between 1st and 3rd quartile.

.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
.. _MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
.. _RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler

- :command:`--scoring`
  *Set scorer for GridSearchCV or cross_val_score. Default: 'accuracy'. See the `scikit-learn_ documentation.*

- :command:`--train_r`
  *Set the train ratio as float [0.0 - 1.0] to split data into train and test datasets. Default: 0.5.*


*'predict'* module
********************

The 'predict' module allows to make predictions of an entire dataset using a pre-trained model, created with the 'train' module. 

Files needed for prediction
============================

To make predictions, 2 files are required:

* The input dataset, as **LAS** or **CSV** (sep=',') formats.
* The pre-trained model, as **'*.model'**. 

Input datasets:
-----------------

**The input dataset must contain the same features used to train the model.**
The 'predict' module starts with comparaison between the features used by the model and the features in the input dataset. If one or more features are missing, the 'predict' module returns an error. If the input dataset contains more features than necessary, these features are discarded for predictions.

The **'target'** field (non case-sensitive), containing the labels as integer, is not mandatory. If a **'target'** field is found in the input dataset, this field is discarded for predictions, but used at the end to make a confusion matrix and compute the scores.

Model:
------

The **'*.model'** file is created during the training phase. It contains the model itself, as well as the scaler used to transform the input data, the features used to train the model and the PCA where applicable. 

Arguments
==========

- :command:`-h, --help`
  *Show this help message and exit.*

- :command:`-c, --config`
  *Give the configuration file with all parameters and selected scalar fields.*

    * **on Windows**: C:\\path\\to\\the\\config.json
    * **on Linux**: /path/to/the/config.json

- :command:`-i, --input_data`
  *Set the input file of the dataset (LAS or CSV).*

    * **on Windows**: C:\\path\\to\\the\\input_data.las
    * **on Linux**: /path/to/the/input_data.las

- :command:`-o, --output`
  *Set the output folder to save all results. Default: Create folder with the path of the input file.*

    * **on Windows**: C:\\path\\to\\the\\output_folder
    * **on Linux**: /path/to/the/output_folder

- :command:`-m, --model`
  *Import the model file to make predictions.*

    * **on Windows**: C:\\path\\to\\the\\model_file.model
    * **on Linux**: /path/to/the/model_file.model


*'segment'* module
********************

The 'segment' module segments an entire dataset according to the number of clusters defined. 

File needed for segmentation
=============================

To segment a dataset, only 1 file is required:

* The input dataset, as **LAS** or **CSV** (sep=',') formats.

Input datasets:
----------------

The input dataset must contain some features that describe the points.

Any fields named **'target'** (non case-sensitive), containing the labels as integer, are discarded for segmentation.

Arguments
==========

- :command:`-h, --help`
  *Show this help message and exit.*

- :command:`-c, --config`
  *Give the configuration file with all parameters and selected scalar fields.*

    * **on Windows**: C:\\path\\to\\the\\config.json
    * **on Linux**: /path/to/the/config.json

- :command:`-i, --input_data`
  *Set the input file of the dataset (LAS or CSV).*

    * **on Windows**: C:\\path\\to\\the\\input_data.las
    * **on Linux**: /path/to/the/input_data.las

- :command:`-o, --output`
  *Set the output folder to save all results. Default: Create folder with the path of the input file.*

    * **on Windows**: C:\\path\\to\\the\\output_folder
    * **on Linux**: /path/to/the/output_folder

- :command:`-f, --features`
  *Set the features to used to segment the dataset, as a list of feature names. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -f=['Anisotropy_5m', 'R', 'G', 'B', ...]

- :command:`-p, --parameters`
  *Set the parameters to pass to the clustering algorithm, as a dictionary. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -p="{'n_estimators':50,'max_depth':5,'max_iter':500}"
