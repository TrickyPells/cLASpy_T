*'predict'* module
===================

The 'predict' module allows to make predictions of an entire dataset using a pre-trained model, created with the 'train' module. 

Files needed for prediction
---------------------------

To make predictions, 2 files are required:

* The input dataset, as **LAS** or **CSV** (sep=',') formats.
* The pre-trained model, as **'*.model'**. 

Input datasets:
~~~~~~~~~~~~~~~

**The input dataset must contain the same features used to train the model.**
The 'predict' module starts with comparaison between the features used by the model and the features in the input dataset. If one or more features are missing, the 'predict' module returns an error. If the input dataset contains more features than necessary, these features are discarded for predictions.

The **'target'** field (non case-sensitive), containing the labels as integer, is not mandatory. If a **'target'** field is found in the input dataset, this field is discarded for predictions, but used at the end to make a confusion matrix and compute the scores.

Model:
~~~~~~

The **'*.model'** file is created during the training phase. It contains the model itself, as well as the scaler used to transform the input data, the features used to train the model and the PCA where applicable. 

Arguments
---------

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