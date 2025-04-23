*'segment'* module
===================

The 'segment' module segments an entire dataset according to the number of clusters defined. 

File needed for segmentation
-----------------------------

To segment a dataset, only 1 file is required:

* The input dataset, as **LAS** or **CSV** (sep=',') formats.

Input datasets:
~~~~~~~~~~~~~~~

The input dataset must contain some features that describe the points.

Any fields named **'target'** (non case-sensitive), containing the labels as integer, are discarded for segmentation.

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

- :command:`-f, --features`
  *Set the features to used to segment the dataset, as a list of feature names. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -f=['Anisotropy_5m', 'R', 'G', 'B', ...]

- :command:`-p, --parameters`
  *Set the parameters to pass to the clustering algorithm, as a dictionary. Caution: Replace whitespaces by underscores '_'.*

.. code-block:: console

  -p="{'n_estimators':50,'max_depth':5,'max_iter':500}"

