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


.. toctree::
   :maxdepth: 1

   cmdline_train.rst
   cmdline_predict.rst
   cmdline_segment.rst




