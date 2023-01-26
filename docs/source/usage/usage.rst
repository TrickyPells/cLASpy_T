Usages
######

Presentation
************

|claspyt| is devided into 3 main modules: :command:`train`, :command:`predict` and :command:`segment`.

* :command:`train`: performs training model according the selected supervised machine learning algorithm and the provided dataset. The data file must contain fields of features that describe each point and the target field of labels as integers.

* :command:`predict`: performs predictions for new dataset according a previous trained model. The new dataset must contain the same fields of features that the dataset used to train the model. |claspyt| ignore any 'target' field if the new dataset has one.

* :command:`segment`: performs cluster segmentation of a dataset according KMeans algorithm (see `scikit-learn`_ documentation).

|claspyt| could be used through :doc:`**command line** <command_line>` or with :doc:`**Graphical User Interface** <gui>`.

.. note::

  In command line mode, use :command:`python cLASpy_T.py train`, :command:`predict` or :command:`segment` with :command:`--help` argument for more details.
  
.. _scikit-learn: https://scikit-learn.org/stable/modules/classes.html

.. toctree::
   :maxdepth: 2

   usage/command_line
   usage/gui
