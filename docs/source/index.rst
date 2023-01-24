.. cLASpy_T documentation master file, created by
   sphinx-quickstart on Mon Jan 23 22:14:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cLASpy_T |version| documentation!
============================================

About cLASpy_T
--------------

**cLASpy_T** means *'Tools for classification of LAS file with python and machine learning algorithms'* or **classification LAS python Tools**.

**cLASpy_T** uses `scikit-learn`_ machine learning algorithms to classify 3D point clouds, such as LiDAR or Photogrammetric point clouds. Data must be provided in LAS ou CSV files. Other formats should be supported later (GEOTIFF or PLY), and other machine learning project too (`TensorFlow`_).

.. _scikit-learn: https://scikit-learn.org/stable/
.. _TensorFlow: https://www.tensorflow.org/

Purpose of cLASpy_T
-------------------

**cLASpy_T** was developped to friendly use machine learning algorithms to classify or segment 3D point clouds.

Roughly, the software formats the input point clouds provided by ALS or CSV files to pandas `DataFrame`_ to be compatible with *Python* machine learning algorythms, such as `scikit-learn`_ or `TensorFlow`_.
**cLASpy_T** writes the output classified point cloud in the same format of the input data, *i.e.* LAS or CSV.

.. _DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   tutorials

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
