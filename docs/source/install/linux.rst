Install |claspyt| on Linux
==========================

Get |claspyt| source code
-------------------------

First, open a terminal and move to the directory in which cLASpy_T source code will be clone. For example, 'Me' user moves to his 'Code' directory, then get the cLASpy_T source code with the :command:`git` command to clone 'cLASpy_T.git':

.. code-block:: console

  me@pc:~$ cd Code
  me@pc:~/Code$ git clone https://github.com/TrickyPells/cLASpy_T.git

.. note::

  If you do not know what :command:`git` is, you can download |claspyt| source code on the `github page <https://github.com/TrickyPells/cLASpy_T>`_. Choose the branch you want to download and click :guilabel:`&Code` on the right, then :guilabel:`&Download ZIP`. Once downloaded, decompress the ZIP file in the directory you want.

Once you clone or download/decompress source code, move to the :file:`cLASpy_T` directory:

.. code-block:: console

  me@pc:~/Code$ cd cLASpy_T

Create a Virtual Environment
----------------------------

Python uses many packages, depending of your usages. To prevent a dirty installation and package incompatibilities, it could be a great idea to use virtual environments. Here, you will create a specific virtual environment for |claspyt|.

First, create a new directory called :file:`.venv` and use :command:`venv` command from python to create a new virtual environment called :file:`claspy_venv`:

.. code-block:: console

  me@pc:~/Code/cLASpy_T$ mkdir .venv
  me@pc:~/Code/cLASpy_T$ python -m venv .venv/claspy_venv

Now, you can use this new virtual environment:

.. code-block:: console

  me@pc:~/Code/cLASpy_T$ source .venv/claspy_venv/bin/activate

Your terminal must return something like this:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$

If you want to deactivate the virtual environment, juste type:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ deactivate

Install all dependencies
------------------------

All required packages are listed in the :file:`requirements.txt` file. We will use :command:`pip` command to install all dependencies automatically.

If no terminal already open, open one, move to the :file:`cLASpy_T` directory and activate the virtual environment created earlier.

Check if :command:`pip` needs to be upgraded:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ python -m pip install --upgrade pip

Once done, install all dependencies:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ python -m pip install -r requirements.txt


Now, with :command:`pip list` command, you should see all packages installed in this *venv*. It should look something like this:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ pip list
  Package                       Version
    ----------------------------- -----------
    colorclass                    2.2.2
    contourpy                     1.2.0
    cycler                        0.12.1
    fonttools                     4.50.0
    joblib                        1.3.2
    kiwisolver                    1.4.5
    laspy                         2.5.3
    matplotlib                    3.8.3
    numpy                         1.26.4
    packaging                     24.0
    pandas                        2.2.1
    pillow                        10.3.0
    pip                           24.3.1
    psutil                        5.9.8
    pyparsing                     3.1.2
    PyQt5                         5.15.10
    PyQt5-Qt                      5.15.2
    PyQt5-Qt5                     5.15.2
    PyQt5-sip                     12.13.0
    PyQt5-stubs                   5.15.6.0
    python-dateutil               2.9.0.post0
    pytz                          2024.1
    PyYAML                        6.0.1
    requests                      2.31.0
    scikit-learn                  1.5.0
    scipy                         1.12.0
    six                           1.16.0
    threadpoolctl                 3.4.0
    tzdata                        2024.1

**Well done ! Your installation of** |claspyt| **is now clomplete !**

You can start by following :doc:`/tutorials/tutorial1` to quickly discover |claspyt| and test its installation. You can also visit the :doc:`/usage/usage` to find out more about |claspyt| commands and usages.