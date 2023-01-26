Install cLASpy_T on Linux
*************************

Get cLASpy_T source code
========================

First, open a terminal and move to the directory in which cLASpy_T source code will be clone. For example, 'Me' user moves to his 'Code' directory, then get the cLASpy_T source code with the :command:`git` command to clone 'cLASpy_T.git':

.. code-block:: console

  me@pc:~$ cd Code
  me@pc:~/Code$ git clone https://github.com/TrickyPells/cLASpy_T.git

.. note::

  If you do not know what :command:`git` is, you also can download **cLASpy_T** source code on the `github page <https://github.com/TrickyPells/cLASpy_T>`_. Choose the branch you want to download and click :guilabel:`&Code` on the right, then :guilabel:`&Download ZIP`. Once downloaded, decompress the ZIP file in the directory you want.

Once you clone or download/decompress source code, move to the :file:`cLASpy_T` directory:

.. code-block:: console

  me@pc:~/Code$ cd cLASpy_T

Create a Virtual Environment
============================

Python uses many packages, depending of your usages. To prevent a dirty installation and package incompatibilities, it could be a great idea to use virtual environments. Here, you will create a specific virtual environment for **cLASpy_T**.

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
========================

All required packages are listed in the :file:`requirements.txt` file. We will use :command:`pip` command to install all dependencies automatically.

If no terminal already open, open one, move to the :file:`cLASpy_T` directory and activate the virtual environment created earlier.

Check if :command:`pip` needs to be upgraded:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ python -m pip install --upgrade pip

Once done, install all dependencies:

.. code-block:: console

  (claspy_venv) me@pc:~/Code/cLASpy_T$ python -m pip install -r requirements.txt

