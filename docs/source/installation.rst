Installation
############

This section describes how to get and install **cLASpy_T**.

Install Python 3
================

**cLASpy_T** is a Python 3 based software. It needs a Python 3.8 64-bit interpreter version installed, or earlier.

See the `Download section of the beginners Guide`_ from the Python documentation.

.. _Download section of the beginners Guide: https://wiki.python.org/moin/BeginnersGuide/Download

Install cLASpy_T on Windows
===========================

Get cLASpy_T source code
------------------------
  
First, open the Command Prompt ``cmd.exe``. You can easily open it by clicking :menuselection:`&Start`, then search `cmd`.
  
Once the Command Prompt open, move to the directory where put the **cLASpy_T** source code. For example, 'Me' user moves to his :file:`Code` directory and gets the **cLASpy_T** source code with the :command:`git` command to clone 'cLASpy_T.git':
  
.. code-block:: doscon

  C:\Users\Me>cd Code
  C:\Users\Me\Code>git clone https://github.com/TrickyPells/cLASpy_T.git
  
.. note::

  If you do not know what :command:`git` is, you also can download **cLASpy_T** source code on the `github page <https://github.com/TrickyPells/cLASpy_T>`_. Choose the branch you want to download and click :guilabel:`&Code` on the right, then :guilabel:`&Download ZIP`. Once downloaded, decompress the ZIP file in the directory you want.
  
Once you cone or download/decompress source code, move to the **cLASpy_T** directory:

.. code-block:: doscon
  
  C:\Users\Me\Code>cd cLASpy_T
 
 
Create a Virtual Environment
----------------------------

Python uses many packages, depending of your usages. To prevent a dirty installation and package incompatibilities, it could be a great idea to use virtual environments. Here, we will create a specific virutal environment for **cLASpy_T**.

First, create a new directory called :file:`.venv` and use :command:`venv` command from python to create a new virtual environment called :file:`claspy_venv`:
 

.. code-block:: doscon
   
  C:\Users\Me\Code\cLASpy_T>mkdir .venv
  C:\Users\Me\Code\cLASpy_T>python -m venv .venv\claspy_venv
  
Now, you can use this new virtual environment:

.. code-block:: doscon

  C:\Users\Me\Code\`cLASpy_T>.venv\claspy_venv\Scripts\activate
  
Your Command Prompt must return something like this:

.. code-block:: doscon

  (claspy_venv) C:\Users\Me\Code\cLASpy_T>
  
To deactivate the virtual environment, juste type:

.. code-block:: doscon

  (claspy_venv) C:\Users\Me\Code\cLASpy_T>deactivate
  
Install all dependancies
------------------------

All required packages are listed in the :file:`requirements.txt` file. We will use :command:`pip` command to install these dependencies automatically.

If no Command Prompt is already open, open one, move to the :file:`cLASpy_T` directory and activate the virtual environment, created earlier.

Check if :command:`pip` needs to be upgraded:

.. code-block:: doscon

  (claspy_venv) C:\Users\Me\Code\cLASpy_T>python -m pip install --upgrade pip
  
Once donce, you can install all dependencies:

.. code-block:: doscon

  (claspy_venv) C:\Users\Me\Code\cLASpy_T>python -m pip install -r requirements.txt


