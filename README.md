# **cLASpy_T**

**cLASpy_T** means '*Tools for classification of LAS file with python and machine
learning algorithms*' or **classification LAS python Tools**.\
cLASpy_T uses scikit-learn machine learning algorithms to classify 3D point clouds,
such as LiDAR or Photogrammetric point clouds.
Data must be provided in LAS ou CSV files. Other formats should be supported
later (GEOTIFF or PLY), and other machine learning project too (TensorFlow).

## cLASpy_T paper
 cLASpy_T is the subject of a scientific article in a special issue of Remote Sensing. Here you'll find a detailed presentation of the software and some practical examples: https://doi.org/10.3390/rs16162891

### Citation
If you use cLASpy_T, for scientific, educational, commercial or other purposes, please cite the article as follows:

Pellerin Le Bas, X.; Froideval, L.; Mouko, A.; Conessa, C.; Benoit, L.; Perez, L. A New Open-Source Software to Help Design Models for Automatic 3D Point Cloud Classification in Coastal Studies. Remote Sens. 2024, 16, 2891. https://doi.org/10.3390/rs16162891

## Limitations

**cLASpy_T** *is in progress: results obtained by or with this software must be analyzed by experts in the field for which the software is used. The same experts must also have a minimum knowledge of automatic classification using machine learning.*

**Results cannot stand alone! They need to be interpreted!**


## **cLASpy_T uses Scikit-learn classifiers**
**Supervised learning**
* **Randomized Decision Trees** : *RandomForestClassifier* and *GradientBoostingClassifier*\
  https://scikit-learn.org/stable/modules/ensemble.html
* **Neural Network** : *MLPClassifier*\
  https://scikit-learn.org/stable/modules/neural_networks_supervised.html
  
**Unsupervised learning**
* **Clustering** : *K-means*\
https://scikit-learn.org/stable/modules/clustering.html#k-means

### **What you need**
1. Point cloud file (CSV or LAS) with **features** describing each point by 
   **spectral** and/or **geometric properties**:
   * **Spectral features** : RGB, LiDAR Intensity, Hyperspectral bands...
   * **Geometric features** : Planarity, Linearity, Sphericity, Verticality...


2. **'Target'** field in the given point cloud file.\
   Supervised algorithms learn to classify each point during the learning step.
   This step requires a field named 'Target' in the point cloud file,
   with integer values corresponding to the classes to be learned.
   

3. Point cloud without **labelled** point.\
   The output of the learning step is a model, which is used to make predictions
   on **non-labelled** point cloud. This point cloud must have the same spectral 
   and/or geometric features as the point cloud used for training.

## **Installation**

### **Install Python 3**
**cLASpy_T** is a Python 3 based program. It requires Python 3.8 64-bit interpreter
or earlier version installed. See the Download section of the Beginners Guide
from the Python documentation (https://wiki.python.org/moin/BeginnersGuide/Download).

To install **cLASpy_T**, you also need *pip* and *venv* python packages.

Once Python 3, *pip* and *venv* installed, get the **cLASpy_T** source code.

### **Get cLASpy_T source code**
First, open a terminal, or a command prompt ('cmd.exe' on Windows). 
Move to the directory where put **cLASpy_T** source code, then clone **cLASpy_T** source code
with following command:

```bash
git clone https://github.com/TrickyPells/cLASpy_T.git
```

If you do not know what Git is, you also can download cLASpy_T source code on
the Github page (https://github.com/TrickyPells/cLASpy_T).
Select the branch you want to download, click on 'Code', then 'Download ZIP'.
Once downloaded, unzip the ZIP file in the directory where you want to put 
the **cLASpy_T** source code.

Once you clone or download/decompress source code, move to the cLASpy_T directory:
```bash
cd cLASpy_T
```

### **Create a Virtual Environment**
Python may use many packages. To prevent a dirty installation and packages
incompatibilities, it could be a great idea to use virtual environment.
Here, we will create a specific virtual environment for **cLASpy_T** with *venv*.
First, create a new directory call '.venv' in the 'cLASpy_T' directory, 
then use *venv* command from python to create a new virtual environment 
named 'claspy_venv':

```bash
mkdir .venv
python -m venv .venv\claspy_venv
```

Now, activate this new virtual environment with:

On Windows:
```bash
.venv\claspy_venv\Scripts\activate
```
and the command prompt returns something like:
```bash
(claspy_venv) ...\cLASpy_T>
```
On Linux:
```bash
source .venv/claspy_venv/bin/activate
```
and the terminal returns something like:
```bash
(claspy_venv) .../cLASpy_T$
```

### **Install all dependencies**
The 'requirements.txt' file list all the required packages. We will use
*pip* command to install all dependencies automatically.

First, check if *pip* needs to be upgraded:
```bash
(claspy_venv) ...\cLASpy_T>python -m pip install --upgrade pip
```

Once done, install all dependencies:

```bash
(claspy_venv) ...\cLASpy_T>python -m pip install -r requirements.txt
```

### **Test the installation**
To check everything is OK, use the following command:
```bash
(claspy_venv) ...\cLASpy_T>python cLASpy_T.py train -a=rf -i=.\Test\Orne_20130525.las
```
You should see returns of the program in the command prompt. 
At the end, a folder named 'Orne_20130525' in the 'Test' folder 
appears with all result files of this training.

If you run into any issue, feel free to discuss it on the Github page : 
https://github.com/TrickyPells/cLASpy_T/issues

## **Usage**
cLASpy_T is divided into 3 main modules: *train*, *predict* and *segment*.

* **train:** performs training according the selected machine learning algorithm,
  and the provided data file. The data file must contain fields of features
  that describe each point, and the target field of label as integer.


* **predict:** performs predictions according a previous trained model
  and file of unknown data. The data file must contain field of features
  that describe each point. **cLASpy_T** ignores any 'target' field.


* **segment:** performs cluster segmentation of dataset according KMeans 
  algorithm (see scikit-learn documentation).

Use *train, predict* or *segment* with *--help* argument for more details, or see the documentation : https://claspy-t-project.readthedocs.io

## **Contributing**
Pull requests are welcome.\
For major changes or bug reports, please open an issue first to discuss
what you would like to change or what does not work.

## **License**
*(See the **'licence_en.txt'** file)*

CeCILL FREE SOFTWARE LICENSE AGREEMENT

Version 2.1 dated 2013-06-21

**Notice**

This Agreement is a Free Software license agreement that is the result
of discussions between its authors in order to ensure compliance with
the two main principles guiding its drafting:

  * firstly, compliance with the principles governing the distribution
    of Free Software: access to source code, broad rights granted to users,
  * secondly, the election of a governing law, French law, with which it
    is conformant, both as regards the law of torts and intellectual
    property law, and the protection that it offers to both authors and
    holders of the economic rights over software.

The authors of the CeCILL (for Ce[a] C[nrs] I[nria] L[ogiciel] L[ibre]) 
license are: 

Commissariat à l'énergie atomique et aux énergies alternatives - CEA, a
public scientific, technical and industrial research establishment,
having its principal place of business at 25 rue Leblanc, immeuble Le
Ponant D, 75015 Paris, France.

Centre National de la Recherche Scientifique - CNRS, a public scientific
and technological establishment, having its principal place of business
at 3 rue Michel-Ange, 75794 Paris cedex 16, France.

Institut National de Recherche en Informatique et en Automatique -
Inria, a public scientific and technological establishment, having its
principal place of business at Domaine de Voluceau, Rocquencourt, BP
105, 78153 Le Chesnay cedex, France.
