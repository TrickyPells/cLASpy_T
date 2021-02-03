# **cLASpy_T**

**cLASpy_T** means '*Tools for classification of LAS file with python and machine learning libraries*' (classification LAS python-Tools). 
cLASpy_T uses Scikit-Learn machine learning algorithms to classify 3D point clouds, such as LiDAR or Photogrammetric point clouds.
The data must be provided in a LAS or CSV file. Other formats should be supported later (GEOTIFF or PLY), and other machine learning libraries too (TensorFlow).

First, create a model with labeled data, according to an available machine learning algorithm in cLASpy-T.
Once the model is created, use it to perform predictions for non-labeled data.

## **Installation**

Use the git command to clone 'cLASpy_T.git'.

```bash
git clone https://github.com/TrickyPells/cLASpy_T.git
```

## **Usage**

```bash
python cLASpy_T.py [optional arguments] algorithm /path/to/the/data.file
```

### **Available algorithms**

* **rf** : *RandomForestClassifier* > Random Forest algorithm
* **gb** : *GradientBoostingClassifier* > Gradient Boosting algorithm
* **ann** : *MLPClassifier* > Artificial Neural Network algorithm
* **kmeans** : *KMeans* > K-Means clustering algorithm

(Refer to Scikit-Learn library for more details)

### **Format of data files**

The input data must be in **LAS** or **CSV**(sep=',') format.

*Example of CSV file:*
```txt
X,Y,Z,Target,Intensity,Red,Green,Blue,Roughness (5),Omnivariance (5),Sphericity (5)...
638.957,916.201,-2.953,1,39.0,104,133,113,0.11013,0.63586,0.00095...
```

**For training, data file must contain:**

* target field named **'target'** (not case-sensitive), contains the labels as integer 
* data fields

**For prediction, data file must contain:**

* data fields


If X, Y and/or Z fields are present, **they are excluded**, but re-used to write the output file.

To use **'Intensity'** field from LAS file, rename it (examples: **'Original_Intensity'** or **'Amplitude'**).

### **Optional arguments:**

**-h, --help :**\
*Show this help message and exit*

**-g, --grid_search :**\
*Perform the training with GridSearchCV* (See the Scikit-Learn documentation).

**-i, --importance :**\
*Export feature importance from **RandomForest** and **GradientBoosting** model as a PNG image file.\
Does not work with **GridSearchCV**.*

**-k, --param_grid [="dict"] :**\
*Set the parameters for the GridSearch as list(sep=',') in dict **with ANY SPACE**.\
Wrong parameters will be ignored. If empty, GridSearchCV uses presets.*

*Example:*
```bash
-k="{'n_estimators':[50,100,500],'loss':['deviance','exponential'],hidden_layer_sizes':[[100,100],[50,100,50]]}"    
```

**-m, --model_to_import [="/path/to.model"] :**\
*The model file to import to make predictions.*

*Examples:*
```bash
-m="/path/to/the/file.model"
```
 **-n, --n_jobs [=int] :**\
*Set the number of CPU used, '-1' means all available CPUs.*

**-p, --parameters [="dict"] :**\
*Set the parameters to pass at the classifier, as dict **with ANY SPACE**.*\
*Wrong parameters will be ignored.*

*Example:*
```bash
-p="{'n_estimators':50,'max_depth':5,'max_iter':500}"
```

**--pca [=int] :**\
*Set the PCA analysis and the number of principal components.*

**-s, --samples [=float (in Mpts)] :**\
*Set the number of samples, in Million points, for large dataset.\
If data length > samples:\
then train + test length = samples*

**--scaler [='Standard','Robust','MinMax']:**\
*Set method to scale the data before training (See the Scikit-Learn documentation).*

**--scoring [='accuracy','balanced_accuracy','average_precision','precision','recall',...]**\
*Set scorer to **GridSearchCV** or **cross_val_score** according to the Scikit-Learn documentation.*

**--test_ratio [0.0-1.0]:**\

*Set the test ratio as float [0.0-1.0] to split into train and test data.\

If train_ratio + test_ratio > 1:\
then test_ratio = 1 - train_ratio*

**--train_ratio [0.0-1.0]:**\
*Set the train ratio as float number to split into train and test data.\
If train_ratio + test_ratio > 1:\
then test_ratio = 1 - train_ratio*

## **Branches**
List all branches (local and remote):
```bash
git branch -a
```

```bash
  cheno
* dev
  master
  remotes/origin/HEAD -> origin/dev
  remotes/origin/beta
  remotes/origin/cheno
  remotes/origin/dev
  remotes/origin/master
```

To create a new local branch from remote:\
*(Here, to create a local branch 'beta')*

```bash
git checkout --track origin/beta
```

```bash
Switched to a new branch 'beta'
Branch 'beta' set up to track remote branch 'beta' from 'origin'.
```

## **Contributing**
Pull requests are welcome.\
For major changes or bug reports, please open an issue first to discuss what you would like to change or what does not work.

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
