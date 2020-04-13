# **cLASser**

Classer is a Scikit-Learn based library in Python3, to classify standard LAS point cloud. 

## **Installation**

Use the git command to clone 'classer.git'.

```bash
git clone ssh://[user]@rsg-m2c.ynh.fr:22986/home/gitea/repositories/remote_sensing_group/classer.git
```


## **Usage**

```bash
python classer.py [optional arguments] algorithm C:/path/to/the/data_file.csv
```
### **algorithm:**

* **rf** : *RandomForestClassifier* > Random Forest algorithm
* **gb** : *GradientBoostingClassifier* > Gradient Boosting algorithm
* **svm** : *LinearSVC* > Support Vector Machine algorithm
* **ann** : *MLPClassifier* > Artificial Neural Network algorithm


(Refer to Scikit-Learn library for more details)

### **data_file.csv**

The input data must be in CSV format.
```txt
X,Y,Z,target,Intensity,Red,Green,Blue,Roughness (5),Omnivariance (5),Sphericity (5)...
638.957,916.201,-2.953,1,39.0,104,133,113,0.11013,0.63586,0.00095...
```

**For training, csv_data_file must contain:**
* target field named *'target'*
* data fields

**For prediction, csv_data_file must contain:**
* data fields


If X, Y and/or Z fields are present, **they are excluded**.
If a field_name contains **'lassif'** such as *'classif'*, *'classification'*, *'raw_classification'*... the field is discarded.


### **optional arguments:**
**-h, --help :**\
*Show this help message and exit*

**-g, --grid_search :**\
*Perform the training with GridSearchCV*

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
 **-m, --model_to_import [="/path/to.file"] :**\
*The model file to import to make predictions:*

*Examples:*
```bash
-m="C:/path/to/the/training/file.model" [WINDOWS]
-m="/path/to/the/training/file.model" [UNIX]
```
 **-n, --n_jobs [1,2,...,-1] :**\
*Set the number of CPU used, '-1' means all CPU available.*

**-p, --parameters [="dict"] :**\
*Set the parameters to pass at the classifier, as dict **with ANY SPACE**.*

*Example:*
```bash
-p="{'n_estimators':50,'max_depth':5,'max_iter':500}"
```

**-s, --samples [float in Mpts] :**\
*Set the number of samples, in Million points, for large dataset.\
If data length* *> samples:\
then train + test length = samples*

**--scaler {Standard,Robust,MinMax}:**\
*Set method to scale the data before training. See the preprocessing documentation of scikit-learn.*

**--scoring [='accuracy','balanced_accuracy','average_precision','precision','recall',...]**\
*Set scorer to **GridSearchCV** or **cross_val_score** according to sckikit-learn documentation.*

**--test_ratio [0.0-1.0]:**\
*Set the test ratio as float [0.0-1.0] to split into train and test data.\
If train_ratio + test_ratio > 1:\
then test_ratio = 1 - train_ratio*

**--train_ratio [0.0-1.0]:**\
*Set the train ratio as float number to split into train and test data.\
If train_ratio + test_ratio > 1:\
then test_ratio = 1 - train_ratio*

## **Contributing**
Pull requests are welcome.\
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## **License**
Any