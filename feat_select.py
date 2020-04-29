#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################
#  #################_###############_###############_###############  #
#  ################/\ \############/ /\############/\ \#############  #
#  ###############/  \ \##########/ /  \##########/  \ \############  #
#  ##############/ /\ \ \########/ / /\ \_#######/ /\ \_\###########  #
#  #############/ / /\ \_\######/ / /\ \__\#####/ / /\/_/###########  #
#  ############/ / /_/ / /######\ \ \#\/__/####/ / /#______#########  #
#  ###########/ / /__\/ /########\ \ \########/ / /#/\_____\########  #
#  ##########/ / /_____/##########\ \ \######/ / /##\/____ /########  #
#  #########/ / /\ \ \######/_/\__/ / /#####/ / /_____/ / /#########  #
#  ########/ / /##\ \ \#####\ \/___/ /#####/ / /______\/ /##########  #
#  ########\/_/####\_\/######\_____\/######\/___________/###########  #
#  ---------- REMOTE -------- SENSING --------- GROUP --------------  #
#  #################################################################  #
#            'feat_select.py' from classer library                    #
#                By Xavier PELLERIN LE BAS and                        #
#                         November 2019                               #
#         REMOTE SENSING GROUP  -- https://rsg.m2c.cnrs.fr/ --        #
#        M2C laboratory (FRANCE)  -- https://m2c.cnrs.fr/ --          #
#  #################################################################  #
#  Description: Feature selection based on the scikit-learn library.  #
#  Run the code and select the data_file.csv to extract the features  #
#  according to SelectKBest or SelectPercentile.                      #
#######################################################################

# --------------------
# --- DEPENDENCIES ---
# --------------------

import argparse

import yaml
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

from tkinter import filedialog
from common import *
from predict import *
from training import *




# -------------------------
# --------- MAIN ----------
# -------------------------

if __name__ == '__main__':
    path_to = filedialog.askopenfilename(
        initialdir="./", title="Select CSV file",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*")))

    # FORMAT DATA as XY & Z & target DataFrames and remove raw_classification from file.
    print("\n1. Formatting data as pandas.Dataframe...")
    data, xy_coord, z_height, target = format_dataset(
        path_to, mode='training', raw_classif='lassif')

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, random_state=0, test_size=0.5)

    print("\n2. Selection of the best features...")
    select = SelectPercentile(percentile=60)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)

    print("X_train Features:\n{}".format(X_train.shape))
    print("X_train_selected: {}".format(X_train_selected.shape))

    features = data.columns.values
    print(features)
    mask = select.get_support()
    print(mask)
    print(features[mask])
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")
    plt.yticks(())
    plt.show()

