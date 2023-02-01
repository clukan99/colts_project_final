### This is a simple model with one middle layer for a logistic regression neural network

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn
from datetime import date
from datetime import datetime
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

random.seed(1234)

print("Reading in the data")
########
now = datetime.now()
print("now =", now)
########
y_train = pd.read_csv("../../PCA_data/y_train_official.csv")
y_test = pd.read_csv("../../PCA_data/y_test_official.csv")
X_train = pd.read_csv("../../PCA_data/X_train_official.csv")
X_test = pd.read_csv("../../PCA_data/X_test_official.csv")

train_labels = pd.read_csv("../../PCA_data/train_label.csv")
test_labels = pd.read_csv("../../PCA_data/test_labels.csv")

print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########
X_train = X_train.drop(labels= ["Unnamed: 0.1","Unnamed: 0",'UniqueID'], axis = 1)
y_train = y_train.drop(labels= ["Unnamed: 0.1","Unnamed: 0", "event_name"], axis=1)
X_test = X_test.drop(labels= ["Unnamed: 0.1","Unnamed: 0","UniqueID"], axis=1)
y_test = y_test.drop(labels =["Unnamed: 0.1","Unnamed: 0","event_name"], axis=1)


test_UniqueID = y_test[['UniqueID']]


y_train = y_train.drop(labels= ['UniqueID'], axis = 1)



X_train_columns = X_train.columns
X_test_columns = X_test.columns
y_train_columns = y_train.columns
#y_test_columns = y_test.columns

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
#y_test_np = y_test.to_numpy()

########
now = datetime.now()
print("now =", now)
########
##################################

##################################


##### Defining the model#####
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=1234)
###############################

#### Grid search ############
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
############################

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
################################
#rf = RandomForestClassifier(n_estimators= 1200, min_samples_split= 5, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 10, bootstrap= False, random_state= 1234, verbose= 2)
#########Random search training#############
#print("Starting random search!!!!!")
print("Searching...")
rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1234)
############################################
print("Fitting.....")
rf_random.fit(X_train_np, y_train_np.ravel())
#######Viewing the best parameters#####
print("Viewing the best parameters")
rf_random.best_params_
######################################
#rf.fit(X_train, y_train)

#y_pred = rf.predict(X_test)

#test_UniqueID['predictions'] = y_pred

#test_UniqueID.to_csv("../../final_data_to_submit/RF_model2.csv")

#######Modelling#####

