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
import sklearn
from datetime import date
from datetime import datetime
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn import svm

from sklearn.model_selection import GridSearchCV
  

  
  
# fitting the model for grid search
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

print(y_train_np)


########
now = datetime.now()
print("now =", now)
########
##################################

##################################
model = svm.SVC()
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear','sigmoid']} 
grid = GridSearchCV(model, param_grid, refit = True, verbose = 3)
grid.fit(X_train_np, y_train_np.ravel())

# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
#clf.fit(X_train_np, y_train_np)