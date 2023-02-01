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
tickets_bare = pd.read_csv("../../Full_Data/tickets_bare_full.csv")
tickets_cleaned = pd.read_csv("../../Full_Data/tickets_cleaned_full.csv")
print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########

########################################This is for the actual train and test set################################################
test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') | (tickets_cleaned['event_name'] == 'CLT22HOU')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU')]
#################################################################################################################################

########################################This is the verification set, trained on all but CLT22LAC ###############################
#test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT22LAC')]
#train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU') & (tickets_cleaned['event_name'] != 'CLT22LAC')]
#################################################################################################################################

########################################Proper formatting must be completed prior to model initiation ###########################

test_UniqueID = test[['UniqueID']]

train = train.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis= 1)
test = test.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis = 1)

y_train = train[['isAttended']]
y_test = test[['isAttended']]

X_train = train.drop(labels= ['isAttended'], axis = 1)
X_test = test.drop(labels = ['isAttended'], axis = 1)

X_train_columns = X_train.columns
X_test_columns = X_test.columns
y_train_columns = y_train.columns
y_test_columns = y_test.columns

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

########
now = datetime.now()
print("now =", now)
########
##################################
test_UniqueID['y_test'] = y_test
test_UniqueID_copy = test_UniqueID.copy()
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
rf = RandomForestClassifier(n_estimators= 1200, min_samples_split= 5, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 10, bootstrap= False, random_state= 42, verbose= 2)
#########Random search training#############
#print("Starting random search!!!!!")
#rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
############################################
#rf_random.fit(X_train, y_train)
#######Viewing the best parameters#####
#print("Viewing the best parameters")
#rf_random.best_params_
######################################
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

test_UniqueID['predictions'] = y_pred

test_UniqueID.to_csv("../../final_data_to_submit/RF_model2.csv")

#######Modelling#####

