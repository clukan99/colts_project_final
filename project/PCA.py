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
from sklearn.decomposition import PCA


random.seed(1234)


print("Reading in the data")
########
now = datetime.now()
print("now =", now)
########
#tickets_bare = pd.read_csv("../../Full_Data/tickets_bare_full.csv")
tickets_cleaned = pd.read_csv("../Full_Data/tickets_cleaned_full.csv")
print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########


test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') | (tickets_cleaned['event_name'] == 'CLT22HOU')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU')]

test_UniqueID = test[['UniqueID']]
train_UniqueID = train[['UniqueID']]

train = train.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis= 1)
test = test.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis = 1)

y_train = train[['isAttended']]
y_test = test[['isAttended']]

X_train = train.drop(labels= ['isAttended'], axis = 1)
X_test = test.drop(labels = ['isAttended'], axis = 1)

pca = PCA(0.95)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

X_train['UniqueID'] = test_UniqueID
X_test['UniqueID'] = train_UniqueID

y_train['UniqueID'] = train_UniqueID
y_test['UniqeID'] = test_UniqueID

X_train.to_csv("../PCA_data/X_train.csv")
X_test.to_csv("../PCA_data/X_test.csv")

y_train.to_csv("../PCA_data/y_train.csv")
y_test.to_csv("../PCA_data/y_test.csv")