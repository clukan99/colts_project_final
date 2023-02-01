### This is a simple model with zero middle layers for a logistic regression neural network



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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
#y_test = y_test.to_numpy()

X_train = torch.tensor(X_train, dtype= torch.float32, requires_grad= True)
X_test =  torch.tensor(X_test, dtype = torch.float32, requires_grad= True)
y_train = torch.tensor(y_train, dtype = torch.float32, requires_grad= True)
#y_test =  torch.tensor(y_test,dtype = torch.float32, requires_grad= True)

input_size = X_train.shape[1]
output_size = 1

########
now = datetime.now()
print("now =", now)
########

print("Starting the training now")

class LogisticRegression(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        
        # Define layers
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x):
        out = self.linear(x)
        y_predicted = self.sigmoid(out)
        return y_predicted


model = LogisticRegression(input_size, output_size)


learning_rate = 0.1

#n_iters = 300 gives 86% accurracy
n_iters = 100
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # Forward
    y_pred = model(X_train)

    # Loss
    l = loss(y_pred,y_train)

    #Gradient = backward pass
    l.backward() # dl/dw

    #Update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch +1}: w = {w[0][0].item()}, loss = {l}')


with torch.no_grad():
    y_test_pred = model(X_test)
########
now = datetime.now()
print("now =", now)
########
print("Finished training; Time to save")
test_UniqueID['y_test_pred'] = y_test_pred.numpy()

print("Saving the data")
test_UniqueID.to_csv("../../final_data_to_submit/ANN0_ver_PCA.csv")

