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
tickets_bare = pd.read_csv("../../Full_Data/tickets_bare_full.csv")
tickets_cleaned = pd.read_csv("../../Full_Data/tickets_cleaned_full.csv")
print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########
#tickets_bare[['isAttended']] = tickets_bare[['isAttended']].replace(1,0)
#tickets_bare[['isAttended']] = tickets_bare[['isAttended']].replace(2,1)

#tickets_cleaned[['isAttended']] = tickets_cleaned[['isAttended']].replace(1,0)
#tickets_cleaned[['isAttended']] = tickets_cleaned[['isAttended']].replace(2,1)

#tickets_cleaned = tickets_cleaned.sample(frac= 0.000005, replace= False, random_state= 1234)
########################################This is for the actual train and test set################################################
test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') | (tickets_cleaned['event_name'] == 'CLT22HOU')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU')]
#################################################################################################################################
#test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21JAX')]
#train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU') & (tickets_cleaned['event_name'] != 'CLT21JAX')]

test_UniqueID = test[['UniqueID']]

train = train.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis= 1)
test = test.drop(labels = ['event_name', 'UniqueID', 'Unnamed: 0', 'SeatUniqueID'], axis = 1)

y_train = train[['isAttended']]
#y_test = test[['isAttended']]

X_train = train.drop(labels= ['isAttended'], axis = 1)
X_test = test.drop(labels = ['isAttended'], axis = 1)

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


learning_rate = 0.009

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
test_UniqueID.to_csv("../../final_data_to_submit/ANN0_ver.csv")

