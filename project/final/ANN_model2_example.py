### This is a simple model with twoo middle layers for a logistic regression neural network
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
########
now = datetime.now()
print("Data is loading =", now)
########
tickets_bare = pd.read_csv("../../Full_Data/tickets_bare_full.csv")
tickets_cleaned = pd.read_csv("../../Full_Data/tickets_cleaned_full.csv")
########
now = datetime.now()
print("Loaded the data =", now)
########
test =  tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') |(tickets_cleaned['event_name'] == 'CLT22HOU')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != "CLT21LV") & (tickets_cleaned['event_name'] != 'CLT22HOU')]


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

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
#y_test_np = y_test.to_numpy()

X_train = torch.tensor(X_train_np,dtype= torch.float32, requires_grad=True)
X_test =  torch.tensor(X_test_np, dtype = torch.float32, requires_grad= True)
y_train = torch.tensor(y_train_np, dtype = torch.float32, requires_grad= True)
#y_test =  torch.tensor(y_test_np,dtype = torch.float32, requires_grad= True)

input_size = X_train.shape[1]
hidden_layer1_size = input_size//4
hidden_layer2_size = hidden_layer1_size//16
output_size = 1
p = 0.1

########
now = datetime.now()
print("Completing the model =", now)
########
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, hidden_layer1, hidden_layer2, output_dim,p):
        super(LogisticRegression,self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_layer1)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(hidden_layer1,hidden_layer2)
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout(p = p)

        self.linear3 = nn.Linear(hidden_layer2, output_dim)
        self.sigmoid = nn.Sigmoid()

    

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear3(out)
        y_predicted = self.sigmoid(out)
        return y_predicted


model = LogisticRegression(input_dim= input_size, hidden_layer1 = hidden_layer1_size, hidden_layer2 = hidden_layer2_size, output_dim = output_size, p=p)
learning_rate = 0.1
n_iters = 60

loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    y_pred = model(X_train)

    l = loss(y_pred, y_train)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch +1}: loss = {l}')

with torch.no_grad():
    y_test_pred = model(X_test)


print("Finished training; Time to save")
test_UniqueID['y_test_pred'] = y_test_pred.numpy()

print("Saving the data")
test_UniqueID.to_csv("../../final_data_to_submit/ANN2_ver.csv")





