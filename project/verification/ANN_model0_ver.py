### This is a simple model with zero middle layers for a logistic regression neural network


from csv import writer
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
#test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') | (tickets_cleaned['event_name'] == 'CLT22HOU')]
#train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU')]
#################################################################################################################################

########################################This is the verification set, trained on all but CLT22LAC ###############################
test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT22LAC')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU') & (tickets_cleaned['event_name'] != 'CLT22LAC')]
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
y_train_np= y_train.to_numpy()
y_test_np = y_test.to_numpy()

##################################
test_UniqueID['y_test'] = y_test
test_UniqueID_copy = test_UniqueID.copy()
##################################



########################################Define the proper hyperparameters########################################################
input_size = X_train.shape[1]
output_size = 1
n_iters = [30]
learning_rate = [0.4,0.3,0.2]
dropout = [0,0.1,0.2,0.3]
#################################################################################################################################



########################################Defining the model#######################################################################
class LogisticRegression(nn.Module):
    def __init__(self,input_dim, output_dim,p):
        super(LogisticRegression, self).__init__()
        
        self.dropout = nn.Dropout(p = p)

        # Define layers
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x):
        out = self.dropout(x)
        out = self.linear(out)
        y_predicted = self.sigmoid(out)
        return y_predicted

loss = nn.BCELoss()

#################################################################################################################################
def fuck_the_decimals(decimals):
    if decimals >= 0.5:
        return 1
    return 0
#################################################################################################################################
print("Starting the training now")
########
now = datetime.now()
print("now =", now)
########
the_list_to_pd = []
for n_iters2 in n_iters:
    for dropout2 in dropout:
        ####Defining the model with the proper dropout here
        model = LogisticRegression(input_size, output_size,p=dropout2)
        for learning_rate2 in learning_rate:
            #####Defining the model with the proper learning rate here
            ####Defining the model with the proper dropout here
            model = LogisticRegression(input_size, output_size,p=dropout2)
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate2)
            ##### Resetting out weights/variables here:
            X_train = torch.tensor(X_train_np, dtype= torch.float32, requires_grad= True)
            X_test =  torch.tensor(X_test_np, dtype = torch.float32, requires_grad= True)
            y_train = torch.tensor(y_train_np, dtype = torch.float32, requires_grad= True)
            y_test =  torch.tensor(y_test_np,dtype = torch.float32, requires_grad= True)
            #####Resetting our output name here
            test_UniqueID = test_UniqueID_copy
            #####Actual training here
            for epoch in range(n_iters2):
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
            #####################################################################################################################
            ####################Prediction off of our specific parameters here###################################################
            with torch.no_grad():
                y_test_pred = model(X_test)
            #####################################################################################################################
            ####################Evaluating the model#############################################################################
            print("Finished training; Time to evaluate")
            test_UniqueID['y_test_pred'] = y_test_pred.numpy()
            test_UniqueID['predicted'] = test_UniqueID['y_test_pred'].apply(fuck_the_decimals)
            accuracyScores = accuracy_score(y_true =list(test_UniqueID['y_test']) ,y_pred= list(test_UniqueID['predicted']))
            ###################Saving the score and all pertinent information to go alongn with it###############################
            list_to_add = [learning_rate2,n_iters2,dropout2,accuracyScores]
            the_list_to_pd.append(list_to_add)
            scores = pd.DataFrame(the_list_to_pd,columns= ['learning_rate','epochs', 'dropout','accuracy_scores'])
            scores.to_csv('../../verification_data/ANN0_accuracy_scores3.csv')
            ########
            now = datetime.now()
            print("Just wrote the results =", now)
            ########


print("Finally complete god damnit!!!!!")
########
now = datetime.now()
print("now =", now)
########


