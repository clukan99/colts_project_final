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
from sklearn.decomposition import PCA

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


pca = PCA(0.95)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


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


########################################Define the proper hyperparameters########################################################
input_size = X_train.shape[1]
output_size = 1
hidden_layer1_size = [input_size//8,input_size//16]
n_iters = [30,50]
learning_rate = [0.1,0.01]
dropout = [0,0.1,0.2,0.3]
#################################################################################################################################


print("Starting the training now")

class LogisticRegression(nn.Module):
    def __init__(self,input_dim, hidden_layer1,output_dim,p):
        super(LogisticRegression, self).__init__()
        
        # Define layers
        self.linear1 = nn.Linear(input_dim, hidden_layer1)
        self.sigmoid = nn.Sigmoid()

        self.linear2 = nn.Linear(hidden_layer1, output_dim)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p = p)
        

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        y_predicted = self.sigmoid(out)
        return y_predicted

def fuck_the_decimals(decimals):
    if decimals >= 0.6:
        return 1
    return 0

loss = nn.BCELoss()

the_list_to_pd = []
for n_iters2 in n_iters:
    for hidden_layer1_size2 in hidden_layer1_size:
        for dropout2 in dropout:
            for learning_rate2 in learning_rate:
                #####Defining the model with the proper learning rate here
                ####Defining the model with the proper dropout here
                model = LogisticRegression(input_dim =input_size,hidden_layer1= hidden_layer1_size2,output_dim= output_size,p=dropout2)
                optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate2)
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
                        print(f'epoch {epoch +1}: loss = {l}')
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
                list_to_add = [learning_rate2,n_iters2,dropout2,hidden_layer1_size2,accuracyScores]
                the_list_to_pd.append(list_to_add)
                scores = pd.DataFrame(the_list_to_pd,columns= ['learning_rate','epochs', 'dropout','hidden_layer1_size','accuracy_scores'])
                scores.to_csv('../../verification_data/ANN1_accuracy_scores_SGD_cutoff_60.csv')
                ########
                now = datetime.now()
                print("Just wrote the results =", now)
                ########


print("Finally complete god damnit!!!!!")
########
now = datetime.now()
print("now =", now)
########








