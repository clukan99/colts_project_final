### This is a simple model with zero middle layers for a logistic regression neural network



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn
import datetime
from datetime import date
from datetime import datetime
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklego.preprocessing import RepeatingBasisFunction

tickets = pd.read_csv("../updated_datasets/Cross Road Analytics Competition Dataset.csv")
extra_predictors = pd.read_csv("../updated_datasets/predictors_extra.csv")
tickets_test = pd.read_csv("../updated_datasets/test.csv")

def to_the_proper_date(bad_date):
    # Converting to datetime object, then converting to string with the format I want
    return datetime.strptime(bad_date, '%m/%d/%y').strftime("%Y-%m-%d")


tickets_test['event_date'] =tickets_test['event_date'].apply(to_the_proper_date)

tickets = pd.concat([tickets, tickets_test])

extra_predictors['Date'] = pd.to_datetime(extra_predictors['Date'], infer_datetime_format=True)

def format_date(datetime):
    return datetime.strftime("%Y-%m-%d")

extra_predictors['Date'] = extra_predictors['Date'].apply(format_date)

extra_predictors['merch_give_away'] = extra_predictors.merch_give_away.replace(np.nan,"No")

extra_encoder_list = ['Coach','Weather','Division', 'Roof_open','Themed_games','Holiday','Holiday_five_days','day_of_week','sunday','merch_give_away','events (Y/N)']

encoder_extra = OneHotEncoder(sparse=  False)
encoder_extra.fit(extra_predictors[extra_encoder_list])
one_hot_encoded_extra_np = encoder_extra.transform(extra_predictors[extra_encoder_list])

out_extra = np.concatenate(encoder_extra.categories_).ravel().tolist()
one_hot_encoded_extra_pd = pd.DataFrame(one_hot_encoded_extra_np, columns = out_extra)

one_hot_encoded_extra_cols = one_hot_encoded_extra_pd.columns

one_hot_encoded_extra_pd['event_date'] = extra_predictors['Date']


def extract_second(s):
    if "21" in s:
        return s.split("21")[1]
    elif "22" in s:
        return s.split("22")[1]
    elif "23" in s:
        return s.split("23")[1]

tickets["opponent"] = tickets["event_name"].apply(extract_second)


tickets_dates = tickets[['event_date']]

def get_the_day(dateyear):
    date = datetime.strptime(dateyear, "%Y-%m-%d")
    return date.timetuple().tm_yday

tickets_dates['day_of_year'] = tickets_dates['event_date'].apply(get_the_day)


rbf = RepeatingBasisFunction(n_periods=12,
                         	column="day_of_year",
                         	input_range=(1,365),
                         	remainder="drop")


rbf.fit(tickets_dates)
tickets_dates_tranformed = pd.DataFrame(index=tickets_dates.event_date,
               	data=rbf.transform(tickets_dates))

tickets[["event_date_0","event_date_1","event_date_2","event_date_3","event_date_4","event_date_5","event_date_6","event_date_7","event_date_8","event_date_9","event_date_10","event_date_11"]] = tickets_dates_tranformed.to_numpy()

def ticket_added(ticket_date_add):
    if (ticket_date_add == 'NaN'):
        return "No"
    else:
        return "Yes"
        
tickets["ticket_added"] = tickets["add_datetime"].apply(ticket_added)

tickets = pd.merge(tickets, one_hot_encoded_extra_pd, on= 'event_date')

tickets = tickets.drop(labels =['acct_id', 'Start Year', 'LastYear','event_date','ResalePrice','ResaleDate', 'add_datetime'], axis = 1)

print("I am here!!!!!!!!!!!")


label_enc = LabelEncoder()
isAttended_label = tickets[['isAttended']]
label_enc.fit(isAttended_label)
tickets['isAttended'] = label_enc.transform(isAttended_label)

tickets[['Resold', 'isSTM','Sales_Source','Term','Tenure']] = tickets[['Resold', 'isSTM','Sales_Source','Term','Tenure']].replace(np.nan,0)

tickets[['Resold']] =  tickets[['Resold']].replace('Yes', 1)
tickets[['Resold']] =  tickets[['Resold']].replace('1', 1)
tickets[['paid']] = tickets[['paid']].replace(np.nan, 'paid_unknown')
tickets[['acct_type_desc']] = tickets[['acct_type_desc']].replace(np.nan, 'acct_desc_unknown')
tickets[['comp_name']] = tickets[['comp_name']].replace(np.nan, 'comp_name_unknown')
tickets[['plan_event_name']] = tickets[['plan_event_name']].replace(np.nan, 'plan_event_name_unknown')
tickets[['ClubExpYear']] = tickets[['ClubExpYear']].replace(np.nan, 'exp_year_unknown')
tickets[["Season"]] = tickets[['Season']].replace(2021, "season_2021")
tickets[["Season"]] = tickets[['Season']].replace(2022, "season_2022")


categorical_encoding_list = ['acct_type_desc','plan_event_name','comp_name','price_code','paid','class_name','status','TicketClass','TicketType','SeatType', 'Sales_Source','opponent' ,'section_name', 'row_name','SeatNum', 'ticket_added', 'ClubExpYear', 'Season', 'PC1']
categorical_one_zero = ['isHost','isSTM','Resold']


tickets[categorical_encoding_list] = tickets[categorical_encoding_list].astype(str)

print("I am here!!!!!!!!!!!    2")


encoder = OneHotEncoder(sparse=  False)
encoder.fit(tickets[categorical_encoding_list])
one_hot_encoded_categorical_np = encoder.transform(tickets[categorical_encoding_list])

out = np.concatenate(encoder.categories_).ravel().tolist()
one_hot_encoded_categorical_pd = pd.DataFrame(one_hot_encoded_categorical_np, columns = out)


one_hot_encoded_categorical_pd[categorical_one_zero] = tickets[categorical_one_zero]

tickets = tickets.drop(labels =categorical_encoding_list, axis = 1)
tickets = tickets.drop(labels =categorical_one_zero, axis = 1)

print("I am here!!!!!!!!!!!    3")



price_scaler = StandardScaler()
price_scaler.fit(tickets[['Price']])
tickets['Price'] = price_scaler.transform(tickets[['Price']])


print("I am here!!!!!!!!!!!    4")

tickets_bare = tickets.copy()

print("I am here!!!!!!!!!!!    5")

bare_cols =tickets_bare.columns

print("I am here!!!!!!!!!!!    6")

tickets_cleaned = one_hot_encoded_categorical_pd.copy()

print("I am here!!!!!!!!!!!    7")

tickets_cleaned = pd.concat([tickets_cleaned, tickets_bare],axis=1)

print("Saving the data here")

#tickets_bare.to_csv("../tickets_bare_full.csv")
#tickets_cleaned.to_csv("../tickets_cleaned_full.csv")

from sklearn.decomposition import PCA

test = tickets_cleaned[(tickets_cleaned['event_name'] == 'CLT21LV') | (tickets_cleaned['event_name'] == 'CLT22HOU')]
train = tickets_cleaned[(tickets_cleaned['event_name'] != 'CLT21LV') & (tickets_cleaned['event_name'] != 'CLT22HOU')]

del tickets_cleaned
del tickets_bare
del tickets
del encoder
del one_hot_encoded_categorical_np
del one_hot_encoded_categorical_pd
del LabelEncoder
del extra_predictors

test_UniqueID = test[['UniqueID']]
train_UniqueID = train[['UniqueID']]

train = train.drop(labels = ['event_name', 'UniqueID', 'SeatUniqueID'], axis= 1)
test = test.drop(labels = ['event_name', 'UniqueID', 'SeatUniqueID'], axis = 1)

y_train = train[['isAttended']]
y_test = test[['isAttended']]

X_train = train.drop(labels= ['isAttended'], axis = 1)
X_test = test.drop(labels = ['isAttended'], axis = 1)

pca = PCA(0.95)
print("Running PCA here")
pca.fit(X_train)
print("Finished running PCA now. Transforminng the data now")
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("Done transforming the data")
print(type(X_train))
print(X_train.shape)


labels = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
#X_train = pd.DataFrame(X_train,columns=labels)
#y_train = pd.DataFrame(y_train, columns= ['y_train'])
X_test = pd.DataFrame(X_test, columns= labels)
#y_test = pd.DataFrame(y_test, columns=['y_test'])


#X_train = np.insert(X_train, list(train_UniqueID), axis = 1)
#X_test = np.insert(X_test, list(test_UniqueID), axis =1)
#X_train['UniqueID'] = train_UniqueID.loc[:,'UniqueID']
X_test['UniqueID'] = test_UniqueID.loc[:,'UniqueID']
#y_train = np.insert(y_train, train_UniqueID)
#y_test = np.insert(y_test, test_UniqueID)
#y_train['UniqueID'] = train_UniqueID.loc[:,'UniqueID']
#y_test['UniqueID'] = test_UniqueID.loc[:,'UniqueID']

#test_UniqueID.to_csv("../PCA_data/test_labels.csv")
#train_UniqueID.to_csv("../PCA_data/train_label.csv")

#X_train.to_csv("../PCA_data/X_train.csv")
X_test.to_csv("../PCA_data/X_test2.csv")

#y_train.to_csv("../PCA_data/y_train.csv")
#y_test.to_csv("../PCA_data/y_test.csv")




