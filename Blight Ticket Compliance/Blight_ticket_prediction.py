#!/usr/bin/env python
# coding: utf-8

# # PROJECT - Blight Ticket Compliance

# ### Required Libraries

# In[1]:


import pandas as pd
import numpy as np

# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV

# Precision and auc - Imbalanced class
from sklearn.metrics import roc_auc_score


# 1. We only take consider features in the test data columns so all common features are cosidered and rest are dropped & not considered 
# 2. From the Test Data : Some of features from initial consideration are as follows:
#     
#     Numerical Data:
#         1. Judgement amount ( Total Net Amount owed by person)
#         3. Late fee         (If any late fee incurred by the poi - person of interest)
#         4. Fine amount      (Original Amount)
#         P.S: All the fees like(state fees,admin fee etc are dropped cause its standard fee and does not help with prediction)
#     
#     Categorical Data:
#         This can have an impact on the prediction, as the person from a 'certain place' might not consider paying
#         and can have better understanding on prediction , atleast provide understanding if people are paying in certain   locations than others
#         
#         1. City
#         2. State
#         3. Pincode (Yet to decide)
#         4. Disposition
#         
# 3.We try to choose features that dont blow up in to 100s and 1000s of features after creating dummy variables.

# #### PART 1: Loading the data

# In[2]:


def load_data():
    
    train = pd.read_csv('data/train.csv', encoding="latin1")
    test = pd.read_csv("data/test.csv")

    print('Length of Training Data ={}'.format(len(train)))
    print('Length of Test Data ={}'.format(len(test)))
    
    return train,test


# #### PART 2: Cleaning up the data
#         1. Removing unwanted rows (NA values) - Training Data ,  Test Data should not have any rows dropped.
#         2. Dropping unnecessary columns that doesnt help with predictions - For Both Test and Training Data.

# In[3]:


def cleaning_data(train,test):
    
    df = train.copy()
    df_test = test.copy()
    df = df[['ticket_id','state','zip_code','disposition','judgment_amount','late_fee','compliance']]
    
    # Among the features we intend to use only disposition ,judgement amout ,late fee(need to be converted to 0 and 1),compliance
    
    # Dropping NA values
    df.dropna(axis = 0,inplace = True)
    
    # Converting compliance to int (0 and 1)
    df['compliance'] = df['compliance'].astype(int)
    
    # Converting late fee in to binary (0 and 1)
    df['late_fee']= df['late_fee'].apply(lambda x : 1 if x > 0 else 0 )
    df_test['late_fee']= df_test['late_fee'].apply(lambda x : 1 if x > 0 else 0 )
    
    
    # Keeping ticket id for final series result
    df_test = df_test[['ticket_id','disposition','judgment_amount','late_fee']]
    
    return df,df_test


# #### PART 3: Creating Dummy variables (if required) and hot encoding categorical data
#         1. We need to hot encode the disposition (it becomes numbered )
#         2. Dont forget to do the same transformation in Test data
#             - We are replacing disposition in to reflect both the Training and Test data
#         3. Ziping up can make it easier to lookup that information (if you want)
#         
#   We are going to consider the following columns for our classifier
#      
#      
#      FOR X:
#      
#      1.(Disposition columns) - after hot encoding -4 columns
#      2. Judgement Amount
#      3. Late fee
#      
#      FOR Y:
#      1. Compliance

# In[4]:


def dummies(df,df_test):
    import pandas as pd
    the_replacement = {'Responsible (Fine Waived) by Deter':'Fine Waived','Responsible (Fine Waived) by Admis':'Fine Waived',
                   'Responsible - Compl/Adj by Default':'Responsible by Default','Responsible - Compl/Adj by Determi':'Responsible by Determination',
                  'Responsible by Dismissal':'Fine Waived'}
    
    # Training data
    df.replace(the_replacement,inplace =True)
    
    # Test Data
    df_test.replace(the_replacement,inplace =True)
    
    
    # Creating Dummies for Training and Test Data
    # Training
    dummies = pd.get_dummies(df['disposition'])
    df2 = pd.concat([df, dummies],axis =1)
    #Test
    dummies = pd.get_dummies(df_test['disposition'])
    df2_test = pd.concat([df_test, dummies],axis =1)
    df2_test.drop(['disposition'],axis =1,inplace =True)
    
    # We get X and y (target) from training dataframe
    # We get the X and y from the cleaned dataframe
    X = df2.drop(['ticket_id','state','zip_code','disposition','compliance'],axis =1)
    y = df2['compliance']
    
    
    
    return X,y,df2_test


# #### PART 4 : Choosing a Classifier and testing the performace of data (HyperParameter Tuning)
#     1. Logistic Regression
#     2. LinearSVM
#     3. Decision Tree
#     4. Random Forest
#     5. Naive Bayes
#     
#     Finding out the best model with the best parameter to give best performace
#     
#     I chose Logistic Regression and calculate auc score, best parameter and score

# In[5]:


def blight_model():
    
    train,test = load_data()
    df,df_test = cleaning_data(train,test)
    X,y,df2_test = dummies(df,df_test)
    
    
    # Logistic Regression
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state =0)
    LogR = LogisticRegression()
    parameters = {'C':[0.1,1,100]}

    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(LogR,param_grid = parameters,scoring ='roc_auc')
    clf.fit(X_train,y_train)

    predicted_log = clf.decision_function(X_test)


    print('Test set AUC:',roc_auc_score(y_test,predicted_log))
    print('Best_parameter:',clf.best_params_)
    print('Best score:',clf.best_score_)
    
    ticket_id = df2_test['ticket_id'].tolist()
    df2_test.drop(['ticket_id'],axis =1,inplace =True)
    
    prob_ticket = clf.predict_proba(df2_test)
    data = pd.Series(prob_ticket[:,1])
    result = pd.DataFrame(data)
    result['ticket_id'] =ticket_id
    result = result.set_index('ticket_id')
    
    
    
    return result

blight_model()

