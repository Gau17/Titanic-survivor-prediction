# In[1]:
# Importing the necessary libraries

import pandas as pd
import numpy as np
import random


# In[2]:
# Importing the Dataset

raw_data = pd.read_csv('train.csv')


# In[3]:


global m
m = raw_data['Age'].mean()    # Calculating the mean of the age values
def preprocess(data):
    drop_list = ['PassengerId','Name','Ticket','Cabin']
    data = data.drop(drop_list,axis = 1)     # Removing columns which will not be used for prediction
   
    data['Sex'] = (data['Sex'] == 'male')    # Converting male to True and Female to False  
    
    data['isS'] = (data['Embarked'] == 'S')  # One-hot encoding of embaeked column. Adding three new columns isS, isC and isQ
    data['isC'] = (data['Embarked'] == 'C')  
    data['isQ'] = (data['Embarked'] == 'Q')
    data = data.drop('Embarked',axis = 1)    # Removing the original embarked column
    
    data.loc[data['Age'].isnull(),'Age'] = m    # Filling the missing values of age 
    
    # Making values of Age and Fare range between 0 and 1
    data[['Age','Fare']] = (data[['Age','Fare']] - data[['Age','Fare']].min())/(data[['Age','Fare']].max() - data[['Age','Fare']].min())   
    
    return data


# In[4]:


Y_train = np.array(raw_data['Survived']).reshape(1,len(raw_data['Survived']))  # Creating the label array Y_train
raw_data1 = raw_data.drop('Survived',axis = 1)    # Dropping the Survived column 
data = preprocess(raw_data1)                 # Preprocessing the data
X_train = np.array(data)             # Creating the input array X_train
X_train[:,[1,-1,-2,-3]] = np.where(X_train[:,[1,-1,-2,-3]],1,0)   # Converting True to 1 and False to 0
X_train = X_train.transpose()
X_train = np.array(np.array(X_train,dtype = np.float32))


# In[5]:


def initializeParameters(X):
    ''' Initializes all parameters to 0 '''
    w = np.zeros((1,X.shape[0]))
    b = 0
    return w,b


# In[6]:


def sigmoid(X):
    return 1/(1+np.exp(-X))


# In[7]:


def ComputeCost(Prediction,Y):
    return -np.sum(Y*np.log(Prediction)+(1-Y)*(np.log(1-Prediction)))/Y.shape[1]


# In[8]:


lr = 0.9
w,b = initializeParameters(X_train)   # Initialize the parameters 
costs = []


# In[7]:


for i in range(1000):    
    T = np.dot(w,X_train) + b   # Calculating T
    Prediction = sigmoid(T)     # Calculating Predition on Training set
    costs.append(ComputeCost(Prediction,Y_train))   # Calculating the loss and appending it to costs list
    dT = Prediction - Y_train   
    dw = dT.dot(X_train.transpose())/X_train.shape[1]         # Calculating derivatives of loss with respect to parameters
    db = np.sum(dT,axis = 1,keepdims = True)/X_train.shape[1]
    w = w - lr*dw    # Updating the parameters
    b = b - lr*db


# In[8]:
# Clacuting the training accuracy

Prediction = np.where(Prediction>0.5,1,0)   # replace every entry greater than 0.5 by 1 and every other entry by 0
accuracy = (np.sum(np.where(Y_train == Prediction,1,0))/Prediction.shape[1]) * 100
print('Accracy on training set -',accuracy,'%')

# In[9]:
# Importing the test Dataset

test_data = pd.read_csv('test.csv')


# In[10]:
# Prepocessing the test data and obtaining X_test and Y_test

data = preprocess(test_data)
X_test = np.array(data)
X_test[:,[1,-1,-2,-3]] = np.where(X_test[:,[1,-1,-2,-3]],1,0)
X_test = X_test.transpose()
X_test = np.array(np.array(X_test,dtype = np.float32))


# In[11]:


T = np.dot(w,X_test) + b
Prediction = sigmoid(T)   # Calculating Prediction for test set
Prediction = np.where(np.isnan(Prediction),0.5,Prediction)   # Replace nan values in Prediction by 0.5 
Prediction = np.where(Prediction>0.5,1,0)  # Replace every entry greater than 0.5 by 1 and every other entry by 0


# In[12]:
# Saving the result in a csv file in the required format

result = pd.DataFrame(Prediction.transpose())
result['PassengerId'] = test_data['PassengerId']
result['Survived'] = result[0]
result = result.drop(0,axis = 1)
result.to_csv('result.csv',index = None)
print('\nPrediction on test set:\n',result)


