#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


dataset=pd.read_csv("Dataset.csv")


# In[11]:


dataset.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 


# In[13]:


dataset['Date']= le.fit_transform(dataset['Date'])


# In[14]:


dataset.tail(10)


# In[15]:


dataset.info()


# In[17]:


X = dataset.iloc[:, 2:6].values
y = dataset.iloc[:, 6:7].values


# In[19]:


X


# In[21]:


X.shape


# In[20]:


y


# In[22]:


y.shape


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[24]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[30]:


#normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[31]:


X_train


# In[32]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train,y_train)


# In[33]:


y_pred = regressor.predict(X_test)
y_pred


# In[34]:


if(y_pred.all()<2.5):
    y_pred=np.round(y_pred-0.5)
else:
    y_pred=np.round(y_pred+0.5)
y_pred


# In[18]:


df1=(y_pred-y_test)/y_test
df1=round(df1.mean()*100,2)
print("Error = ",df1,"%") 
a=100-df1
print("Accuracy= ",a,"%")


# In[19]:


#SupportVectorMachine
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)


# In[20]:


y_pred = regressor.predict(X_test)
y_pred


# In[21]:


if(y_pred.all()<2.5):
    y_pred=np.round(y_pred-0.5)
    
else:
    y_pred=np.round(y_pred+0.5)

y_pred


# In[22]:


df1=(y_pred-y_test)/y_test
df1=round(df1.mean()*100,2)
print("Error = ",df1,"%") 


# In[23]:


a=100-df1
print("Accuracy= ",a,"%")


# In[ ]:




