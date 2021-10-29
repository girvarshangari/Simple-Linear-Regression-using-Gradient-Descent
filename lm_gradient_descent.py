#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from numpy.random import randn


# In[53]:


train_data = pa.read_csv("C:/Users/gshangar/Documents/ML_projects_practice/Linear Regression/train.csv")
test_data  = pa.read_csv("C:/Users/gshangar/Documents/ML_projects_practice/Linear Regression/test.csv")
x_train = train_data['x']
y_train = train_data['y']
x_train = np.array(x_train)
x_train = x_train.reshape(-1,1)
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)
plt.scatter(x_train,y_train)

x_test = test_data['x']
y_test = test_data['y']
x_test = np.array(x_test)
x_test = x_test.reshape(-1,1)
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)


# In[54]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))


# In[55]:


x2 = sum(x_train**2)[0]
y2 = sum(y_train**2)[0]
x  = sum(x_train)[0]
y  = sum(y_train)[0]
xy = sum(x_train*y_train)[0]


# In[56]:


m = 0
c = 0
alpha = 0.001
arr = np.empty(shape=[0, 1])
arr2 = np.empty(shape=[0, 1])
for i in range(1,25):
    cost = ((m*m*x2)+(len(x_train)*c**2)+(2*m*c*x)+y2-(2*m*xy)-(2*m*c*xy))/len(x_train)
    arr = np.append(arr, cost)
    arr2 = np.append(arr2, m)
    print(cost,m,c)
    dm = ((2*x2)+(2*c*x*len(x_train))-(2*xy))/len(x_train)
    dc = ((2*x)+(2*c*len(x_train))-(2*y))/len(x_train)
    m  = m-(alpha*dm)
    c  = c-(alpha*dc)


# In[57]:


m = 0.9883548555451995
c = -0.002202115235561737
y_pred = m*x_train+c
error = y_pred-y_train
#plt.scatter(y_train,err)
#print(r2_score(y_train,y_pred))
plt.hist(error)
plt.show()


# In[40]:




