#!/usr/bin/env python
# coding: utf-8

# In[0]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error

# In[1]:

print(os.listdir("."))

# In[2]:

train = pd.read_csv('../input/tr1.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

print("Train loaded! ")
train.head()

# In[3]:

# pandas doesn't show us all the decimals
pd.options.display.precision = 15
train.head()

# In[4]:


# Create a training file with simple derived features
rows = 2
segments = int( np.floor(train.shape[0]) / rows) 

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min'])



y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])


for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()


# In[5]:


X_train.head()


# In[6]:

#normalize train data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled


# In[6]:
#apply model 

svm = NuSVR()
svm.fit(X_train_scaled, y_train.values.flatten())
y_pred = svm.predict(X_train_scaled)


# In[7]:
plt.figure(figsize=(6, 6))
plt.scatter(y_train.values.flatten(), y_pred)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('actual', fontsize=12)
plt.ylabel('predicted', fontsize=12)
plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
plt.show()
