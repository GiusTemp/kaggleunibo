#!/usr/bin/env python
# coding: utf-8

# In[0]:

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# In[1]:

#os.listdir(".")

# In[2]:

train = pd.read_csv('../../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("Train loaded! ")

# In[3]:

# pandas doesn't show us all the decimals
pd.options.display.precision = 15
print(train.head())

# In[4]:

# Create a training file with simple derived features


rows = 150000 
segments = int( np.floor(train.shape[0]) / rows) 
columns_X= ['std','mad','q01','q05', 'q95','q99','abs_std','abs_mad','abs_q95', 'abs_q99']

X_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=columns_X)

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])
 
for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    

    x = seg['acoustic_data'] 
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'mad'] = x.mad()
    X_train.loc[segment, 'q01'] = np.quantile(x, 0.01)
    X_train.loc[segment, 'q05'] = np.quantile(x, 0.05)
    X_train.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X_train.loc[segment, 'q99'] = np.quantile(x, 0.99) 
    X_train.loc[segment, 'abs_std'] = x.abs().std()
    X_train.loc[segment, 'abs_mad'] = x.abs().mad()
    X_train.loc[segment, 'abs_q95'] = np.quantile(x.abs(), 0.95)
    X_train.loc[segment, 'abs_q99'] = np.quantile(x.abs(), 0.99)
    
# In[5]:

#X_train.to_csv("X_train.csv")
#print(X_train.head())

# In[6]:

#normalize train data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled)

# In[6]:
#apply model

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(verbose=True,random_state=10, tol=0.0000001, 
        max_iter=2500,hidden_layer_sizes=2000, shuffle=False,
        warm_start=True, solver='sgd',n_iter_no_change=500)
#LinearRegression()
model.fit(X_train_scaled, y_train.values.flatten())

y_pred = model.predict(X_train_scaled)

# In[7]:
plt.figure(figsize=(16, 8))
plt.plot(y_train, color='b', label='y_train')
plt.plot(y_pred, color='gold', label='our_model')
plt.legend();
plt.title('Predictions vs actual');
plt.show();



# In[8]:
score = mean_absolute_error(y_train.values, y_pred)
print(score)


# In[9]:
print("reading all segments")
submission = pd.read_csv('../../input/sample_submission.csv', index_col='seg_id')


# In[10]:

X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

for seg_id in X_test.index:
    seg = pd.read_csv('../../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'] 
    
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'mad'] = x.mad()
    X_test.loc[seg_id, 'q01'] = np.quantile(x, 0.01)
    X_test.loc[seg_id, 'q05'] = np.quantile(x, 0.05)
    X_test.loc[seg_id, 'q95'] = np.quantile(x, 0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x, 0.99)
    X_test.loc[seg_id, 'abs_std'] = x.abs().std()
    X_test.loc[seg_id, 'abs_mad'] = x.abs().mad()
    X_test.loc[seg_id, 'abs_q95'] = np.quantile(x.abs(), 0.95)
    X_test.loc[seg_id, 'abs_q99'] = np.quantile(x.abs(), 0.99)
    
X_test_scaled = scaler.transform(X_test)
submission['time_to_failure'] = model.predict(X_test_scaled)
submission.to_csv('submission.csv')

