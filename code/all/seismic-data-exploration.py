#!/usr/bin/env python
# coding: utf-8

# <h2>1. Overview</h2>
# 
# As stated in the description, the data for this competition comes from a experimental set-up used to study earthquake physics. Our goal is to predict the time remaining before the next laboratory earthquake. The only feature we have is the seismic signal (acoustic data) which has integer values in a limited range.
# 
# * Training data: single, continuous segment of experimental data.
# 
# * Test data: consists of a folder containing many small segments. The data within each test file is continuous, but **the test files do not represent a continuous segment of the experiment**.
# 
# There are a lot of files in this competition, so let's start with the folders structure:

# In[ ]:


import os
from random import shuffle
import numpy as np
import pandas as pd

import dask.dataframe as dd

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numba
import warnings
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

@numba.jit
def get_stats(arr):
    """Memory efficient stats (min, max and mean). """
    size  = len(arr)
    min_value = max_value = arr[0]
    mean_value = 0
    for i in numba.prange(size):
        if arr[i] < min_value:
            min_value = arr[i]
        if arr[i] > max_value:
            max_value = arr[i]
        mean_value += arr[i]
    return min_value, max_value, mean_value/size


# In[ ]:


print(os.listdir("C:/Users/PC-Casa/input"))


# The test folder has many csv files:

# In[ ]:


test_folder_files = os.listdir("C:/Users/PC-Casa/input/test")
print(test_folder_files[:10])  # print first 10
print("\nNumber of files in the test folder", len(test_folder_files))


# There is one file in the test folder for each prediction (seg_id) in sample_submission:

# In[ ]:


sample_sub = pd.read_csv('C:/Users/PC-Casa/input/sample_submission.csv')
print("Submission shape", sample_sub.shape)
sample_sub.head()


# <h2>2. Training data</h2>
# 
# One huge csv file has all the training data, which is a single continuous experiment. There are only two columns in this file:
# * Acoustic data (int16): the seismic signal
# * Time to failure (float64): the time until the next laboratory earthquake  (in seconds)
# * **No missing values for both columns**

# In[ ]:


#first pd PandaDataframe, now dd DaskDataframe
train = dd.read_csv('C:/Users/PC-Casa/input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", train.shape)
#pd.set_option("display.precision", 15)  # show more decimals

train.acoustic_data.describe()
train.time_to_failure.describe()

train.head()


# <h2>Acoustic data</h2>
# 
# Our single feature are integers in the range [-5515, 5444] with mean 4.52

# In[ ]:


#pd.set_option("display.precision", 8)



# The plot below is using a 1% random sample (~6M rows):

# In[ ]:


train_sample = train.sample(frac=0.01)
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
ax = sns.distplot(train_sample.acoustic_data, label='Train (1% sample)')


# There are outliers in both directions; let's try to plot the same distribution with x in the range -25 to 25. The black line is the closest normal distribution (gaussian) possible.

# In[ ]:


train_sample = train.sample(frac=0.01)
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
tmp = train_sample.acoustic_data[train_sample.acoustic_data.between(-25, 25)]
ax = sns.distplot(tmp, label='Train (1% sample)', kde=False, fit=stats.norm)


# <h2>Time to failure</h2>
# 
# Now let's check the target variable, which is given in seconds:

# In[ ]:


tmin, tmax, tmean = get_stats(train.time_to_failure.values)
print("min value: {:.6f}, max value: {:.2f}, mean: {:.4f}".format(tmin, tmax, tmean))


# The min value is very close to zero (around 10^-5) and the max is 16 seconds. Now the distribution for the random sample:

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Time to failure distribution")
ax = sns.kdeplot(train_sample.time_to_failure, label='Train (1% sample)')


# <h2>Timeseries</h2>
# 
# Let's see how both variables change over time. The orange line is the acoustic data and the blue one is the time to failure:

# In[ ]:


def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue'):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx].acoustic_data.values, ax=ax1, color=color1)
    p2 = sns.lineplot(data=train.iloc[idx].time_to_failure.values, ax=ax2, color=color2)


def double_timeseries(final_idx1, final_idx2, init_idx1=0, init_idx2=0, step=1, title=""):
    idx1 = [i for i in range(init_idx1, final_idx1, step)]
    idx2 = [i for i in range(init_idx2, final_idx2, step)]
    
    fig, (ax1a, ax2a) = plt.subplots(1,2, figsize=(12,5))
    fig.subplots_adjust(wspace=0.4)
    ax1b = ax1a.twinx()
    ax2b = ax2a.twinx()
    
    ax1a.set_xlabel('index')
    ax1a.set_ylabel('Acoustic data')
    ax2a.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx1].acoustic_data.values, ax=ax1a, color='orange')
    p2 = sns.lineplot(data=train.iloc[idx1].time_to_failure.values, ax=ax1b, color='blue')
    
    p3 = sns.lineplot(data=train.iloc[idx2].acoustic_data.values, ax=ax2a, color='orange')
    p4 = sns.lineplot(data=train.iloc[idx2].time_to_failure.values, ax=ax2b, color='blue')
    
single_timeseries(100000, title="First hundred thousand rows")


# Time to failure looks like a stairway with many steps where the time is constant. However, this is not the case as we can see when plotting only the first thousand rows:

# In[ ]:


single_timeseries(1000, title="First thousand rows")


# The time is decreasing linearly between each "step" or "jump". However, when plotting the first million rows the steps are not visible anymore:

# In[ ]:


single_timeseries(1000000, title="First million rows")


# It's as if we have stretched the line and the steps disappear, but they still there. Now let's check 10 million rows, but plotting only every 10 points:

# In[ ]:


single_timeseries(10000000, step=10, title="Ten million rows")


# There is a huge spike in acoustic data around half second before the earthquake. Let's have a closer look at the data right after the spike to the earthquake (index 5 to 6 million, step 1).

# In[ ]:


single_timeseries(final_idx=6000000, init_idx=5000000, title="Five to six million index")


# Finally, the plot for all the data (629.145 million rows):

# In[ ]:


single_timeseries(629145000, step=1000, title="All training data")


# There are 16 earthquakes in the training data. The shortest time to failure is 1.5 seconds for the first earthquake and 6s to the 7º, while the longest is around 16 seconds.
# 
# In the plot below I just changed the time_to_failure color to white. Now it's possible to see a peak in acoustic data before each earthquake:

# In[ ]:


single_timeseries(629145000, step=1000, title="All training data", color2='white')


# This is not a perfect rule though: there are some peaks far from earthquakes (e.g. between the 14º and 15º). Another interesting thing to check is the time between these high levels of seismic signal and the earthquakes. I'm considering any acoustic data with **absolute value greater than 500 as a high level:**

# In[ ]:


peaks = train[train.acoustic_data.abs() > 500]
peaks.time_to_failure.describe()


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Cumulative distribution - time to failure with high signal")
ax = sns.distplot(peaks.time_to_failure, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))


# **More than 80% of high acoustic values are around 0.31 seconds before an earthquake!**

# <h2>Time between measurements</h2>
# 
# Let's check the difference between the time to failure for each row and the previous (for the first 5M rows).

# In[ ]:


diff = train.iloc[:5000000].time_to_failure.diff()
counts = diff.value_counts()

plt.figure(figsize=(10,5))
plt.title("Time to failure difference")
ax = sns.scatterplot(data=diff.values, color='red')


# We can see three different ranges of values. Two are around 0.001, while the other is very close to zero. This agrees with the first timeseries plot, where we had those steps. Let's check the distribution:

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Time to failure difference")
ax = sns.kdeplot(data=diff.values)


# Indeed most values are less than 0.0001, but there are a few around 0.0010, which means that there are different time steps. Note that I didn't include any earthquake here since the first is only around index 7 million

# <h2>3. Test data</h2>
# 
# Each file is considered an experiment with 150,000 values for acoustic data (single column). Let's have a look at the first test file:

# In[ ]:


pd.set_option("display.precision", 4)
test1 = pd.read_csv('../input/test/seg_37669c.csv', dtype='int16')
print(test1.describe())
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
ax = sns.distplot(test1.acoustic_data, label='seg_37669c', kde=False)


# Now the distribution for 10 files choosen at random:

# In[ ]:


fig, axis = plt.subplots(5, 2, figsize=(12,14))
shuffle(test_folder_files)
xrow = xcol = 0
for f in test_folder_files[:10]:
    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')
    ax = sns.distplot(tmp.acoustic_data, label=f.replace('.csv',''), ax=axis[xrow][xcol], kde=False)
    if xcol == 0:
        xcol += 1
    else:
        xcol = 0
        xrow += 1


# Timeseries (index as x-axis) for the same ten files:

# In[ ]:


fig, axis = plt.subplots(5, 2, figsize=(12,14))
xrow = xcol = 0
for f in test_folder_files[:10]:
    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')
    ax = sns.lineplot(data=tmp.acoustic_data.values,
                      label=f.replace('.csv',''),
                      ax=axis[xrow][xcol],
                      color='orange')
    if xcol == 0:
        xcol += 1
    else:
        xcol = 0
        xrow += 1


# <h2>4. Statistics for chunks</h2>
# 
# <h2>Rolling mean</h2>
# Most kernels are grouping the training data every 150,000 rows and calculating statistics about that chunk. The next plot shows the **mean absolute value for every 150,000 data points** and the** time to failure for the last point within each chunk**.

# In[ ]:


rolling_mean = []
rolling_std = []
last_time = []
init_idx = 0
for _ in range(4194):  # 629M / 150k = 4194
    x = train.iloc[init_idx:init_idx + 150000]
    last_time.append(x.time_to_failure.values[-1])
    rolling_mean.append(x.acoustic_data.abs().mean())
    rolling_std.append(x.acoustic_data.abs().std())
    init_idx += 150000
    
rolling_mean = np.array(rolling_mean)
last_time = np.array(last_time)

# plot rolling mean
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.suptitle('Mean for chunks with 150k samples of training data', fontsize=14)

ax2 = ax1.twinx()
ax1.set_xlabel('index')
ax1.set_ylabel('Acoustic data')
ax2.set_ylabel('Time to failure')
p1 = sns.lineplot(data=rolling_mean, ax=ax1, color='orange')
p2 = sns.lineplot(data=last_time, ax=ax2, color='gray')


# Removing high values (mean < 8):

# In[ ]:


# plot rolling mean
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.suptitle('Mean (< 8) for chunks of 150k samples', fontsize=14)

ax2 = ax1.twinx()
ax1.set_xlabel('index')
ax1.set_ylabel('Acoustic data')
ax2.set_ylabel('Time to failure')
p1 = sns.lineplot(data=rolling_mean[rolling_mean < 8], ax=ax1, color='orange')
p2 = sns.lineplot(data=last_time, ax=ax2, color='gray')


# We can see that the mean absolute value is increasing as we get closer to an earthquake.

# <h2>Standard deviation</h2>
# 
# Now the same plot for the standard deviation in each chunk:

# In[ ]:


frame = pd.DataFrame({'rolling_std': rolling_std, 'time': np.around(last_time, 1)})
s = frame.groupby('time').rolling_std.mean()
s = s[s < 20]  # remove one outlier
plt.figure(figsize=(10, 5))
plt.title("Std for chunks with 150k samples of training data")
plt.xlabel("Time to failure")
plt.ylabel("Acoustic data")
ax = sns.lineplot(x=s.index, y=s.values)


# The standard deviation is also higher for chunks that are closer to an earthquake (in general).
