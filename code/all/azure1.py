# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 17:49:46 2019

@author: PC-Casa
"""
# In[0]:

import azureml.core

azureml.core.VERSION

# In[1]:

from azureml.core import Workspace

ws = Workspace.create(name='kaggleunibo19',
                      subscription_id='2959a0ad-f6b2-4666-9bd4-59be8e887da1', 
                      resource_group='resourcesgroupkaggleunibo19',
                      create_resource_group=True,
                      location='WestEurope ' # Or other supported Azure region   
        )
        
ws.get_details()

# In[2]:

from azureml.core import Experiment

# Create a new experiment in your workspace.
exp = Experiment(workspace=ws, name='kaggleExpUnibo19')

# Start a run and start the logging service.
run = exp.start_logging()

# Log a single  number.
run.log('my magic number', 42)

# Log a list (Fibonacci numbers).
run.log_list('my list', [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]) 

# Finish the run.
run.complete()










