'''
Group-11
OEP Definition: Detect credit card fraud using TPOT.
(TPOT is a Python Automated Machine Learning tool that
optimizes machine learning pipelines using genetic programming)
'''
#!/usr/bin/env python
# coding: utf-8

# In[20]:
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 7)
import warnings
warnings.filterwarnings('ignore')

# In[21]:
dataset = pd.read_csv('D:/Desktop/Sem-8/Python Oep/creditcard.csv')
dataset.head()

# In[22]:
dataset.shape

# In[23]:
dataset['Class'].value_counts

# In[42]:
X = dataset.iloc[:1400,0:29].values
y = dataset.iloc[:1400,30].values
print(type(X))

# In[25]:
X.shape

# In[26]:
y.shape

# In[27]:
dataset['Class'].value_counts()

# In[28]:
from sklearn.model_selection import train_test_split

# In[29]:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=43)

# In[30]:
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[31]:
from tpot import TPOTClassifier

# In[32]:
tpot = TPOTClassifier(generations=5, population_size=20,
                          offspring_size=None, mutation_rate=0.9,
                          crossover_rate=0.1,
                          scoring='accuracy', cv=5,
                          subsample=1.0, n_jobs=1,
                          max_time_mins=None, max_eval_time_mins=5,
                          random_state=None, config_dict=None,
                          warm_start=False,
                          memory=None,
                          periodic_checkpoint_folder=None,
                          early_stop=None,
                          verbosity=3,
                          disable_update_check=False)

# In[33]:
tpot.fit(X_train, y_train)

# In[34]:
tpot.score(X_test, y_test)

# In[35]:
print(tpot.score(X_test, y_test))

# In[36]:
tpot.export('D:/Desktop/Sem-8/Python Oep/tpot_pipeline.py')

'''
# In[ ]:
Considered notes=>
1. If we done run tpot for long enough, it may not find the best possible pipeline for our dataset.

2. It may not even find any suitable pipeline at all, and a runtime error occurs

3.Typically TPOT runs for hours to days to finish, but you can always interrupt the run partway through and see the best
results so far.

4.It provides a warm_start parameter that lets us restart a TPOT run from where it left off.

5.If we are working with a reasonably complex dataset and run TPOT for a short amount of time, different TPOT runs may
result in different pipeline recommendations.

6. TPOT\'s optimization algorithm is stochastic in nature- i.e.it uses randomness to search the possible pipeline space.

7. when two TPOT runs recommend different pipelines, this means that TPOT runs didnt converge due to lack of time or 
that multiple pipelines perform more-or-less the same on your dataset.

8. max_time_mins: how many minutes TPOT has to optimize the pipeline. If not None,
this setting will override the generations parameter and allow TPOT to run until max_time_mins
minutes elapsed.

9. use this in the classifier max_time_mins=None, max_eval_time_mins=5
'''

# In[25]:
#get_ipython().system('pip install xgboost-0.82-cp37-cp37m-win32.whl')
