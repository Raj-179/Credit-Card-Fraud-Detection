# In[81]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'Class' in the data file
tpot_data = pd.read_csv('D:/Desktop/Sem-8/Python Oep/creditcard.csv',nrows=1400,dtype=np.float64)
features = tpot_data.drop('Class', axis=1).values
training_features, testing_features, training_target, testing_target =             train_test_split(features, tpot_data['Class'].values, random_state=42)


# In[82]:


# Average CV score on the training set was:0.9979695431472081
exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=11, min_samples_split=12)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


# In[83]:


results


# In[84]:


one = pd.DataFrame(results,testing_target)
one.tail(100)


# In[71]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testing_target, results)
cm


# In[89]:


print('Please note\n0=>Not Fraud\n1=>Fraud')
one = pd.DataFrame(results)
print(testing_target[22])
print(testing_target[1])


# In[88]:


results1 = exported_pipeline.predict([testing_features[22]])
results2 = exported_pipeline.predict([testing_features[1]])
print("Prediction result :",results1)
print("Prediction result :",results2)


# In[65]:


import pandas as pd
import matplotlib.pyplot as plt

count_classes = pd.value_counts(results, sort = True).sort_index()
count_classes.plot(kind = 'pie')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[66]:


count_classes = pd.value_counts(results, sort = True)
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
