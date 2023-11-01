#!/usr/bin/env python
# coding: utf-8

# # Python

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier


# In[7]:


data = pd.read_csv(r"C:\Users\mca1\Downloads\creditcard.csv")
print(data.shape)
data.head()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[15]:


fraudulent = data[data['Class'] == 1]
non_fraudulent = data[data['Class'] == 0]
data["Amount"].plot()
plt.show()


# In[16]:


def sample_equally(data,col):
    min_sam = data[col].value_counts().min()
    sample_idxs = []
    for x in 0,1:
        sample_idxs += list(data[data[col] == x].sample(min_sam + (300
if x == 0 else 0)).index)
    return data.loc[sample_idxs]

data_ = sample_equally(data,"Class")
print(data_.shape)
data_["Class"].value_counts().plot(kind="bar")
plt.show()
(1284, 31)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(data_.drop(columns=["Time","Class"]).values,data_["Class"],shuffle=True,random_state=69420)
model = RandomForestClassifier(21)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}")
print(classification_report(y_test,y_pred))


# In[18]:


print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))


# In[29]:


import pickle
pickle.dump(model,open("cc-rf-1-11-23.pkl","wb"))


# In[ ]:




