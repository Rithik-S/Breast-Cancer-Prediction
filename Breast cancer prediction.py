#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv("C:\\Users\\Ritzzz\\Downloads\\cancer.csv")
df.head()


# In[8]:



df.info()


# In[9]:


df.isna().sum()


# In[16]:


df=df.drop(["Unnamed: 32"],axis=1)


# In[17]:


df.hist(figsize=(12,14), bins=30)


# In[18]:


df.columns


# In[19]:


columns = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
for column in columns:
    plt.figure(figsize=(5, 3))
    sns.histplot(data=df, x=column, hue='diagnosis')


# In[23]:


df1 = df.copy()


# In[24]:


df1.drop('id', axis=1, inplace=True)


# In[25]:


plt.figure(figsize=[10,9])
sns.heatmap(df1.corr(), cmap='coolwarm')


# In[26]:


df1['diagnosis'].value_counts()


# In[27]:


X = df1.drop('diagnosis', axis=1)
y = df1['diagnosis']


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# In[29]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[30]:


y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score, confusion_matrix
train_accuracy = accuracy_score(y_train_pred, y_train)

matrix_train = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(matrix_train, annot=True)
plt.show()

print('Train Accuracy: {}'.format(train_accuracy))


# In[32]:


train_accuracy = accuracy_score(y_test_pred, y_test)

matrix_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(matrix_test, annot=True)
plt.show()


# In[33]:


print('Test Accuracy: {}'.format(train_accuracy))


# In[34]:


test_TP = matrix_test[0][0]
test_FP = matrix_test[0][1]
test_FN = matrix_test[1][0]
test_TN = matrix_test[1][1]

test_precision = test_TP/(test_TP+test_FP)
test_recall = test_TP/(test_TP+test_FN)

print('Recall and Precision: {}, {}'.format(round(test_recall, 4), round(test_precision, 4)))


# In[36]:


feature_importance = clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})                         .sort_values(by='Importance', ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(data=feature_importance_df[:10], y='Feature', x='Importance')


# In[ ]:




