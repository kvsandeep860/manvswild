#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[58]:


df=pd.read_csv("C:\\Users\\kvsan\\Desktop\\data analysis\\Student_Performance.csv")


# In[60]:


df.head()


# In[61]:


df.shape


# In[63]:


df.info()


# In[65]:


df.isnull().sum() ## no null values


# In[70]:


df.duplicated().sum() ## 127 duplicates


# In[71]:


df.drop_duplicates(inplace=True)


# In[72]:


df.head(2)


# In[94]:


plt.figure(figsize=(10,8))
plt.subplot(3,2,1)
sns.histplot(df["Hours Studied"])
plt.subplot(3,2,2)
sns.histplot(df["Previous Scores"])
plt.subplot(3,2,3)
sns.histplot(df["Sleep Hours"])
plt.subplot(3,2,4)
sns.histplot(df["Sample Question Papers Practiced"])
plt.subplot(3,2,5)
df["Extracurricular Activities"].value_counts().plot(kind="pie",autopct="%2.2f")
plt.tight_layout()
plt.show()


# In[96]:


df.head(2)


# In[95]:


from sklearn.model_selection import train_test_split


# In[101]:


x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,0:5],df.iloc[:,-1],test_size=0.2,random_state=2)


# ## One Hot Encoding

# In[109]:


x_train_trans=pd.get_dummies(x_train,columns=["Extracurricular Activities"])
x_test_trans=pd.get_dummies(x_test,columns=["Extracurricular Activities"])


# In[113]:


x_train_trans.head()


# ## With Considering Extracurricular Activities

# In[116]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[117]:


lr.fit(x_train_trans,y_train)


# In[118]:


lr.coef_


# In[119]:


lr.intercept_


# In[121]:


lr.predict(x_test_trans).shape


# In[124]:


from sklearn.metrics import r2_score 
r2_score(y_test,lr.predict(x_test_trans))


# ## With Considering Extracurricular Activities

# In[148]:


lr1=LinearRegression()


# In[153]:


lr1.fit(x_train,y_train)


# In[155]:


lr1.predict(x_test).shape


# In[156]:


r2_score(y_test,lr1.predict(x_test))


# In[157]:


## r2 score with and with out extracaricular activities is same which means extra ciricular activities has no impact


# In[165]:


sns.violinplot(df,x="Extracurricular Activities",y="Performance Index")
plt.show()


# In[166]:


## the distribution is all most the same


# In[167]:


sns.boxplot(df,x="Extracurricular Activities",y="Performance Index")
plt.show()


# In[ ]:




