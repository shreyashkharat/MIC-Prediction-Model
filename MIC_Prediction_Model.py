#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# ### Importing data and preprocessing

# In[2]:


pop_data = pd.read_csv('/media/shreyashkharat/Storage Drive/Machine Learning/Python/Projects/MIC Prediction Model/insurance.csv', header = 0)
pop_data_fixed = pd.read_csv('/media/shreyashkharat/Storage Drive/Machine Learning/Python/Projects/MIC Prediction Model/insurance.csv', header = 0)
pop_data.info()


# In[3]:


pop_data.describe()


# * The means and medians of non-categorical independent variables are quite close, so we expect that there are no outliers.
# * The distribution of the quantile value seem ordinary, hence there is no skewness in data.

# In[4]:


# Dummy Variable creation
pop_data = pd.get_dummies(pop_data)
pop_data.head()


# In[5]:


del pop_data['smoker_no']
del pop_data['region_southwest']
del pop_data['sex_female']


# ### Graphical Analysis of Data

# In[6]:


sns.jointplot(x = 'age', y = 'charges', data = pop_data)


# * Clearly, the above plot suggests that the variable charges increases with age.
# * But the relation between age and charges isn't linear or for that matter the relation has some other aspects.

# In[7]:


sns.jointplot(x = 'children', y = 'charges', data = pop_data)


# ### Variable sepration, Train Test Split and Correlation Matrix

# In[8]:


x = pop_data.loc[:, pop_data.columns != 'charges']
y = pop_data['charges']


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[10]:


pop_data.corr()


# * There is no multi-colineraity in the above data.

# ## Linear Regression Model

# In[11]:


from sklearn.linear_model import LinearRegression
model_linear = LinearRegression()
model_linear.fit(x_train, y_train)


# In[12]:


print(model_linear.intercept_, model_linear.coef_)


# In[13]:


lm_predict = model_linear.predict(x_test)


# In[14]:


from sklearn.metrics import r2_score
rsq_lm = r2_score(y_test, lm_predict)
rsq_lm


# * The Linear Regression Model has an R-squared value of 0.7999 on above test_set, so the model is modefrately accurate.

# ## Regression Tree

# In[15]:


from sklearn import tree
reg_tree = tree.DecisionTreeRegressor(max_depth = 5)
reg_tree.fit(x_train, y_train)


# ### Predictions and R-square calculation

# In[16]:


reg_predict_train = reg_tree.predict(x_train)
reg_predict_test = reg_tree.predict(x_test)


# In[17]:


from sklearn.metrics import r2_score
rsq_reg_train = r2_score(y_train, reg_predict_train)
rsq_reg_test = r2_score(y_test, reg_predict_test)
rsq_reg_train, rsq_reg_test


# * The Simple Regression Tree has an R-squared value of 0.8426 on above test_set, so the model is highly accurate.

# ### Ploting the Tree

# In[18]:


import graphviz
dot_data_reg = tree.export_graphviz(reg_tree, out_file = None, feature_names = x_train.columns, rounded = True, filled = True)
from IPython.display import Image
import pydotplus
graph_reg = pydotplus.graph_from_dot_data(dot_data_reg)
Image(graph_reg.create_png())


# ### Exporting Graph to jpg file

# In[19]:


dd_reg_export = tree.export_graphviz(reg_tree, out_file = 'graph_reg.dot', feature_names = x_train.columns, rounded = True, filled = True)
# Converting into jpg
from subprocess import call
call(['dot', '-Tjpg', 'graph_reg.dot', '-o', 'graph_reg.jpg', '-Gdpi=600'])


# ### Trees with other Growth Control Parameter

# In[20]:


reg_tree1 = tree.DecisionTreeRegressor(min_samples_split = 50)
reg_tree1.fit(x_train, y_train)


# In[21]:


# R-square calculation.
reg1_predict_test = reg_tree1.predict(x_test)
rsq_reg1_test = r2_score(y_test, reg1_predict_test)
reg1_predict_train = reg_tree1.predict(x_train)
rsq_reg1_train = r2_score(y_train, reg1_predict_train)
rsq_reg1_train, rsq_reg1_test


# In[22]:


# Plot of the tree
dot_data_reg1 = tree.export_graphviz(reg_tree1, out_file = None, feature_names = x_train.columns, rounded = True, filled = True)
graph_reg1 = pydotplus.graph_from_dot_data(dot_data_reg1)
Image(graph_reg1.create_png())


# In[23]:


# Exporting Graph to jpg format
dd_reg1_export = tree.export_graphviz(reg_tree1, out_file = 'graph_reg1.dot', feature_names = x_train.columns, rounded = True, filled = True)
# Converting into jpg
from subprocess import call
call(['dot', '-Tjpg', 'graph_reg1.dot', '-o', 'graph_reg1.jpg', '-Gdpi=600'])


# The Model has got quite accurate with R-squared value 0.8738 on test_set.
# Let's just apply Random Forest Ensemble technique and see the results.

# ## Random Forest

# In[24]:


from sklearn.ensemble import RandomForestRegressor
random_tree = RandomForestRegressor(n_estimators = 5000, n_jobs = -1, random_state = 0, max_features = 4)
random_tree.fit(x_train, y_train)


# In[25]:


# R-squared calculation
random_predict_train = random_tree.predict(x_train)
random_predict_test = random_tree.predict(x_test)
rsq_random_train = r2_score(y_train, random_predict_train)
rsq_random_test = r2_score(y_test, random_predict_test)
rsq_random_train, rsq_random_test


# After applying Random Forest we get the R-squared value as 0.8980 on test_set, implies the model is pretty good.
