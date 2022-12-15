#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns


# In[2]:


df_drought = pd.read_csv('droughtdatacsv.csv')
# droughtdata = df[df['Year']!=2022].copy()
df_snow = pd.read_csv('Snowdatacsv.csv')


# In[3]:


df_drought.shape


# In[4]:


df_drought.sort_values(by=['Year', 'Month', 'Day'], inplace = True)


# In[5]:


df_drought.head()


# In[6]:


df_drought.isna().sum()


# In[7]:


#aggregate drought data to month level for modeling
# df_drought_mo = df_drought.groupby(['Year', 'Month']).agg({'D0':'mean','D1':'mean','D2':'mean','D3':'mean','D4':'mean'}).reset_index()
df_drought_mo = df_drought.groupby(['Year', 'Month'])[['D0', 'D1', 'D2', 'D3', 'D4']].aggregate('mean').reset_index()


# In[8]:


for f in ['D0', 'D1', 'D2', 'D3', 'D4']:
    perct_0 = round(df_drought_mo[df_drought_mo[f] == 0].shape[0]/df_drought_mo.shape[0] * 100)
    perct_100 = round(df_drought_mo[df_drought_mo[f] == 100].shape[0]/df_drought_mo.shape[0] * 100)
    print(f'{f} has {perct_0} percentage of records with value of 0, and {perct_100} percentage of records with value of 100')


# In[9]:


for f in ['D0', 'D1', 'D2', 'D3', 'D4']:
    df_drought_mo.hist(column=f)


# In[10]:


df_drought_mo.tail(10)


# In[11]:


def make_lags(df, feature, lags):
    for i in range(1, lags+1):
        new_name = feature + '_lag_' + str(i)
#         print(new_name)
        df[new_name] = df[feature].shift(i)


# In[12]:


for f in ['D1', 'D2', 'D3']:
    make_lags(df_drought_mo, f, 12)


# In[13]:


df_drought_mo.head(20)


# In[14]:


df_drought_mo.tail(20)


# In[15]:


# 2022 snow data is not available (0)
df_snow = df_snow[df_snow['Year']!=2022].copy()


# In[16]:


df_snow.shape


# In[17]:


df_snow['SWE_Prior'] = df_snow.SWE.shift(1)


# In[18]:


df_snow.head()


# In[19]:


df_snow['SWE_Prior_MA'] = df_snow['SWE_Prior'].rolling(3).mean()


# In[20]:


df_snow.head()


# In[21]:


df_snow = df_snow.dropna()


# In[22]:


df_snow.head()


# In[23]:


df_snow_drought = pd.merge(df_drought_mo, df_snow, how='inner', on=['Year'])


# In[24]:


df_snow_drought.head(20)


# In[25]:


df_snow_drought.tail(20)


# In[26]:


df_snow_drought.shape


# In[27]:


df_snow_drought.isna().sum()


# In[28]:


#change Month to nominal data to use as seasonality
df_snow_drought['Month'] = df_snow_drought['Month'].astype('category')


# In[29]:


for d in ['D1', 'D2', 'D3']:
    corr_swe = df_snow_drought[d].corr(df_snow_drought['SWE_Prior'])
    corr_swe_ma = df_snow_drought[d].corr(df_snow_drought['SWE_Prior_MA'])          
    print(f'Correlation between SWE and {d} is {corr_swe}')
    print(f'Correlation between SWE moving average and {d} is {corr_swe_ma}')


# In[30]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_snow_drought[['D1', 'D2', 'D3', 'SWE_Prior', 'SWE_Prior_MA']].corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# # Modeling D1

# In[32]:


#discard 0's
#df_snow_drought = df_snow_drought[df_snow_drought['D1']!=0].copy()
#df_snow_drought.shape


# In[33]:


df_snow_drought.head()


# ### D1 one-year ahead model

# In[34]:


# one-year ahead model
X = df_snow_drought[['Month', 'SWE_Prior', 'SWE_Prior_MA', 'D1_lag_12']]
y = df_snow_drought['D1']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)


# In[36]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[37]:


lr_D1 = LinearRegression()
lr_D1.fit(X_train, y_train)


# In[38]:


predictions = lr_D1.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D1 one-quarter ahead model

# In[39]:


X = df_snow_drought[['Month', 'SWE_Prior', 'SWE_Prior_MA', 'D1_lag_3', 'D1_lag_4', 'D1_lag_5', 'D1_lag_6', 'D1_lag_9', 'D1_lag_12']]
y = df_snow_drought['D1']


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[41]:


lr_D1 = LinearRegression()
lr_D1.fit(X_train, y_train)


# In[42]:


predictions = lr_D1.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D1 one-month ahead model

# In[43]:


X = df_snow_drought[['Month', 'SWE_Prior', 'SWE_Prior_MA', 'D1_lag_1', 'D1_lag_3', 'D1_lag_6', 'D1_lag_9', 'D1_lag_12']]
y = df_snow_drought['D1']


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[45]:


lr_D1 = LinearRegression()
lr_D1.fit(X_train, y_train)


# In[46]:


predictions = lr_D1.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ## Modeling D2

# In[47]:


#discard 0's
#df_snow_drought = df_snow_drought[df_snow_drought['D2']!=0].copy()
#df_snow_drought.shape


# In[48]:


df_snow_drought.head()


# ### D2 one-year ahead model

# In[49]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D2_lag_12']]
y = df_snow_drought['D2']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[51]:


lr_D2 = LinearRegression()
lr_D2.fit(X_train, y_train)


# In[52]:


predictions = lr_D2.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D2 one-quarter ahead model

# In[53]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D2_lag_3', 'D2_lag_4', 'D2_lag_5', 'D2_lag_6', 'D2_lag_9', 'D2_lag_12']]
y = df_snow_drought['D2']


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[55]:


lr_D2 = LinearRegression()
lr_D2.fit(X_train, y_train)


# In[56]:


predictions = lr_D2.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D2 one-month ahead model

# In[57]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D2_lag_1', 'D2_lag_2', 'D2_lag_3']]
y = df_snow_drought['D2']


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[59]:


lr_D2 = LinearRegression()
lr_D2.fit(X_train, y_train)


# In[60]:


predictions = lr_D2.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# In[ ]:





# ## Modeling D3

# In[61]:


#discard 0's
#df_snow_drought = df_snow_drought[df_snow_drought['D3']!=0].copy()
#df_snow_drought.shape


# In[62]:


df_snow_drought.head()


# ### D3 one-year ahead model

# In[63]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D3_lag_12']]
y = df_snow_drought['D3']


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[65]:


lr_D3 = LinearRegression()
lr_D3.fit(X_train, y_train)


# In[66]:


predictions = lr_D3.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D3 one-quarter ahead model

# In[67]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D3_lag_3', 'D3_lag_4', 'D3_lag_5', 'D3_lag_6', 'D3_lag_9', 'D3_lag_12']]
y = df_snow_drought['D3']


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[69]:


lr_D3 = LinearRegression()
lr_D3.fit(X_train, y_train)


# In[70]:


predictions = lr_D3.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# ### D3 one-month ahead model

# In[71]:


X = df_snow_drought[['Month', 'SWE_Prior_MA', 'SWE_Prior', 'D3_lag_1', 'D3_lag_2', 'D3_lag_3']]
y = df_snow_drought['D3']


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[73]:


lr_D3 = LinearRegression()
lr_D3.fit(X_train, y_train)


# In[74]:


predictions = lr_D3.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('The r2 is: ', r2)
print('The rmse is: ', rmse)


# In[ ]:




