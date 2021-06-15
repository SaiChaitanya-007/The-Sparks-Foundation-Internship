#!/usr/bin/env python
# coding: utf-8

# ## Name - Sai Chaitanya K
# ### THE SPARKS FOUNDATION INTERNSHIP
# #### TASK 1
# ##### Supervised ML - Student scores Dataset

# In[46]:


# Importing all libraries necessary for task
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


# Reading data
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported ")

s_data.head(15)


# Let's plot our data points on 2-D graph to see if there is any  direct relatinship between data.We need to plot percentage vs hours

# In[51]:


# Plotting 
s_data.plot(x='Hours', y='Scores', style='.',color='m')  
plt.title('Percentage vs Hours')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From graph,we can see that as no. of hours increased percentage also increased.Therefore,there is a positive linear relation between the number of hours studied and percentage of score.

# #### Preparing the data
# We have to divide the data into input and output.

# In[52]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# Since we have input and output.We will now split it into training and testing sets.

# In[53]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ## *Training the Algorithm*
# We have split our data into training and testing sets, and we will train our algorithm. 

# In[54]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training completed")


# In[55]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### *Predicting*
# Now since we have trained our algorithm, it's time to make some predictions.

# In[56]:


print(X_test) # Testing data-hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[72]:


# Lets Compare Actual vs Predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[73]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,7),color={'Actual':'blue','Predicted':'red'})

plt.show()


# The actual and predicted values don't differ by large difference ,so our algorithm worked good

# In[74]:


# prediction
hours = 9.25
test = np.array([hours])
test = test.reshape(-1,1)
own_predict = regressor.predict(test)
print(f"No of Hours = {hours}")
print(f"Predicted Score = {own_predict}%")


# In[75]:


#Lets findout the  error
from sklearn import metrics

print("Mean Absolute Error : ", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error : ", metrics.mean_squared_error(y_test,y_pred))
print("RMSE : ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Explained Variance Score : ", metrics.explained_variance_score(y_test,y_pred))
print("Maximum Error : ", metrics.max_error(y_test,y_pred))


# ### Thank you
