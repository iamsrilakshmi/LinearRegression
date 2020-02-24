#Let's import the necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error   


#Let's Read the data into pandas Dataframe
myDataFrame = pd.read_csv('MyDataset.csv')

#Let's Analyze the Data
print(myDataFrame.columns)
print(myDataFrame.describe())
print(myDataFrame.head())
print(myDataFrame.dtypes)

#We Select the features: independent variable(x) & dependent variable(y)
#As it is a simple Linear Regression, we select only 1 column(Maths) for x
x = myDataFrame['Maths'].values.reshape(-1,1)
y = myDataFrame['total_marks'].values.reshape(-1,1)


#Now, we will split our data into 2 categories:Training & validation data
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=1)

#Let's build our Machine Learning Model
myLRModel = LinearRegression()

#We train the model using our Training Data
myLRModel.fit(train_x,train_y)

#It's Time to predict the total_marks using our model
prediction = myLRModel.predict(val_x)

#Let's take a look at the values of Slope & Intercept
print(myLRModel.coef_)
print(myLRModel.intercept_)

#Let's evaluate the error
error = mean_squared_error(val_y,prediction)
print(error)

#TO BE CONTINUED...!!!




