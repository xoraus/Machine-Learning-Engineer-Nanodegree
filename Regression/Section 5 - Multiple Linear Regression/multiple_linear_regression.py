# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/mpandey1/Desktop/ML using Python Training/day4/Section 5 - Multiple Linear Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# regression coefficients 
print('Coefficients: \n', regressor.coef_) 

print(regressor.intercept_)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred)#mse

print(np.sqrt(mean_squared_error(y_test, y_pred))) #rmse

# print the R-squared value for the model
regressor.score(X_test, y_test)#rsquare

sst=sum(np.power(y_test-statistics.mean(y_test),2))

sse=sum(np.power((y_test-y_pred),2))


rsquared=(sst-sse)/sst
adjusted_r_squared = 1 - (1-rsquared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)  


from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test) 
    
mean_squared_error(y_test,pred)#mse 

ridgeReg.score(X_test, y_test) 



from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.8, normalize=True)

lassoReg.fit(X_train,y_train)

predl = lassoReg.predict(X_test)

mean_squared_error(y_test,predl)#mse 

lassoReg.score(X_test, y_test)



                 

                  