# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:20:32 2018

@author: MPandey1
"""
'''
 Department of Buildings Job Application Filings dataset from the NYC Open Data portal. 
 This dataset consists of job applications filed on January 22, 2017.
'''
# Import pandas
import pandas as pd
import numpy as np

df=pd.read_csv('/Users/sajjadsalaria/Programming/ML-WorkShop/Data Pre-processing/dob_job_application_filings_subset.csv')
# Read the file into a DataFrame: df


# Print the head of df
print(df.head(2))

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)


##.describe() method to calculate summary statistics of your data
print(df.describe())

df.info()
##.value_counts() method, which returns the frequency counts for each unique value in a column!
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))

# Print the value_counts for 'State'
print(df['State'].value_counts(dropna=True))


# Print the value counts for 'Site Fill'
print(df['Site Fill'].value_counts(dropna=True))

df.dtypes
df.get_dtype_counts()


##seperating string and numerical columns

df_string=df.select_dtypes(include=['object'])
df_numerical=df.select_dtypes(exclude=['object'])

df_string.shape
df_numerical.shape

df_string.info()

##ensuring all categorical variables in a DataFrame are of type category reduces memory usage.
df_string['Job Type'] = df_string['Job Type'].astype('category')

##converting datatypes

tips = pd.read_csv('/Users/sajjadsalaria/Programming/ML-WorkShop/Data Pre-processing/tips.csv')
print(tips.info())

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

##for numeric directly
#df['column'] = df['column'].to_numeric()


# Print the info of tips
print(tips.info())
##numerica data conversion

tips.tip = tips.tip.astype('object')

#tips.tip = tips.tip.astype('float')
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce') ##incase some char values result into nan


def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'
    elif gender == 'Male':
        return 1
        
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head())    

'''lambda functions. Instead of using the def syntax that you used in the previous exercise, 
lambda functions let you make simple, one-line functions.
''' 

'''
def my_square(x):
    return x ** 2

df.apply(my_square)

The equivalent code using a lambda function is:

df.apply(lambda x: x ** 2)

'''
##apply function using lambda
tips['tip1']=tips.tip.apply(lambda x : x+1)

   
# Write the lambda function using replace
tips.tip = tips.tip.astype('object')
tips['day1'] = tips.day.apply(lambda x: x.replace('Sun', 'Fri'))

##drop duplicates
tips.shape[0]

#last row
tips.iloc[-1:,]

tips.append(tips.iloc[-1:,])

tips.drop_duplicates()

#missing values
tips.info()

tips.isnull().any()

tips.isnull().values.any()

tips.isnull().sum().sum()

tips.tip=tips.tip.fillna(tips.tip.mean())
    

##better option
tips.loc[tips.day=='Sun'].tip
len(tips.loc[tips.day=='Sun'].tip)
mn=tips.loc[tips.day=='Sun'].tip.mean()
tips.tip=tips.tip.fillna(mn)        
        

##Validation

##Holdout
from sklearn.cross_validation import train_test_split
train, validation = train_test_split(tips, test_size=0.50, random_state = 5)




from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.split(tips)

from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[4,5],[5,6]]) # create an array
y = np.array([1,1,2,2,2,1])
loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X):
        print("train:", train_index, "validation:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        
        
from sklearn.model_selection import KFold # import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[4,5],[5,6]]) # create an array
y = np.array([1,1,2,2,2,1]) # Create another array
kf = KFold(n_splits=3 ,random_state=None, shuffle=False) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)        
        
for train_index, test_index in kf.split(X):
 print('TRAIN:', train_index, 'TEST:', test_index)
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
 
 
 
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, random_state=None)
# X is the feature set and y is the target
for train_index, test_index in skf.split(X,y): 
    print('Train:', train_index, 'Validation:', test_index) 
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]

