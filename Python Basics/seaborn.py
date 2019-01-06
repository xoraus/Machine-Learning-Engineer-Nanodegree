# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:25:38 2018

@author: MPandey1
"""

import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns

iris=pd.read_csv('C:/Users/mpandey1/Desktop/ML using Python Training/iris1.csv')
setosa=iris.loc[iris['Species']=='setosa']
versicolor=iris.loc[iris['Species']=='versicolor']
virginica=iris.loc[iris['Species']=='virginica']

sns.barplot(iris['Species'] ,iris['Sepal.Length']  )
plt.show()


sns.distplot(iris['Petal.Length'])
plt.show()



tips =sns.load_dataset('tips')

sns.stripplot(y= 'tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

sns.stripplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

##spreading our strip plot
sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=True)
plt.ylabel('tip ($)')
plt.show()

##swarm plot
sns.swarmplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

sns.swarmplot(x='day', y='tip', data=tips, hue='sex')
plt.show()

sns.swarmplot(x='tip', y='day', data=tips, hue='sex', orient='h')
plt.show()

## violin plot and box plot 
plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.tight_layout()
plt.show()


##Combining plots

sns.violinplot(x='day', y='tip', data=tips, inner=None,color='lightgray')
sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=True)
plt.ylabel('tip ($)')
plt.show()


##jointplot
sns.jointplot(x= 'total_bill', y= 'tip', data=tips)
plt.show()

sns.jointplot(x='total_bill', y= 'tip', data=tips,kind='kde')
plt.show()

sns.pairplot(tips)
plt.show()

sns.pairplot(tips, hue='sex')
plt.show()

sns.heatmap(tips.corr())
plt.show()




