# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:27:37 2018

@author: MPandey1
"""
import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(1,11 ,2) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
#plt.plot(x,y,'y--') 
plt.plot(x,y,'ro')
plt.grid(True)
plt.axis([0, 25, 0, 25])
#xlim([])
#ylim([])
plt.text(6, 15, 'collision pt.')
plt.plot(x, y, 'r--', x, x**2, 'bs', y, y**3, 'g^')
plt.show() 


#plt.cla()
#plt.clf()
#plt.close()
# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 3 * np.pi, 0.1) 
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
plt.show()

###subplot 
   
# Compute the x and y coordinates for points on sine and cosine curves 
x = np.arange(0, 3 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
#plt.subplot(nrow,ncols,index)
plt.subplot(2, 2, 1)
   
# Make the first plot 
plt.plot(x, y_sin) 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 2, 2) 
plt.plot(x, y_cos) 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()


###bar graph


city = ['NY','London','Delhi'] 
population = [30,33,50] #in millions 
#plt.bar(city, population, align = 'center') 
             
x2 = [6,9,11] 
y2 = [6,15,7] 

plt.bar(x2, y2, color = 'g', align = 'edge' , width=1) 
plt.title('Bar graph') 
#plt.ylabel('Population') 
#plt.xlabel('City')  

plt.show()

import seaborn as sns

sns.barplot(x=city, y=population)
plt.show()


def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(city))
    plt.bar(index, population)
    plt.xlabel('City', fontsize=25)
    plt.ylabel('Population', fontsize=25)
    plt.xticks(index, city, fontsize=15)
    plt.title('Bar graph')
    plt.show()
##you see for xticks I used both index and label. '
##Labels will be placed on each tick that is generated due to index sequence


##histogram
##Categorical VS Discrete variable
##graphical representation of the frequency distribution of data
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
np.histogram(a,bins = [0,20,40,60,80,100]) 
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100]) 
print(hist) 
print(bins) 


plt.hist(a, bins = [0,20,40,60,80,100]) 
plt.title("histogram") 
plt.show()

##Scatter plot
#Legends:provide lables for points and curves
import pandas as pd
iris=pd.read_csv('C:/Users/mpandey1/Desktop/ML using Python Training/iris1.csv')
setosa=iris.loc[iris['Species']=='setosa']
versicolor=iris.loc[iris['Species']=='versicolor']
virginica=iris.loc[iris['Species']=='virginica']


plt.scatter(setosa['Sepal.Length'],setosa['Sepal.Width'],marker='o', color='red', label='setosa')
plt.scatter(versicolor['Sepal.Length'],versicolor['Sepal.Width'],marker='o', color='green', label='versicolor')
plt.scatter(virginica['Sepal.Length'],virginica['Sepal.Width'],marker='o', color='blue', label='virginica')

plt.legend(loc='upper right')
plt.title('Iris data')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
####

