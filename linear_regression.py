# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:37:07 2018

@author: RAJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_hypothesis = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,y_hypothesis,color="blue")
plt.title("Salary vs Experience ( Training set )")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,y_pred_test,color="blue")
plt.title("Salary vs Experience ( Test set )")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()