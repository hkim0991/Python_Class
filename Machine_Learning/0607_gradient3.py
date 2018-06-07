# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:11:55 2018

@author: kimi
"""
# Import libraries & funtions -------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create random values for x --------------------------------------------------
x1 = np.random.uniform(low=-1.0, high=1, size= 100)
x2 = np.random.uniform(low=-1.0, high=1, size= 100)


# create signoid function -----------------------------------------------------
def sigmoid(n):
    return 1/(1+np.exp(-n))

print(sigmoid(-9))
print(sigmoid(100))
print(sigmoid(9))


# Plot the sigmoid function ---------------------------------------------------
xx = np.linspace(start=-100.0, stop=100.0)
yy = sigmoid(xx)

plt.plot(xx, yy)


# Gradient Descent Algorithm Application --------------------------------------
y = x1 * 0.3 + x2 * 0.5 + 0.5
y = y >= 0.3  # to make the target value class 


num_epoch = 100

w1 = np.random.uniform(low = 0.0, high = 1.0)
w2 = np.random.uniform(low = 0.0, high = 1.0)

b = np.random.uniform(low = 0.0, high = 1.0)

for epoch in range(num_epoch):
    y_predict = (x1 * w1) + (x2 * w2) + b
    y_predict = sigmoid(y_predict)
    
    predict = (y_predict >= 0.5)
    actual = y
    
    error = (predict != actual).mean()
    if epoch % 10 == 0:
        print("{0} error = {1}".format(epoch,error))
    
    w1 = w1 - ((y_predict-y) * x1).mean()
    w2 = w2 - ((y_predict-y) * x2).mean()
    b = b - (y_predict - y).mean()
    
print("--------" * 10)
print("{0} error = {1}".format(epoch,error))

