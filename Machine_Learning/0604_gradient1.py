# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:39:22 2018

@author: kimi
"""

import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
%matplotlib inline

boston = load_boston()
X = boston['data']
y = boston['target']
print(boston['DESCR'])
print(boston['feature_names'])

data = pd.DataFrame(X, columns=boston['feature_names'])
data.head()
data['PRICE'] = y


x1 = data['CRIM'].values
x2 = data['ZN'].values

w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)


num_epoch = 10000
learning_rate = 0.000005

for epoch in range(num_epoch):
    y_predict = (x1 * w1) + (x2 * w2)
    
    # error check : actual value - predicted value
    error = np.abs(y_predict - y).mean()
    if error < 3:
        break
    
    w1 = w1 - learning_rate * ((y_predict-y) * x1).mean()
    w2 = w2 - learning_rate * ((y_predict-y) * x2).mean()
    
    if epoch % 1000 == 0:
        print("{0:5} error = {1:.5f}".format(epoch, error))

print("----" * 10)
print("{0:5} error = {1:.5f}".format(epoch, error))


# when there are five features 
x1 = data['CRIM'].values
x2 = data['ZN'].values
x3 = data['INDUS'].values
x4 = data['CHAS'].values
x5 = data['NOX'].values

w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)
w3 = np.random.uniform(low=0.0, high=1.0)
w4 = np.random.uniform(low=0.0, high=1.0)
w5 = np.random.uniform(low=0.0, high=1.0)

num_epoch = 10000
learning_rate = 0.000005

for epoch in range(num_epoch):
    y_predict = (x1 * w1) + (x2 * w2) + (x3 * w3) + (x4 * w4) + (x5 * w5)
    
    # error check : actual value - predicted value
    error = ((y_predict -y) * (y_predict-y) ).mean()
    if error < 7:
        break
    
    w1 = w1 - learning_rate * ((y_predict-y) * x1).mean()
    w2 = w2 - learning_rate * ((y_predict-y) * x2).mean()
    w3 = w3 - learning_rate * ((y_predict-y) * x3).mean()
    w4 = w4 - learning_rate * ((y_predict-y) * x4).mean()
    w5 = w5 - learning_rate * ((y_predict-y) * x5).mean()
    
    if epoch % 1000 == 0:
        print("{0:5} error = {1:.5f}".format(epoch, error))

print("----" * 10)
print("{0:5} error = {1:.5f}".format(epoch, error))



# when there are five features 
x1 = data['CRIM'].values
x2 = data['ZN'].values
x3 = data['INDUS'].values
x4 = data['CHAS'].values
x5 = data['NOX'].values
x6 = data['RM'].values
x7 = data['AGE'].values
x8 = data['DIS'].values
x9 = data['RAD'].values
x10 = data['TAX'].values
x11 = data['PTRATIO'].values
x12 = data['B'].values
x13 = data['LSTAT'].values

w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)
w3 = np.random.uniform(low=0.0, high=1.0)
w4 = np.random.uniform(low=0.0, high=1.0)
w5 = np.random.uniform(low=0.0, high=1.0)
w6 = np.random.uniform(low=0.0, high=1.0)
w7 = np.random.uniform(low=0.0, high=1.0)
w8 = np.random.uniform(low=0.0, high=1.0)
w9 = np.random.uniform(low=0.0, high=1.0)
w10 = np.random.uniform(low=0.0, high=1.0)
w11 = np.random.uniform(low=0.0, high=1.0)
w12 = np.random.uniform(low=0.0, high=1.0)
w13 = np.random.uniform(low=0.0, high=1.0)


num_epoch = 10000
learning_rate = 0.000005

for epoch in range(num_epoch):
    y_predict = (x1 * w1) + \
                (x2 * w2) + \
                (x3 * w3) + \
                (x4 * w4) + \
                (x5 * w5) + \
                (x6 * w6) + \
                (x7 * w7) + \
                (x8 * w8) + \
                (x9 * w9) + \
                (x10 * w10) + \
                (x11 * w11) + \
                (x12 * w12) + \
                (x13 * w13)
    
    # error check : actual value - predicted value
    #error = ((y_predict -y) * (y_predict-y) ).mean()
    error = np.abs(y_predict -y).mean()
    if error < 7:
        break
    
    w1 = w1 - learning_rate * ((y_predict-y) * x1).mean()
    w2 = w2 - learning_rate * ((y_predict-y) * x2).mean()
    w3 = w3 - learning_rate * ((y_predict-y) * x3).mean()
    w4 = w4 - learning_rate * ((y_predict-y) * x4).mean()
    w5 = w5 - learning_rate * ((y_predict-y) * x5).mean()
    w6 = w6 - learning_rate * ((y_predict-y) * x6).mean()
    w7 = w7 - learning_rate * ((y_predict-y) * x7).mean()
    w8 = w8 - learning_rate * ((y_predict-y) * x8).mean()
    w9 = w9 - learning_rate * ((y_predict-y) * x9).mean()
    w10 = w10 - learning_rate * ((y_predict-y) * x10).mean()
    w11 = w11 - learning_rate * ((y_predict-y) * x11).mean()
    w12 = w12 - learning_rate * ((y_predict-y) * x12).mean()
    w13 = w13 - learning_rate * ((y_predict-y) * x13).mean()
    
    if epoch % 1000 == 0:
        print("{0:5} error = {1:.5f}".format(epoch, error))

print("----" * 10)
print("{0:5} error = {1:.5f}".format(epoch, error))


























