# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 09:26:41 2018

@author: kimi
"""
# Import libraries & funtions ------------------------------------------------
import pandas as pd
import numpy as np


# Load dataset ----------------------------------------------------------------
house = pd.read_csv('C:/Users/202-22/Documents/Python - Hyesu/Machine_Learning/data/house_prices/train.csv', engine='python')
house.head()
house.info()
house.isnull().sum()

y = house['SalePrice']
house = house.drop('SalePrice', 1)  # drop row(0) or column(1)


#%%
# Gradient Descent Algorithm Apply 1 ------------------------------------------
x1 = house['MSSubClass'].values
x2 = house['LotArea'].values
x3 = house['OverallQual'].values
x4 = house['BsmtFinSF1'].values
x5 = house['FullBath'].values


w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)
w3 = np.random.uniform(low=0.0, high=1.0)
w4 = np.random.uniform(low=0.0, high=1.0)
w5 = np.random.uniform(low=0.0, high=1.0)


#%%
num_epoch = 30000
learning_rate = 0.0000000001

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


#%%
# Gradient Descent Algorithm Apply 2 ------------------------------------------
x1 = house['MSSubClass'].values
x2 = house['LotArea'].values
x3 = house['OverallQual'].values
x4 = house['BsmtFinSF1'].values
x5 = house['FullBath'].values
x6 = house['Fireplaces'].values
x7 = house['GarageArea'].values
x8 = house['WoodDeckSF'].values
x9 = house['3SsnPorch'].values
x10 = house['ScreenPorch'].values
x11 = house['TotalBsmtSF'].values
x12 = house['PoolArea'].values
x13 = house['MiscVal'].values

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


#%%
num_epoch = 10000
learning_rate = 0.0000000001

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


#%%
# Gradient Descent Algorithm Apply 3 ------------------------------------------

selcol = ['MSSubClass', 'LotArea', 'OverallQual', 'BsmtFinSF1', 'FullBath',
          'Fireplaces', 'GarageArea', 'WoodDeckSF', '3SsnPorch', 
          'ScreenPorch', 'TotalBsmtSF', 'PoolArea', 'MiscVal']

X = house[selcol]
y = house['SalePrice'].values
w = np.random.uniform(low=0.0, high=1.0, size=13)


#%%
num_epoch = 10000
learning_rate = 0.0000000000001

for epoch in range(num_epoch):
    y_predict = X.dot(w)
    error = np.abs(y_predict - y).mean()
    
    w =  w - learning_rate * X.T.dot((y_predict - y)).mean()
   
    if epoch % 1000 == 0:
        print("{0:5} error = {1:.5f}".format(epoch, error))
        print(w)

