# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:18:31 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
import pandas as pd
from sklearn import preprocessing
from numpy import argmax


# Load dataset ----------------------------------------------------------------
data = { "target": ['spring', 'summer', 'autumn', 'winter', 'spring', 'spring', 
                    'summer', 'autumn', 'winter', 'winter'],
        "work": [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        "wday": [5, 6, 7, 1, 2, 3, 4, 5, 6, 7]}

df = pd.DataFrame(data)


# one_hot_encoding - Label_Encoding -------------------------------------------
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_y =label_encoder.fit_transform(df['target'])
print(train_y)
print(train_y.shape)

train_y = train_y.reshape(len(train_y), 1)
print(train_y.shape)


# one_hot_encoding - one_hot_Encoding -----------------------------------------
train_y = onehot_encoder.fit_transform(train_y)
print(train_y)
print(train_y.shape)
print(train_y)


# one_hot_encoding - inverse_transform ----------------------------------------
train_y[2,:]
argmax(train_y[2,:])
label_encoder.inverse_transform(argmax(train_y[2,:]))



