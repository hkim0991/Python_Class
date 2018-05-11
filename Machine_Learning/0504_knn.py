# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:20:49 2018

@author: kimi
"""

# Import libraries & funtions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib
import sklearn

from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Check the version of the software
import sys
print('Python version: {}'.format(sys.version))
print('pandas version: {}'.format(pd.__version__))
print('matplotlib version: {}'.format(matplotlib.__version__))
print('NumPy version: {}'.format(np.__version__))
print('SciPy version: {}'.format(sp.__version__))
print('scikit-learn version: {}'.format(sklearn.__version__))


# Load dataset
from sklearn.datasets import load_iris


# EDA
iris_dataset = load_iris()
iris_dataset.keys()
iris_dataset.values()
iris_dataset.items()

iris_dataset['data']
iris_dataset['feature_names']
iris_dataset['target_names']


# Training/Testing data separation 
print('feature_names : {}'.format(iris_dataset['feature_names']))
print('target_names : {}'.format(iris_dataset['target_names']))

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset.data,      # input features 
        iris_dataset.target,    # output features
        test_size=0.1,          # test dataset proportion
        train_size=0.9,         # train dataset proportion 
        random_state=0)

print('Size of X_train : {}'.format(X_train.shape))
print('Size of X_test : {}'.format(X_test.shape))
print('Size of y_train : {}'.format(y_train.shape))
print('Size of y_test : {}'.format(y_test.shape))


# Modeling 
!pip install mglearn 
import mglearn

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# Testing the model with new data
X_new = np.array([[5, 2.9, 1.0, 2]]) 
print(X_new.shape)   # make an array in a shape as input features

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("The name of predicted target feature: {}".format(
        iris_dataset['target_names'][prediction]))


# Model evaluation
y_pred =knn.predict(X_test)
print("Prediction: {}".format(y_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(y_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(knn.score(X_test, y_test)))

print("Accuracy of the prediction:{}".format(np.sum(y_pred==y_test)))

