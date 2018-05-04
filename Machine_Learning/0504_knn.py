# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:20:49 2018

@author: 202-22
"""

# import libraries & funtions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib
import sklearn

from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# check the version of the software
import sys
print('Python 버전: {}'.format(sys.version))
print('pandas 버전: {}'.format(pd.__version__))
print('matplotlib 버전: {}'.format(matplotlib.__version__))
print('NumPy 버전: {}'.format(np.__version__))
print('SciPy 버전: {}'.format(sp.__version__))
print('scikit-learn 버전: {}'.format(sklearn.__version__))


# load dataset
from sklearn.datasets import load_iris


# Data exploration & analysis
iris_dataset = load_iris()
iris_dataset.keys()
iris_dataset.values()
iris_dataset.items()

iris_dataset['data']
iris_dataset['target_names']
iris_dataset['feature_names']

print('feature_names : {}'.format(iris_dataset['feature_names']))
print('target_names : {}'.format(iris_dataset['target_names']))

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'],   # input features 
        iris_dataset['target'], # output features
        test_size=0.1,          # test dataset proportion
        train_size=0.9,         # train dataset proportion 
        random_state=0)

print('Size of X_train : {}'.format(X_train.shape))
print('Size of X_test : {}'.format(X_test.shape))
print('Size of y_train : {}'.format(y_train.shape))
print('Size of y_test : {}'.format(y_test.shape))


!pip install mglearn
import mglearn

# from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1.0, 2]]) 
print(X_new.shape) # make an array in a shape as input features

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("The name of predicted target feature: {}".format(
        iris_dataset['target_names'][prediction]))

# model evaluation
y_pred =knn.predict(X_test)
print("Prediction: {}".format(y_pred))

print("Accuracy of the prediction1:{:.3f}".format(np.mean(y_pred==y_test)))
print("Accuracy of the prediction2:{:.3f}".format(knn.score(X_test, y_test)))

print("Accuracy of the prediction:{}".format(np.sum(y_pred==y_test)))









