# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:26:25 2018

@author: 202-22
"""

# Assesment 1 ------------------------------------------------------
# Question: Using iris dataset, make a knnn model to predict the class of 
# iris data and find out which cv (k_fold) drive the best accuracy. 

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
clf = KNeighborsClassifier(n_neighbors=3, random_state=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=3)

score= []
for i in range(2,51):
   kfold = model_selection.KFold(n_splits=i, random_state=0)
   scores = cross_val_score(clf, iris.data, iris.target, cv=kfold)
   score.append(scores)
   msg = "%s: %f " % (i, scores.mean())
   print(msg)


# Assesment 2 ------------------------------------------------------
# Question: With Boston dataset, make a linear regression model to predict 
# the price of a house and find out which partition rate between train/test 
# drive the best accuracy. 

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


def load_extended_boston():
    boston = load_boston()  # load the dataset
    X = boston.data         # input data
    y = boston.target       # output/target data 
    
    X = MinMaxScaler().fit_transform(boston.data)  # data normalization (0~1)
    # PolynomialFeatures
    print(X.shape, y.shape)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, y


boston = load_boston()
X, y = load_extended_boston()


for i in range(1,6):
    for j in range(0,31):
        rate=i/10
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rate, random_state=j)
        lr = LinearRegression().fit(X_train, y_train)
        print("Random_state = {}".format(j))
        print("test data rate : {}".format(rate))
        print("accuracy with the train dataset: {:.2f}".format(lr.score(X_train, y_train)))
        print("accuracy with the test dataset: {:.2f}".format(lr.score(X_test, y_test)))
    
    






