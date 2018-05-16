# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:14:46 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures # add more features
from sklearn.linear_model import Ridge # Ridge regression model
import matplotlib.pyplot as plt


# Load dataset ----------------------------------------------------------------
def load_extended_boston(): # adding more features from existing features
    boston = load_boston()
    X = boston.data
    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target
X, y = load_extended_boston()
print(X.shape, y.shape)


# Training/Testing data separation --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Modeling - LinearRegression model -------------------------------------------
lr = LinearRegression().fit(X_train, y_train)
print("score of the model with train data: {:.2f}".format(lr.score(X_train, y_train)))
print("score of the model with test data: {:.2f}".format(lr.score(X_test, y_test)))


# Modeling - Ridge Regression model - L2 regularization -----------------------
ridge1 = Ridge().fit(X_train, y_train) # alpha = 1.0 (default value)
print("score of the model with train data: {:.2f}".format(ridge1.score(X_train, y_train)))
print("score of the model with test data: {:.2f}".format(ridge1.score(X_test, y_test)))


ridge000001 = Ridge(alpha=0.00001).fit(X_train, y_train) # alpha = 0.00001 (default value)
print("score of the model with train data: {:.2f}".format(ridge000001.score(X_train, y_train)))
print("score of the model with test data: {:.2f}".format(ridge000001.score(X_test, y_test)))


ridge001 = Ridge(alpha=0.001).fit(X_train, y_train) # alpha = 0.01 (default value)
print("score of the model with train data: {:.2f}".format(ridge001.score(X_train, y_train)))
print("score of the model with test data: {:.2f}".format(ridge001.score(X_test, y_test)))


ridge100 = Ridge(alpha=100).fit(X_train, y_train) # alpha = 100 (default value)
print("score of the model with train data: {:.2f}".format(ridge100.score(X_train, y_train)))
print("score of the model with train data: {:.2f}".format(ridge100.score(X_test, y_test)))


# Compare coefficient between linear model and ridge(alpha=1) models ---------- 
lr.coef_.mean()
ridge1.coef_.mean()
ridge100.coef_.mean()  # the bigger alpha is, the smaller coefficient is 
ridge001.coef_.mean()
ridge000001.coef_.mean()


plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.plot(ridge1.coef_, '^', label="Ridge alpha=1")
plt.plot(ridge100.coef_, 's', label="Ridge alpha=100")
plt.plot(ridge000001.coef_, '^', label="Ridge alpha=0.00001")
plt.xlabel('coefficient')
plt.ylabel('coefficient size')
plt.hlines(0,0, len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()