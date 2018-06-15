# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:31:18 2018

@author: kimi
"""


# Import libraries & funtions -------------------------------------------------
import pandas as pd


# Load dataset ----------------------------------------------------------------
train = pd.read_csv("C:/Users/202-22/Documents/Python - Hyesu/data/Titanic/train.csv", 
                      index_col=['PassengerId'])
test = pd.read_csv("C:/Users/202-22/Documents/Python - Hyesu/data/Titanic/test.csv",
                    index_col=['PassengerId'])
test.head()
train.head()

print(train.info())
print(test.info())

# Preprocessing - missing values ----------------------------------------------
tr_mean_age = train["Age"].mean()
test_mean_age = test["Age"].mean()

train.loc[pd.isnull(train["Age"]), "Age"] = tr_mean_age
train[pd.isnull(train["Age"])]

test.loc[pd.isnull(test["Age"]), "Age"] = test_mean_age
test[pd.isnull(test["Age"])]


test_mean_Fare = test["Fare"].mean()
test.loc[pd.isnull(test["Fare"]), "Fare"] = test_mean_Fare
test[pd.isnull(test["Fare"])]

train.info()
test.info()


# Preprocessing - One Hot Encoding --------------------------------------------
sel_col = ['Pclass' , 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked' ]
X_train = train[sel_col]
X_train.head()


X_train = pd.get_dummies(X_train)
print(X_train.columns)


X_test = test[sel_col]
X_test = pd.get_dummies(X_test)
print(X_test.columns)


# Modeling --------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

label_name = "Survived"
y_train = train[label_name]

X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, random_state=0)

print(X_tr.shape, y_tr.shape)
model = LogisticRegression()
model.fit(X_tr, y_tr)

prediction = model.predict(X_te)
prediction[1:5]
accuracy_score(y_te, prediction)


# Prediction for test data ----------------------------------------------------
prediction = model.predict(X_test)
prediction[1:5]

sbm = pd.read_csv("C:/Users/202-22/Documents/Python - Hyesu/data/Titanic/gender_submission.csv", 
                      index_col=['PassengerId'])

sbm['Survived'] = prediction
sbm.head()
