# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:37:33 2018

@author: 202-22
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))  #569 x 30

cancer.feature_names
print("feature_names:{}".format(cancer['feature_names']))
print("target_names:{}".format(cancer['target_names']))

X_train, X_test, y_train, y_test = train_test_split(cancer['data'],
                                                    cancer['target'],
                                                    test_size=0.3,
                                                    random_state=77)

training_accuracy=[]
test_accuracy = []
neighbors_settings = range(1,11)
for n in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)
    score_tr=clf.score(X_train, y_train)
    score_test=clf.score(X_test, y_test)
    training_accuracy.append(score_tr)
    test_accuracy.append(score_test)
    
    print("training_accuracy: {:.2f}".format(score_tr))
    print("test_accuracy: {:.2f}".format(score_test))

plt.plot(neighbors_settings, training_accuracy, label= "training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label= "test_accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.legend()


import mglearn
%matplotlib inline

mglearn.plots.plot_knn_regression(n_neighbors=3)

from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
reg.predict(X_test)
reg.score(X_test, y_test)



plt.scatter(neighbors_settings, training_accuracy, c = "g", alpha = 0.2, label = "")
plt.xlabel(neighbors_settings)
plt.ylabel(training_accuracy)
plt.legend()
plt.show()











