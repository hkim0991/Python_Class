# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:41:27 2018

@author: 202-22
"""


import mglearn
mglearn.plots.plot_animal_tree()

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


# Import libraries & funtions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Load dataset
from sklearn.datasets import load_breast_cancer


# Training/Testing data separation + modeling with DecisionTreeClassifier
def testTreeModel(TestSize=0.3, treedepth=3, max_leaf=10):
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                        cancer.target,
                                                        stratify=cancer.target,
                                                        test_size = TestSize,
                                                        random_state=77)
    tree = DecisionTreeClassifier(max_depth=treedepth, random_state=0, max_leaf_nodes=max_leaf)
    tree.fit(X_train, y_train)
    print("accuracy with the train dataset: {:.3f}".format(tree.score(X_train, y_train)))
    print("accuracy with the test dataset: {:.3f}".format(tree.score(X_test, y_test)))

testTreeModel(0.3, 3, 9)


for i in range(1,11):
    print("when the size of the test data is", i/10)
    print("when the depth of the tree is:", i)
    testTreeModel(i/10, i)




# Import libraries & funtions
from sklearn.ensemble import RandomForestClassifier


# Load dataset
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print(X.shape, y.shape)


# Training/Testing data separation
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=77)


# Modeling with RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, random_state=2)
forest.fit(X_train, y_train)

forest
print(forest.estimators_)
print(forest.score)

print("\n\n{}".format(forest.base_estimator))

print("\n\n{}".format(forest.bootstrap))
print("\n\n{}".format(forest.criterion))
print("\n\n{}".format(forest.oob_score))




