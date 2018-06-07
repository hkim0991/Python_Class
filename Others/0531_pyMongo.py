# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:33:33 2018

@author: kimi

"""
# load iris data
from sklearn.datasets import load_iris
import pymongo

iris = load_iris()
iris.data
iris.feature_names
iris.target
iris.target_names

mongo = pymongo.MongoClient("localhost", 27017)
mongo.database_names()

for i in range(0, len(iris.data)):
    mongo.testA.iris.insert( {'sepal length': iris.data[i][0],
                              'sepal width': iris.data[i][1],
                              'petal length': iris.data[i][2],
                              'petal width': iris.data[i][3],
                              'species': iris.target_names[int(iris.target[i])] } )


cursor = mongo.testA.iris.find()
for doc in cursor:
    print(doc)

