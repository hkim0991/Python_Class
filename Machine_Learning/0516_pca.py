# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:37:33 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
from pandas.tools.plotting import scatter_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib
import sklearn
import mglearn

from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Load dataset ----------------------------------------------------------------
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)


# EDA -------------------------------------------------------------------------
image_shape = people.images[0].shape # (87, 65) : 87 x 65 pix

print(people.DESCR)
print(people.data.shape) # 3023 x 5655
print(people.target.shape) # 3023
print(people.target_names.shape) # 62 names of people
print(people.images.shape) # 3023 (87 x 65)

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks':(), 'yticks':()})

a1 =people.target # a number to indicate people's name
a2 = people.images # information of the image
a3 = axes.ravel()

for target, image, ax in zip(a1, a2, a3):
    ax.imshow(image)
    ax.set_title(people.target_names[target])


# To avoid the bias, we only select 50 images of each people
mask = np.zeros(people.target.shape, dtype=np.bool)
print(mask)
print(np.unique(people.target))
print(np.where(people.target==target)[0][:50])


for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1  # 해당 위치의 값 중에 50개만 선택(유일함)

X_people = people.data[mask]
y_people = people.target[mask]
print(X_people.shape, y_people.shape)

X_people = X_people / 255 # normalization (maximum value - minimum value)
X_people


# Training/Testing data separation --------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, 
                                                    stratify=y_people, random_state=0)

X_train.shape  # 1547 x 5655


# Modeling 1 - KNeighborsClassifier ---------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("accuracy with the train dataset: {:.3f}".format(knn.score(X_train, y_train))) # 1.00
print("accuracy with the test dataset: {:.3f}".format(knn.score(X_test, y_test))) # 0.23

import mglearn
mglearn.plots.plot_pca_whitening()


# Modeling 2.1 - PCA ------------------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train) # no y_train(unsupervised learning)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train_pca.shape : {}".format(X_train_pca.shape)) # 1547 x 100


# Modeling 2.2 - PCA -> KNN ---------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("accuracy with the train dataset: {:.3f}".format(knn.score(X_test_pca, y_test))) #0.31

print("pca.components_.shape : {}".format(pca.components_.shape)) # 100 x 5655

pca.components_[0]

fig, axes = plt.subplots(3,5, figsize=(15,12), 
                         subplot_kw={'xticks' : (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("PCA {}".format((i+1)))
    
    
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)


###############################################################################

# zip function review
a = ["1", "2", "3"]
b = ["a", "b", "c"]
for (a,b) in zip(a,b):
    print(a,b)

# enumerate function review
a = ["1", "2", "3"]
b = ["a", "b", "c"]
for i, (a,b) in enumerate(zip(a,b)):
    print(i, a, b)

###############################################################################

