# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:07:07 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
import mglearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

%matplotlib inline

mglearn.plots.plot_kmeans_algorithm()
mglearn.plots.plot_kmeans_boundaries()


# Load dataset ----------------------------------------------------------------
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=1)
X
y
print(X.shape, y.shape)


# Modeling - KNN --------------------------------------------------------------

# k = 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print("cluster label:\n{}".format(kmeans.labels_))
print(kmeans.predict(X))


# Ploting ---------------------------------------------------------------------

mglearn.discrete_scatter(X[:,0], X[:,1],  # X, y 
                         kmeans.labels_,  # color
                         markers='o')     # shape

cen = kmeans.cluster_centers_

mglearn.discrete_scatter(cen[:, 0], cen[:, 1],
                         [2,0,1],
                         markers='^')



# Modeling & Ploting ----------------------------------------------------------

# when k = 2 or k = 5
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(X)

kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(X)

cen2 = kmeans2.cluster_centers_
cen5 = kmeans5.cluster_centers_


fig, axes = plt.subplots(1,2, figsize=(10,5))

# k =2
mglearn.discrete_scatter(X[:, 0], X[:, 1],
                         kmeans2.labels_,
                         ax=axes[0],
                         markers='o')

mglearn.discrete_scatter(cen2[:, 0], cen2[:, 1],
                         [1,0],
                         ax=axes[0],
                         markers='^')

# k = 5
mglearn.discrete_scatter(X[:, 0], X[:, 1],
                         kmeans5.labels_,
                         ax=axes[1],
                         markers='o')

mglearn.discrete_scatter(cen5[:, 0], cen5[:, 1],
                         [4,3,1,2,0],
                         ax=axes[1],
                         markers='^')


# random data creation -------------------------------------------------------
-
X, y = make_blobs(random_state=170, n_samples=600)
print(X.shape, y.shape)

kmeans_ran = KMeans(n_clusters=3)
kmeans_ran.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1],
                          kmeans_ran.labels_,
                          markers='o')

rng = np.random.RandomState(74)
transformation = rng.normal(size=(2,2))
X = np.dot(X, transformation)

kmeans_ran = KMeans(n_clusters=3)
kmeans_ran.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1],
                          kmeans_ran.labels_,
                          markers='o')
