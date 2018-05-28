# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:05:29 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
import mglearn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering


# Load dataset ----------------------------------------------------------------
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=1)
print(X.shape, y.shape)
print(X[:,0].shape)
print(X[:,1].shape)

fig,axes = plt.subplots(1,2, figsize=(10,5))
axes[0].plot(X[:,0], X[:,1], "o")
axes[1].plot(y, "o")


# AgglomerativeClustering -----------------------------------------------------
agg3 = AgglomerativeClustering(n_clusters=3)
assignment3 = agg3.fit_predict(X)

plot3 = mglearn.discrete_scatter(X[:,0], X[:,1], assignment3)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc="best")
plt.xlabel("attribute 0")
plt.ylabel("attribute 1")

# n_clusters = 5
agg5 = AgglomerativeClustering(n_clusters=5)
assignment5 = agg5.fit_predict(X)

plot5 = mglearn.discrete_scatter(X[:,0], X[:,1], assignment5)
plt.legend(["cluster 0", "cluster 1", "cluster 2", "cluster 3", "cluster 4"], loc="best")
plt.xlabel("attribute 0")
plt.ylabel("attribute 1")


# plots when n_clusters is 3 and 5
fig,axes = plt.subplots(1,2, figsize=(10,5))
fig.add_subplot(1,2,1)
mglearn.discrete_scatter(X[:,0], X[:,1], assignment3)
fig.add_subplot(1,2,2)
mglearn.discrete_scatter(X[:,0], X[:,1], assignment5)


# Dendrogram ------------------------------------------------------------------
from scipy.cluster.hierarchy import dendrogram, ward
X, y = make_blobs()
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
df.columns 

linkage_array = ward(df[['x', 'y']])
linkage_array

dendrogram(linkage_array)


# DBSCAN ----------------------------------------------------------------------
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X1, y1 = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler= StandardScaler() # standardization
scaler.fit(X1)

X1_scaled = scaler.transform(X1)

print(X1[1:5], X1_scaled[1:5])
print(np.std(X1_scaled))
print(np.mean(X1_scaled))


plt.scatter(X1_scaled[:, 0],
            X1_scaled[:, 1])

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X1_scaled)

plt.scatter(X1_scaled[:, 0],
            X1_scaled[:, 1],
            c=clusters)
plt.xlabel('attribute 0')
plt.ylabel('attribute 1')
