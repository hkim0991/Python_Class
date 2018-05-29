# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:39:40 2018

@author: kimi
"""

# Clustering algorithms comparison and evaluation #

# Import libraries & funtions -------------------------------------------------
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import matplotlib


# Load dataset ----------------------------------------------------------------
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler # standardization
from sklearn.metrics.cluster import adjusted_rand_score 

X, y = make_moons(n_samples=200, noise=0.05, random_state=1)
print(X[1:15], y [1:15])
print(np.mean(X))
print(np.std(X))


# Standardization of the data(X, y) -------------------------------------------
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled[1:15]
print(np.mean(X), np.mean(X_scaled))
print(np.std(X), np.std(X_scaled))
print(np.var(X), np.var(X_scaled))


# Plot ------------------------------------------------------------------------
fig, axes = plt.subplots(1,2, figsize=(15,3),
                        subplot_kw= {'xticks':(), 'yticks':()})

axes[0].scatter(X[:,0], X[:,1])
axes[1].scatter(X_scaled[:,0], X_scaled[:,1])


# random allocation
print(mglearn.cm3)
plt.scatter(X_scaled[:,0], X_scaled[:,1],
            c=random_clusters,
            cmap=mglearn.cm3,  # color
            s=60,              # size of the dot 
            edgecolors='black')
plt.title("random allocation - ARI:{:.2f}".format(adjusted_rand_score(y, random_clusters))) 


# Comparison between different clustering models ------------------------------

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN

algor = [KMeans(n_clusters=2),
         AgglomerativeClustering(n_clusters=2),
         DBSCAN()]

random_state= np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
random_clusters


fig, axes = plt.subplots(1,4, figsize=(15,3),
                        subplot_kw= {'xticks':(), 'yticks':()})

axes[0].scatter(X_scaled[:,0], X_scaled[:,1],
            c=random_clusters,
            cmap=mglearn.cm3,  # color
            s=60,              # size of the dot 
            edgecolors='black')
axes[0].set_title("random allocation - ARI:{:.2f}".format(adjusted_rand_score(y, random_clusters))) 

for ax, algorithm in zip(axes[1:], algor):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap=mglearn.cm3,
               s=60, edgecolors='black')
    ax.set_title("{} - ARI:{:.2F}".format(algorithm.__class__.__name__, adjusted_rand_score(y,clusters)))