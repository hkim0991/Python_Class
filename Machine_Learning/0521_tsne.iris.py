# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:24:45 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Load dataset ----------------------------------------------------------------

from sklearn.datasets import load_iris
iris = load_iris()

iris.data.shape  # 150 x 4
iris.target.shape  # 150 x 1


# Model - PCA & TSNE ----------------------------------------------------------
X_pca = PCA(n_components=2).fit_transform(iris.data)
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)

plt.figure(figsize=(10, 5))
plot(X_pca)


iris['PCA1'] = iris_pca[:, 0]
iris['PCA2'] = iris_pca[:, 1]


# Plot - PCA vs TSNE ----------------------------------------------------------

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(iris['PCA1'], iris['PCA2'], c=iris['target'])

plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris['target'])


