# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:42:40 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline


# Load dataset ----------------------------------------------------------------
iris = sns.load_dataset('iris')
iris.head()
print(type(iris))


# Data seperation -------------------------------------------------------------
X_iris = iris.drop('species', axis=1)
y_iris = iris['species']
X_iris


# Modeling - PCA --------------------------------------------------------------
pca = PCA(n_components=2, whiten=True, random_state=0).fit(X_iris) 
X_iris2 = pca.transform(X_iris)

print(X_iris.head())
print(X_iris2)

X_iris2[:, 0]
X_iris2[:, 1]


iris['PCA1'] = X_iris2[:, 0]
iris['PCA2'] = X_iris2[:, 1]


# Plot ------------------------------------------------------------------------
sns.lmplot("PCA1", "PCA2", hue="species", data=iris, fit_reg=False);