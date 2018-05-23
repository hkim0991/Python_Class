# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:06:41 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

%matplotlib inline

# Load dataset ----------------------------------------------------------------
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7) 
image_shape = people.images[0].shape


# EDA -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks':(),
                                     'yticks':()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])


# To avoid the bias, we only select 50 images of each people ------------------
mask = np.zeros(people.target.shape, dtype=np.bool)

for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]] = 1 

X_people = people.data[mask]
y_people = people.target[mask]
print(X_people.shape, y_people.shape)

X_people = X_people / 255 # normalization (maximum value - minimum value)
X_people


# Train/test data partitiaion -------------------------------------------------
X_train, X_test, y_train, y_test =train_test_split(
        X_people, y_people,
        stratify=y_people,
        random_state=0)


# Modeling 1 - KNN ------------------------------------------------------------

kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)


# Modeling 2 - PCA ------------------------------------------------------------

pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

d = pca.transform(X_test)


# Data reconstitution ---------------------------------------------------------

X_re_pca = pca.inverse_transform(d)
X_re_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

print(X_re_pca.shape, X_re_kmeans.shape)


# Ploting ---------------------------------------------------------------------

# With the extracted components from KNN and PCA

fig, axes = plt.subplots(2,5, figsize=(8,8),
                         subplot_kw={'xticks':(), 'yticks':()})
fig.suptitle("extracted components")

axes.T.shape
kmeans_c = kmeans.cluster_centers_
pca_com = pca.components_

kmeans_c.shape
pca_com.shape

for ax, comp_kmeans, comp_pca in zip(axes.T, kmeans_c, pca_com):
    ax[0].imshow(comp_kmeans.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape))


# after reconsitution from KNN and PCA
    
fig, axes = plt.subplots(3,5, figsize=(8,8),
                         subplot_kw={'xticks':(), 'yticks':()})

fig.suptitle('reconstituion')
for ax, orig, rec_kmeans, rec_pca in zip(
    axes.T, X_test, X_re_kmeans, X_re_pca):
    ax[0].imshow(orig.reshape(image_shape))
    ax[1].imshow(rec_kmeans.reshape(image_shape))
    ax[2].imshow(rec_pca.reshape(image_shape))

axes[0,0].set_ylabel("original")
axes[1,0].set_ylabel("kmeans")
axes[2,0].set_ylabel("pca")


