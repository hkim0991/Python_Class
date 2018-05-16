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
print(X_people, y_people)









