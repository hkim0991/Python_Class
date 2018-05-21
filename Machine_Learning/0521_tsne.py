# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:07:28 2018

@author: kimi
"""

# Import libraries & funtions -------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline


# Load dataset ----------------------------------------------------------------

from sklearn.datasets import load_digits
digits = load_digits()
digits.images
digits.images.shape

fig, axes = plt.subplots(3, 5,
                         figsize=(10, 5),
                         subplot_kw = {'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
    
print(digits.data.shape) # (1797, 8, 8) - 1797 x (8x8) 
print(digits.data[1:5])
print(digits.DESCR)

print(digits.images[1:5]) 
print(digits.images.shape) # (1797, 8, 8) - 1797 x (8x8)

print(digits.target[1:15])
print(digits.target.shape)
print(digits.target_names)


# Model - PCA -----------------------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(digits.data)

digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
"#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]

plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

# scatter plot with digit texts 
for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("1st PCA")
plt.ylabel("2nd PCA")


# Model - t-SNE ---------------------------------------------------------------
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)

d_tsne = tsne.fit_transform(digits.data)
print(d_tsne[:, 0].shape)
print(d_tsne[:, 0].min(), d_tsne[:, 0].max())
print(d_tsne[:, 1].min(), d_tsne[:, 1].max())

plt.figure(figsize=(10,10)) 
plt.xlim(d_tsne[:,0].min(), d_tsne[:,0].max() + 1)
plt.ylim(d_tsne[:,1].min(), d_tsne[:,1].max() + 1)
for i in range(len(digits.data)):
    plt.text(d_tsne[i,0],
             d_tsne[i,1],
             str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight':'bold', 'size':9})
plt.xlabel("t-SNE attribute 0")
plt.xlabel("t-SNE attribute 1")


