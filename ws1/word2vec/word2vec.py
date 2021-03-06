"""
Author: Amol Kapoor
Description: Implementation of word2vec PCA.
"""

import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import manifold


def load_vects(samples, path='../../../GoogleNews-vectors-negative300.bin'):
    """
    Loads gensim model and gets vectors of samples
    """
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    vectors = []

    for word in samples:
        vectors.append(model.wv[word])

    return vectors

def get_mean(input_vects):
    """
    Gets the mean of the inputs
    """
    count = len(input_vects)
    ave = np.zeros(input_vects[0].shape)
    for arr in input_vects:
        ave = ave + arr/count
    return ave

def eigensystem(vects, mean):
    """
    Gets eigenvalues and vectors from a set of vectors.
    """
    diff_vects = np.array([x - mean for x in vects])
    U, S, V = np.linalg.svd(np.transpose(diff_vects), full_matrices=False)
    return U

def project_img(image, mean, evectors):
    """
    Projects an img onto a basis of evectors
    """
    image = image_to_vector(image)
    mean = image_to_vector(mean)
    diff = image - mean
    return np.dot(diff, evectors)

def distances(points):
    dist = np.zeros((len(points), len(points)))
    for x in range(len(points)):
        for y in range(len(points)):
            dist[x][y] = np.linalg.norm(points[x] - points[y])
    return dist

def plot(labels, x, y):
    """
    Plot a scatter plot.
    """
    plt.scatter(x, y, marker='o')

    for i, w in enumerate(labels):
        plt.annotate(
            w.decode('utf-8', 'ignore'),
            xy = (x[i], y[i]), xytext = (3, 3),
            textcoords = 'offset points', ha = 'left', va = 'top')

    plt.show()



labels = ['pizza', 'tacos', 'sub', 'cheese', 'dessert', 'lunch', 'pie', 'steak',
           'ribs', 'carrot', 'car', 'train', 'plane', 'boat', 'ship', 'bike',
           'skates', 'bus', 'subway', 'scooter', 'pen', 'pencil', 'notebook',
           'paper', 'highlighter', 'eraser', 'paperclip', 'stapler', 'staples',
           'lead']

# PCA Plot
vects = load_vects(labels)
mean = get_mean(vects)
evects = eigensystem(vects, mean)
reduced_vects = np.dot(np.transpose(evects[:, 0:2]), np.transpose(vects))
plot(labels, reduced_vects[0], reduced_vects[1])

# MDS Plot
points = np.transpose(reduced_vects)
dist = distances(points)
mds = manifold.MDS(n_components=2)
results = mds.fit(dist)
coords = results.embedding_
plot(labels, coords[:, 0], coords[:, 1])

