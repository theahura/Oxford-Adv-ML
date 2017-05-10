"""
Author: Amol Kapoor
Description: Implementation of Eigenfaces for ws1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import scipy.misc

DEBUG = False

def load_images(path):
    """
    Returns array of images as greyscale matrices.
    """
    images = []
    for dirname in os.listdir(path):
        for f in os.listdir(path + '/' + dirname):
            filepath = path + '/' + dirname + '/' + f
            images.append(scipy.misc.imread(filepath, True))
            break

    return images

def image_to_vector(matrix):
    return matrix.flatten()

def vector_to_image(vector, shape=(200, 180)):
    return np.reshape(vector, shape)

def get_mean_face(images):
    """
    Shows the mean face and returns the array corresponding to that face.
    """
    count = len(images)
    ave = np.zeros(images[0].shape)
    for arr in images:
        ave = ave + arr/count
    if DEBUG:
        plt.matshow(ave)
        plt.show()
    return ave

def diff_vectors(images, mean):
    """
    Returns matrix of vector differences from the mean
    """
    difffaces = [x - mean for x in images]
    return map(image_to_vector, difffaces)

def eigensystem(images, mean):
    """
    Gets eigenvalues and vectors from a set of images.
    """
    diff_vects = diff_vectors(images, mean)
    diff_vects = np.array(diff_vects)
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

images = load_images('faces94/male/')
mean = get_mean_face(images)
efaces = eigensystem(images, mean)
test_im = scipy.misc.imread('test.jpg', True)
coeffs = project_img(test_im, mean, efaces)
reconst = image_to_vector(mean) + np.dot(coeffs, np.transpose(efaces))
img = vector_to_image(reconst)
plt.imshow(test_im)
plt.show()
plt.imshow(img)
plt.show()
