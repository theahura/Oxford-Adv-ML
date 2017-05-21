"""
Author: Amol Kapoor
Description: Loads the data to be used by the mnist gan
"""

from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    """
    Returns the mnist data. See mnist beginner tensorflow tutorial.
    """
    return input_data.read_data_sets("MNIST_data/", one_hot=True)
