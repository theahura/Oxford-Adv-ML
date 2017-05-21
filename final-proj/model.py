"""
Author: Amol Kapoorr
Description: The model for the mnist gan.
"""

import numpy as np
import tensorflow as tf

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=1):
    """
    Defines 2d convolution layer. Influenced by starter agent.
    """
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        w_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
                   num_filters]
        b_shape = [1, 1, 1, num_filters]

        # initialize weights with random weights, see ELU paper
        # which cites He initialization
        # filter w * h * channels in
        fan_in = filter_size[0] * filter_size[1] * int(x.get_shape()[3])
        # filter w * h * channels out
        fan_out = filter_size[0] * filter_size[1] * num_filters

        w_bound = np.sqrt(12. / (fan_in + fan_out))

        w = tf.get_variable("W", w_shape, tf.float32,
                            tf.random_uniform_initializer(-w_bound, w_bound))

        b = tf.get_variable("b", b_shape,
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(x, w, stride_shape, padding="SAME") + b

def linear(x, size, name, initializer=None):
    """
    Defines a linear layer in tf.
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size],
                        initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(0))
    return tf.matmul(x, w) + b

class GAN(object):
    """
    Model for DCGAN.
    """

    def __init__(self, image_shape):
        """
        Builds the DCGAN model.
        """
        x = tf.placeholder(tf.float32, [None] + image_shape)

        for name, filter_shape, stride in c.CONV_SETUP:
            x = tf.nn.elu(conv2d(x, c.OUTPUT_CHANNELS, name, filter_shape,
                                 stride))
