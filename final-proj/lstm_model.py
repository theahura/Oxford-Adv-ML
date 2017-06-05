"""
Author: Amol Kapoorr
Description: The model for the mnist gan.
"""

import numpy as np
import tensorflow as tf

import constants as c

def get_model(sess):
    """
    Loads the tf model or inits a new one.
    """
    gan = LSTMGAN(c.IM_SIZE, c.Z_SIZE)

    ckpt = tf.train.get_checkpoint_state(c.CKPT_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        gan.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    return gan

def flatten(x):
    """
    Flattens input across batches.
    """
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def batchnormalize(x, eps=1e-8):
    mean = tf.reduce_mean(x, [0, 1, 2])
    std = tf.reduce_mean( tf.square(x - mean), [0, 1, 2])
    x = (x - mean) / tf.sqrt(std + eps)
    return x

def linear(x, size, name, initializer=None):
    """
    Defines a linear layer in tf.
    """
    if not initializer:
        initializer = tf.truncated_normal([int(x.get_shape()[1]), size],
                                          stddev=0.1)
    w = tf.get_variable(name + "/w", initializer=initializer)
    b = tf.get_variable(name + "/b", [size],
                        initializer=tf.constant_initializer(0))
    return tf.matmul(x, w) + b

def single_cell(units):
    cell = tf.contrib.rnn.LSTMCell(units)
    cell = tf.contrib.rnn.DropoutWrapper(cell, c.LSTM_KEEP_PROB,
                                         c.LSTM_KEEP_PROB)
    return cell

def build_generator(x):
    """
    Builds a linear feedforward generator network.
    """
    with tf.variable_scope('gen'):
        cell = tf.contrib.rnn.MultiRNNCell([single_cell(c.LSTM_UNITS) for _ in
                                            range(c.LSTM_LAYERS)])
        x = tf.expand_dims(x, [0])
        outputs, _ = tf.nn.dynamic_rnn(cell, x,
                                       sequence_length=[tf.shape(x)[0]],
                                       dtype=tf.float32, time_major=False)

        # Make this a column vector to make the linear math easier
        x = tf.reshape(outputs, [-1, c.LSTM_UNITS])
        x = linear(x, c.IM_SIZE, 'glin')
        return tf.nn.tanh(x)

def build_discriminator(x, reuse=False):
    """
    Builds a linear feedforward discriminator network.
    """
    with tf.variable_scope('discrim', reuse=reuse):
        cell = tf.contrib.rnn.MultiRNNCell([single_cell(c.LSTM_UNITS) for _ in
                                            range(c.LSTM_LAYERS)])
        x = tf.expand_dims(x, [0])
        outputs, _ = tf.nn.dynamic_rnn(cell, x,
                                       sequence_length=[tf.shape(x)[0]],
                                       dtype=tf.float32, time_major=False)

        # Make this a column vector to make the linear math easier
        x = tf.reshape(outputs, [-1, c.LSTM_UNITS])
        x = linear(x, 1, 'glin')
        return tf.nn.sigmoid(x)

def cost(logits, labels):
    logits = tf.clip_by_value(logits, 1e-7, 1.0 - 1e-7)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

class LSTMGAN(object):

    def __init__(self, input_size, z_size):
        # Construct networks
        self.x = x = tf.placeholder(tf.float32, [None, input_size], name="x")
        self.z = z = tf.placeholder(tf.float32, [None, z_size], name="z")
        self.generator = build_generator(z)
        self.discrim_data = build_discriminator(x)
        self.discrim_gen = build_discriminator(self.generator, True)

        # Training
        d_cost_real = cost(self.discrim_data, tf.ones_like(self.discrim_data))
        d_cost_gen = cost(self.discrim_gen, tf.zeros_like(self.discrim_gen))
        self.d_loss = d_loss = d_cost_real + d_cost_gen

        self.g_loss = g_loss = cost(self.discrim_gen,
                                    tf.ones_like(self.discrim_gen))

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discrim')

        self.g_grads = g_grads = tf.gradients(self.g_loss, self.g_vars)
        self.d_grads = d_grads = tf.gradients(self.d_loss, self.d_vars)

        g_grads, _ = tf.clip_by_global_norm(g_grads, c.MAX_GRAD_NORM)
        d_grads, _ = tf.clip_by_global_norm(d_grads, c.MAX_GRAD_NORM)

        d_grads_and_vars = list(zip(d_grads, self.d_vars))
        g_grads_and_vars = list(zip(g_grads, self.g_vars))

        train = tf.train.AdamOptimizer(learning_rate=c.LEARNING_RATE)

        if not c.ADAM:
            train = tf.train.GradientDescentOptimizer(c.LEARNING_RATE)

        self.d_train = train.apply_gradients(d_grads_and_vars)
        self.g_train = train.apply_gradients(g_grads_and_vars)

        # Summary ops
        self.image = tf.expand_dims(tf.reshape(self.generator, c.IM_RESHAPE), -1)
        im_sum = tf.summary.image('model/generated', self.image)
        d_loss_sum = tf.summary.scalar('model/d_loss', tf.reduce_mean(d_loss))
        g_loss_sum = tf.summary.scalar('model/g_loss', tf.reduce_mean(g_loss))
        var_g_sum = tf.summary.scalar('model/g_norm',
                                      tf.global_norm(self.g_vars))
        var_d_sum = tf.summary.scalar('model/d_norm',
                                      tf.global_norm(self.d_vars))
        self.summary_op = tf.summary.merge([im_sum, d_loss_sum, g_loss_sum,
                                            var_g_sum, var_d_sum])
        self.summary_writer = tf.summary.FileWriter(c.LOGDIR)

        # Misc ops
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_inc = tf.assign(self.global_step, self.global_step + 1)
        self.saver = tf.train.Saver()

    def train(self, sess, x, z, get_summary=False):
        """
        Trains GAN.
        """
        feed_dict = {
            self.x: x,
            self.z: z
        }

        output_feed = [self.d_train, self.g_train]

        if get_summary:
            output_feed = output_feed + [self.summary_op]

        if c.DEBUG:
            output_feed = output_feed + [self.d_loss, self.g_loss, self.image,
                                         self.d_grads, self.g_grads]

        fetched = sess.run(output_feed, feed_dict)

        if get_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[2]),
                                            sess.run(self.global_step))
            self.summary_writer.flush()

            print 'GOT SUMMARY'

        return fetched

    def generate(self, sess, z):
        """
        Generates a batch of numbers.
        """
        feed_dict = {self.z: z}

        return sess.run(self.generator, feed_dict)

