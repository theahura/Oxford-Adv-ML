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
    gan = GAN(c.IM_SIZE, c.Z_SIZE, c.GEN_NETWORK, c.DISCRIM_NETWORK)

    ckpt = tf.train.get_checkpoint_state(c.CKPT_PATH)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        gan.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    return gan

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

def build_generator(x, gen_network):
    """
    Builds a linear feedforward generator network.
    """
    with tf.variable_scope('gen'):
        for i, (name, size) in enumerate(gen_network):
            if i == len(gen_network) - 1:
                x = linear(x, size, name)
            else:
                x = tf.nn.relu(linear(x, size, name))
        return tf.nn.tanh(x)

def build_discriminator(x_data, x_generated, discrim_network):
    """
    Builds a linear feedforward discriminator network.
    """
    with tf.variable_scope('discrim'):
        x = tf.concat([x_data, x_generated], 0)
        for i, (name, size) in enumerate(discrim_network):
            if i == len(discrim_network) - 1:
                x = linear(x, size, name)
            else:
                x = tf.nn.relu(linear(x, size, name))
                x = tf.nn.dropout(x, c.KEEP_PROB)
        y_data = tf.nn.sigmoid(tf.slice(x, [0, 0], [c.BATCH_SIZE, -1],
                                        name=None))
        y_generated = tf.nn.sigmoid(tf.slice(x, [c.BATCH_SIZE, 0], [-1, -1],
                                             name=None))
        return y_data, y_generated

class GAN(object):

    def __init__(self, input_size, z_size, gen_network, discrim_network):

        # Construct networks
        self.x = x = tf.placeholder(tf.float32, [None, input_size], name="x")
        self.z = z = tf.placeholder(tf.float32, [None, z_size], name="z")
        self.generator = build_generator(z, gen_network)
        discriminator = build_discriminator(x, self.generator, discrim_network)
        self.discrim_data = discriminator[0]
        self.discrim_gen = discriminator[1]

        # Training
        d_loss = - (tf.log(self.discrim_data) + tf.log(1 - self.discrim_gen))
        g_loss = - tf.log(self.discrim_gen)

        adam_opt = tf.train.GradientDescentOptimizer(c.LEARNING_RATE)
        if c.ADAM:
            adam_opt = tf.train.AdamOptimizer(c.LEARNING_RATE)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discrim')

        self.d_train = adam_opt.minimize(d_loss, var_list=self.d_vars)
        self.g_train = adam_opt.minimize(g_loss, var_list=self.g_vars)

        # Summary ops
        image = tf.expand_dims(tf.reshape(self.generator, c.IM_RESHAPE), -1)
        im_sum = tf.summary.image('model/generated', image)
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

        fetched = sess.run(output_feed, feed_dict)

        if get_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[2]),
                                            sess.run(self.global_step))
            self.summary_writer.flush()

        return fetched

    def generate(self, sess, z):
        """
        Generates a batch of numbers.
        """
        feed_dict = {self.z: z}

        return sess.run(self.generator, feed_dict)

