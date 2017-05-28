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
    gan = CONVGAN(c.IM_SIZE, c.Z_SIZE)

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

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=2):
    """
    Defines 2d convolution layer.
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

        return batchnormalize(tf.nn.conv2d(x, w, stride_shape,
                                           padding="SAME") + b)

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

def build_generator(x):
    """
    Builds a linear feedforward generator network.
    """
    with tf.variable_scope('gen'):
        for i in range(c.LAYERS):
            x = tf.nn.elu(conv2d(x, c.OUTPUT_CHANNELS, 'l{}'.format(i + 1),
                                 c.FILTER_SHAPE, c.STRIDE))
            x = tf.nn.dropout(x, c.CONV_KEEP_PROB)
        x = flatten(x)
        x = linear(x, c.IM_SIZE, 'glin')
        return tf.nn.tanh(x)

def build_discriminator(x_data, x_generated):
    """
    Builds a linear feedforward discriminator network.
    """
    with tf.variable_scope('discrim'):
        x = tf.concat([x_data, x_generated], 0)
        for i in range(c.LAYERS):
            x = tf.nn.elu(conv2d(x, c.OUTPUT_CHANNELS, 'l{}'.format(i + 1),
                                 c.FILTER_SHAPE, c.STRIDE))
            x = tf.nn.dropout(x, c.CONV_KEEP_PROB)

        x = flatten(x)
        x = linear(x, 1, 'glin')
        y_data = tf.nn.sigmoid(tf.slice(x, [0, 0], [c.BATCH_SIZE, -1],
                                        name=None))
        y_generated = tf.nn.sigmoid(tf.slice(x, [c.BATCH_SIZE, 0], [-1, -1],
                                             name=None))
        return y_data, y_generated

def cost(logits, labels):
    logits = tf.clip_by_value(logits, 1e-7, 1.0 - 1e-7)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

class CONVGAN(object):

    def __init__(self, input_size, z_size):

        # Construct networks
        self.x = x = tf.placeholder(tf.float32, [None, c.W, c.H], name="x")
        x = tf.expand_dims(x, -1)
        self.z = z = tf.placeholder(tf.float32, [None, c.zW, c.zH], name="z")
        z = tf.expand_dims(z, -1)
        self.generator = build_generator(z)
        generated_image = tf.reshape(self.generator, [-1, c.W, c.H, 1])
        discriminator = build_discriminator(x, generated_image)
        self.discrim_data = discriminator[0]
        self.discrim_gen = discriminator[1]

        # Training
        d_cost_real = cost(self.discrim_data, tf.ones_like(self.discrim_data))
        d_cost_gen = cost(self.discrim_gen, tf.zeros_like(self.discrim_gen))
        self.d_loss = d_loss = d_cost_real + d_cost_gen

        self.g_loss = g_loss = cost(self.discrim_data,
                                    tf.ones_like(self.discrim_data))

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

