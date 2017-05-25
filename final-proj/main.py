"""
Author: Amol Kapoor
Description: Runs the mnist gan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import shutil

import constants as c
import load_data
import model

sess = tf.Session()
gan = model.get_model(sess)
data = load_data.get_data()

batch_size = c.BATCH_SIZE
total_size = c.TOTAL_SIZE
z_size = c.Z_SIZE

checkpoint_path = os.path.join(c.CKPT_PATH, 'gan.ckpt')

for i in range(sess.run(gan.global_step), c.MAX_EPOCH):
    for j in range(total_size / batch_size):
        print "epoch:%s, iter:%s" % (i, j)
        x, _ = data.train.next_batch(batch_size)
        x = 2 * x.astype(np.float32) - 1
        z = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

        get_summary = True if j % c.SUM_STEPS == 0 else False
        gan.train(sess, x, z, get_summary)

    sess.run(gan.global_inc)
    gan.saver.save(sess, checkpoint_path, global_step=gan.global_step)
