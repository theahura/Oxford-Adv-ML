"""
Author: Amol Kapoor
Description: Runs the mnist gan
"""

import os

import tensorflow as tf
import numpy as np

import constants as c
import load_data
import model
import conv_model

sess = tf.Session()
gan = model.get_model(sess) if c.LINEAR else conv_model.get_model(sess)
batch_size = c.BATCH_SIZE
total_size = c.TOTAL_SIZE
z_size = c.Z_SIZE

for var in gan.d_vars:
    print var.name
for var in gan.g_vars:
    print var.name

if c.TRAIN:
    data = load_data.get_data()

    checkpoint_path = os.path.join(c.CKPT_PATH, 'gan.ckpt')

    for i in range(sess.run(gan.global_step), c.MAX_EPOCH):
        for j in range(total_size / batch_size):
            print "epoch:%s, iter:%s" % (i, j)
            x, _ = data.train.next_batch(batch_size)
            x = 2 * x.astype(np.float32) - 1
            z = np.random.normal(0, 1, size=(batch_size,
                                             z_size)).astype(np.float32)

            if not c.LINEAR:
                x = np.reshape(x, (batch_size, c.W, c.H))

                z = np.random.normal(0, 1,
                                     size=(batch_size,
                                           c.zW, c.zH)).astype(np.float32)

            get_summary = True if j % c.SUM_STEPS == 0 else False

            fetched = gan.train(sess, x, z, get_summary)

            if c.DEBUG and get_summary:
                print fetched[3]
                print sess.run(tf.reduce_mean(fetched[3]))
                print sess.run(tf.reduce_mean(fetched[4]))
                print fetched[5]
                print np.array(fetched[5]).shape
                print fetched[6]

        sess.run(gan.global_inc)
        gan.saver.save(sess, checkpoint_path, global_step=gan.global_step)
