"""
Author: Amol Kapoor
Description: Runs the mnist gan
"""

import os

import tensorflow as tf
import numpy as np
from skimage.io import imsave

import constants as c
import load_data
import model
import conv_model
import lstm_model

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], c.H, c.W)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

sess = tf.Session()

print c.CKPT_PATH

gan = None
if c.MODEL == 'linear':
    gan = model.get_model(sess)
elif c.MODEL == 'conv':
    gan = conv_model.get_model(sess)
else:
    gan = lstm_model.get_model(sess)
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

            if c.MODEL == 'conv':
                x = np.reshape(x, (batch_size, c.W, c.H))

                z = np.random.normal(0, 1,
                                     size=(batch_size,
                                           c.zW, c.zH)).astype(np.float32)

            get_summary = True if j % c.SUM_STEPS == 0 else False

            fetched = gan.train(sess, x, z, get_summary)

            if c.DEBUG and get_summary and c.MODEL != 'linear':
                print fetched[3]
                print sess.run(tf.reduce_mean(fetched[3]))
                print sess.run(tf.reduce_mean(fetched[4]))
                print fetched[5]
                print np.array(fetched[5]).shape
                print fetched[6]

        sess.run(gan.global_inc)
        gan.saver.save(sess, checkpoint_path, global_step=gan.global_step)
else:
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    if c.MODEL == 'conv':
        z_test_value = z_test_value.reshape(batch_size, c.zW, c.zH)
    x_gen_val = gan.generate(sess, z_test_value)
    show_result(x_gen_val, c.SAVE_NAME)
