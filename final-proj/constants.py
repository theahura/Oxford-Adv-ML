"""
Author: Amol Kapoor
Description: Constants used for the mnist gan.
"""

# Network Params
#       Input Constants
H = 28
W = 28
IM_SIZE = H*W
BATCH_SIZE = 256
IM_RESHAPE = [BATCH_SIZE, H, W]
TOTAL_SIZE = 60000
Z_SIZE = 100
zW = zH = 10
#       Construction vars
MODEL = 'conv' # linear, conv, or lstm
USE_POOL = True
ADAM = True
#           Linear model
LAYER1 = 150
LAYER2 = 300
GEN_NETWORK = [('g1', LAYER1), ('g2', LAYER2), ('g3', IM_SIZE)]
DISCRIM_NETWORK = [('d1', LAYER2), ('d2', LAYER1), ('d3', 1)]
KEEP_PROB = 0.5
#           Conv model
LAYERS = 2
OUTPUT_CHANNELS = 5
FILTER_SHAPE = [2, 2]
STRIDE = 1
CONV_KEEP_PROB = 0.5
#           Conv Pool model
KSIZE = [1, 2, 2, 1]
POOL_STRIDE = [1, 1, 1, 1]
#           LSTM Model
LSTM_LAYERS = 2
LSTM_UNITS = 128
LSTM_KEEP_PROB = 0.5

#       Optimizations and Convergence
LEARNING_RATE = 0.0001
MAX_GRAD_NORM = 5.0

# Debugging
CKPT_PATH = './data/backups/convnet-1-2-pool/'
LOGDIR = './data/backups/convnet-1-2-pool/'
DEBUG = True
SAVE_NAME = './data/images/convnet-1-2-pool.png'
TRAIN = True

# Misc
MAX_EPOCH = 500
SUM_STEPS = TOTAL_SIZE / BATCH_SIZE - 1
