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
LINEAR = False
ADAM = False
#           Linear model
LAYER1 = 300
LAYER2 = 600
GEN_NETWORK = [('g1', LAYER1), ('g2', LAYER2), ('g3', IM_SIZE)]
DISCRIM_NETWORK = [('d1', LAYER2), ('d2', LAYER1), ('d3', 1)]
#           Conv model
LAYERS = 2
OUTPUT_CHANNELS = 10
FILTER_SHAPE = [3, 3]
STRIDE = 1
CONV_KEEP_PROB = 0.5
#       Optimizations and Convergence
LEARNING_RATE = 0.0001
KEEP_PROB = 0.5

# Debugging
CKPT_PATH = './data/ckpt/'
LOGDIR = './data/logs/'

# Misc
MAX_EPOCH = 500
SUM_STEPS = TOTAL_SIZE / BATCH_SIZE - 1
TRAIN = True
