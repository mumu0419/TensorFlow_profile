import sys
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[1,14,14,3])
TRAN_PARAMETERS = {
    "filters":32,
    "kernel_size":3,
    "padding":"valid",
    "strides":2,
    "use_bias":True
}
TRAN_LAYER = tf.layers.conv2d_transpose

out = TRAN_LAYER(**TRAN_PARAMETERS,inputs = x)

