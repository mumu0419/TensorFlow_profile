import sys
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[1,14,14,3])
ker_size = 3
C_o = 32
filter_shape =[ker_size, ker_size, x.shape[3],C_o]
filter = tf.get_variable('atrous', shape=filter_shape, dtype=tf.float32)
ATR_PARAMETERS = {
    "filters":filter,
    "rate":2,
    "padding":'VALID',
    "name":'atrous'
}
ATR_LAYER = tf.nn.atrous_conv2d

out = ATR_LAYER(**ATR_PARAMETERS,value=x)
print(out.shape)
