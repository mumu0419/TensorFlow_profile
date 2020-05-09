import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

with tf.Session(graph=tf.Graph()) as sess:
    input = tf.placeholder(tf.float32, [1, 14, 14, 3],name='input_data')

    out_1 = tf.layers.conv2d(
        inputs=input, filters=64, kernel_size=3, strides=1, padding="SAME", activation=None, name="conv_1")
    print(out_1.shape)

    out_2 = tf.layers.conv2d(
        inputs=out_1, filters=32, kernel_size=3, strides=1, padding="SAME", activation=None, name="conv_2")
    print(out_2.shape)

    w = weight_variable([5, 5, 1, 32])
    out_2=tf.nn.conv2d(out_2, w, strides=[1, 1, 1, 1], padding="SAME",name='conv_3')



    # 这里的输出需要加上name属性
    #op = tf.add(xy, b, name='op_to_store')

    sess.run(tf.global_variables_initializer())

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["conv_3"])



    # 测试 OP
    # feed_dict = {input:[1,14,14,3]}
    #print(sess.run(out_2, feed_dict))
    #print(sess.run(out_2))

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # 输出
    # INFO:tensorflow:Froze 1 variables.
    # Converted 1 variables to const ops.
    # 31