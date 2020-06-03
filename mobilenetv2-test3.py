#! /usr/bin/env python
# coding=utf-8


import core.common_mobilenetv2 as common
import tensorflow as tf
import numpy as np
import time
import os

INPUT_SHAPE = [1,256,256,3]
RUN_TIMES = 1000
trainable = tf.cast(True, tf.bool)

# 定义 MobilenetV2 backbone
def MobilenetV2(input_data, trainable):
    with tf.variable_scope('MobilenetV2'):
        conv = common.convolutional(name='Conv', input_data=input_data, filters_shape=(3, 3, 3, 32),
                             trainable=trainable, downsample=True, activate=True, bn=True)
        conv = common.inverted_residual(name='expanded_conv', input_data=conv, input_c=32, output_c=16,
                                 trainable=trainable, t=1)

        conv = common.inverted_residual(name='expanded_conv_1', input_data=conv, input_c=16, output_c=24, downsample=True,
                                 trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_2', input_data=conv, input_c=24, output_c=24, trainable=trainable)

        conv = common.inverted_residual(name='expanded_conv_3', input_data=conv, input_c=24, output_c=32, downsample=True,
                                 trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_4', input_data=conv, input_c=32, output_c=32, trainable=trainable)
        feature_map_s = common.inverted_residual(name='expanded_conv_5', input_data=conv, input_c=32, output_c=32,
                                          trainable=trainable)

        conv = common.inverted_residual(name='expanded_conv_6', input_data=feature_map_s, input_c=32, output_c=64,
                                 downsample=True, trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_7', input_data=conv, input_c=64, output_c=64, trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_8', input_data=conv, input_c=64, output_c=64, trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_9', input_data=conv, input_c=64, output_c=64, trainable=trainable)

        conv = common.inverted_residual(name='expanded_conv_10', input_data=conv, input_c=64, output_c=96, trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_11', input_data=conv, input_c=96, output_c=96, trainable=trainable)
        feature_map_m = common.inverted_residual(name='expanded_conv_12', input_data=conv, input_c=96, output_c=96,
                                          trainable=trainable)

        conv = common.inverted_residual(name='expanded_conv_13', input_data=feature_map_m, input_c=96, output_c=160,
                                 downsample=True, trainable=trainable)
        conv = common.inverted_residual(name='expanded_conv_14', input_data=conv, input_c=160, output_c=160, trainable=trainable)
        # conv = common.inverted_residual(name='expanded_conv_15', input_data=conv, input_c=160, output_c=160, trainable=trainable)

        conv = common.inverted_residual(name='expanded_conv_16', input_data=conv, input_c=160, output_c=320, trainable=trainable)

        feature_map_l = common.convolutional(name='Conv_1', input_data=conv, filters_shape=(1, 1, 320, 1280),
                                      trainable=trainable, downsample=False, activate=True, bn=True)
    return feature_map_s, feature_map_m, feature_map_l


def ckpt_to_frozen_pb(sess,frozen_pb_file):
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names=output_node_names)
    # with open(frozen_pb_file, 'wb') as f:
    #     f.write(frozen_graph_def.SerializeToString())
    with tf.gfile.GFile(frozen_pb_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())


def create_model():
    # out = tf.identity(out,name=output_node_names)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # for v in sess.graph.get_operations():
    #     print(v.name)
    saver.save(sess, 'ckpt/123')
    return sess

def cal_parameters():
    # 创建Profiler实例作为记录、处理、显示数据的主体
    profiler = tf.profiler.Profiler(graph=sess.graph)
    # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
    #run_metadata = tf.RunMetadata()
    # 统计模型的参数量
    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    param_stats = profiler.profile_name_scope(options=opts)
    # 总参数量
    print('总参数：', param_stats.total_parameters)
    # 各scope参数量
    for x in param_stats.children:
        print(x.name, 'scope参数：', x.total_parameters)

def cal_flops():
    # 创建Profiler实例作为记录、处理、显示数据的主体
    profiler = tf.profiler.Profiler(graph=sess.graph)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    float_stats = profiler.profile_operations(opts)
    # 总浮点运算数
    print('总浮点运算数：', float_stats.total_float_ops)

    for x in float_stats.children:
        print(x.name, 'scope参数：', x.float_ops)



def pb_performance(model_path,times):
    tf.reset_default_graph()
    from pb_loader import PBModelLoader
    pbloader = PBModelLoader(model_path)
    total_time = 0
    data_input = np.random.rand(*INPUT_SHAPE)
    data_output_list = pbloader.run_pb([data_input])
    for i in range(times):
        data_input = np.random.rand(*INPUT_SHAPE)
        t0 = time.time()
        data_output_list = pbloader.run_pb([data_input])
        total_time += time.time() - t0
    print(f"Average inference time: {total_time/times} ,Times: {times}")



layer_input = tf.placeholder(tf.float32, shape=tuple(INPUT_SHAPE), name='input')
output_node_names = ["MobilenetV2/expanded_conv_5/add", "MobilenetV2/expanded_conv_12/add", "MobilenetV2/Conv_1/LeakyRelu"
                   ]


if __name__ == "__main__":
    out = MobilenetV2(layer_input, trainable)
    print(out)
    sess = create_model()
    ckpt_to_frozen_pb(sess, '123.pb')

    cal_flops()
    cal_parameters()
    pb_performance('123.pb', RUN_TIMES)


# def ckpt_to_frozen_pb(sess,frozen_pb_file):
#     frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names=['output'])
#     with open(frozen_pb_file, 'wb') as f:
#         f.write(frozen_graph_def.SerializeToString())
#
# sess = create_model()
# ckpt_to_frozen_pb(sess,123)
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #
# #     # 创建Profiler实例作为记录、处理、显示数据的主体
# #     profiler = tf.profiler.Profiler(graph=sess.graph)
# #
# #     # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
# #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# #     # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
# #     run_metadata = tf.RunMetadata()
# #
# #
# # #统计模型的参数量
# # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
# # param_stats = profiler.profile_name_scope(options=opts)
# # # 总参数量
# # print('总参数：', param_stats.total_parameters)
# # # 各scope参数量
# # for x in param_stats.children:
# #   print(x.name, 'scope参数：', x.total_parameters)
