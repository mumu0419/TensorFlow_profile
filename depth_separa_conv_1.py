import tensorflow as tf
import sys

input = tf.placeholder(tf.float32,[1,14,14,3])

#实现深度卷积
with tf.variable_scope('depthwise'):
    filter = tf.get_variable('w',shape=[3,3,3,20],dtype=tf.float32)
out_1 = tf.nn.depthwise_conv2d(
    input=input, filter=filter,  strides=[1,1,1,1],  rate=[1,1], padding='VALID', name="depth_1")
print(out_1.shape)

#实现1x1卷积
# point_filter = tf.placeholder(tf.float32,[1,1,3,60])
# point_filter = tf.get_variable('123',shape=[1,1,3,4],dtype=tf.float32)
# out_img = tf.nn.conv2d(input=input, filter=point_filter, strides=[1,1,1,1], padding='SAME',name='conv_1')
out_2 = tf.layers.conv2d(
    inputs=out_1, filters=64, kernel_size=(1,1), strides=1, padding="valid", activation=None, name="conv_1")

# get tracing data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 创建Profiler实例作为记录、处理、显示数据的主体
    profiler = tf.profiler.Profiler(graph=sess.graph)

    # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
    run_metadata = tf.RunMetadata()


#统计模型的参数量
opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
param_stats = profiler.profile_name_scope(options=opts)
# 总参数量
print('总参数：', param_stats.total_parameters)
# 各scope参数量
for x in param_stats.children:
  print(x.name, 'scope参数：', x.total_parameters)


#统计模型的浮点运算数
# 统计运算量
opts = tf.profiler.ProfileOptionBuilder.float_operation()
float_stats = profiler.profile_operations(opts)
# 总参数量
print('总浮点运算数：', float_stats.total_float_ops)
#
for x in float_stats.children:
  print(x.name, 'scope参数：', x.float_ops)

