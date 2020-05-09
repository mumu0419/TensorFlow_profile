import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer


input = tf.placeholder(tf.float32,[1,14,14,3])
print(input.shape)

out_1 = tf.layers.conv2d(
    inputs=input, filters=64, kernel_size=(9,9), strides=3, padding="valid", activation=None, name="conv_1")
print(out_1.shape)

# out_2 = tf.layers.conv2d(
#     inputs=out_1, filters=32, kernel_size=3, strides=1, padding="SAME", activation=None, name="conv_2")
# print(out_2.shape)
# out_flatten = tf.layers.flatten(out_1,name='flatten')
# print(out_flatten.shape)
#
# out_2=tf.layers.dense(inputs=out_flatten,units=10,activation=None,
#                                name="dense")
# print("shape of logits:", out_2.shape)

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



# 将模型变量的统计情况输出到 stdout。
builder = tf.profiler.ProfileOptionBuilder
param_stats1 = tf.profiler.profile(
    tf.get_default_graph(),
    options=builder.trainable_variables_parameter())
# or
# 以 Python code 视图来显示统计数据。
opts =builder(builder.trainable_variables_parameter())
opts.with_node_names(show_name_regexes=['.*my_code1.py.*', '.*my_code2.py.*'])
opts = opts.build()
param_stats2 = tf.profiler.profile(
    tf.get_default_graph(),
    cmd='code', # 通过该参数可以控制显示方式
    options=opts)
# param_stats can be tensorflow.tfprof.GraphNodeProto or
# tensorflow.tfprof.MultiGraphNodeProto, depending on the view.
# Let's print the root below.
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
#for x in param_stats.children: # 输出各部分的参数量
#  sys.stdout.write('%s: %d\n' % (x.name, x.total_parameters))



#统计模型的浮点运算数
# 统计运算量
opts = tf.profiler.ProfileOptionBuilder.float_operation()
float_stats = profiler.profile_operations(opts)
# 总参数量
print('总浮点运算数：', float_stats.total_float_ops)
#
for x in float_stats.children:
  print(x.name, 'scope参数：', x.total_float_ops)

# Print to stdout an analysis of the number of floating point operations in the
# model broken down by individual operations.
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.float_operation())
# opts.with_accounted_types(account_type_regexes=['.*MatMul','.*BiasAdd','.*Conv2D']) # 选择只统计Conv和FC的计算量
opts = opts.build()
float_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=opts)
sys.stdout.write('total_floats: %d\n' % float_stats.total_float_ops)
# 注意事项：
#    这个统计不太准确
#    例如：flatten后一层的MatMul不会统计进去

#
# 统计模型内存和耗时情况
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory())
print(opts)
#opts.with_step(1)
opts.with_timeline_output('timeline.json')
opts = opts.build()



#profiler.profile_name_scope(opts) # 只能保存单step的timeline
profiler.profile_graph(opts) # 保存各个step的timeline

#给出使用profile工具给出建议
opts = {'AcceleratorUtilizationChecker': {},
        'ExpensiveOperationChecker': {},
        'JobChecker': {},
        'OperationChecker': {}}
profiler.advise(opts)