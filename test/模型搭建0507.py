import os
import tempfile

import tensorflow as tf
import  common
from tensorflow.examples.tutorials.mnist import input_data


#建立模型
batch_size = 100

# placeholder
inputs = tf.placeholder(tf.float32, name='inputs')
targets = tf.placeholder(tf.float32, name='targets')

# model
out_1 = common.convolutional(inputs, filters_shape=(3, 3,  3,  32), trainable=True, name='conv0')
logits = tf.layers.dense(out_1, 10, activation=None)

# loss + train_op
loss = tf.losses.softmax_cross_entropy(onehot_labels=targets, logits=logits)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#加载数据，并获取程序运行数据
# load data
mnist_save_dir = os.path.join(tempfile.gettempdir(), 'MNIST_data')
mnist = input_data.read_data_sets(mnist_save_dir, one_hot=True)

# get tracing data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 创建Profiler实例作为记录、处理、显示数据的主体
    profiler = tf.profiler.Profiler(graph=sess.graph)

    # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
    run_metadata = tf.RunMetadata()

    for i in range(10):
        batch_input, batch_target = mnist.train.next_batch(batch_size)
        feed_dict = {inputs: batch_input,
                     targets: batch_target}
        _ = sess.run(train_op,
                     feed_dict=feed_dict,
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata)

        # 将当前step的统计数据添加到Profiler实例中
        profiler.add_step(step=i, run_meta=run_metadata)


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
# 统计模型内存和耗时情况
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory())
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