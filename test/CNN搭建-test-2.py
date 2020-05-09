# coding:utf-8
# 导入本次需要的模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME",name='conv')  # padding="SAME"用零填充边界


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# #################处理图片##################################
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
# 定义dropout的输入，解决过拟合问题
keep_prob = tf.placeholder(tf.float32)
# 处理xs，把xs的形状变成[-1,28,28,1]
# -1代表先不考虑输入的图片例子多少这个维度。
# 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1。如果是RGB图像，那么channel就是3.
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape) #[n_samples, 28,28,1]
# #################处理图片##################################

## convl layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # kernel 5*5, channel is 1, out size 32
b_conv1 = bias_variable([32])
out_1 = conv2d(x_image, W_conv1)
h_conv1 = tf.nn.relu(out_1 + b_conv1)  # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # kernel 5*5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

## funcl layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# [n_samples,7,7,64]->>[n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# #################优化神经网络##################################
# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())
# #################优化神经网络##################################

# get tracing data
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 创建Profiler实例作为记录、处理、显示数据的主体
    profiler = tf.profiler.Profiler(graph=sess.graph)

    # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
    run_metadata = tf.RunMetadata()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            # print(sess.run(prediction,feed_dict={xs:batch_xs}))
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

    # for i in range(10):
    #     batch_input, batch_target = mnist.train.next_batch(batch_size)
    #     feed_dict = {inputs: batch_input,
    #                  targets: batch_target}
    #     _ = sess.run(train_op,
    #                  feed_dict=feed_dict,
    #                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #                  run_metadata=run_metadata)
    #
    #     # 将当前step的统计数据添加到Profiler实例中
    #     profiler.add_step(step=i, run_meta=run_metadata)


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


