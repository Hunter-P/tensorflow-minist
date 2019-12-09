# -*- encoding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import ssl

"""
用双层lstm来做手写数字识别
"""
ssl._create_default_https_context = ssl._create_unverified_context

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)

lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐含层的节点数, 可随意设置
hidden_size = 256
# LSTM layer 的层数
layer_num = 2
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

X = tf.reshape(_X, [-1, 28, 28])


def lstm_cell(hidden_size, keep_prob):
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


mlstm_cell = rnn.MultiRNNCell([lstm_cell(hidden_size, keep_prob) for _ in range(layer_num)], state_is_tuple=True)

# 用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# 按时间步展开计算，这里输出双层lstm的计算结果
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]  # 最后一个时间步

# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W)+bias)

# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1) % 200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch[0], y:batch[1], keep_prob:1.0, batch_size:_batch_size
        })
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
    sess.run(train_op, feed_dict={_X:batch[0], y:batch[1], keep_prob:0.5,
                                  batch_size:_batch_size})

# 计算测试数据的准确率
test_accuracy = sess.run(accuracy, feed_dict={_X: mnist.test.images,
                                              y: mnist.test.labels,
                                              keep_prob: 1.0,
                                              batch_size:mnist.test.images.shape[0]})

print("test accuracy %g" % test_accuracy)

