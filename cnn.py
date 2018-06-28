# -*- coding: utf-8 -*-

"""
简单卷积神经网络实现手写数字识别， 与LeNet-5模型类似
"""

'''
使用python解析二进制文件
'''

import tensorflow as tf
from nn import X_train, X_test, X_validation, y_train, y_test, y_validation


sess = tf.InteractiveSession()


# 初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 填充0，使得卷积的输入和输出保持相同的尺寸


# 2x2的池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1代表样本数量不固定

# 定义第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 6])  # 卷积核的尺寸为5x5,1个颜色通道， 32个不同的卷积核
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 池化

# 定义第二层卷积层
W_conv2 = weight_variable([5, 5, 6, 12])  # 卷积核的尺寸为5x5,1个颜色通道， 64个不同的卷积核
b_conv2 = bias_variable([12])
h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 池化

# 经过两次池化层，图片尺寸变为7x7
# 定义节点数量为1024的全连接层
W_fc1 = weight_variable([7*7*12, 200])
b_fc1 = bias_variable([200])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*12])
h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 将dropout层的输出连接一个softmax层
W_fc2 = weight_variable([200, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv, labels=tf.argmax(y_, 1))
cross_entropy = tf.reduce_mean(cross_entropy)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),
#                                               reduction_indices=[1]))


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
learning_rate_base = 0.005
learning_rate_decay = 0.999
batch_size = 200
global_step = tf.Variable(0, trainable=False)
# 学习率的衰减
# learning_rate = tf.train.exponential_decay(learning_rate_base,
#                                            global_step,
#                                            len(X_train)/batch_size,
#                                            learning_rate_decay)
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(10000):
    if i % 200 == 0:
        validation_dict = {x: X_validation, y_: y_validation, keep_prob: 1.0}
        val_accuracy = accuracy.eval(feed_dict=validation_dict)
        print("step %d, validation accuracy is %g" % (i, val_accuracy))
        # print(cross_entropy.eval())

    start = (i*batch_size) % len(X_train)
    end = min(start+batch_size, len(X_train))
    train_step.run(feed_dict={x: X_train[start:end], y_: y_train[start:end], keep_prob: 0.5})

test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
print("test accuracy is %g" % test_accuracy)




