# -*- coding: utf-8 -*-

"""
tensorflow实现softmax regression识别手写数字
"""
# -------------------------------------------
'''
使用python解析二进制文件
'''
import numpy as np
import struct
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


class LoadData(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    # 载入训练集
    def loadImageSet(self):
        binfile = open(self.file1, 'rb')  # 读取二进制文件
        buffers = binfile.read()  # 缓冲
        head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组
        offset = struct.calcsize('>IIII')  # 定位到data开始的位置

        imgNum = head[1]  # 图像个数
        width = head[2]  # 行数，28行
        height = head[3]  # 列数，28

        bits = imgNum*width*height  # data一共有60000*28*28个像素值
        bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
        imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

        binfile.close()
        imgs = np.reshape(imgs, [imgNum, width*height])
        return imgs, head

    # 载入训练集标签
    def loadLabelSet(self):
        binfile = open(self.file2, 'rb')  # 读取二进制文件
        buffers = binfile.read()  # 缓冲
        head = struct.unpack_from('>II', buffers, 0)  # 取前2个整数，返回一个元组
        offset = struct.calcsize('>II')  # 定位到label开始的位置

        labelNum = head[1]  # label个数
        numString = '>' + str(labelNum) + 'B'
        labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

        binfile.close()
        labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)
        return labels, head

    # 将标签拓展为10维向量
    def expand_lables(self):
        labels, head = self.loadLabelSet()
        expand_lables = []
        for label in labels:
            zero_vector = np.zeros((1, 10))
            zero_vector[0, label] = 1
            expand_lables.append(zero_vector)
        return expand_lables

    # 将样本与标签组合成数组[[array(data), array(label)], []...]
    def loadData(self):
        imags, head = self.loadImageSet()
        expand_lables = self.expand_lables()
        data = []
        for i in range(imags.shape[0]):
            imags[i] = imags[i].reshape((1, 784))
            data.append([imags[i], expand_lables[i]])
        return data


file1 = r'D:\机器学习资料汇总\手写数字识别数据\数据2\train-images.idx3-ubyte'
file2 = r'D:\机器学习资料汇总\手写数字识别数据\数据2\train-labels.idx1-ubyte'
trainingData = LoadData(file1, file2)
training_data = trainingData.loadData()
file3 = r'D:\机器学习资料汇总\手写数字识别数据\数据2\t10k-images.idx3-ubyte'
file4 = r'D:\机器学习资料汇总\手写数字识别数据\数据2\t10k-labels.idx1-ubyte'
testData = LoadData(file3, file4)
test_data = testData.loadData()
X_train = [i[0] for i in training_data]
y_train = [i[1][0] for i in training_data]
X_test = [i[0] for i in test_data]
y_test = [i[1][0] for i in test_data]

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=7)

# 将session注册为默认的session，之后的运算也默认跑在这个session里面
# sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w)+b)  # tf.matmul是矩阵乘法函数

y_ = tf.placeholder(tf.float32, [None, 10])
# 交叉熵，损伤函数
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 选择梯度下降作为优化算法， 学习率设为0.01
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entroy)

# 计算预测值与标签是否一样
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 统计全部样本预测的准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 100
TRAINING_STEPS = 5000

# 迭代计算
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(TRAINING_STEPS):
        if i % 20 == 0:
            validate_acc = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
            print("after %d training step(s), validation accuracy "
                  "using average model is %g" % (i, validate_acc))
        start = (i * BATCH_SIZE) % len(X_train)
        end = min(start + BATCH_SIZE, len(X_train))
        sess.run(train_step, feed_dict={x: X_train[start:end], y_: y_train[start:end]})
        # print('loss:', sess.run([cross_entroy]))
