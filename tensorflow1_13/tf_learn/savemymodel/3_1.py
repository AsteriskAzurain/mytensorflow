from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   # 读取数据
x_data = tf.placeholder(tf.float32, [None, 784])  # 占位符：样本自变量
y_data = tf.placeholder(tf.float32, [None, 10])   # 占位符：样本目标变量

w = tf.Variable(tf.zeros([784, 10]))   # 网络权值矩阵
bias = tf.Variable(tf.zeros([10]))     # 网络阈值
y = tf.nn.softmax(tf.matmul(x_data, w) + bias)   # 网络输出

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data*tf.log(y), axis=1))  # 交叉熵（损失函数）
optimizer = tf.train.GradientDescentOptimizer(0.03)   # 梯度下降法优化器
train = optimizer.minimize(cross_entropy)   # 训练节点
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_data, axis=1)), dtype=tf.float32))  # 模型预测值与样本实际值比较的精度

sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 执行变量初始化操作
for i in range(20000):
    x_s, y_s = mnist.train.next_batch(100)
    if i%1000 == 0:
        acc_tr = sess.run(acc, feed_dict={x_data: x_s, y_data: y_s})
        print(i, '轮训练的精度', acc_tr)
    sess.run(train, feed_dict={x_data:x_s, y_data:y_s})   # 模型训练

acc_te = sess.run(acc, feed_dict={x_data:mnist.test.images, y_data:mnist.test.labels})  # 测试集精度
print('模型测试精度：', acc_te)
sess.close()




# # MNIST数据探索
# mnist.train.images[0]   # 训练集样本自变量（灰度值）
# mnist.train.labels[0]
#
# mnist.train.images.shape   # (55000, 784) 55000张图片，每张图片有784个像素值
# mnist.train.labels.shape   # 一共有55000张图片，每张图片都有一个标签
#
# mnist.test.images.shape        # 测试集样本自变量（灰度值）
# mnist.test.labels.shape
# # '还原某张图片'
# import matplotlib.pyplot as plt
# a = mnist.train.images[2]
# b = a.reshape([28, 28])*255
# plt.imshow(b)

# '''
# 交叉熵
# '''
# # real: 0   [1,  0,  0,  0,  0,0,0,0,0,0]
# # p1: 1     [0.4,0.5,0.1,0,  0,0,0,0,0,0]
# # p2: 2     [0,  0.4,0.5,0.1,0,0,0,0,0,0]
#
# import numpy as np
#
# real = np.array([1,  0,  0,  0,  0,0,0,0,0,0])
# p1 =   np.array([0.4,0.5,0.1,0,  0,0,0,0,0,0])
# p2 =   np.array([0,  0.4,0.5,0.1,0,0,0,0,0,0])
#
# -sum(real*np.log(p1+0.0000001))
# -sum(real*np.log(p2+0.0000001))











