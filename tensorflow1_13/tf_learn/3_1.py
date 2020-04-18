from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 使用tf的input_data模块
mnist = input_data.read_data_sets('tf_learn/MNIST_data', one_hot=True)  # 读取数据
# 使用placeholder 不设具体值 便于后续存放样本数据
#       def placeholder(dtype数据类型, shape=None维度, name=None)
x_data = tf.placeholder(tf.float32, [None, 784])  # 占位符：样本自变量
y_data = tf.placeholder(tf.float32, [None, 10])  # 占位符：样本目标变量
# 等真的放入数据 才能进行计算

# # MNIST数据探索
# 导入的是一个文件夹 里面是打包好的数据集
# 含有训练集和测试集
# mnist.train.images[0]   # 训练集样本自变量（灰度值）
# mnist.train.labels[0]
#
# mnist.train.images.shape   # (55000, 784) 55000张图片，每张图片有784个像素值 784=28^2 每个像素格取值0-255(灰度)
# mnist.train.labels.shape   # (55000, )一共有55000张图片，每张图片都有一个标签
#                              (55000, 10)读取数据时 设置独热"one_hot=True" 改为10个标签
# mnist.test.images.shape        # 测试集样本自变量（灰度值）
# mnist.test.labels.shape

# # '还原某张图片'
# import matplotlib.pyplot as plt
# a = mnist.train.images[2]
# b = a.reshape([28, 28])*255
# plt.imshow(b)

# '''
# 交叉熵 cross-entropy
# '''
# 本身 作为标签 0,1,2...是没有大小可言的
# 要衡量模型的准确度就不能使用0 1 2本身的数值大小
# 而需要以下的表现形式(概率分布)来衡量模型的判断对错以及准确性
# 标签一共有十个(0~9) 模型对每个标签预测可能性(0~1) 10个标签中可能性最大的即为预测结果
# # real: 0   [1,  0,  0,  0,  0,0,0,0,0,0]
# # p1: 1     [0.4,0.5,0.1,0,  0,0,0,0,0,0]
# # p2: 2     [0,  0.4,0.5,0.1,0,0,0,0,0,0]
# 真实值为0 p1预测标签40%可能为0 p2预测标签不可能为0
# 显然 模型1比模型2更为准确
#
# import numpy as np
#
# real = np.array([1,  0,  0,  0,  0,0,0,0,0,0])
# p1 =   np.array([0.4,0.5,0.1,0,  0,0,0,0,0,0])
# p2 =   np.array([0,  0.4,0.5,0.1,0,0,0,0,0,0])
#
# ↓ 交叉熵的公式 通过交叉熵来衡量模型的准确性
# -sum(real*np.log(p1+0.0000001))
# -sum(real*np.log(p2+0.0000001))
# 因为log(0)=-∞ 所以给0值加上一个可以忽略的0.0000001

w = tf.Variable(tf.zeros([784, 10]))  # 网络权值矩阵
bias = tf.Variable(tf.zeros([10]))  # 网络阈值
y = tf.nn.softmax(tf.matmul(x_data, w) + bias)  # 网络输出

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), axis=1))  # 交叉熵（损失函数） axis=1为按行求和 否则是全部求和
optimizer = tf.train.GradientDescentOptimizer(0.03)  # 梯度下降法优化器
train = optimizer.minimize(cross_entropy)  # 训练节点
acc = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_data, axis=1)), dtype=tf.float32))  # 模型预测值与样本实际值比较的精度
# cast 将结果转换为float
# equal 比较两个tensor对象
# argmax axis=1 按行返回最大值

sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 执行变量初始化操作
for i in range(20001):  # 通过提升训练的轮数来增大精度
    x_s, y_s = mnist.train.next_batch(100)
    # next_batch 随机的从train中取出一小批(100个)数据
    if i % 1000 == 0:
        acc_tr = sess.run(acc, feed_dict={x_data: x_s, y_data: y_s})
        print(i, '轮训练的精度', acc_tr)
    sess.run(train, feed_dict={x_data: x_s, y_data: y_s})  # 模型训练
    # fees_dict的值是dict key是placeholder value是喂入的值

# test数据 得到测试精度
# 由于placeholder预留了数据接口 所以可以方便的把test数据喂入
acc_te = sess.run(acc, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels})  # 测试集精度
print('模型测试精度：', acc_te)

sess.close()

'''
output:
0 轮训练的精度 0.23
1000 轮训练的精度 0.9
2000 轮训练的精度 0.94
3000 轮训练的精度 0.91
4000 轮训练的精度 0.92
5000 轮训练的精度 0.93
6000 轮训练的精度 0.9
7000 轮训练的精度 0.94
8000 轮训练的精度 0.92
9000 轮训练的精度 0.95
10000 轮训练的精度 0.93
11000 轮训练的精度 0.9
12000 轮训练的精度 0.9
13000 轮训练的精度 0.94
14000 轮训练的精度 0.9
15000 轮训练的精度 0.85
16000 轮训练的精度 0.9
17000 轮训练的精度 0.94
18000 轮训练的精度 0.94
19000 轮训练的精度 0.87
20000 轮训练的精度 0.96
模型测试精度： 0.922
'''
