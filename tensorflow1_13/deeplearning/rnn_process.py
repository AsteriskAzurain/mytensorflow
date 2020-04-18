import tensorflow as tf
import cv2
import numpy as np

img = np.float32(cv2.imread('deeplearning/0.jpg')[:, :, 0] / 255)  # 读取示例图片并进行归一化处理
#  (28, 28) → (1, 28, 28) 指的是一张28*28的图片
# -1表示我懒得计算该填什么数字，由python通过原array和已给维数(28*28)推测出来
img = img.reshape([-1, 28, 28])  # 转换图片维度
# 图片被分割为28个向量(1,28)作为最初的输入
# 所以模型共有28个rnn_cell
# h1:(0,28), h2:(0:1,28), h3:(0:2,28) ... h28(全部,28)
# h1到h28的过程相当于从上到下按行扫描了一遍图片
# 每个cell有100个神经元(num_units=100)
'''
假设在我们的训练数据中，每一个样本 x 是 28*28 维的一个矩阵，
那么将这个样本的每一行当成一个输入，通过28个时间步骤展开，
在每一个单元(run_cell)，我们输入一行维度为28的向量,
那么，对每一个cell，参数 num_units=100的话，就是每一个cell的输出为 100*1 的向量，
在展开的网络维度来看，如下图所示，对于每一个输入28维的向量，cell都把它映射到100维的维度，
在下一个cell时，
cell会接收上一个100维的输出，和新的28维的输入，
处理之后再映射成一个新的100维的向量输出，
就这么一直处理下去，直到网络中最后一个cell，输出一个100维的向量。
'''
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(100)  # RNN隐层神经元
# tf封装好的rnn复杂的计算过程 ↓
output, states = tf.nn.dynamic_rnn(rnn_cell, img, dtype=tf.float32)  # RNN网络传输过程
# output是隐层的输出

sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 执行变量初始化操作
res = sess.run(output)  # 执行RNN传输计算
# 输出res.shape=(1, 28, 100)
# 输入28对应输出28 隐层100 => 输出28*100
sess.close()

