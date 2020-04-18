import numpy as np
import tensorflow as tf

# numpy导入的数据都是float64 tensorflow用的是float32 不转型会报错
data = np.float32(np.load('tf_learn/line_fit_data.npy'))  # 导入100个样本数据
x_data = data[:, :2]   # 样本自变量
y_data = data[:, 2:]   # 样本实际值

'''
定义计算（计算图）
'''
# y=ax1+bx2+c
w = tf.Variable(tf.zeros([2, 1]))  # 列向量(a,b) x1 x2的参数
bias = tf.Variable(tf.zeros([1]))  # c
# data(100*2) w(2*1) 相乘后(100*1)
y = tf.matmul(x_data, w) + bias   # 构造一个线性模型
# 使用均方误差
loss = tf.reduce_mean(tf.square(y_data - y))  # 定义损失函数

optimizer = tf.train.GradientDescentOptimizer(0.5)   # 构建梯度下降法优化器
train = optimizer.minimize(loss)   # 定义训练函数

'''
执行计算（会话中）
'''
# 启动会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())   # 初始化变量
# sess.run(y)
for i in range(100):
    print('第', i, '轮训练后的模型损失值：', sess.run(loss))
    sess.run(train)   # 开始训练
'''
output:
第 0 轮训练后的模型损失值： 0.20766991
第 1 轮训练后的模型损失值： 0.055225402
第 2 轮训练后的模型损失值： 0.01477379
...
第 98 轮训练后的模型损失值： 3.062836e-11
第 99 轮训练后的模型损失值： 2.7425253e-11
loss(误差)已经变得很小了
'''

# 得到现在的参数 最后的模型 ↓
sess.run([w, bias])   # y = 0.1*x1 + 0.2*x2 + 0.3
''' output:
    [array([[0.09999494],
        [0.19998254]], dtype=float32),
     array([0.30001214], dtype=float32)]
 '''

sess.close()


