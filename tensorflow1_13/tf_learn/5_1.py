from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# plt.switch_backend('agg')

data, label = load_iris(True)  # 导入数据

with tf.Session() as sess:  # 将样本标签转为独热编码的形式
    label = sess.run(tf.one_hot(label, 3))
global_step = tf.Variable(0, trainable=False)  # 动态变化的学习轮数
learning_rate = tf.train.exponential_decay(0.45, global_step, 10, 0.96)  # 动态学习速率
'''
global_step = tf.Variable(0, trainable=False)   
训练轮数，在训练过程中会发生变化，但它不属于被优化的变量
learning_rate = tf.train.exponential_decay(0.2, global_step, 100, 0.96)：以指数衰减的动态学习速率
0.2：初始学习速率
global_step：训练轮数，随着训练进行而动态变化
100：每训练100轮更新一次学习速率
0.96：衰减系数，即每100轮后学习速率乘以0.96
注：一般来说初始学习速率、衰减系数和衰减速度都是根据工程经验进行设置。
'''
data_tr, data_te, label_tr, label_te = train_test_split(data, label, test_size=0.2)  # 将数据集拆成训练集和测试集

'''
搭建神经网络（定义计算）
'''
x_data = tf.placeholder(tf.float32, [None, 4])  # 占位符：网络的输入
y_data = tf.placeholder(tf.float32, [None, 3])  # 占位符：网络的目标输出

w0 = tf.Variable(tf.zeros([4, 6]))  # 隐层神经元的权值矩阵
w1 = tf.Variable(tf.zeros([6, 3]))  # 输出层神经元的权值矩阵
b0 = tf.Variable(tf.zeros([6]))  # 隐神经元的阈值
b1 = tf.Variable(tf.zeros([3]))  # 输出层神经元的阈值

H = tf.sigmoid(tf.matmul(x_data, w0) + b0)  # 隐层神经元的输出
y = tf.nn.softmax(tf.matmul(H, w1) + b1)  # 网路的输出
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), axis=1))  # 交叉熵

# 梯度下降法优化器，传入的是动态学习速率
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(cross_entropy, global_step=global_step)  # 训练节点
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_data, 1), tf.argmax(y, 1)), dtype=tf.float32))  # 精度

sess = tf.Session()
sess.run(tf.global_variables_initializer())
Learning = []  # 记录学习率的变化
for i in range(1000):
    sess.run(train, feed_dict={x_data: data_tr, y_data: label_tr})  # 模型训练
    acc_tr = sess.run(acc, feed_dict={x_data: data_tr, y_data: label_tr})
    print(i, 'global_step:', sess.run(global_step), 'acc_tr', acc_tr)
    Learning.append(sess.run(learning_rate))
'''
output:
0 global_step: 1 acc_tr 0.36666667
999 global_step: 1000 acc_tr 0.975
'''
acc_te = sess.run(acc, feed_dict={x_data: data_te, y_data: label_te})
print('测试样本的精度：', acc_te)
sess.close()
# output: 测试样本的精度： 0.96666664

plt.plot(Learning)
plt.show()
# 输出结果在learningrate.png
