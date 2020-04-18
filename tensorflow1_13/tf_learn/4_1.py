from sklearn.datasets import load_iris
import tensorflow as tf
from sklearn.model_selection import train_test_split

data, label = load_iris(True)  # 导入数据
'''原label
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ......
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
'''

with tf.Session() as sess:  # 将样本标签转为独热编码的形式
    label = sess.run(tf.one_hot(label, 3))  # 使用tf的onehot 3代表需要转化成的编码长度([1., 0., 0.])
'''独热形式的label
array([[1., 0., 0.],
       [1., 0., 0.],
        ......
       [0., 0., 1.],
       [0., 0., 1.]], dtype=float32)
'''
# 将数据集拆成训练集和测试集
data_tr, data_te, label_tr, label_te = train_test_split(data, label, test_size=0.2)

'''
搭建神经网络（定义计算）
'''
x_data = tf.placeholder(tf.float32, [None, 4])  # 占位符：网络的输入
y_data = tf.placeholder(tf.float32, [None, 3])  # 占位符：网络的目标输出

w0 = tf.Variable(tf.zeros([4, 6]))  # 隐层神经元的权值矩阵
w1 = tf.Variable(tf.zeros([6, 3]))  # 输出层神经元的权值矩阵
b0 = tf.Variable(tf.zeros([6]))  # 隐神经元的阈值
b1 = tf.Variable(tf.zeros([3]))  # 输出层神经元的阈值
'''
网络
输入  →  隐层  →  输出
n*4 4*6 n*6 6*3 n*3
隐层即为中间层,多层神经网络提高模型性能
'''
# 1 输入→隐层 需要输入x_data,x的参数w0,阈值b0
H = tf.sigmoid(tf.matmul(x_data, w0) + b0)  # 隐层神经元的输出
# sigmoid的输出在0和1之间,我们在二分类任务中,采用sigmoid的输出的是事件概率

# 2 隐层→输出 需要输入隐层的输出H,H的参数w1,阈值b1
y = tf.nn.softmax(tf.matmul(H, w1) + b1)  # 网络的输出
# softmax,此处作为激活函数,把一些输入映射为0-1之间的实数,并且归一化保证和为1,因此多分类的概率之和也刚好为1

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), axis=1))  # 交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.35)
train = optimizer.minimize(cross_entropy)  # 训练节点
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_data, 1), tf.argmax(y, 1)), dtype=tf.float32))  # 精度

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    # 只要run train 整个模型都会运作
    sess.run(train, feed_dict={x_data: data_tr, y_data: label_tr})
    acc_tr = sess.run(acc, feed_dict={x_data: data_tr, y_data: label_tr})
    print(acc_tr)
# output: 0.98333335 ←训练精度

acc_te = sess.run(acc, feed_dict={x_data: data_te, y_data: label_te})
print('测试样本的精度：', acc_te)
sess.close()
# output: 测试样本的精度： 0.9
