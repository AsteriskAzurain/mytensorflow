from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
tf.reset_default_graph()   # 保存model前重置计算图
mnist = input_data.read_data_sets('tf_learn/savemymodel/MNIST_data', one_hot=True)   # 读取数据
x_data = tf.placeholder(tf.float32, [None, 784], name='input')  # 占位符：样本自变量
y_data = tf.placeholder(tf.float32, [None, 10])   # 占位符：样本目标变量

w = tf.Variable(tf.zeros([784, 10]))   # 网络权值矩阵
bias = tf.Variable(tf.zeros([10]))     # 网络阈值
y = tf.nn.softmax(tf.matmul(x_data, w) + bias, name='output')   # 网络输出

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data*tf.log(y), axis=1))  # 交叉熵（损失函数）
optimizer = tf.train.GradientDescentOptimizer(0.03)   # 梯度下降法优化器
train = optimizer.minimize(cross_entropy)   # 训练节点
# 模型预测值与样本实际值比较的精度
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_data, axis=1)), dtype=tf.float32))

saver = tf.train.Saver()  # 必须在会话关闭前保存模型

sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 执行变量初始化操作
for i in range(20001):
    x_s, y_s = mnist.train.next_batch(100)
    if i%1000 == 0:
        acc_tr = sess.run(acc, feed_dict={x_data: x_s, y_data: y_s})
        print(i, '轮训练的精度', acc_tr)
    sess.run(train, feed_dict={x_data:x_s, y_data:y_s})   # 模型训练

acc_te = sess.run(acc, feed_dict={x_data:mnist.test.images, y_data:mnist.test.labels})  # 测试集精度
print('模型测试精度：', acc_te)
saver.save(sess, 'tf_learn/savemymodel/model/softmax_model_new')
sess.close()
