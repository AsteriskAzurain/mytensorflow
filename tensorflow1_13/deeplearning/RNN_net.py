from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('deeplearning/MNIST_data', one_hot=True)   # 读取mnist数据集

learning_rate = 0.001    # 学习速率
train_step = 10000       # 最大训练轮数
batch_size = 500         # 每批训练样本个数
displayer_step = 100     # 间隔打印轮数

frame_size = 28          # 每个输入向量中的元素个数
sequence_num = 28        # 输入向量的个数
hidden_size = 100        # RNN隐层神经元个数
n_class = 10             # 样本类别

tf.reset_default_graph()   # 重置计算图

x_data = tf.placeholder(tf.float32, [None, frame_size*sequence_num])   # 占位符：模型输入
y_data = tf.placeholder(tf.float32, [None, n_class])                   # 占位符：模型目标输出

weights = tf.Variable(tf.truncated_normal(shape=[hidden_size, n_class]))    # 输出层神经元权值
bias = tf.Variable(tf.zeros(shape=[n_class]))                               # 输出层神经元阈值

x = tf.reshape(x_data, [-1, sequence_num, frame_size])              # 改变样本外观形状
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(100)                         # RNN隐层神经元
output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)   # RNN传输过程
# 最后一个隐层的输出: output[:, -1, :], "-1"是最后一个的index
y = tf.nn.softmax(tf.matmul(output[:, -1, :], weights) + bias)      # 整个网络的输出值
'''
>>> print(output.shape,y.shape)
output: (?, 28, 100) (?, 10)
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y))   # 交叉熵
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)                              # 优化器,训练节点
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))))                 # 精度
# tf.argmax(y, 1) 即tf.argmax(input=y, axis=1) 按行取最大值

sess = tf.Session()                           # 启动会话
sess.run(tf.global_variables_initializer())   # 执行变量初始化操作
step = 1
while step <= train_step:
    x_s, y_s = mnist.train.next_batch(batch_size)     # 获取训练样本
    loss, _ = sess.run([cross_entropy, train], feed_dict={x_data: x_s, y_data: y_s})   # 模型训练
    if step % displayer_step == 0:
        acc_tr, loss = sess.run([acc, cross_entropy], feed_dict={x_data: x_s, y_data: y_s})   # 模型训练精度
        print('第', step, '次训练：', '训练精度: ', acc_tr, '交叉熵：', loss)
    step += 1
'''output
第 9800 次训练： 训练精度:  0.984 交叉熵： 1.477399
第 9900 次训练： 训练精度:  0.988 交叉熵： 1.473029
第 10000 次训练： 训练精度:  0.988 交叉熵： 1.4776862
'''

acc_te = sess.run(acc, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels})   # 模型测试精度
print('模型在测试集上的预测精度：', acc_te)
# output: 模型在测试集上的预测精度： 0.9742
sess.close()
















