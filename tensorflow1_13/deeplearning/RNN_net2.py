import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)  # 读取数据

learning_rate = 0.001  # 学习速率
train_step = 10000  # 最大训练步长
batch_size = 1280  # 分支样本数
display_step = 100  # 打印间隔

frame_size = 28  # 向量长度
sequence_length = 28  # 向量个数
hidden_num = 100  # 隐层神经元个数
n_classes = 10  # 类别数

tf.reset_default_graph()  # 重置计算图
# 定义输入,输出
x = tf.placeholder(tf.float32, [None, sequence_length * frame_size])  # 占位符：模型输入
y = tf.placeholder(tf.float32, [None, n_classes])  # 占位符：目标输出

weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes]))  # 初始化输出层神经元权值
# 从截断的正态分布中输出随机值，生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
bias = tf.Variable(tf.zeros(shape=[n_classes]))  # 初始化输出层神经元阈值


# 定义RNN网络
def RNN(x, weights, bias):
    """
    :param x: RNN接收的输入
    :param weights: 输出层神经元权值
    :param bias: 输出层神经元阈值
    :return: 网络输出值（模型预测值）
    """
    x = tf.reshape(x, shape=[-1, sequence_length, frame_size])
    # 先把输入转换为dynamic_rnn接受的形状：batch_size,sequence_length,frame_size这样子的
    # reshape -1 代表自适应，这里按照图像每一列的长度为reshape后的列长度
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    # 生成有hidden_num个隐层神经元的RNN网络,rnn_cell.output_size等于隐层个数，state_size也是等于隐层个数，但是对于LSTM单元来说这两个size又是不一样的。
    # 这是一个深度RNN网络,对于每一个长度为sequence_length的序列[x1,x2,x3,...,]的每一个xi,都会在深度方向跑一遍RNN,每一个都会被这hidden_num个隐层单元处理。

    output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)  # 循环神经网络操作
    # 此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor
    return tf.nn.softmax(tf.matmul(output[:, -1, :], weights) + bias, 1)  # 从隐层到输出层传递
    # 我们取出每个样本的最后一个序列（最后一行）output[:,-1,:],它的形状为[batch_size,rnn_cell.output_size]也就是:[batch_size,hidden_num]所以它可以和weights相乘。这就是2.5中weights的形状初始化为[hidden_num,n_classes]的原因。然后再经softmax归一化。


predy = RNN(x, weights, bias)  # 模型输出节点
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=y))  # 交叉熵
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)  # 训练节点

correct_pred = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))  # 预测值与实际值比较
accuracy = tf.reduce_mean(tf.to_float(correct_pred))  # 输出精度

sess = tf.Session()  # 启动会话
sess.run(tf.global_variables_initializer())  # 执行变量初始化操作
step = 1
while step <= train_step:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # batch_x=tf.reshape(batch_x,shape=[batch_size,sequence_length,frame_size])
    loss, _ = sess.run([cross_entropy, train], feed_dict={x: batch_x, y: batch_y})
    if step % display_step == 0:
        acc, loss = sess.run([accuracy, cross_entropy],
                             feed_dict={x: mnist.test.images, y: mnist.test.labels})  # 将训练样本放入模型
        print(step, acc, loss)

    step += 1
