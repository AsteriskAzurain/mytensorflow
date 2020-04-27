# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# 定义参数
num_words = 20000
maxlen = 80

# 解决ValueError: Object arrays cannot be loaded when allow_pickle=False
# 原因: numpy版本的问题,numpy的的版本和keras没有完全兼容
old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# 加载数据  自路透社的11,228条新闻，分为了46个主题
print('Loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(path='./reuters.npz', num_words=num_words)
print(len(x_train), 'train sequences')  # output:8982 train sequences
print(len(x_test), 'test sequences')  # output:2246 test sequences
print(x_train[0])
print(y_train[:10])

'''
# 借鉴
total_words = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=total_words)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 导入本地的数据
data = np.load("data/reuters.npz", allow_pickle=True)
data.files
from sklearn.model_selection import train_test_split

data_x = data['x']
data_y = data['y']
train_X, test_X, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=5)

print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
# 导入结束
'''

# word_index = tf.keras.datasets.reuters.get_word_index('./reuters_word_index.json')# 单词--下标 对应字典
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])# 下标-单词对应字典
#
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# print(decoded_newswire)


# 数据对齐
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
print('Pad sequences x_train shape:', x_train.shape)
# output: Pad sequences x_train shape: (8982, 80)

leng = np.count_nonzero(x_train, axis=1)  # 获取长度
print(leng[:3])  # [80 56 80]

tf.reset_default_graph()

BATCH_SIZE = 128  # 批次
# 定义数据集
dataset = tf.data.Dataset.from_tensor_slices(((x_train, leng), y_train)).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def mkMask(input_tensor, maxLen):  # 计算变长RNN的掩码
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


# 定义函数，将输入转化成uhat
def shared_routing_uhat(caps,  # 输入 shape(b_sz, maxlen, caps_dim)
                        out_caps_num,  # 输出胶囊个数
                        out_caps_dim, scope=None):  # 输出胶囊维度

    batch_size, maxlen = tf.shape(caps)[0], tf.shape(caps)[1]  # 获取批次和长度

    with tf.variable_scope(scope or 'shared_routing_uhat'):  # 转成uhat
        caps_uhat = tf.layers.dense(caps, out_caps_num * out_caps_dim, activation=tf.tanh)
        caps_uhat = tf.reshape(caps_uhat, shape=[batch_size, maxlen, out_caps_num, out_caps_dim])

    return caps_uhat  # 输出batch_size, maxlen, out_caps_num, out_caps_dim


def masked_routing_iter(caps_uhat, seqLen, iter_num):  # 动态路由计算

    assert iter_num > 0
    batch_size, maxlen = tf.shape(caps_uhat)[0], tf.shape(caps_uhat)[1]  # 获取批次和长度
    out_caps_num = int(caps_uhat.get_shape()[2])
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
    mask = mkMask(seqLen, maxlen)  # shape(batch_size, maxlen)
    floatmask = tf.cast(tf.expand_dims(mask, axis=-1), dtype=tf.float32)  # shape(batch_size, maxlen, 1)

    # shape(b_sz, maxlen, out_caps_num)
    B = tf.zeros([batch_size, maxlen, out_caps_num], dtype=tf.float32)
    for i in range(iter_num):
        C = tf.nn.softmax(B, axis=2)  # shape(batch_size, maxlen, out_caps_num)
        C = tf.expand_dims(C * floatmask, axis=-1)  # shape(batch_size, maxlen, out_caps_num, 1)
        weighted_uhat = C * caps_uhat  # shape(batch_size, maxlen, out_caps_num, out_caps_dim)

        S = tf.reduce_sum(weighted_uhat, axis=1)  # shape(batch_size, out_caps_num, out_caps_dim)

        V = _squash(S, axes=[2])  # shape(batch_size, out_caps_num, out_caps_dim)
        V = tf.expand_dims(V, axis=1)  # shape(batch_size, 1, out_caps_num, out_caps_dim)
        B = tf.reduce_sum(caps_uhat * V, axis=-1) + B  # shape(batch_size, maxlen, out_caps_num)

    V_ret = tf.squeeze(V, axis=[1])  # shape(batch_size, out_caps_num, out_caps_dim)
    S_ret = S
    return V_ret, S_ret


def _squash(in_caps, axes):  # 定义_squash激活函数
    _EPSILON = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSILON)
    vec_squashed = scalar_factor * in_caps  # element-wise
    return vec_squashed


# 定义函数，使用动态路由对RNN结果信息聚合
def routing_masked(in_x, xLen, out_caps_dim, out_caps_num, iter_num=3,
                   dropout=None, is_train=False, scope=None):
    assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None
    b_sz = tf.shape(in_x)[0]
    with tf.variable_scope(scope or 'routing'):
        caps_uhat = shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope='rnn_caps_uhat')
        attn_ctx, S = masked_routing_iter(caps_uhat, xLen, iter_num)
        attn_ctx = tf.reshape(attn_ctx, shape=[b_sz, out_caps_num * out_caps_dim])
        if dropout is not None:
            attn_ctx = tf.layers.dropout(attn_ctx, rate=dropout, training=is_train)
    return attn_ctx


x = tf.placeholder("float", [None, maxlen])  # 定义输入占位符
x_len = tf.placeholder(tf.int32, [None, ])  # 定义输入序列长度占位符
y = tf.placeholder(tf.int32, [None, ])  # 定义输入分类标签占位符

nb_features = 128  # 词嵌入维度
embeddings = tf.keras.layers.Embedding(num_words, nb_features)(x)

# 定义带有IndyLSTMCell的RNN网络
hidden = [100, 50, 30]  # RNN单元个数
stacked_rnn = []
for i in range(3):
    cell = tf.contrib.rnn.IndyLSTMCell(hidden[i])
    stacked_rnn.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.8))
mcell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn)

rnnoutputs, _ = tf.nn.dynamic_rnn(mcell, embeddings, dtype=tf.float32)
out_caps_num = 5  # 定义输出的胶囊个数
n_classes = 46  # 分类个数

outputs = routing_masked(rnnoutputs, x_len, int(rnnoutputs.get_shape()[-1]), out_caps_num, iter_num=3)
print(outputs.get_shape())  # output:(?, 150)
pred = tf.layers.dense(outputs, n_classes, activation=tf.nn.relu)

# 定义优化器
learning_rate = 0.001
cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

iterator1 = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
one_element1 = iterator1.get_next()  # 获取一个元素
# one_element1:
# ( 1.tuple (<tf.Tensor 'IteratorGetNext:0' shape=(100, 80) dtype=int32>,
#           <tf.Tensor 'IteratorGetNext:1' shape=(100,) dtype=int64>),
#   2.<tf.Tensor 'IteratorGetNext:2' shape=(100,) dtype=int64>)

# 训练网络
with tf.Session() as sess:
    sess.run(iterator1.make_initializer(dataset))  # 初始化迭代器
    sess.run(tf.global_variables_initializer())
    EPOCHS = 20
    for ii in range(EPOCHS):
        alloss = []  # 数据集迭代两次
        while True:  # 通过for循环打印所有的数据
            try:
                inp, target = sess.run(one_element1)
                _, loss = sess.run([optimizer, cost], feed_dict={x: inp[0], x_len: inp[1], y: target})
                alloss.append(loss)

            except tf.errors.OutOfRangeError:
                print("step", ii + 1, ": loss=", np.mean(alloss))
                sess.run(iterator1.make_initializer(dataset))  # 从头再来一遍
                break
'''
output:
step 1 : loss= 2.4825401
step 2 : loss= 1.5677279
step 3 : loss= 1.1372739
step 4 : loss= 0.8962817
step 5 : loss= 0.7359249
step 6 : loss= 0.61192006
step 7 : loss= 0.5228155
step 8 : loss= 0.45577016
step 9 : loss= 0.3954943
step 10 : loss= 0.34166512
step 11 : loss= 0.30633175
step 12 : loss= 0.2729049
step 13 : loss= 0.24777739
step 14 : loss= 0.22667328
step 15 : loss= 0.21350874
step 16 : loss= 0.19517484
step 17 : loss= 0.18156888
step 18 : loss= 0.1729429
step 19 : loss= 0.16100404
step 20 : loss= 0.15217435
'''
