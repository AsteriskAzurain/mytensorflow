import tensorflow as tf

# data preparation
num_epochs = 200
batch_size = 128
learning_rate = 0.001
num_classes = 46
total_words = 40000
max_news_words = 80
embedding_len = 200

# 解决ValueError: Object arrays cannot be loaded when allow_pickle=False
# 原因: numpy版本的问题,numpy的的版本和keras没有完全兼容
# old = np.load
# np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# 通过tf.keras.datasets.reuters接口加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=total_words)

# 每句保留80个词 不足的补0
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_news_words)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_news_words)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# output:(8982, 80) (8982,) (2246, 80) (2246,)

# 将标签y转为one hot编码
# y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes, 'int64')
# y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes, 'int64')

# 定义数据集
# 将样本数据按照指定批次制作成tf.data.Dataset接口的数据集 并将不足一批次的剩余数据丢弃
# 批次(batch_size)设为128
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))


# ↑ tf.data.Dataset.from_tensor_slices
# ↑ 该函数是dataset核心函数之一，它的作用是把给定的元组、列表和张量等数据进行特征切片
# ↑ 元素逐个转换为Tensor对象然后依次放入Dataset中


def preprocess(x, y):  # 将标签 y 进行了 one hot 编码
    y = tf.one_hot(y, depth=num_classes)
    return x, y


# batch 处理数据的时候让 drop_remainder = True, 这样可以丢弃掉最后一个数量不足 batch size 的 batch.
# shuffle and batch dataset and drop the last batch shorter than batch_size
ds_train = ds_train.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder=True)
ds_test = ds_test.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder=True)


# output: <BatchDataset shapes: ((128, 80), (128, 46)), types: (tf.int32, tf.float32)>

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# output:(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# 使用动态路由(Dynamic Routing)聚合信息

# 1 将data(x)转换为vendor

# 2 使用rnn计算
def build_model(total_words, embedding_len, rnn_units, batch_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_news_words),
        tf.keras.layers.LSTM(units=rnn_units, return_sequences=True),
        tf.keras.layers.LSTM(units=rnn_units, return_sequences=False)
        # tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()
    return model


model = build_model(total_words=total_words, embedding_len=embedding_len, rnn_units=100, batch_size=batch_size)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)

# 未经训练过的预测结果
# for input_example_batch, target_example_batch in ds_train.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape)
#
for ds_input, ds_target in ds_train:
    ds_predict = model(ds_input)
    print(ds_predict)
    break

ds_train = ds_train.repeat()
it = ds_train.__iter__()
for i in range(70):
    ds_input, ds_target = it.next()
    ds_predict = model(ds_input)


# 3 rnn输出 -全连接层-> 动态路由中的uhat
def shared_routing_uhat(caps,  # 输入 shape(b_sz, maxlen, caps_dim)
                        out_caps_num):  # 输出胶囊个数

    batch_size, maxlen = tf.shape(caps)[0], tf.shape(caps)[1]  # 获取批次和长度
    out_caps_dim = caps.get_shape()[-1]  # 输出胶囊维度
    print(batch_size,maxlen,out_caps_num,out_caps_dim)
    net = tf.keras.layers.Dense(out_caps_num * out_caps_dim, activation=tf.nn.tanh)
    caps_uhat = net(caps)
    caps_uhat = tf.reshape(caps_uhat, shape=[batch_size, maxlen, out_caps_num, out_caps_dim])

    return caps_uhat  # 输出batch_size, maxlen, out_caps_num, out_caps_dim
